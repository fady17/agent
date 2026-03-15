"""
agent/background/consolidation.py

Consolidation engine — the mechanism by which the agent gets smarter.

Runs nightly via the scheduler. Reads recent episodic events,
sends them to the LLM using the CONSOLIDATE template, parses the
returned patterns, and upserts each pattern into the skill store.

Pipeline:
    1. Load consolidation state (last processed event ID + timestamp)
    2. List episodic events since last run
    3. If nothing new, skip
    4. Format events into a readable string for the LLM
    5. Call LLM with CONSOLIDATE template → JSON patterns array
    6. Validate and parse the patterns
    7. Upsert each pattern into the skill store
    8. Update consolidation state with new watermark

The consolidation state lives at:
    ~/.agent/consolidation/state.json

Pattern extraction is idempotent — running consolidation twice over
the same events will boost confidence on existing patterns rather than
creating duplicates (upsert_skill handles this).

Design decisions:
    - Consolidation runs on the LOCAL LLM (force_local=True).
      It's a batch job with no latency requirement. Cloud would add cost
      for every nightly run with no quality benefit over a capable local model.
    - Events are batched to stay within the LLM's context window.
      Default batch size is 50 events per consolidation call.
    - The full consolidation run is wrapped in try/except — a failure
      updates the error log but does not corrupt the state file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.memory.episodic import EpisodicEvent, EventType, list_events, write_event
from agent.memory.skills import upsert_skill

log = get_logger(__name__)

# Maximum events per LLM consolidation call
BATCH_SIZE = 50

# Minimum events required to bother calling the LLM
MIN_EVENTS = 3


# ── ConsolidationState ────────────────────────────────────────────────────────

class ConsolidationState(BaseModel):
    """Persisted state that tracks the consolidation watermark."""

    last_run_utc:             datetime | None = None
    last_processed_event_id:  str | None      = None
    total_runs:                int             = 0
    total_patterns_extracted:  int             = 0
    last_error:               str | None      = None

    @field_validator("last_run_utc", mode="before")
    @classmethod
    def parse_dt(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(str(v)).replace(tzinfo=timezone.utc)

    def mark_complete(
        self,
        last_event_id: str | None,
        patterns_extracted: int,
    ) -> "ConsolidationState":
        return self.model_copy(update={
            "last_run_utc":            datetime.now(timezone.utc),
            "last_processed_event_id": last_event_id,
            "total_runs":              self.total_runs + 1,
            "total_patterns_extracted": self.total_patterns_extracted + patterns_extracted,
            "last_error":              None,
        })

    def mark_error(self, error: str) -> "ConsolidationState":
        return self.model_copy(update={
            "last_run_utc": datetime.now(timezone.utc),
            "last_error":   error,
        })


def _state_path() -> Path:
    return cfg.consolidation_dir / "state.json"


def _load_state() -> ConsolidationState:
    path = _state_path()
    if not path.exists():
        return ConsolidationState()
    try:
        return ConsolidationState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("consolidation.state_load_failed", error=str(exc))
        return ConsolidationState()


def _save_state(state: ConsolidationState) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


# ── ExtractedPattern ──────────────────────────────────────────────────────────

class ExtractedPattern(BaseModel):
    """One pattern returned by the LLM consolidation call."""

    task_type:        str
    pattern:          str
    confidence:       float = Field(ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list)

    @field_validator("task_type")
    @classmethod
    def normalise_task_type(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return 0.5


# ── Event formatting ──────────────────────────────────────────────────────────

def _format_events_for_llm(events: list[EpisodicEvent]) -> str:
    """
    Format a list of episodic events into a compact string for the LLM.
    Each line is: [type] project: summary
    """
    lines: list[str] = []
    for evt in events:
        project_tag = f" [{evt.project}]" if evt.project else ""
        ts = evt.timestamp.strftime("%Y-%m-%d")
        lines.append(f"- [{ts}] [{evt.event_type}]{project_tag}: {evt.summary}")
    return "\n".join(lines)


# ── Pattern parsing ───────────────────────────────────────────────────────────

def parse_patterns_from_llm(raw_content: str) -> list[ExtractedPattern]:
    """
    Parse the LLM's JSON response into a list of ExtractedPattern objects.
    Handles both {"patterns": [...]} and bare [...] formats.
    Silently drops malformed entries rather than failing the whole batch.
    """
    content = raw_content.strip()

    # Strip markdown fences if present
    for fence in ("```json", "```"):
        if content.startswith(fence):
            content = content[len(fence):]
            break
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.warning("consolidation.json_parse_failed", error=str(exc))
        return []

    # Unwrap {"patterns": [...]} envelope
    if isinstance(data, dict):
        data = data.get("patterns", [])

    if not isinstance(data, list):
        log.warning("consolidation.unexpected_shape", type=type(data).__name__)
        return []

    patterns: list[ExtractedPattern] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            pattern = ExtractedPattern.model_validate(item)
            patterns.append(pattern)
        except Exception as exc:
            log.warning("consolidation.pattern_parse_error", item=item, error=str(exc))

    return patterns


# ── Core consolidation logic ──────────────────────────────────────────────────

async def _consolidate_batch(
    events: list[EpisodicEvent],
    router: Any,
    engine: Any,
    validator: Any,
) -> list[ExtractedPattern]:
    """
    Send one batch of events to the LLM and return extracted patterns.
    Returns empty list on failure.
    """
    from agent.llm.prompt_engine import TemplateID

    events_text = _format_events_for_llm(events)

    messages = engine.build(
        TemplateID.CONSOLIDATE,
        variables={"events": events_text},
    )

    try:
        response = await router.complete(
            messages,
            force_local=True,
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as exc:
        log.error("consolidation.llm_call_failed", error=str(exc))
        return []

    validation = await validator.validate_json(response)
    if not validation.success:
        log.warning("consolidation.validation_failed", error=validation.error)
        # Try parsing raw content anyway — LLM may have returned valid JSON
        # that failed Pydantic validation for a minor reason
        return parse_patterns_from_llm(response.content)

    # validator returned parsed dict — re-parse into ExtractedPattern list
    raw_content = response.content
    return parse_patterns_from_llm(raw_content)


async def run_consolidation(
    router: Any | None = None,
    engine: Any | None = None,
    validator: Any | None = None,
) -> ConsolidationState:
    """
    Run one complete consolidation pass.

    Args:
        router:    LLMRouter (injected for testing; auto-created if None)
        engine:    PromptEngine (injected for testing; auto-created if None)
        validator: ResponseValidator (injected for testing; auto-created if None)

    Returns:
        Updated ConsolidationState after the run.
    """
    from agent.llm.prompt_engine import PromptEngine
    from agent.llm.router import LLMRouter
    from agent.llm.validator import ResponseValidator

    _router    = router    or LLMRouter()
    _engine    = engine    or PromptEngine()
    _validator = validator or ResponseValidator(max_retries=2)

    state = _load_state()
    log.info(
        "consolidation.start",
        last_run=state.last_run_utc.isoformat() if state.last_run_utc else "never",
        total_runs=state.total_runs,
    )

    try:
        # Fetch events since last run
        since = state.last_run_utc
        events = list_events(since=since, limit=BATCH_SIZE * 2)

        # Filter out consolidation events themselves (avoid self-referential loops)
        events = [
            e for e in events
            if e.event_type != EventType.MEMORY_CONSOLIDATE
        ]

        if len(events) < MIN_EVENTS:
            log.info(
                "consolidation.skipped",
                reason=f"only {len(events)} new events (min {MIN_EVENTS})",
            )
            state = state.mark_complete(
                last_event_id=state.last_processed_event_id,
                patterns_extracted=0,
            )
            _save_state(state)
            return state

        log.info("consolidation.processing", event_count=len(events))

        # Process in batches
        total_patterns = 0
        last_event_id  = state.last_processed_event_id

        for i in range(0, len(events), BATCH_SIZE):
            batch = events[i : i + BATCH_SIZE]
            patterns = await _consolidate_batch(batch, _router, _engine, _validator)

            for pattern in patterns:
                # Link pattern to the event IDs in this batch if not already set
                source_ids = pattern.source_event_ids or [e.id for e in batch[:5]]
                upsert_skill(
                    task_type=pattern.task_type,
                    pattern=pattern.pattern,
                    event_id=source_ids[0] if source_ids else None,
                )
                total_patterns += 1

            if batch:
                last_event_id = batch[-1].id

        # Write consolidation event to episodic store
        write_event(EpisodicEvent(
            event_type=EventType.MEMORY_CONSOLIDATE,
            summary=f"Consolidation run: {len(events)} events → {total_patterns} patterns",
            data={
                "events_processed": len(events),
                "patterns_extracted": total_patterns,
            },
        ))

        state = state.mark_complete(
            last_event_id=last_event_id,
            patterns_extracted=total_patterns,
        )
        _save_state(state)

        log.info(
            "consolidation.complete",
            events_processed=len(events),
            patterns_extracted=total_patterns,
        )
        return state

    except Exception as exc:
        log.error("consolidation.failed", error=str(exc))
        state = state.mark_error(str(exc))
        _save_state(state)
        return state