"""
tests/test_consolidation.py

Tests for the consolidation engine — state management, pattern parsing,
LLM call wiring, skill store upserts, and full run_consolidation() pipeline.
All LLM calls and filesystem paths are mocked or redirected to tmp_path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.background.consolidation import (
    BATCH_SIZE,
    MIN_EVENTS,
    ConsolidationState,
    ExtractedPattern,
    _format_events_for_llm,
    parse_patterns_from_llm,
    run_consolidation,
)
from agent.llm.lm_studio import LLMResponse
from agent.llm.validator import ValidationResult
from agent.memory.episodic import EpisodicEvent, EventType, write_event


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_event(
    summary: str = "Wrote FastAPI endpoint",
    project: str | None = "agrivision",
    event_type: EventType = EventType.CODE_WRITE,
    ts: datetime | None = None,
) -> EpisodicEvent:
    return EpisodicEvent(
        event_type=event_type,
        summary=summary,
        project=project,
        timestamp=ts or datetime.now(timezone.utc),
    )


def make_router(content: str = "") -> MagicMock:
    router = MagicMock()
    router.complete = AsyncMock(
        return_value=LLMResponse(content=content, model="test-model")
    )
    return router


def make_engine() -> MagicMock:
    engine = MagicMock()
    engine.build = MagicMock(return_value=[])
    return engine


def make_validator(success: bool = True) -> MagicMock:
    validator = MagicMock()
    validator.validate_json = AsyncMock(
        return_value=ValidationResult(
            success=success,
            value={} if success else None,
            error="" if success else "parse error",
            attempts=1,
        )
    )
    return validator


# ── ConsolidationState ────────────────────────────────────────────────────────

def test_state_defaults() -> None:
    s = ConsolidationState()
    assert s.total_runs == 0
    assert s.last_run_utc is None
    assert s.last_processed_event_id is None


def test_state_mark_complete() -> None:
    s = ConsolidationState()
    updated = s.mark_complete(last_event_id="evt-001", patterns_extracted=5)
    assert updated.total_runs == 1
    assert updated.total_patterns_extracted == 5
    assert updated.last_processed_event_id == "evt-001"
    assert updated.last_error is None
    assert updated.last_run_utc is not None


def test_state_mark_complete_accumulates() -> None:
    s = ConsolidationState()
    s = s.mark_complete("evt-001", 3)
    s = s.mark_complete("evt-002", 7)
    assert s.total_runs == 2
    assert s.total_patterns_extracted == 10


def test_state_mark_error() -> None:
    s = ConsolidationState()
    updated = s.mark_error("LLM timeout")
    assert updated.last_error == "LLM timeout"
    assert updated.last_run_utc is not None
    assert updated.total_runs == 0  # error doesn't increment runs


def test_state_returns_new_instance() -> None:
    s = ConsolidationState()
    updated = s.mark_complete("evt-001", 1)
    assert s is not updated
    assert s.total_runs == 0


def test_state_round_trips_json() -> None:
    s = ConsolidationState(
        total_runs=3,
        total_patterns_extracted=12,
        last_processed_event_id="evt-abc",
        last_run_utc=datetime(2026, 3, 14, 2, 0, 0, tzinfo=timezone.utc),
    )
    raw = s.model_dump_json()
    restored = ConsolidationState.model_validate_json(raw)
    assert restored.total_runs == 3
    assert restored.total_patterns_extracted == 12
    assert restored.last_processed_event_id == "evt-abc"


# ── ExtractedPattern ──────────────────────────────────────────────────────────

def test_extracted_pattern_normalises_task_type() -> None:
    p = ExtractedPattern(task_type="Create API Endpoint", pattern="use FastAPI", confidence=0.8)
    assert p.task_type == "create_api_endpoint"


def test_extracted_pattern_clamps_confidence_above_1() -> None:
    p = ExtractedPattern(task_type="task", pattern="pattern", confidence=1.5)
    assert p.confidence == 1.0


def test_extracted_pattern_clamps_confidence_below_0() -> None:
    p = ExtractedPattern(task_type="task", pattern="pattern", confidence=-0.1)
    assert p.confidence == 0.0


def test_extracted_pattern_defaults_confidence() -> None:
    p = ExtractedPattern(task_type="task", pattern="pattern", confidence="invalid")  # type: ignore
    assert p.confidence == 0.5


# ── _format_events_for_llm ────────────────────────────────────────────────────

def test_format_events_includes_summary() -> None:
    events = [make_event("Wrote FastAPI route", project="agrivision")]
    text = _format_events_for_llm(events)
    assert "Wrote FastAPI route" in text


def test_format_events_includes_project() -> None:
    events = [make_event(project="agrivision")]
    text = _format_events_for_llm(events)
    assert "agrivision" in text


def test_format_events_includes_event_type() -> None:
    events = [make_event(event_type=EventType.CODE_WRITE)]
    text = _format_events_for_llm(events)
    assert "code.write" in text


def test_format_events_multiple_events() -> None:
    events = [make_event(f"event {i}") for i in range(5)]
    text = _format_events_for_llm(events)
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == 5


def test_format_events_empty_list() -> None:
    assert _format_events_for_llm([]) == ""


# ── parse_patterns_from_llm ───────────────────────────────────────────────────

def test_parse_patterns_envelope_format() -> None:
    raw = json.dumps({
        "patterns": [
            {"task_type": "create_api", "pattern": "use FastAPI", "confidence": 0.9},
        ]
    })
    patterns = parse_patterns_from_llm(raw)
    assert len(patterns) == 1
    assert patterns[0].task_type == "create_api"


def test_parse_patterns_bare_list_format() -> None:
    raw = json.dumps([
        {"task_type": "export_asset", "pattern": "use GLB", "confidence": 0.8},
    ])
    patterns = parse_patterns_from_llm(raw)
    assert len(patterns) == 1


def test_parse_patterns_strips_code_fences() -> None:
    raw = '```json\n{"patterns": [{"task_type": "t", "pattern": "p", "confidence": 0.7}]}\n```'
    patterns = parse_patterns_from_llm(raw)
    assert len(patterns) == 1


def test_parse_patterns_multiple() -> None:
    raw = json.dumps({"patterns": [
        {"task_type": "api", "pattern": "FastAPI", "confidence": 0.9},
        {"task_type": "3d", "pattern": "GLB export", "confidence": 0.7},
        {"task_type": "test", "pattern": "pytest", "confidence": 0.85},
    ]})
    patterns = parse_patterns_from_llm(raw)
    assert len(patterns) == 3


def test_parse_patterns_invalid_json_returns_empty() -> None:
    patterns = parse_patterns_from_llm("not json at all")
    assert patterns == []


def test_parse_patterns_missing_required_field_skipped() -> None:
    raw = json.dumps({"patterns": [
        {"task_type": "good_task", "pattern": "valid pattern", "confidence": 0.8},
        {"pattern": "missing task_type field"},  # malformed — skipped
    ]})
    patterns = parse_patterns_from_llm(raw)
    assert len(patterns) == 1
    assert patterns[0].task_type == "good_task"


def test_parse_patterns_empty_list() -> None:
    raw = json.dumps({"patterns": []})
    patterns = parse_patterns_from_llm(raw)
    assert patterns == []


def test_parse_patterns_source_event_ids() -> None:
    raw = json.dumps({"patterns": [
        {
            "task_type": "debug",
            "pattern": "add logging",
            "confidence": 0.75,
            "source_event_ids": ["evt-001", "evt-002"],
        }
    ]})
    patterns = parse_patterns_from_llm(raw)
    assert patterns[0].source_event_ids == ["evt-001", "evt-002"]


# ── run_consolidation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_consolidation_skips_with_few_events(tmp_path: Path) -> None:
    """Fewer than MIN_EVENTS → skip LLM call entirely."""
    router = make_router()
    engine = make_engine()
    validator = make_validator()

    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state"), \
         patch("agent.background.consolidation.list_events", return_value=[make_event()]):

        state = await run_consolidation(router, engine, validator)

    router.complete.assert_not_called()
    assert state.total_runs == 1  # mark_complete still called


@pytest.mark.asyncio
async def test_run_consolidation_calls_llm_with_enough_events(tmp_path: Path) -> None:
    events = [make_event(f"event {i}") for i in range(MIN_EVENTS + 2)]
    llm_response = json.dumps({"patterns": [
        {"task_type": "code_task", "pattern": "use async", "confidence": 0.8}
    ]})
    router = make_router(llm_response)
    engine = make_engine()
    validator = make_validator()

    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state"), \
         patch("agent.background.consolidation.list_events", return_value=events), \
         patch("agent.background.consolidation.write_event"), \
         patch("agent.background.consolidation.upsert_skill") as mock_upsert:

        state = await run_consolidation(router, engine, validator)

    router.complete.assert_called_once()
    mock_upsert.assert_called_once()
    assert state.total_patterns_extracted == 1


@pytest.mark.asyncio
async def test_run_consolidation_upserts_each_pattern(tmp_path: Path) -> None:
    events = [make_event(f"event {i}") for i in range(MIN_EVENTS + 1)]
    llm_response = json.dumps({"patterns": [
        {"task_type": "api_task", "pattern": "FastAPI", "confidence": 0.9},
        {"task_type": "3d_task",  "pattern": "GLB export", "confidence": 0.7},
    ]})
    router = make_router(llm_response)
    engine = make_engine()
    validator = make_validator()

    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state"), \
         patch("agent.background.consolidation.list_events", return_value=events), \
         patch("agent.background.consolidation.write_event"), \
         patch("agent.background.consolidation.upsert_skill") as mock_upsert:

        state = await run_consolidation(router, engine, validator)

    assert mock_upsert.call_count == 2
    assert state.total_patterns_extracted == 2


@pytest.mark.asyncio
async def test_run_consolidation_filters_consolidation_events() -> None:
    """Consolidation events must not be fed back into consolidation."""
    events = [
        make_event(f"real event {i}") for i in range(MIN_EVENTS + 1)
    ] + [
        EpisodicEvent(
            event_type=EventType.MEMORY_CONSOLIDATE,
            summary="previous consolidation run",
        )
    ]
    router = make_router(json.dumps({"patterns": []}))
    engine = make_engine()
    validator = make_validator()

    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state"), \
         patch("agent.background.consolidation.list_events", return_value=events), \
         patch("agent.background.consolidation.write_event"):

        await run_consolidation(router, engine, validator)

    # Engine.build was called — check the events text doesn't include consolidation
    call_args = engine.build.call_args
    events_text = call_args[1]["variables"]["events"]
    assert "previous consolidation run" not in events_text


@pytest.mark.asyncio
async def test_run_consolidation_writes_episodic_event() -> None:
    events = [make_event(f"e {i}") for i in range(MIN_EVENTS + 1)]
    router = make_router(json.dumps({"patterns": []}))
    engine = make_engine()
    validator = make_validator()

    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state"), \
         patch("agent.background.consolidation.list_events", return_value=events), \
         patch("agent.background.consolidation.write_event") as mock_write, \
         patch("agent.background.consolidation.upsert_skill"):

        await run_consolidation(router, engine, validator)

    mock_write.assert_called_once()
    written_event = mock_write.call_args[0][0]
    assert written_event.event_type == EventType.MEMORY_CONSOLIDATE


@pytest.mark.asyncio
async def test_run_consolidation_saves_state() -> None:
    events = [make_event(f"e {i}") for i in range(MIN_EVENTS + 1)]
    router = make_router(json.dumps({"patterns": []}))
    engine = make_engine()
    validator = make_validator()

    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state") as mock_save, \
         patch("agent.background.consolidation.list_events", return_value=events), \
         patch("agent.background.consolidation.write_event"), \
         patch("agent.background.consolidation.upsert_skill"):

        await run_consolidation(router, engine, validator)

    mock_save.assert_called_once()
    saved_state: ConsolidationState = mock_save.call_args[0][0]
    assert saved_state.total_runs == 1


@pytest.mark.asyncio
async def test_run_consolidation_handles_llm_failure_gracefully() -> None:
    events = [make_event(f"e {i}") for i in range(MIN_EVENTS + 1)]
    router = MagicMock()
    router.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    engine = make_engine()
    validator = make_validator()

    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state"), \
         patch("agent.background.consolidation.list_events", return_value=events), \
         patch("agent.background.consolidation.write_event"), \
         patch("agent.background.consolidation.upsert_skill"):

        # Should not raise
        state = await run_consolidation(router, engine, validator)

    # Zero patterns extracted but run completed
    assert state.total_patterns_extracted == 0


@pytest.mark.asyncio
async def test_run_consolidation_handles_catastrophic_failure() -> None:
    """Even if list_events crashes, run_consolidation returns a state."""
    with patch("agent.background.consolidation._load_state", return_value=ConsolidationState()), \
         patch("agent.background.consolidation._save_state"), \
         patch("agent.background.consolidation.list_events", side_effect=OSError("disk error")):

        state = await run_consolidation(make_router(), make_engine(), make_validator())

    assert state.last_error is not None
    assert "disk error" in state.last_error