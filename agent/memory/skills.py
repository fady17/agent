"""
agent/memory/skills.py

Skill memory — the agent's learned patterns and preferences.

Each SkillRecord captures a single repeatable behaviour the agent has
observed the user perform. Records are stored as individual JSON files
under:

    ~/.agent/memory/skills/{task_type}.json

Key design decisions:
  - task_type is the primary key and the filename stem. One file per
    skill. Simple, grep-able, no index needed for lookup.
  - confidence is a float [0.0, 1.0]. It rises toward 1.0 each time the
    pattern is observed and decays slowly over time when unused.
    This means old irrelevant patterns naturally fade without manual cleanup.
  - source_event_ids links back to the episodic events that produced this
    pattern — full audit trail from skill back to raw observation.
  - pattern is a human-readable description of the behaviour. The agent
    uses this directly in prompts: "User prefers: {pattern}".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0

# How much confidence rises per new observation (additive, capped at 1.0)
CONFIDENCE_BOOST: float = 0.15

# How much confidence decays per nightly pass (multiplicative)
# 0.98 means ~10% decay over 5 days of no use — slow enough to be stable,
# fast enough that truly abandoned patterns fade within weeks.
CONFIDENCE_DECAY_FACTOR: float = 0.98

# Records below this threshold are considered unreliable and excluded
# from context injection by the retrieval pipeline.
CONFIDENCE_PRUNE_THRESHOLD: float = 0.10


# ── SkillRecord ───────────────────────────────────────────────────────────────

class SkillRecord(BaseModel):
    """
    A single learned pattern for a specific task type.

    Example:
        task_type   = "create_api_endpoint"
        pattern     = "Use FastAPI with async def, Pydantic request/response
                       models, snake_case routes, no ORM"
        confidence  = 0.91
        last_used   = 2026-03-14T10:00:00Z
        source_event_ids = ["evt-001", "evt-017", "evt-043"]
    """

    task_type: str
    pattern: str
    confidence: float = Field(default=0.5, ge=CONFIDENCE_MIN, le=CONFIDENCE_MAX)
    last_used: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    observation_count: int = Field(default=1, ge=1)
    source_event_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("task_type")
    @classmethod
    def normalise_task_type(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_")
        if not v:
            raise ValueError("task_type must not be empty")
        if len(v) > 100:
            raise ValueError(f"task_type too long ({len(v)} chars, max 100)")
        return v

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("pattern must not be empty")
        if len(v) > 2000:
            raise ValueError(f"pattern too long ({len(v)} chars, max 2000)")
        return v

    @field_validator("last_used", "created_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse datetime: {v!r}")

    @model_validator(mode="after")
    def created_not_after_last_used(self) -> "SkillRecord":
        if self.created_at > self.last_used:
            # Fix silently — can happen on first observation
            object.__setattr__(self, "created_at", self.last_used)
        return self

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_reliable(self) -> bool:
        """True if confidence is above the prune threshold."""
        return self.confidence >= CONFIDENCE_PRUNE_THRESHOLD

    @property
    def days_since_used(self) -> float:
        delta = datetime.now(timezone.utc) - self.last_used
        return delta.total_seconds() / 86_400

    # ── Mutation helpers (return new instance — records are immutable) ─────────

    def with_observation(self, event_id: str | None = None) -> "SkillRecord":
        """
        Return a new record updated as if this pattern was just observed again.
        Boosts confidence, updates last_used, increments observation_count.
        """
        new_confidence = min(
            CONFIDENCE_MAX,
            self.confidence + CONFIDENCE_BOOST,
        )
        new_ids = list(self.source_event_ids)
        if event_id and event_id not in new_ids:
            new_ids.append(event_id)

        return self.model_copy(update={
            "confidence": round(new_confidence, 6),
            "last_used": datetime.now(timezone.utc),
            "observation_count": self.observation_count + 1,
            "source_event_ids": new_ids,
        })

    def with_decay(self) -> "SkillRecord":
        """
        Return a new record with confidence decayed by one nightly pass.
        Does not update last_used — decay is passive, not an observation.
        """
        new_confidence = max(
            CONFIDENCE_MIN,
            self.confidence * CONFIDENCE_DECAY_FACTOR,
        )
        return self.model_copy(update={
            "confidence": round(new_confidence, 6),
        })

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "SkillRecord":
        return cls.model_validate_json(raw)


# ── File path ─────────────────────────────────────────────────────────────────

def _skill_path(task_type: str, base_dir: Path | None = None) -> Path:
    base = base_dir or cfg.skills_dir
    # task_type is already normalised by the validator — safe as filename
    return base / f"{task_type}.json"


# ── Read ──────────────────────────────────────────────────────────────────────

def get_skill(
    task_type: str,
    base_dir: Path | None = None,
) -> SkillRecord | None:
    """
    Load a skill record by task_type.
    Returns None if no record exists for this task_type.
    """
    path = _skill_path(task_type.strip().lower().replace(" ", "_"), base_dir)
    if not path.exists():
        return None
    try:
        record = SkillRecord.from_json(path.read_text(encoding="utf-8"))
        log.debug("skills.get", task_type=task_type, confidence=record.confidence)
        return record
    except Exception as exc:
        log.error("skills.read_error", task_type=task_type, error=str(exc))
        return None


def list_skills(
    base_dir: Path | None = None,
    *,
    min_confidence: float = 0.0,
) -> list[SkillRecord]:
    """
    Return all skill records, sorted by confidence descending.
    Optionally filtered by minimum confidence.
    """
    base = base_dir or cfg.skills_dir
    if not base.exists():
        return []

    records: list[SkillRecord] = []
    for path in base.glob("*.json"):
        try:
            record = SkillRecord.from_json(path.read_text(encoding="utf-8"))
            if record.confidence >= min_confidence:
                records.append(record)
        except Exception as exc:
            log.warning("skills.list_error", path=str(path), error=str(exc))

    return sorted(records, key=lambda r: r.confidence, reverse=True)


# ── Write / upsert ────────────────────────────────────────────────────────────

def save_skill(record: SkillRecord, base_dir: Path | None = None) -> Path:
    """Persist a skill record. Overwrites if the file already exists."""
    base = base_dir or cfg.skills_dir
    base.mkdir(parents=True, exist_ok=True)
    path = _skill_path(record.task_type, base_dir)
    path.write_text(record.to_json(), encoding="utf-8")
    log.debug(
        "skills.save",
        task_type=record.task_type,
        confidence=record.confidence,
        observations=record.observation_count,
    )
    return path


def upsert_skill(
    task_type: str,
    pattern: str,
    *,
    event_id: str | None = None,
    base_dir: Path | None = None,
) -> SkillRecord:
    """
    Insert or update a skill record.

    - If no record exists: create with confidence=0.5, observation_count=1.
    - If record exists with same pattern: boost confidence, increment count.
    - If record exists with different pattern: update pattern, reset
      observation_count to 1, set confidence to 0.5 (new pattern, unknown
      reliability until re-observed).

    Returns the saved record.
    """
    existing = get_skill(task_type, base_dir)

    if existing is None:
        # First observation
        record = SkillRecord(
            task_type=task_type,
            pattern=pattern,
            confidence=0.5,
            observation_count=1,
            source_event_ids=[event_id] if event_id else [],
        )
        log.info("skills.create", task_type=task_type, confidence=record.confidence)

    elif existing.pattern == pattern.strip():
        # Same pattern observed again — boost confidence
        record = existing.with_observation(event_id)
        log.info(
            "skills.boost",
            task_type=task_type,
            old_confidence=existing.confidence,
            new_confidence=record.confidence,
        )

    else:
        # Pattern changed — user's preference evolved
        record = SkillRecord(
            task_type=task_type,
            pattern=pattern,
            confidence=0.5,
            observation_count=1,
            source_event_ids=[event_id] if event_id else [],
        )
        log.info(
            "skills.pattern_changed",
            task_type=task_type,
            old_pattern=existing.pattern[:60],
            new_pattern=pattern[:60],
        )

    save_skill(record, base_dir)
    return record


# ── Decay pass ────────────────────────────────────────────────────────────────

def decay_pass(
    base_dir: Path | None = None,
    *,
    prune: bool = True,
) -> dict[str, int]:
    """
    Apply one decay step to every skill record.
    Called nightly by the consolidation engine.

    If prune=True, deletes records that fall below CONFIDENCE_PRUNE_THRESHOLD
    after decay is applied.

    Returns a summary dict: {"decayed": N, "pruned": N, "remaining": N}
    """
    base = base_dir or cfg.skills_dir
    if not base.exists():
        return {"decayed": 0, "pruned": 0, "remaining": 0}

    decayed = 0
    pruned = 0

    for path in base.glob("*.json"):
        try:
            record = SkillRecord.from_json(path.read_text(encoding="utf-8"))
            decayed_record = record.with_decay()

            if prune and not decayed_record.is_reliable:
                path.unlink()
                pruned += 1
                log.info(
                    "skills.pruned",
                    task_type=record.task_type,
                    confidence=decayed_record.confidence,
                )
            else:
                path.write_text(decayed_record.to_json(), encoding="utf-8")
                decayed += 1

        except Exception as exc:
            log.warning("skills.decay_error", path=str(path), error=str(exc))

    remaining = len(list(base.glob("*.json")))
    log.info(
        "skills.decay_pass_complete",
        decayed=decayed,
        pruned=pruned,
        remaining=remaining,
    )
    return {"decayed": decayed, "pruned": pruned, "remaining": remaining}