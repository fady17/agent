"""
agent/memory/episodic.py

Episodic memory — the agent's event log.

Every significant action the agent takes or observes is written as an
EpisodicEvent JSON file under:

    ~/.agent/memory/episodic/YYYY/MM/DD/evt-{ulid}.json

Design decisions:
  - One file per event. No append-only log file that requires seeking.
    Each event is independently readable, deletable, and grep-able.
  - ULID as ID: lexicographically sortable by creation time, collision-free,
    no database required.
  - UTC everywhere. No local timezone stored — display layer converts.
  - The `data` payload is a free-form dict. Each EventType documents its
    own expected keys — enforced at the call site, not in the base schema.
    This keeps the schema stable as new event types are added.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)


# ── ULID-lite ─────────────────────────────────────────────────────────────────
# Full ULID library is a dependency we don't need — a time-prefixed UUID
# gives us the same two properties: sortable + unique.

def _new_event_id() -> str:
    """
    Generate a sortable unique event ID.
    Format: evt-{unix_ms_10digits}-{random_8hex}
    Example: evt-1741910400123-a3f2c1b0
    Lexicographic sort == chronological sort.
    """
    ms = int(time.time() * 1000)
    rand = uuid4().hex[:8]
    return f"evt-{ms:013d}-{rand}"


# ── EventType ─────────────────────────────────────────────────────────────────

class EventType(StrEnum):
    # Developer actions
    CODE_WRITE      = "code.write"       # file written / modified
    CODE_RUN        = "code.run"         # script / test executed
    CODE_DEBUG      = "code.debug"       # debugging session / fix applied
    GIT_COMMIT      = "git.commit"       # commit made
    GIT_BRANCH      = "git.branch"       # branch created / switched

    # Design / 3D actions
    DESIGN_EXPORT   = "design.export"    # asset exported (GLB, FBX, etc.)
    DESIGN_MODIFY   = "design.modify"    # scene / mesh modified via bpy
    DESIGN_RENDER   = "design.render"    # render job started or completed

    # Shell / system
    SHELL_RUN       = "shell.run"        # arbitrary shell command executed

    # LLM interactions
    LLM_CALL        = "llm.call"         # LLM request made (local or cloud)
    LLM_TOOL_USE    = "llm.tool_use"     # LLM invoked a tool

    # File system
    FILE_WATCH      = "file.watch"       # watchdog event observed
    FILE_READ       = "file.read"        # agent read a file
    FILE_WRITE      = "file.write"       # agent wrote a file

    # Memory operations
    MEMORY_CONSOLIDATE = "memory.consolidate"  # consolidation run completed
    MEMORY_RETRIEVE    = "memory.retrieve"     # retrieval pipeline queried

    # Agent lifecycle
    SESSION_START   = "session.start"
    SESSION_END     = "session.end"
    TASK_COMPLETE   = "task.complete"    # a routed task finished

    # Errors
    ERROR           = "error"            # any caught exception worth recording


# ── EpisodicEvent ─────────────────────────────────────────────────────────────

class EpisodicEvent(BaseModel):
    """
    A single timestamped agent action or observation.

    Required fields:
      event_type  — what kind of action this was
      summary     — one human-readable sentence describing what happened

    Optional but encouraged:
      project     — which project context this belongs to (e.g. "agrivision")
      data        — structured payload specific to the event_type

    Auto-generated:
      id          — sortable unique identifier
      timestamp   — UTC ISO-8601
    """

    id: str = Field(default_factory=_new_event_id)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)  # noqa: UP017
    )
    event_type: EventType
    summary: str
    project: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("summary must not be empty")
        if len(v) > 500:
            raise ValueError(f"summary too long ({len(v)} chars, max 500)")
        return v

    @field_validator("project")
    @classmethod
    def normalise_project(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        if not v:
            return None
        return v

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                # Assume UTC if naive
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse timestamp: {v!r}")

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "EpisodicEvent":
        return cls.model_validate_json(raw)

    # ── File path ─────────────────────────────────────────────────────────────

    def file_path(self, base_dir: Path | None = None) -> Path:
        """
        Canonical path for this event:
            {base_dir}/YYYY/MM/DD/{id}.json
        """
        base = base_dir or cfg.episodic_dir
        day = self.timestamp.strftime("%Y/%m/%d")
        return base / day / f"{self.id}.json"


# ── Writer ────────────────────────────────────────────────────────────────────

def write_event(event: EpisodicEvent, base_dir: Path | None = None) -> Path:
    """
    Persist an event to the filesystem.
    Creates the date-partitioned directory if it does not exist.
    Returns the path written.
    """
    path = event.file_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(event.to_json(), encoding="utf-8")

    log.debug(
        "episodic.write",
        event_id=event.id,
        event_type=event.event_type,
        project=event.project,
        path=str(path),
    )
    return path


# ── Reader ────────────────────────────────────────────────────────────────────

def read_event(path: Path) -> EpisodicEvent:
    """Load a single event from its JSON file."""
    return EpisodicEvent.from_json(path.read_text(encoding="utf-8"))


def list_events(
    base_dir: Path | None = None,
    *,
    project: str | None = None,
    event_type: EventType | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int | None = None,
) -> list[EpisodicEvent]:
    """
    Return events from the episodic store, newest first.

    All filters are applied in-memory after reading — this is fast enough
    for personal-scale event volumes (thousands, not millions).
    For larger volumes the metadata-filter + ripgrep path in the retrieval
    pipeline handles pre-filtering before events are fully deserialised.

    Args:
        project:    filter to a specific project slug
        event_type: filter to a specific EventType
        since:      only events at or after this UTC datetime
        until:      only events at or before this UTC datetime
        limit:      cap the number of results returned
    """
    base = base_dir or cfg.episodic_dir

    if not base.exists():
        return []

    # Collect all .json files — pathlib rglob, sorted by name (== time order)
    paths = sorted(base.rglob("evt-*.json"), reverse=True)

    events: list[EpisodicEvent] = []
    for p in paths:
        try:
            evt = read_event(p)
        except Exception as exc:
            log.warning("episodic.read_error", path=str(p), error=str(exc))
            continue

        # Apply filters
        if project is not None and evt.project != project.strip().lower():
            continue
        if event_type is not None and evt.event_type != event_type:
            continue
        if since is not None and evt.timestamp < since:
            continue
        if until is not None and evt.timestamp > until:
            continue

        events.append(evt)

        if limit is not None and len(events) >= limit:
            break

    return events