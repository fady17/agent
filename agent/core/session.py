"""
agent/core/session.py

Working memory — the agent's in-session state.

SessionState holds everything the agent needs to carry across turns
within a single running session. It is:

    - Persisted to disk on every mutation (atomic write — no partial state)
    - Reloaded automatically at startup if a prior session file exists
    - Reset cleanly when a new session is explicitly started

File location:
    ~/.agent/memory/working/session.json

What goes in session state vs episodic memory:
    SessionState  — transient, current-session context. Overwritten each run.
                    "What am I doing right now?"
    EpisodicEvent — permanent log. Append-only, never overwritten.
                    "What did I do historically?"

On shutdown the orchestrator writes a SESSION_END episodic event, then
the session file is left in place so the next startup can show a
"resuming from last session" summary before resetting.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)


# ── SessionMetrics ────────────────────────────────────────────────────────────

class SessionMetrics(BaseModel):
    """Accumulated cost and usage metrics for the current session."""

    total_llm_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    local_calls: int = 0   # LM Studio calls — zero cost
    cloud_calls: int = 0   # Anthropic API calls — tracked for cost

    def add_call(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        is_local: bool,
    ) -> "SessionMetrics":
        """Return updated metrics after one LLM call."""
        return self.model_copy(update={
            "total_llm_calls": self.total_llm_calls + 1,
            "total_tokens_in":  self.total_tokens_in  + tokens_in,
            "total_tokens_out": self.total_tokens_out + tokens_out,
            "total_cost_usd":   round(self.total_cost_usd + cost_usd, 6),
            "local_calls":  self.local_calls  + (1 if is_local  else 0),
            "cloud_calls":  self.cloud_calls  + (1 if not is_local else 0),
        })


# ── SessionState ──────────────────────────────────────────────────────────────

class SessionState(BaseModel):
    """
    Complete working memory for one agent session.

    Immutable update pattern — every mutation returns a new instance,
    which is then persisted. This prevents partial-write corruption and
    makes state transitions explicit and testable.
    """

    # Identity
    session_id: str
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_active_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Current context
    active_project: str | None = None
    active_task_type: str | None = None
    active_task_summary: str | None = None

    # Turn tracking
    turn_count: int = 0

    # Accumulated metrics
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)

    # Free-form scratch space for the orchestrator
    # (temporary values that don't warrant a dedicated field)
    scratch: dict[str, Any] = Field(default_factory=dict)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("session_id must not be empty")
        return v

    @field_validator("started_at", "last_active_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse datetime: {v!r}")

    @field_validator("turn_count")
    @classmethod
    def validate_turn_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("turn_count must be non-negative")
        return v

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def with_project(self, project: str | None) -> "SessionState":
        return self.model_copy(update={
            "active_project": project.strip().lower() if project else None,
            "last_active_at": datetime.now(timezone.utc),
        })

    def with_task(
        self,
        task_type: str | None,
        summary: str | None = None,
    ) -> "SessionState":
        return self.model_copy(update={
            "active_task_type":    task_type,
            "active_task_summary": summary,
            "last_active_at":      datetime.now(timezone.utc),
        })

    def with_turn(self) -> "SessionState":
        """Increment turn counter and update last_active_at."""
        return self.model_copy(update={
            "turn_count":      self.turn_count + 1,
            "last_active_at":  datetime.now(timezone.utc),
        })

    def with_metrics(self, updated_metrics: SessionMetrics) -> "SessionState":
        return self.model_copy(update={
            "metrics":        updated_metrics,
            "last_active_at": datetime.now(timezone.utc),
        })

    def with_scratch(self, key: str, value: Any) -> "SessionState":
        return self.model_copy(update={
            "scratch": {**self.scratch, key: value},
        })

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def duration_seconds(self) -> float:
        return (self.last_active_at - self.started_at).total_seconds()

    @property
    def is_fresh(self) -> bool:
        """True if no turns have been processed yet."""
        return self.turn_count == 0


# ── SessionManager ────────────────────────────────────────────────────────────

class SessionManager:
    """
    Manages the lifecycle of SessionState — create, load, save, reset.

    Single instance per agent process. The orchestrator holds a reference
    and calls save() after every state mutation.
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self._path = state_path or (cfg.working_dir / "session.json")
        self._state: SessionState | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        if self._state is None:
            raise RuntimeError(
                "SessionManager not initialised. Call load() or new_session() first."
            )
        return self._state

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def new_session(self, session_id: str | None = None) -> SessionState:
        """
        Create a fresh session, persist it, and return it.
        Overwrites any existing session file.
        """
        import uuid
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        state = SessionState(session_id=sid)
        self._state = state
        self.save()
        log.info("session.new", session_id=sid)
        return state

    def load(self) -> SessionState | None:
        """
        Load the last session from disk.
        Returns the loaded state, or None if no session file exists.
        Does not create a new session — call new_session() for that.
        """
        if not self._path.exists():
            log.debug("session.no_file", path=str(self._path))
            return None

        try:
            raw = self._path.read_text(encoding="utf-8")
            state = SessionState.model_validate_json(raw)
            self._state = state
            log.info(
                "session.loaded",
                session_id=state.session_id,
                turns=state.turn_count,
                project=state.active_project,
            )
            return state
        except Exception as exc:
            log.error("session.load_failed", path=str(self._path), error=str(exc))
            return None

    def save(self, state: SessionState | None = None) -> Path:
        """
        Atomically persist session state to disk.

        Uses a temp file + rename so a crash mid-write never produces
        a corrupt session file. The previous session is always recoverable.
        """
        target_state = state or self._state
        if target_state is None:
            raise RuntimeError("No state to save.")

        if state is not None:
            self._state = state

        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file in the same directory, then rename
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=".session_tmp_",
            suffix=".json",
        )
        tmp_path = Path(tmp_path_str)

        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(target_state.model_dump_json(indent=2))
            tmp_path.replace(self._path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        log.debug(
            "session.saved",
            session_id=target_state.session_id,
            turns=target_state.turn_count,
            path=str(self._path),
        )
        return self._path

    def update(self, state: SessionState) -> SessionState:
        """
        Replace in-memory state and persist atomically.
        The canonical way to apply a mutation:

            mgr.update(mgr.state.with_turn())
        """
        self._state = state
        self.save(state)
        return state

    def reset(self) -> None:
        """
        Clear in-memory state without deleting the session file.
        Call before new_session() if you want a clean start.
        """
        self._state = None
        log.debug("session.reset")