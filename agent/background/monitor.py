"""
agent/background/monitor.py

Proactive monitor — runs every 15 minutes via the scheduler.

Checks three signal sources for events that warrant unprompted agent
action. Each trigger produces a ProactiveAction queued for the
orchestrator to surface at the next user interaction.

Signal sources:
    git     — new commits since last check in watched project repos
    fs      — new/modified files of interest (.blend, .py, .dart)
    calendar— upcoming events in the next N minutes (local .ics or stub)

Design:
    - The monitor never acts directly — it only produces ProactiveActions.
    - Actions are written to a pending queue file so they survive restarts.
    - The orchestrator drains the queue at the start of each turn and
      prepends relevant actions to the system context.
    - Signals that fire repeatedly are deduplicated by a seen-set keyed
      on (signal_type, source_id). The seen-set resets each midnight.
    - The monitor is safe to run with no projects configured — it just
      produces no actions.

ProactiveAction priority:
    HIGH   — standup in < 10 min, test suite failing
    NORMAL — new commits, new blend file
    LOW    — lint suggestions, docs out of date
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.memory.episodic import EpisodicEvent, EventType, write_event
from agent.tools.code import git_log

log = get_logger(__name__)

# Path to the pending actions queue file
_PENDING_PATH = lambda: cfg.agent_memory_root / "monitor" / "pending.json"

# Path to monitor state (last-seen watermarks)
_STATE_PATH = lambda: cfg.agent_memory_root / "monitor" / "state.json"


# ── ProactiveAction ───────────────────────────────────────────────────────────

class ActionPriority(StrEnum):
    HIGH   = "high"
    NORMAL = "normal"
    LOW    = "low"


class ActionType(StrEnum):
    GIT_NEW_COMMITS    = "git.new_commits"
    FS_NEW_BLEND       = "fs.new_blend"
    FS_NEW_PYTHON      = "fs.new_python"
    FS_MODIFIED_PYTHON = "fs.modified_python"
    CALENDAR_UPCOMING  = "calendar.upcoming"
    TESTS_FAILING      = "tests.failing"


@dataclass
class ProactiveAction:
    """
    An unprompted suggestion the agent will surface at the next user turn.

    action_type: What kind of signal triggered this.
    summary:     One-sentence description for the user.
    suggestion:  What the agent suggests doing about it.
    priority:    How urgently to surface this.
    data:        Extra context (commit hashes, file paths, event details).
    created_at:  When this action was generated.
    """
    action_type: ActionType
    summary:     str
    suggestion:  str
    priority:    ActionPriority = ActionPriority.NORMAL
    data:        dict[str, Any] = field(default_factory=dict)
    created_at:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "summary":     self.summary,
            "suggestion":  self.suggestion,
            "priority":    self.priority,
            "data":        self.data,
            "created_at":  self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProactiveAction":
        return cls(
            action_type=ActionType(d["action_type"]),
            summary=d["summary"],
            suggestion=d["suggestion"],
            priority=ActionPriority(d.get("priority", "normal")),
            data=d.get("data", {}),
            created_at=datetime.fromisoformat(d["created_at"]),
        )


# ── MonitorState ──────────────────────────────────────────────────────────────

@dataclass
class MonitorState:
    """Watermarks for each signal source — persisted between runs."""

    last_git_commit_hash: dict[str, str]  = field(default_factory=dict)
    # project_path → last seen commit hash

    last_fs_check_utc:    str | None      = None
    last_calendar_check:  str | None      = None
    seen_action_keys:     list[str]       = field(default_factory=list)
    # dedup keys: "{action_type}:{source_id}"

    def to_dict(self) -> dict:
        return {
            "last_git_commit_hash": self.last_git_commit_hash,
            "last_fs_check_utc":    self.last_fs_check_utc,
            "last_calendar_check":  self.last_calendar_check,
            "seen_action_keys":     self.seen_action_keys,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorState":
        return cls(
            last_git_commit_hash=d.get("last_git_commit_hash", {}),
            last_fs_check_utc=d.get("last_fs_check_utc"),
            last_calendar_check=d.get("last_calendar_check"),
            seen_action_keys=d.get("seen_action_keys", []),
        )

    def is_seen(self, key: str) -> bool:
        return key in self.seen_action_keys

    def mark_seen(self, key: str) -> None:
        if key not in self.seen_action_keys:
            self.seen_action_keys.append(key)

    def reset_seen(self) -> None:
        """Reset seen keys — called at midnight to re-enable daily signals."""
        self.seen_action_keys = []


def _load_state() -> MonitorState:
    path = _STATE_PATH()
    if not path.exists():
        return MonitorState()
    try:
        return MonitorState.from_dict(json.loads(path.read_text()))
    except Exception as exc:
        log.warning("monitor.state_load_failed", error=str(exc))
        return MonitorState()


def _save_state(state: MonitorState) -> None:
    path = _STATE_PATH()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


# ── Pending queue ─────────────────────────────────────────────────────────────

def load_pending_actions() -> list[ProactiveAction]:
    """Load all pending actions from the queue file."""
    path = _PENDING_PATH()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
        return [ProactiveAction.from_dict(d) for d in raw]
    except Exception as exc:
        log.warning("monitor.pending_load_failed", error=str(exc))
        return []


def save_pending_actions(actions: list[ProactiveAction]) -> None:
    """Persist the pending action queue."""
    path = _PENDING_PATH()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([a.to_dict() for a in actions], indent=2))


def drain_pending_actions() -> list[ProactiveAction]:
    """
    Return all pending actions and clear the queue.
    Called by the orchestrator at the start of each turn.
    """
    actions = load_pending_actions()
    if actions:
        save_pending_actions([])
        log.debug("monitor.drained", count=len(actions))
    return actions


def queue_action(action: ProactiveAction) -> None:
    """Append one action to the persistent queue."""
    existing = load_pending_actions()
    existing.append(action)
    save_pending_actions(existing)


# ── Signal checkers ───────────────────────────────────────────────────────────

async def _check_git(
    state: MonitorState,
    project_paths: list[Path],
) -> list[ProactiveAction]:
    """
    Check each project repo for new commits since last seen hash.
    """
    # from agent.tools.code import git_log

    actions: list[ProactiveAction] = []

    for project_path in project_paths:
        if not (project_path / ".git").exists():
            continue

        result = await git_log(cwd=project_path, limit=5)
        if not result.success or not result.commits:
            continue

        latest_hash = result.commits[0].hash
        project_key = str(project_path)
        last_seen   = state.last_git_commit_hash.get(project_key)

        if last_seen is None:
            # First time seeing this repo — record hash, no action
            state.last_git_commit_hash[project_key] = latest_hash
            continue

        if latest_hash == last_seen:
            continue

        # New commits found
        new_commits = []
        for commit in result.commits:
            if commit.hash == last_seen:
                break
            new_commits.append(commit)

        if not new_commits:
            state.last_git_commit_hash[project_key] = latest_hash
            continue

        dedup_key = f"{ActionType.GIT_NEW_COMMITS}:{project_key}:{latest_hash}"
        if not state.is_seen(dedup_key):
            project_name = project_path.name
            actions.append(ProactiveAction(
                action_type=ActionType.GIT_NEW_COMMITS,
                summary=(
                    f"{len(new_commits)} new commit(s) in {project_name}: "
                    f"{new_commits[0].message[:60]}"
                ),
                suggestion=(
                    f"Update the {project_name} context doc with recent changes, "
                    f"or run the test suite to verify everything still passes."
                ),
                priority=ActionPriority.NORMAL,
                data={
                    "project":     project_name,
                    "project_path": project_key,
                    "new_count":   len(new_commits),
                    "commits":     [
                        {"hash": c.hash, "message": c.message}
                        for c in new_commits[:3]
                    ],
                },
            ))
            state.mark_seen(dedup_key)

        state.last_git_commit_hash[project_key] = latest_hash

    return actions


async def _check_filesystem(
    state: MonitorState,
    project_paths: list[Path],
) -> list[ProactiveAction]:
    """
    Check for newly created files of interest in the last 15 minutes.
    Uses direct filesystem scan rather than the watchdog queue
    (the watcher may not be running during background checks).
    """
    import time as _time

    actions: list[ProactiveAction] = []
    now = datetime.now(timezone.utc)

    # Window: files created/modified in the last 20 minutes
    window_seconds = 20 * 60

    for project_path in project_paths:
        if not project_path.exists():
            continue

        for path in project_path.rglob("*"):
            if not path.is_file():
                continue

            # Skip ignored paths
            path_str = str(path)
            if any(p in path_str for p in (
                ".git", "__pycache__", ".venv", "node_modules", ".agent"
            )):
                continue

            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue

            age_s = _time.time() - mtime
            if age_s > window_seconds:
                continue

            ext = path.suffix.lower()

            if ext == ".blend":
                dedup_key = f"{ActionType.FS_NEW_BLEND}:{path_str}"
                if not state.is_seen(dedup_key):
                    actions.append(ProactiveAction(
                        action_type=ActionType.FS_NEW_BLEND,
                        summary=f"New Blender file detected: {path.name}",
                        suggestion=(
                            f"Run the LOD pipeline on {path.name} "
                            f"or export as GLB for web use."
                        ),
                        priority=ActionPriority.NORMAL,
                        data={"path": path_str, "name": path.name},
                    ))
                    state.mark_seen(dedup_key)

            elif ext == ".py":
                dedup_key = f"{ActionType.FS_NEW_PYTHON}:{path_str}"
                if not state.is_seen(dedup_key):
                    actions.append(ProactiveAction(
                        action_type=ActionType.FS_NEW_PYTHON,
                        summary=f"New Python file: {path.name}",
                        suggestion=f"Run ruff linter on {path.name} to check for issues.",
                        priority=ActionPriority.LOW,
                        data={"path": path_str, "name": path.name},
                    ))
                    state.mark_seen(dedup_key)

    return actions


def _check_calendar() -> list[ProactiveAction]:
    """
    Check for calendar events coming up soon.

    Reads from ~/.agent/config/calendar.json if it exists.
    Format: [{"title": "...", "start_utc": "ISO", "duration_min": 30}]

    This is a simple stub — a full implementation would read from
    a .ics file or system calendar API. The structure is in place
    for task 39 (proactive monitor) and can be extended later.
    """
    calendar_path = cfg.agent_memory_root / "config" / "calendar.json"
    if not calendar_path.exists():
        return []

    actions: list[ProactiveAction] = []
    now = datetime.now(timezone.utc)

    try:
        events = json.loads(calendar_path.read_text())
    except Exception as exc:
        log.warning("monitor.calendar_load_failed", error=str(exc))
        return []

    for event in events:
        try:
            start = datetime.fromisoformat(event["start_utc"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)

            minutes_until = (start - now).total_seconds() / 60

            if 0 < minutes_until <= 15:
                title    = event.get("title", "Event")
                duration = event.get("duration_min", 30)
                priority = ActionPriority.HIGH if minutes_until <= 10 else ActionPriority.NORMAL

                actions.append(ProactiveAction(
                    action_type=ActionType.CALENDAR_UPCOMING,
                    summary=f"{title} starts in {int(minutes_until)} minutes",
                    suggestion=(
                        f"Wrap up current work — {title} starts soon "
                        f"({duration} min duration)."
                    ),
                    priority=priority,
                    data={
                        "title":         title,
                        "start_utc":     event["start_utc"],
                        "minutes_until": round(minutes_until, 1),
                        "duration_min":  duration,
                    },
                ))
        except Exception as exc:
            log.warning("monitor.calendar_event_parse_error", error=str(exc))

    return actions


# ── Main monitor run ──────────────────────────────────────────────────────────

async def run_proactive_monitor(
    project_paths: list[Path] | None = None,
) -> list[ProactiveAction]:
    """
    Run one complete monitor pass and queue any detected actions.

    Args:
        project_paths: Repo directories to monitor. If None, reads from
                       cfg.projects_dir (each subdirectory is a project).

    Returns:
        List of ProactiveAction objects queued in this pass.
    """
    log.debug("monitor.start")

    try:
        state = _load_state()

        # Resolve project paths
        _paths: list[Path] = []
        if project_paths is not None:
            _paths = project_paths
        else:
            projects_dir = cfg.projects_dir
            if projects_dir.exists():
                _paths = [p for p in projects_dir.iterdir() if p.is_dir()]

        all_actions: list[ProactiveAction] = []

        # ── Git signal ────────────────────────────────────────────────────────
        try:
            git_actions = await _check_git(state, _paths)
            all_actions.extend(git_actions)
        except Exception as exc:
            log.warning("monitor.git_check_failed", error=str(exc))

        # ── Filesystem signal ─────────────────────────────────────────────────
        try:
            fs_actions = await _check_filesystem(state, _paths)
            all_actions.extend(fs_actions)
        except Exception as exc:
            log.warning("monitor.fs_check_failed", error=str(exc))

        # ── Calendar signal ───────────────────────────────────────────────────
        try:
            cal_actions = _check_calendar()
            all_actions.extend(cal_actions)
        except Exception as exc:
            log.warning("monitor.calendar_check_failed", error=str(exc))

        # Persist state and queue actions
        _save_state(state)

        if all_actions:
            existing = load_pending_actions()
            save_pending_actions(existing + all_actions)
            log.info(
                "monitor.actions_queued",
                count=len(all_actions),
                types=[a.action_type for a in all_actions],
            )
        else:
            log.debug("monitor.no_actions")

        return all_actions

    except Exception as exc:
        log.error("monitor.failed", error=str(exc))
        return []