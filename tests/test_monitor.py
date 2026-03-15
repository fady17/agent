"""
tests/test_monitor.py

Tests for the proactive monitor — ProactiveAction schema,
pending queue, dedup logic, and signal checkers.
All filesystem I/O uses tmp_path. Git calls are mocked.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.background.monitor import (
    ActionPriority,
    ActionType,
    MonitorState,
    ProactiveAction,
    _check_calendar,
    _check_filesystem,
    _check_git,
    drain_pending_actions,
    load_pending_actions,
    queue_action,
    run_proactive_monitor,
    save_pending_actions,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_action(
    action_type: ActionType = ActionType.FS_NEW_BLEND,
    summary: str = "test summary",
    suggestion: str = "test suggestion",
    priority: ActionPriority = ActionPriority.NORMAL,
    **kwargs: Any,  # Added this
) -> ProactiveAction:
    return ProactiveAction(
        action_type=action_type,
        summary=summary,
        suggestion=suggestion,
        priority=priority,
        **kwargs,   # Added this
    )


def make_git_log_result(commits: list[dict]) -> MagicMock:
    from agent.tools.code import GitCommit, GitLogResult
    result = MagicMock(spec=GitLogResult)
    result.success = True
    result.branch = "main"
    result.commits = [
        GitCommit(
            hash=c["hash"],
            author=c.get("author", "Dev"),
            date=c.get("date", "2026-03-14T10:00:00Z"),
            message=c["message"],
        )
        for c in commits
    ]
    return result


# ── ProactiveAction ───────────────────────────────────────────────────────────

def test_action_round_trips_dict() -> None:
    original = make_action(
        action_type=ActionType.FS_NEW_BLEND,
        summary="New blend file",
        data={"path": "/project/model.blend"}, # type: ignore
    )
    restored = ProactiveAction.from_dict(original.to_dict())
    assert restored.action_type == original.action_type
    assert restored.summary     == original.summary
    assert restored.data        == original.data
    assert restored.priority    == original.priority


def test_action_created_at_is_utc() -> None:
    action = make_action()
    assert action.created_at.tzinfo == timezone.utc


def test_action_to_dict_has_all_keys() -> None:
    d = make_action().to_dict()
    for key in ("action_type", "summary", "suggestion", "priority", "data", "created_at"):
        assert key in d


# ── MonitorState ──────────────────────────────────────────────────────────────

def test_monitor_state_defaults() -> None:
    s = MonitorState()
    assert s.last_git_commit_hash == {}
    assert s.seen_action_keys == []


def test_monitor_state_mark_seen() -> None:
    s = MonitorState()
    s.mark_seen("git:project:abc123")
    assert s.is_seen("git:project:abc123") is True


def test_monitor_state_not_seen() -> None:
    s = MonitorState()
    assert s.is_seen("git:project:abc123") is False


def test_monitor_state_no_duplicates_in_seen() -> None:
    s = MonitorState()
    s.mark_seen("key")
    s.mark_seen("key")
    assert s.seen_action_keys.count("key") == 1


def test_monitor_state_reset_seen() -> None:
    s = MonitorState()
    s.mark_seen("key1")
    s.mark_seen("key2")
    s.reset_seen()
    assert s.seen_action_keys == []


def test_monitor_state_round_trips_dict() -> None:
    s = MonitorState(
        last_git_commit_hash={"/project": "abc123"},
        seen_action_keys=["key1", "key2"],
    )
    restored = MonitorState.from_dict(s.to_dict())
    assert restored.last_git_commit_hash == {"/project": "abc123"}
    assert restored.seen_action_keys == ["key1", "key2"]


# ── Pending queue ─────────────────────────────────────────────────────────────

def test_load_pending_empty_when_no_file(tmp_path: Path) -> None:
    with patch("agent.background.monitor._PENDING_PATH", return_value=tmp_path / "pending.json"):
        actions = load_pending_actions()
    assert actions == []


def test_queue_and_load_action(tmp_path: Path) -> None:
    pending_path = tmp_path / "pending.json"
    with patch("agent.background.monitor._PENDING_PATH", return_value=pending_path):
        action = make_action(summary="test action")
        queue_action(action)
        loaded = load_pending_actions()
    assert len(loaded) == 1
    assert loaded[0].summary == "test action"


def test_queue_multiple_actions(tmp_path: Path) -> None:
    pending_path = tmp_path / "pending.json"
    with patch("agent.background.monitor._PENDING_PATH", return_value=pending_path):
        queue_action(make_action(summary="first"))
        queue_action(make_action(summary="second"))
        loaded = load_pending_actions()
    assert len(loaded) == 2


def test_drain_returns_all_and_clears(tmp_path: Path) -> None:
    pending_path = tmp_path / "pending.json"
    with patch("agent.background.monitor._PENDING_PATH", return_value=pending_path):
        save_pending_actions([make_action(summary="a"), make_action(summary="b")])
        drained = drain_pending_actions()
        remaining = load_pending_actions()
    assert len(drained) == 2
    assert remaining == []


def test_drain_empty_queue_returns_empty(tmp_path: Path) -> None:
    pending_path = tmp_path / "pending.json"
    with patch("agent.background.monitor._PENDING_PATH", return_value=pending_path):
        result = drain_pending_actions()
    assert result == []


# ── _check_git ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_git_produces_action_on_new_commits(tmp_path: Path) -> None:
    project = tmp_path / "agrivision"
    project.mkdir()
    (project / ".git").mkdir()

    state = MonitorState(last_git_commit_hash={str(project): "old_hash"})

    git_result = make_git_log_result([
        {"hash": "new_hash", "message": "feat: add detection endpoint"},
        {"hash": "old_hash", "message": "chore: initial setup"},
    ])

    with patch("agent.background.monitor.git_log", new=AsyncMock(return_value=git_result)):
        actions = await _check_git(state, [project])

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.GIT_NEW_COMMITS
    assert "agrivision" in actions[0].summary
    assert state.last_git_commit_hash[str(project)] == "new_hash"


@pytest.mark.asyncio
async def test_check_git_no_action_when_no_new_commits(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".git").mkdir()

    state = MonitorState(last_git_commit_hash={str(project): "abc123"})
    git_result = make_git_log_result([{"hash": "abc123", "message": "same commit"}])

    with patch("agent.background.monitor.git_log", new=AsyncMock(return_value=git_result)):
        actions = await _check_git(state, [project])

    assert actions == []


@pytest.mark.asyncio
async def test_check_git_records_hash_on_first_seen(tmp_path: Path) -> None:
    project = tmp_path / "new_proj"
    project.mkdir()
    (project / ".git").mkdir()

    state = MonitorState()  # no prior hash
    git_result = make_git_log_result([{"hash": "first_hash", "message": "init"}])

    with patch("agent.background.monitor.git_log", new=AsyncMock(return_value=git_result)):
        actions = await _check_git(state, [project])

    # First time — no action, just records the hash
    assert actions == []
    assert state.last_git_commit_hash[str(project)] == "first_hash"


@pytest.mark.asyncio
async def test_check_git_skips_non_git_dirs(tmp_path: Path) -> None:
    project = tmp_path / "not_git"
    project.mkdir()
    # No .git directory

    state = MonitorState()
    with patch("agent.background.monitor.git_log", new=AsyncMock()) as mock_git:
        actions = await _check_git(state, [project])

    mock_git.assert_not_called()
    assert actions == []


@pytest.mark.asyncio
async def test_check_git_deduplicates_actions(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".git").mkdir()

    state = MonitorState(last_git_commit_hash={str(project): "old"})
    git_result = make_git_log_result([
        {"hash": "new_hash", "message": "new commit"},
        {"hash": "old", "message": "old commit"},
    ])

    with patch("agent.background.monitor.git_log", new=AsyncMock(return_value=git_result)):
        actions1 = await _check_git(state, [project])
        # State now has new_hash as last seen and dedup key marked
        state.last_git_commit_hash[str(project)] = "old"  # simulate rollback for second call
        actions2 = await _check_git(state, [project])

    assert len(actions1) == 1
    assert len(actions2) == 0  # deduped


# ── _check_filesystem ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_filesystem_detects_new_blend(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    blend = project / "model.blend"
    blend.write_text("fake blend content")

    state = MonitorState()
    actions = await _check_filesystem(state, [project])

    blend_actions = [a for a in actions if a.action_type == ActionType.FS_NEW_BLEND]
    assert len(blend_actions) == 1
    assert "model.blend" in blend_actions[0].summary


@pytest.mark.asyncio
async def test_check_filesystem_detects_new_python(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    py = project / "new_module.py"
    py.write_text("print('hello')")

    state = MonitorState()
    actions = await _check_filesystem(state, [project])

    py_actions = [a for a in actions if a.action_type == ActionType.FS_NEW_PYTHON]
    assert len(py_actions) == 1


@pytest.mark.asyncio
async def test_check_filesystem_deduplicates(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    blend = project / "scene.blend"
    blend.write_text("x")

    state = MonitorState()
    actions1 = await _check_filesystem(state, [project])
    actions2 = await _check_filesystem(state, [project])

    assert len(actions1) == 1
    assert len(actions2) == 0  # deduped


@pytest.mark.asyncio
async def test_check_filesystem_ignores_old_files(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    old_file = project / "old.blend"
    old_file.write_text("x")

    # Set mtime to 25 minutes ago (outside the 20-min window)
    old_mtime = time.time() - (25 * 60)
    import os
    os.utime(old_file, (old_mtime, old_mtime))

    state = MonitorState()
    actions = await _check_filesystem(state, [project])
    assert actions == []


@pytest.mark.asyncio
async def test_check_filesystem_ignores_git_dirs(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    git_dir = project / ".git"
    git_dir.mkdir()
    git_file = git_dir / "something.py"
    git_file.write_text("x")

    state = MonitorState()
    actions = await _check_filesystem(state, [project])
    assert not any(".git" in a.data.get("path", "") for a in actions)


# ── _check_calendar ───────────────────────────────────────────────────────────

def test_check_calendar_upcoming_event(tmp_path: Path) -> None:
    soon = (datetime.now(timezone.utc) + timedelta(minutes=8)).isoformat()
    calendar_data = [{"title": "Team standup", "start_utc": soon, "duration_min": 30}]

    cal_path = tmp_path / "config" / "calendar.json"
    cal_path.parent.mkdir(parents=True)
    cal_path.write_text(json.dumps(calendar_data))

    with patch("agent.background.monitor.cfg") as mock_cfg:
        mock_cfg.agent_memory_root = tmp_path
        actions = _check_calendar()

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.CALENDAR_UPCOMING
    assert "standup" in actions[0].summary.lower()
    assert actions[0].priority == ActionPriority.HIGH  # <10 min away


def test_check_calendar_future_event_not_surfaced(tmp_path: Path) -> None:
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    cal_path = tmp_path / "config" / "calendar.json"
    cal_path.parent.mkdir(parents=True)
    cal_path.write_text(json.dumps([{"title": "Later meeting", "start_utc": future}]))

    with patch("agent.background.monitor.cfg") as mock_cfg:
        mock_cfg.agent_memory_root = tmp_path
        actions = _check_calendar()

    assert actions == []


def test_check_calendar_no_file_returns_empty(tmp_path: Path) -> None:
    with patch("agent.background.monitor.cfg") as mock_cfg:
        mock_cfg.agent_memory_root = tmp_path
        actions = _check_calendar()
    assert actions == []


def test_check_calendar_normal_priority_between_10_and_15(tmp_path: Path) -> None:
    soon = (datetime.now(timezone.utc) + timedelta(minutes=12)).isoformat()
    cal_path = tmp_path / "config" / "calendar.json"
    cal_path.parent.mkdir(parents=True)
    cal_path.write_text(json.dumps([{"title": "Review", "start_utc": soon}]))

    with patch("agent.background.monitor.cfg") as mock_cfg:
        mock_cfg.agent_memory_root = tmp_path
        actions = _check_calendar()

    assert len(actions) == 1
    assert actions[0].priority == ActionPriority.NORMAL


# ── run_proactive_monitor ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_proactive_monitor_returns_list(tmp_path: Path) -> None:
    with patch("agent.background.monitor._load_state", return_value=MonitorState()), \
         patch("agent.background.monitor._save_state"), \
         patch("agent.background.monitor._PENDING_PATH", return_value=tmp_path / "p.json"), \
         patch("agent.background.monitor._check_git", new=AsyncMock(return_value=[])), \
         patch("agent.background.monitor._check_filesystem", new=AsyncMock(return_value=[])), \
         patch("agent.background.monitor._check_calendar", return_value=[]):

        actions = await run_proactive_monitor(project_paths=[])

    assert isinstance(actions, list)


@pytest.mark.asyncio
async def test_run_proactive_monitor_queues_actions(tmp_path: Path) -> None:
    pending_path = tmp_path / "pending.json"
    action = make_action(summary="new commits")

    with patch("agent.background.monitor._load_state", return_value=MonitorState()), \
         patch("agent.background.monitor._save_state"), \
         patch("agent.background.monitor._PENDING_PATH", return_value=pending_path), \
         patch("agent.background.monitor.load_pending_actions", return_value=[]), \
         patch("agent.background.monitor.save_pending_actions") as mock_save, \
         patch("agent.background.monitor._check_git", new=AsyncMock(return_value=[action])), \
         patch("agent.background.monitor._check_filesystem", new=AsyncMock(return_value=[])), \
         patch("agent.background.monitor._check_calendar", return_value=[]):

        actions = await run_proactive_monitor(project_paths=[])

    mock_save.assert_called_once()
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_run_proactive_monitor_never_raises(tmp_path: Path) -> None:
    with patch("agent.background.monitor._load_state", side_effect=OSError("disk error")):
        actions = await run_proactive_monitor(project_paths=[])

    assert actions == []


@pytest.mark.asyncio
async def test_run_proactive_monitor_git_failure_does_not_stop_fs_check(tmp_path: Path) -> None:
    fs_action = make_action(action_type=ActionType.FS_NEW_BLEND, summary="new blend")

    with patch("agent.background.monitor._load_state", return_value=MonitorState()), \
         patch("agent.background.monitor._save_state"), \
         patch("agent.background.monitor._PENDING_PATH", return_value=tmp_path / "p.json"), \
         patch("agent.background.monitor.load_pending_actions", return_value=[]), \
         patch("agent.background.monitor.save_pending_actions"), \
         patch("agent.background.monitor._check_git", new=AsyncMock(side_effect=RuntimeError("git gone"))), \
         patch("agent.background.monitor._check_filesystem", new=AsyncMock(return_value=[fs_action])), \
         patch("agent.background.monitor._check_calendar", return_value=[]):

        actions = await run_proactive_monitor(project_paths=[])

    assert len(actions) == 1
    assert actions[0].action_type == ActionType.FS_NEW_BLEND