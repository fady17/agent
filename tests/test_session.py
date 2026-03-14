"""
tests/test_session.py

Tests for SessionState (working memory model) and SessionManager
(lifecycle + persistence). All filesystem I/O uses pytest's tmp_path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.core.session import SessionManager, SessionMetrics, SessionState


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**kwargs: object) -> SessionState:
    defaults = {"session_id": "sess-test001"}
    return SessionState(**{**defaults, **kwargs})  # type: ignore[arg-type]


def make_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(state_path=tmp_path / "session.json")


# ── SessionMetrics ────────────────────────────────────────────────────────────

def test_metrics_initial_values() -> None:
    m = SessionMetrics()
    assert m.total_llm_calls == 0
    assert m.total_cost_usd == 0.0


def test_metrics_add_local_call() -> None:
    m = SessionMetrics()
    updated = m.add_call(tokens_in=100, tokens_out=50, cost_usd=0.0, is_local=True)
    assert updated.total_llm_calls == 1
    assert updated.local_calls == 1
    assert updated.cloud_calls == 0
    assert updated.total_tokens_in == 100
    assert updated.total_tokens_out == 50


def test_metrics_add_cloud_call() -> None:
    m = SessionMetrics()
    updated = m.add_call(tokens_in=500, tokens_out=200, cost_usd=0.003, is_local=False)
    assert updated.cloud_calls == 1
    assert updated.local_calls == 0
    assert updated.total_cost_usd == pytest.approx(0.003)


def test_metrics_accumulate_across_calls() -> None:
    m = SessionMetrics()
    m = m.add_call(tokens_in=100, tokens_out=50,  cost_usd=0.001, is_local=False)
    m = m.add_call(tokens_in=200, tokens_out=100, cost_usd=0.002, is_local=False)
    assert m.total_llm_calls == 2
    assert m.total_tokens_in == 300
    assert m.total_cost_usd == pytest.approx(0.003)


def test_metrics_returns_new_instance() -> None:
    m = SessionMetrics()
    updated = m.add_call(tokens_in=10, tokens_out=10, cost_usd=0.0, is_local=True)
    assert m is not updated
    assert m.total_llm_calls == 0


# ── SessionState construction ─────────────────────────────────────────────────

def test_state_constructs_with_defaults() -> None:
    s = make_state()
    assert s.session_id == "sess-test001"
    assert s.turn_count == 0
    assert s.active_project is None
    assert s.is_fresh is True


def test_state_empty_session_id_raises() -> None:
    with pytest.raises(ValidationError, match="session_id"):
        make_state(session_id="  ")


def test_state_negative_turn_count_raises() -> None:
    with pytest.raises(ValidationError, match="turn_count"):
        make_state(turn_count=-1)


def test_state_timestamps_are_utc() -> None:
    s = make_state()
    assert s.started_at.tzinfo == timezone.utc
    assert s.last_active_at.tzinfo == timezone.utc


def test_state_naive_timestamp_assumed_utc() -> None:
    naive = datetime(2026, 3, 14, 10, 0, 0)
    s = make_state(started_at=naive)
    assert s.started_at.tzinfo == timezone.utc


# ── Mutation helpers ──────────────────────────────────────────────────────────

def test_with_project_sets_project() -> None:
    s = make_state().with_project("AgriVision")
    assert s.active_project == "agrivision"  # normalised


def test_with_project_none_clears() -> None:
    s = make_state().with_project("agrivision").with_project(None)
    assert s.active_project is None


def test_with_task_sets_task() -> None:
    s = make_state().with_task("create_api_endpoint", "Write detect endpoint")
    assert s.active_task_type == "create_api_endpoint"
    assert s.active_task_summary == "Write detect endpoint"


def test_with_task_none_clears() -> None:
    s = make_state().with_task("some_task").with_task(None)
    assert s.active_task_type is None


def test_with_turn_increments_count() -> None:
    s = make_state()
    s = s.with_turn().with_turn().with_turn()
    assert s.turn_count == 3


def test_with_turn_updates_last_active() -> None:
    s = make_state()
    before = s.last_active_at
    s2 = s.with_turn()
    assert s2.last_active_at >= before


def test_with_metrics_replaces_metrics() -> None:
    s = make_state()
    new_metrics = s.metrics.add_call(tokens_in=10, tokens_out=5, cost_usd=0.0, is_local=True)
    s2 = s.with_metrics(new_metrics)
    assert s2.metrics.total_llm_calls == 1


def test_with_scratch_sets_value() -> None:
    s = make_state().with_scratch("last_file", "api/detect.py")
    assert s.scratch["last_file"] == "api/detect.py"


def test_with_scratch_merges_values() -> None:
    s = make_state().with_scratch("a", 1).with_scratch("b", 2)
    assert s.scratch == {"a": 1, "b": 2}


def test_mutations_return_new_instances() -> None:
    original = make_state()
    mutated = original.with_turn()
    assert original is not mutated
    assert original.turn_count == 0


def test_is_fresh_false_after_turn() -> None:
    s = make_state().with_turn()
    assert s.is_fresh is False


def test_duration_seconds_positive() -> None:
    s = make_state()
    s = s.with_turn().with_turn()
    assert s.duration_seconds >= 0.0


# ── JSON round-trip ───────────────────────────────────────────────────────────

def test_state_round_trips_json() -> None:
    original = (
        make_state()
        .with_project("agrivision")
        .with_task("create_api_endpoint", "Write plant detection route")
        .with_turn()
        .with_scratch("key", "value")
    )
    restored = SessionState.model_validate_json(original.model_dump_json())
    assert restored.session_id == original.session_id
    assert restored.active_project == original.active_project
    assert restored.turn_count == original.turn_count
    assert restored.scratch == original.scratch


# ── SessionManager: new_session ───────────────────────────────────────────────

def test_new_session_creates_state(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    state = mgr.new_session(session_id="sess-abc")
    assert state.session_id == "sess-abc"
    assert mgr.is_loaded


def test_new_session_auto_generates_id(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    state = mgr.new_session()
    assert state.session_id.startswith("sess-")


def test_new_session_creates_file(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    mgr.new_session()
    assert (tmp_path / "session.json").exists()


def test_new_session_file_is_valid_json(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    mgr.new_session(session_id="sess-json-test")
    raw = (tmp_path / "session.json").read_text()
    data = json.loads(raw)
    assert data["session_id"] == "sess-json-test"


# ── SessionManager: load ──────────────────────────────────────────────────────

def test_load_returns_none_when_no_file(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    result = mgr.load()
    assert result is None
    assert not mgr.is_loaded


def test_load_restores_saved_state(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    original = mgr.new_session(session_id="sess-restore")
    mgr.update(original.with_project("agrivision").with_turn().with_turn())

    mgr2 = make_manager(tmp_path)
    loaded = mgr2.load()
    assert loaded is not None
    assert loaded.session_id == "sess-restore"
    assert loaded.active_project == "agrivision"
    assert loaded.turn_count == 2


def test_load_handles_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    path.write_text("{ this is not valid json }", encoding="utf-8")
    mgr = SessionManager(state_path=path)
    result = mgr.load()
    assert result is None


# ── SessionManager: save / update ─────────────────────────────────────────────

def test_save_is_atomic_via_temp_file(tmp_path: Path) -> None:
    """Verify no .session_tmp_ files are left behind after save."""
    mgr = make_manager(tmp_path)
    mgr.new_session()
    mgr.update(mgr.state.with_turn())
    tmp_files = list(tmp_path.glob(".session_tmp_*"))
    assert tmp_files == []


def test_update_persists_and_returns_state(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    mgr.new_session(session_id="sess-update")
    new_state = mgr.update(mgr.state.with_project("plant-chat"))
    assert new_state.active_project == "plant-chat"
    # Verify it was written
    mgr2 = make_manager(tmp_path)
    loaded = mgr2.load()
    assert loaded is not None
    assert loaded.active_project == "plant-chat"


def test_save_without_state_raises(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    with pytest.raises(RuntimeError, match="No state to save"):
        mgr.save()


# ── SessionManager: state property ────────────────────────────────────────────

def test_state_property_raises_before_init(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    with pytest.raises(RuntimeError, match="not initialised"):
        _ = mgr.state


def test_state_property_returns_after_new_session(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    mgr.new_session()
    assert mgr.state is not None


# ── SessionManager: reset ─────────────────────────────────────────────────────

def test_reset_clears_in_memory_state(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    mgr.new_session()
    mgr.reset()
    assert not mgr.is_loaded


def test_reset_does_not_delete_file(tmp_path: Path) -> None:
    mgr = make_manager(tmp_path)
    mgr.new_session()
    mgr.reset()
    assert (tmp_path / "session.json").exists()