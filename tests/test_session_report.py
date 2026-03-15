"""
tests/test_session_report.py

Tests for session_report.py — data loading helpers,
summary computation, and CLI output modes.
All file I/O uses tmp_path.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from scripts.session_report import (
    SessionSummary,
    _build_current_session_summary,
    _build_event_stats,
    _load_consolidation_state,
    _load_episodic_events,
    _load_session_state,
    app,
)

runner = CliRunner()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session_json(
    session_id: str = "sess-abc",
    turn_count: int = 5,
    tokens_in: int = 1000,
    tokens_out: int = 500,
    cost: float = 0.002,
    local: int = 4,
    cloud: int = 1,
) -> dict:
    return {
        "session_id": session_id,
        "started_at": "2026-03-14T10:00:00+00:00",
        "last_active_at": "2026-03-14T10:30:00+00:00",
        "turn_count": turn_count,
        "active_project": "agrivision",
        "metrics": {
            "total_llm_calls": local + cloud,
            "local_calls": local,
            "cloud_calls": cloud,
            "total_tokens_in": tokens_in,
            "total_tokens_out": tokens_out,
            "total_cost_usd": cost,
        },
    }


def write_session(root: Path, data: dict) -> None:
    path = root / "memory" / "working" / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_consolidation(root: Path, data: dict) -> None:
    path = root / "consolidation" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_event(root: Path, event_type: str, project: str | None = None, data: dict | None = None) -> None:
    from datetime import datetime, timezone
    ts   = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    year, month, day = ts.split("/")
    dir_ = root / "memory" / "episodic" / year / month / day
    dir_.mkdir(parents=True, exist_ok=True)
    import time, random, string
    eid  = f"evt-{int(time.time()*1000)}-{''.join(random.choices(string.hexdigits[:16], k=8))}"
    path = dir_ / f"{eid}.json"
    path.write_text(json.dumps({
        "id": eid,
        "event_type": event_type,
        "project": project,
        "summary": f"test event {event_type}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data or {},
    }))


# ── SessionSummary ────────────────────────────────────────────────────────────

def test_total_tokens() -> None:
    s = SessionSummary("s1", tokens_in=300, tokens_out=100)
    assert s.total_tokens == 400


def test_duration_minutes() -> None:
    s = SessionSummary(
        "s1",
        started_at=datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 14, 10, 30, tzinfo=timezone.utc),
    )
    assert s.duration_minutes == pytest.approx(30.0)


def test_duration_no_timestamps() -> None:
    s = SessionSummary("s1")
    assert s.duration_minutes == 0.0


# ── _load_session_state ───────────────────────────────────────────────────────

def test_load_session_state_returns_dict(tmp_path: Path) -> None:
    write_session(tmp_path, make_session_json())
    result = _load_session_state(tmp_path)
    assert result is not None
    assert result["session_id"] == "sess-abc"


def test_load_session_state_returns_none_when_missing(tmp_path: Path) -> None:
    result = _load_session_state(tmp_path)
    assert result is None


def test_load_session_state_returns_none_on_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "working" / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")
    result = _load_session_state(tmp_path)
    assert result is None


# ── _load_consolidation_state ─────────────────────────────────────────────────

def test_load_consolidation_state_returns_dict(tmp_path: Path) -> None:
    write_consolidation(tmp_path, {"total_runs": 3, "total_patterns_extracted": 12})
    result = _load_consolidation_state(tmp_path)
    assert result["total_runs"] == 3


def test_load_consolidation_state_empty_when_missing(tmp_path: Path) -> None:
    result = _load_consolidation_state(tmp_path)
    assert result == {}


# ── _load_episodic_events ─────────────────────────────────────────────────────

def test_load_episodic_events_returns_events(tmp_path: Path) -> None:
    write_event(tmp_path, "code.write", project="agrivision")
    write_event(tmp_path, "llm.call",   project="agrivision")
    events = _load_episodic_events(tmp_path, project=None, limit=100)
    assert len(events) == 2


def test_load_episodic_events_project_filter(tmp_path: Path) -> None:
    write_event(tmp_path, "code.write", project="agrivision")
    write_event(tmp_path, "code.write", project="plant-chat")
    events = _load_episodic_events(tmp_path, project="agrivision", limit=100)
    assert all(e["project"] == "agrivision" for e in events)


def test_load_episodic_events_respects_limit(tmp_path: Path) -> None:
    for i in range(5):
        write_event(tmp_path, f"event.{i}")
    events = _load_episodic_events(tmp_path, project=None, limit=3)
    assert len(events) == 3


def test_load_episodic_events_empty_dir(tmp_path: Path) -> None:
    events = _load_episodic_events(tmp_path, project=None, limit=100)
    assert events == []


# ── _build_current_session_summary ───────────────────────────────────────────

def test_build_session_summary_fields() -> None:
    data = make_session_json(
        session_id="sess-xyz",
        turn_count=7,
        tokens_in=2000,
        tokens_out=800,
        cost=0.005,
        local=6,
        cloud=1,
    )
    s = _build_current_session_summary(data)
    assert s.session_id == "sess-xyz"
    assert s.turn_count == 7
    assert s.tokens_in == 2000
    assert s.tokens_out == 800
    assert s.cost_usd == pytest.approx(0.005)
    assert s.local_calls == 6
    assert s.cloud_calls == 1
    assert s.total_llm_calls == 7


def test_build_session_summary_parses_timestamps() -> None:
    data = make_session_json()
    s = _build_current_session_summary(data)
    assert s.started_at is not None
    assert s.started_at.tzinfo == timezone.utc


def test_build_session_summary_handles_missing_metrics() -> None:
    data = {"session_id": "sess-minimal", "turn_count": 0}
    s = _build_current_session_summary(data)
    assert s.session_id == "sess-minimal"
    assert s.total_llm_calls == 0
    assert s.cost_usd == 0.0


# ── _build_event_stats ────────────────────────────────────────────────────────

def test_build_event_stats_counts_types() -> None:
    events = [
        {"event_type": "code.write", "data": {}},
        {"event_type": "code.write", "data": {}},
        {"event_type": "llm.call",   "data": {}},
    ]
    stats = _build_event_stats(events)
    assert stats["type_counts"]["code.write"] == 2
    assert stats["type_counts"]["llm.call"] == 1


def test_build_event_stats_aggregates_tokens() -> None:
    events = [
        {"event_type": "llm.call", "data": {"tokens_in": 100, "tokens_out": 50, "is_local": True}},
        {"event_type": "llm.call", "data": {"tokens_in": 200, "tokens_out": 80, "is_local": False, "cost_usd": 0.002}},
    ]
    stats = _build_event_stats(events)
    assert stats["total_tokens_in"]  == 300
    assert stats["total_tokens_out"] == 130
    assert stats["local_calls"]  == 1
    assert stats["cloud_calls"]  == 1
    assert stats["total_cost"]   == pytest.approx(0.002)


def test_build_event_stats_counts_projects() -> None:
    events = [
        {"event_type": "code.write", "project": "agrivision", "data": {}},
        {"event_type": "code.write", "project": "agrivision", "data": {}},
        {"event_type": "code.write", "project": "plant-chat", "data": {}},
    ]
    stats = _build_event_stats(events)
    assert stats["project_counts"]["agrivision"] == 2
    assert stats["project_counts"]["plant-chat"] == 1


def test_build_event_stats_empty_events() -> None:
    stats = _build_event_stats([])
    assert stats["total_tokens_in"] == 0
    assert stats["total_cost"] == 0.0
    assert stats["type_counts"] == {}


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_runs_with_no_data(tmp_path: Path) -> None:
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, [])
    assert result.exit_code == 0


def test_cli_shows_session_id(tmp_path: Path) -> None:
    write_session(tmp_path, make_session_json(session_id="sess-report-test"))
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "sess-report-test" in result.output


def test_cli_shows_token_counts(tmp_path: Path) -> None:
    write_session(tmp_path, make_session_json(tokens_in=1234, tokens_out=567))
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, [])
    assert "1,234" in result.output
    assert "567" in result.output


def test_cli_shows_cost(tmp_path: Path) -> None:
    write_session(tmp_path, make_session_json(cost=0.0042))
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, [])
    assert "0.0042" in result.output


def test_cli_shows_consolidation_runs(tmp_path: Path) -> None:
    write_consolidation(tmp_path, {"total_runs": 7, "total_patterns_extracted": 42})
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, [])
    assert "7" in result.output
    assert "42" in result.output


def test_cli_project_filter(tmp_path: Path) -> None:
    write_event(tmp_path, "code.write", project="agrivision")
    write_event(tmp_path, "code.write", project="plant-chat")
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, ["--project", "agrivision"])
    assert result.exit_code == 0
    assert "agrivision" in result.output


def test_cli_csv_output(tmp_path: Path) -> None:
    write_session(tmp_path, make_session_json(tokens_in=500, tokens_out=200, cost=0.001))
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, ["--csv"])
    assert result.exit_code == 0
    # Should be valid CSV with a header row
    lines = [l for l in result.output.strip().splitlines() if l]
    assert len(lines) >= 2
    reader = csv.DictReader(io.StringIO(result.output))
    rows = list(reader)
    assert len(rows) == 1
    assert "session_id" in rows[0]
    assert rows[0]["session_id"] == "sess-abc"


def test_cli_csv_no_session(tmp_path: Path) -> None:
    """CSV should still produce valid output even with no session state."""
    with patch("scripts.session_report._agent_root", return_value=tmp_path):
        result = runner.invoke(app, ["--csv"])
    assert result.exit_code == 0
    lines = [l for l in result.output.strip().splitlines() if l]
    assert len(lines) >= 1  # at least headers