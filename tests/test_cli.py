"""
tests/test_cli.py

Tests for CLI commands using typer's CliRunner.
The `chat` command is not tested here (it requires an interactive loop
and live orchestrator) — it is covered by the end-to-end smoke test.
Tested: memory show, memory search, memory skills, memory delete, status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from agent.interface.cli import app
from agent.memory.episodic import EpisodicEvent, EventType
from agent.memory.skills import SkillRecord

runner = CliRunner()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_event(
    summary: str = "Wrote FastAPI endpoint",
    project: str | None = "agrivision",
    event_type: EventType = EventType.CODE_WRITE,
) -> EpisodicEvent:
    return EpisodicEvent(
        event_type=event_type,
        summary=summary,
        project=project,
        timestamp=datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc),
    )


def make_skill(
    task_type: str = "create_api_endpoint",
    pattern: str   = "Use FastAPI with async def",
    confidence: float = 0.85,
) -> SkillRecord:
    return SkillRecord(task_type=task_type, pattern=pattern, confidence=confidence)


# ── memory show ───────────────────────────────────────────────────────────────

def test_memory_show_no_events() -> None:
    with patch("agent.interface.cli.memory_show.__wrapped__", side_effect=None), \
         patch("agent.memory.episodic.list_events", return_value=[]):
        result = runner.invoke(app, ["memory", "show"])
    assert result.exit_code == 0


def test_memory_show_with_events() -> None:
    events = [make_event(f"event {i}") for i in range(3)]
    with patch("agent.memory.episodic.list_events", return_value=events):
        result = runner.invoke(app, ["memory", "show"])
    assert result.exit_code == 0
    assert "code.write" in result.output


def test_memory_show_project_filter() -> None:
    events = [make_event("agrivision event", project="agrivision")]
    with patch("agent.memory.episodic.list_events", return_value=events) as mock_list:
        result = runner.invoke(app, ["memory", "show", "--project", "agrivision"])
    assert result.exit_code == 0
    mock_list.assert_called_once()
    _, kwargs = mock_list.call_args
    assert kwargs.get("project") == "agrivision"


def test_memory_show_limit_flag() -> None:
    with patch("agent.memory.episodic.list_events", return_value=[]) as mock_list:
        runner.invoke(app, ["memory", "show", "--limit", "5"])
    _, kwargs = mock_list.call_args
    assert kwargs.get("limit") == 5


def test_memory_show_invalid_event_type() -> None:
    result = runner.invoke(app, ["memory", "show", "--type", "invalid.type"])
    assert result.exit_code != 0
    assert "Unknown event type" in result.output


def test_memory_show_valid_event_type() -> None:
    with patch("agent.memory.episodic.list_events", return_value=[]):
        result = runner.invoke(app, ["memory", "show", "--type", "code.write"])
    assert result.exit_code == 0


# ── memory search ─────────────────────────────────────────────────────────────

def test_memory_search_no_results() -> None:
    with patch("agent.memory.search.search_and_load", new=AsyncMock(return_value=[])):
        result = runner.invoke(app, ["memory", "search", "nonexistent"])
    assert result.exit_code == 0
    assert "No matching" in result.output


def test_memory_search_with_results() -> None:
    events = [make_event("Found FastAPI result")]
    with patch("agent.memory.search.search_and_load", new=AsyncMock(return_value=events)):
        result = runner.invoke(app, ["memory", "search", "FastAPI"])
    assert result.exit_code == 0
    assert "FastAPI" in result.output


def test_memory_search_shows_query() -> None:
    with patch("agent.memory.search.search_and_load", new=AsyncMock(return_value=[])):
        result = runner.invoke(app, ["memory", "search", "my query"])
    assert "my query" in result.output


# ── memory skills ─────────────────────────────────────────────────────────────

def test_memory_skills_no_records() -> None:
    with patch("agent.memory.skills.list_skills", return_value=[]):
        result = runner.invoke(app, ["memory", "skills"])
    assert result.exit_code == 0
    assert "No skill" in result.output


def test_memory_skills_shows_records() -> None:
    skills = [
        make_skill("create_api", "use FastAPI", 0.9),
        make_skill("export_3d",  "GLB format",  0.6),
    ]
    with patch("agent.memory.skills.list_skills", return_value=skills):
        result = runner.invoke(app, ["memory", "skills"])
    assert result.exit_code == 0
    assert "create_api" in result.output
    assert "0.90" in result.output


def test_memory_skills_min_confidence_passed() -> None:
    with patch("agent.memory.skills.list_skills", return_value=[]) as mock_list:
        runner.invoke(app, ["memory", "skills", "--min", "0.7"])
    mock_list.assert_called_once_with(min_confidence=0.7)


# ── memory delete ─────────────────────────────────────────────────────────────

def test_memory_delete_not_found(tmp_path: Path) -> None:
    with patch("agent.interface.cli.cfg") as mock_cfg:
        mock_cfg.episodic_dir = tmp_path
        result = runner.invoke(app, ["memory", "delete", "evt-nonexistent", "--yes"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_memory_delete_found_and_confirmed(tmp_path: Path) -> None:
    event_id = "evt-1741910400123-a3f2c1"
    event_file = tmp_path / f"{event_id}.json"
    event_file.write_text('{"test": true}')

    with patch("agent.interface.cli.cfg") as mock_cfg:
        mock_cfg.episodic_dir = tmp_path
        result = runner.invoke(app, ["memory", "delete", event_id, "--yes"])

    assert result.exit_code == 0
    assert not event_file.exists()
    assert "Deleted" in result.output


def test_memory_delete_cancelled_without_yes_flag(tmp_path: Path) -> None:
    event_id = "evt-test-12345678"
    event_file = tmp_path / f"{event_id}.json"
    event_file.write_text("{}")

    with patch("agent.interface.cli.cfg") as mock_cfg:
        mock_cfg.episodic_dir = tmp_path
        # Simulate user typing 'n' at confirmation prompt
        result = runner.invoke(app, ["memory", "delete", event_id], input="n\n")

    assert event_file.exists()  # not deleted
    assert "Cancelled" in result.output


# ── status ────────────────────────────────────────────────────────────────────

def test_status_no_active_session() -> None:
    from agent.background.consolidation import ConsolidationState
    from agent.core.session import SessionManager

    mock_mgr = MagicMock(spec=SessionManager)
    mock_mgr.load.return_value = None

    with patch("agent.interface.cli.SessionManager", return_value=mock_mgr), \
         patch("agent.memory.skills.list_skills", return_value=[]), \
         patch("agent.memory.episodic.list_events", return_value=[]), \
         patch("agent.interface.cli.load_consolidation_state",
               return_value=ConsolidationState()):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Session" in result.output


def test_status_with_active_session() -> None:
    from agent.background.consolidation import ConsolidationState
    from agent.core.session import SessionManager, SessionState

    state = SessionState(session_id="sess-test-abc")
    state = state.with_project("agrivision").with_turn().with_turn()

    mock_mgr = MagicMock(spec=SessionManager)
    mock_mgr.load.return_value = state

    with patch("agent.interface.cli.SessionManager", return_value=mock_mgr), \
         patch("agent.memory.skills.list_skills", return_value=[make_skill()]), \
         patch("agent.memory.episodic.list_events", return_value=[make_event()]), \
         patch("agent.interface.cli.load_consolidation_state",
               return_value=ConsolidationState(total_runs=3, total_patterns_extracted=12)), \
         patch("agent.interface.cli.cfg") as mock_cfg:
        mock_cfg.graph_path   = Path("/nonexistent/graph.json")
        mock_cfg.faiss_index_path = Path("/nonexistent/index.faiss")
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "sess-test-abc" in result.output
    assert "agrivision"    in result.output
    assert "3"             in result.output   # total runs


def test_status_shows_memory_counts() -> None:
    from agent.background.consolidation import ConsolidationState
    from agent.core.session import SessionManager

    mock_mgr = MagicMock(spec=SessionManager)
    mock_mgr.load.return_value = None

    events = [make_event() for _ in range(5)]
    skills = [make_skill(f"task_{i}") for i in range(3)]

    with patch("agent.interface.cli.SessionManager", return_value=mock_mgr), \
         patch("agent.memory.skills.list_skills", return_value=skills), \
         patch("agent.memory.episodic.list_events", return_value=events), \
         patch("agent.interface.cli.load_consolidation_state",
               return_value=ConsolidationState()), \
         patch("agent.interface.cli.cfg") as mock_cfg:
        mock_cfg.graph_path       = Path("/dev/null/graph.json")
        mock_cfg.faiss_index_path = Path("/dev/null/index.faiss")
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "5" in result.output   # event count
    assert "3" in result.output   # skill count


def test_status_consolidation_never_run() -> None:
    from agent.background.consolidation import ConsolidationState
    from agent.core.session import SessionManager

    mock_mgr = MagicMock(spec=SessionManager)
    mock_mgr.load.return_value = None

    with patch("agent.interface.cli.SessionManager", return_value=mock_mgr), \
         patch("agent.memory.skills.list_skills", return_value=[]), \
         patch("agent.memory.episodic.list_events", return_value=[]), \
         patch("agent.interface.cli.load_consolidation_state",
               return_value=ConsolidationState()), \
         patch("agent.interface.cli.cfg") as mock_cfg:
        mock_cfg.graph_path       = Path("/dev/null/g.json")
        mock_cfg.faiss_index_path = Path("/dev/null/i.faiss")
        result = runner.invoke(app, ["status"])

    assert "never" in result.output