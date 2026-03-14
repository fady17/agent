"""
tests/test_episodic.py

Tests for the episodic memory schema and filesystem operations.
All I/O is redirected to pytest's tmp_path — never touches ~/.agent/.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent.memory.episodic import (
    EpisodicEvent,
    EventType,
    list_events,
    read_event,
    write_event,
    _new_event_id,
)


# ── ID generation ─────────────────────────────────────────────────────────────

def test_event_id_format() -> None:
    eid = _new_event_id()
    assert eid.startswith("evt-")
    parts = eid.split("-")
    assert len(parts) == 3
    assert len(parts[1]) == 13   # unix ms, 10^13 range
    assert len(parts[2]) == 8    # random hex


def test_event_ids_are_sortable() -> None:
    # Mock time.time to return strictly increasing seconds
    with patch("agent.memory.episodic.time.time", side_effect=[1.0, 2.0, 3.0, 4.0, 5.0]):
        ids =[_new_event_id() for _ in range(5)]
    assert ids == sorted(ids), "IDs must be lexicographically sortable by time"
def test_event_ids_are_unique() -> None:
    ids = {_new_event_id() for _ in range(100)}
    assert len(ids) == 100


# ── Schema construction ───────────────────────────────────────────────────────

def test_minimal_event_constructs() -> None:
    evt = EpisodicEvent(
        event_type=EventType.CODE_WRITE,
        summary="Wrote FastAPI endpoint for plant detection",
    )
    assert evt.event_type == EventType.CODE_WRITE
    assert evt.id.startswith("evt-")
    assert evt.timestamp.tzinfo is not None
    assert evt.project is None
    assert evt.data == {}


def test_full_event_constructs() -> None:
    evt = EpisodicEvent(
        event_type=EventType.LLM_CALL,
        summary="Called Qwen3 for task classification",
        project="agrivision",
        data={"model": "qwen3", "tokens": 512, "latency_ms": 340},
    )
    assert evt.project == "agrivision"
    assert evt.data["tokens"] == 512


def test_timestamp_defaults_to_utc() -> None:
    evt = EpisodicEvent(event_type=EventType.SESSION_START, summary="Agent started")
    assert evt.timestamp.tzinfo == timezone.utc


def test_naive_timestamp_assumed_utc() -> None:
    naive = datetime(2026, 3, 14, 10, 0, 0)
    evt = EpisodicEvent(
        event_type=EventType.CODE_RUN,
        summary="ran tests",
        timestamp=naive,
    )
    assert evt.timestamp.tzinfo == timezone.utc


def test_project_normalised_to_lowercase() -> None:
    evt = EpisodicEvent(
        event_type=EventType.CODE_WRITE,
        summary="wrote something",
        project="  AgriVision  ",
    )
    assert evt.project == "agrivision"


def test_empty_project_becomes_none() -> None:
    evt = EpisodicEvent(
        event_type=EventType.CODE_WRITE,
        summary="wrote something",
        project="   ",
    )
    assert evt.project is None


# ── Validation errors ─────────────────────────────────────────────────────────

def test_empty_summary_raises() -> None:
    with pytest.raises(ValidationError, match="summary"):
        EpisodicEvent(event_type=EventType.ERROR, summary="   ")


def test_summary_too_long_raises() -> None:
    with pytest.raises(ValidationError, match="summary"):
        EpisodicEvent(event_type=EventType.ERROR, summary="x" * 501)


def test_invalid_event_type_raises() -> None:
    with pytest.raises(ValidationError):
        EpisodicEvent(event_type="not.a.real.type", summary="test")  # type: ignore


# ── Serialisation ─────────────────────────────────────────────────────────────

def test_round_trip_json() -> None:
    original = EpisodicEvent(
        event_type=EventType.GIT_COMMIT,
        summary="chore: task 07 complete",
        project="agent",
        data={"hash": "abc1234", "files_changed": 3},
    )
    raw = original.to_json()
    restored = EpisodicEvent.from_json(raw)

    assert restored.id == original.id
    assert restored.event_type == original.event_type
    assert restored.summary == original.summary
    assert restored.project == original.project
    assert restored.data == original.data
    assert restored.timestamp == original.timestamp


def test_json_is_valid() -> None:
    evt = EpisodicEvent(event_type=EventType.SHELL_RUN, summary="ran ruff linter")
    parsed = json.loads(evt.to_json())
    assert "id" in parsed
    assert "timestamp" in parsed
    assert "event_type" in parsed
    assert "summary" in parsed


# ── File path ─────────────────────────────────────────────────────────────────

def test_file_path_structure(tmp_path: Path) -> None:
    evt = EpisodicEvent(
        event_type=EventType.CODE_WRITE,
        summary="wrote config",
        timestamp=datetime(2026, 3, 14, 10, 30, 0, tzinfo=timezone.utc),
    )
    path = evt.file_path(tmp_path)
    assert path.parent == tmp_path / "2026" / "03" / "14"
    assert path.name == f"{evt.id}.json"
    assert path.suffix == ".json"


# ── Writer / reader ───────────────────────────────────────────────────────────

def test_write_creates_file(tmp_path: Path) -> None:
    evt = EpisodicEvent(event_type=EventType.SESSION_START, summary="session began")
    path = write_event(evt, base_dir=tmp_path)
    assert path.exists()
    assert path.is_file()


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    evt = EpisodicEvent(
        event_type=EventType.CODE_RUN,
        summary="ran pytest",
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    path = write_event(evt, base_dir=tmp_path)
    assert (tmp_path / "2026" / "06" / "01").is_dir()


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    original = EpisodicEvent(
        event_type=EventType.DESIGN_EXPORT,
        summary="Exported plant_v02.glb",
        project="agrivision",
        data={"format": "glb", "poly_count": 48200},
    )
    path = write_event(original, base_dir=tmp_path)
    restored = read_event(path)

    assert restored.id == original.id
    assert restored.summary == original.summary
    assert restored.data["poly_count"] == 48200


# ── list_events ───────────────────────────────────────────────────────────────

def make_evt(
    tmp: Path,
    event_type: EventType,
    summary: str,
    project: str | None = None,
    ts: datetime | None = None,
) -> EpisodicEvent:
    evt = EpisodicEvent(
        event_type=event_type,
        summary=summary,
        project=project,
        timestamp=ts or datetime.now(timezone.utc),
    )
    write_event(evt, base_dir=tmp)
    return evt


def test_list_events_returns_all(tmp_path: Path) -> None:
    for i in range(5):
        make_evt(tmp_path, EventType.CODE_WRITE, f"wrote file {i}")
    results = list_events(tmp_path)
    assert len(results) == 5


def test_list_events_newest_first(tmp_path: Path) -> None:
    base = datetime(2026, 3, 14, tzinfo=timezone.utc)
    
    # Generator that yields an endlessly increasing timestamp
    def time_gen():
        t = 100.0
        while True:
            yield t
            t += 1.0

    # Pass the instantiated generator to side_effect
    with patch("agent.memory.episodic.time.time", side_effect=time_gen()):
        for i in range(3):
            make_evt(tmp_path, EventType.CODE_RUN, f"run {i}",
                     ts=base + timedelta(minutes=i))
            
    results = list_events(tmp_path)
    timestamps =[r.timestamp for r in results]
    assert timestamps == sorted(timestamps, reverse=True)

    
def test_list_events_filter_by_project(tmp_path: Path) -> None:
    make_evt(tmp_path, EventType.CODE_WRITE, "agrivision task", project="agrivision")
    make_evt(tmp_path, EventType.CODE_WRITE, "other task", project="plant-chat")
    make_evt(tmp_path, EventType.CODE_WRITE, "no project task")

    results = list_events(tmp_path, project="agrivision")
    assert len(results) == 1
    assert results[0].project == "agrivision"


def test_list_events_filter_by_type(tmp_path: Path) -> None:
    make_evt(tmp_path, EventType.LLM_CALL, "called llm")
    make_evt(tmp_path, EventType.CODE_WRITE, "wrote code")
    make_evt(tmp_path, EventType.LLM_CALL, "called llm again")

    results = list_events(tmp_path, event_type=EventType.LLM_CALL)
    assert len(results) == 2
    assert all(r.event_type == EventType.LLM_CALL for r in results)


def test_list_events_filter_by_since(tmp_path: Path) -> None:
    base = datetime(2026, 3, 14, tzinfo=timezone.utc)
    make_evt(tmp_path, EventType.CODE_WRITE, "old event", ts=base)
    make_evt(tmp_path, EventType.CODE_WRITE, "new event", ts=base + timedelta(hours=2))

    results = list_events(tmp_path, since=base + timedelta(hours=1))
    assert len(results) == 1
    assert results[0].summary == "new event"


def test_list_events_filter_by_until(tmp_path: Path) -> None:
    base = datetime(2026, 3, 14, tzinfo=timezone.utc)
    make_evt(tmp_path, EventType.CODE_WRITE, "early", ts=base)
    make_evt(tmp_path, EventType.CODE_WRITE, "late", ts=base + timedelta(hours=5))

    results = list_events(tmp_path, until=base + timedelta(hours=3))
    assert len(results) == 1
    assert results[0].summary == "early"


def test_list_events_limit(tmp_path: Path) -> None:
    for i in range(10):
        make_evt(tmp_path, EventType.SHELL_RUN, f"ran command {i}")
    results = list_events(tmp_path, limit=3)
    assert len(results) == 3


def test_list_events_empty_dir(tmp_path: Path) -> None:
    results = list_events(tmp_path / "nonexistent")
    assert results == []


def test_list_events_combined_filters(tmp_path: Path) -> None:
    base = datetime(2026, 3, 14, tzinfo=timezone.utc)
    make_evt(tmp_path, EventType.CODE_WRITE, "agri old",
             project="agrivision", ts=base)
    make_evt(tmp_path, EventType.CODE_WRITE, "agri new",
             project="agrivision", ts=base + timedelta(hours=2))
    make_evt(tmp_path, EventType.LLM_CALL, "agri llm",
             project="agrivision", ts=base + timedelta(hours=3))
    make_evt(tmp_path, EventType.CODE_WRITE, "other new",
             project="plant-chat", ts=base + timedelta(hours=2))

    results = list_events(
        tmp_path,
        project="agrivision",
        event_type=EventType.CODE_WRITE,
        since=base + timedelta(hours=1),
    )
    assert len(results) == 1
    assert results[0].summary == "agri new"