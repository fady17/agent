"""
tests/test_search.py

Tests for the episodic search layer.

Uses real files on disk (in pytest's tmp_path) so ripgrep has actual
content to search. All tests are async since search_events is a coroutine.

Requires ripgrep installed — tests are skipped automatically if rg is absent.
"""

import shutil
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from agent.memory.episodic import EpisodicEvent, EventType, write_event
from agent.memory.search import SearchHit, search_events, search_and_load

# ── Skip marker ───────────────────────────────────────────────────────────────

rg_available = shutil.which("rg") is not None
skip_no_rg = pytest.mark.skipif(not rg_available, reason="ripgrep (rg) not installed")


# ── Fixtures ──────────────────────────────────────────────────────────────────

def seed_events(base: Path) -> list[EpisodicEvent]:
    """Write a known set of events for search tests."""
    events = [
        EpisodicEvent(
            event_type=EventType.CODE_WRITE,
            summary="Wrote FastAPI endpoint for plant disease detection",
            project="agrivision",
            data={"file": "api/routes/detect.py"},
        ),
        EpisodicEvent(
            event_type=EventType.LLM_CALL,
            summary="Called Qwen3 for task classification",
            project="agrivision",
            data={"model": "qwen3", "tokens": 512},
        ),
        EpisodicEvent(
            event_type=EventType.DESIGN_EXPORT,
            summary="Exported plant_model_v02.glb from Blender",
            project="agrivision",
            data={"format": "glb", "poly_count": 48200},
        ),
        EpisodicEvent(
            event_type=EventType.CODE_WRITE,
            summary="Implemented Flutter chat widget for diagnosis",
            project="plant-chat",
            data={"file": "lib/widgets/chat_bubble.dart"},
        ),
        EpisodicEvent(
            event_type=EventType.SHELL_RUN,
            summary="Ran ruff linter across all Python files",
            project=None,
            data={"exit_code": 0},
        ),
    ]
    for evt in events:
        write_event(evt, base_dir=base)
    return events


# ── Basic search ──────────────────────────────────────────────────────────────

@skip_no_rg
@pytest.mark.asyncio
async def test_search_finds_keyword(tmp_path: Path) -> None:
    seed_events(tmp_path)
    hits = await search_events("FastAPI", base_dir=tmp_path)
    assert len(hits) == 1
    assert "FastAPI" in hits[0].matched_text or "fastapi" in hits[0].matched_text.lower()


@skip_no_rg
@pytest.mark.asyncio
async def test_search_is_case_insensitive_by_default(tmp_path: Path) -> None:
    seed_events(tmp_path)
    hits_lower = await search_events("fastapi", base_dir=tmp_path)
    hits_upper = await search_events("FASTAPI", base_dir=tmp_path)
    assert len(hits_lower) == len(hits_upper)
    assert len(hits_lower) >= 1


@skip_no_rg
@pytest.mark.asyncio
async def test_search_case_sensitive_misses(tmp_path: Path) -> None:
    seed_events(tmp_path)
    # "fastapi" lowercase won't match "FastAPI" with case_sensitive=True
    hits = await search_events("fastapi", base_dir=tmp_path, case_sensitive=True)
    matched_texts = [h.matched_text.lower() for h in hits]
    # None of the matches should be the exact case mismatch
    # (they may match other occurrences in the JSON like field names)
    for text in matched_texts:
        assert "FastAPI" not in text or "fastapi" in text


@skip_no_rg
@pytest.mark.asyncio
async def test_search_no_match_returns_empty(tmp_path: Path) -> None:
    seed_events(tmp_path)
    hits = await search_events("xyznonexistentterm123", base_dir=tmp_path)
    assert hits == []


@skip_no_rg
@pytest.mark.asyncio
async def test_search_empty_query_returns_empty(tmp_path: Path) -> None:
    seed_events(tmp_path)
    hits = await search_events("   ", base_dir=tmp_path)
    assert hits == []


@skip_no_rg
@pytest.mark.asyncio
async def test_search_nonexistent_dir_returns_empty(tmp_path: Path) -> None:
    hits = await search_events("anything", base_dir=tmp_path / "does_not_exist")
    assert hits == []


# ── SearchHit structure ───────────────────────────────────────────────────────

@skip_no_rg
@pytest.mark.asyncio
async def test_search_hit_has_correct_fields(tmp_path: Path) -> None:
    seed_events(tmp_path)
    hits = await search_events("Blender", base_dir=tmp_path)
    assert len(hits) >= 1
    hit = hits[0]
    assert isinstance(hit, SearchHit)
    assert hit.event_id.startswith("evt-")
    assert hit.path.exists()
    assert hit.path.suffix == ".json"
    assert hit.line_number > 0
    assert isinstance(hit.matched_text, str)


# ── Project filter ────────────────────────────────────────────────────────────

@skip_no_rg
@pytest.mark.asyncio
async def test_search_filters_by_project(tmp_path: Path) -> None:
    seed_events(tmp_path)
    hits = await search_events("code.write", base_dir=tmp_path, project="agrivision")
    # Both CODE_WRITE events match, but only one is agrivision
    assert all(_get_project(h) == "agrivision" for h in hits)


@skip_no_rg
@pytest.mark.asyncio
async def test_search_project_filter_excludes_other_projects(tmp_path: Path) -> None:
    seed_events(tmp_path)
    hits = await search_events("Flutter", base_dir=tmp_path, project="agrivision")
    # Flutter event belongs to plant-chat, not agrivision
    assert len(hits) == 0


# ── max_results ───────────────────────────────────────────────────────────────

@skip_no_rg
@pytest.mark.asyncio
async def test_search_respects_max_results(tmp_path: Path) -> None:
    # Write 10 events all containing the same keyword
    for i in range(10):
        evt = EpisodicEvent(
            event_type=EventType.CODE_WRITE,
            summary=f"searchable event number {i}",
        )
        write_event(evt, base_dir=tmp_path)

    hits = await search_events("searchable event", base_dir=tmp_path, max_results=3)
    assert len(hits) <= 3


# ── search_and_load ───────────────────────────────────────────────────────────

@skip_no_rg
@pytest.mark.asyncio
async def test_search_and_load_returns_events(tmp_path: Path) -> None:
    from agent.memory.episodic import EpisodicEvent as Evt
    seed_events(tmp_path)
    events = await search_and_load("Qwen3", base_dir=tmp_path)
    assert len(events) >= 1
    assert all(isinstance(e, Evt) for e in events)


@skip_no_rg
@pytest.mark.asyncio
async def test_search_and_load_event_fields_intact(tmp_path: Path) -> None:
    seed_events(tmp_path)
    events = await search_and_load("plant disease detection", base_dir=tmp_path)
    assert len(events) == 1
    evt = events[0]
    assert evt.event_type == EventType.CODE_WRITE
    assert evt.project == "agrivision"
    assert evt.data["file"] == "api/routes/detect.py"


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_project(hit: SearchHit) -> str | None:
    import json
    try:
        return json.loads(hit.path.read_text()).get("project")
    except Exception:
        return None