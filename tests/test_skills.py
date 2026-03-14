"""
tests/test_skills.py

Tests for the skill memory system.
All filesystem I/O uses pytest's tmp_path — never touches ~/.agent/memory/skills/.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.memory.skills import (
    CONFIDENCE_BOOST,
    CONFIDENCE_DECAY_FACTOR,
    CONFIDENCE_MAX,
    CONFIDENCE_PRUNE_THRESHOLD,
    SkillRecord,
    decay_pass,
    get_skill,
    list_skills,
    save_skill,
    upsert_skill,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_record(**kwargs) -> SkillRecord:
    defaults = dict(
        task_type="create_api_endpoint",
        pattern="Use FastAPI with async def and Pydantic models",
    )
    return SkillRecord(**{**defaults, **kwargs})


# ── Schema construction ───────────────────────────────────────────────────────

def test_minimal_record_constructs() -> None:
    r = make_record()
    assert r.task_type == "create_api_endpoint"
    assert r.confidence == 0.5
    assert r.observation_count == 1
    assert r.source_event_ids == []


def test_task_type_normalised() -> None:
    r = make_record(task_type="  Create API Endpoint  ")
    assert r.task_type == "create_api_endpoint"


def test_task_type_spaces_become_underscores() -> None:
    r = make_record(task_type="write flutter widget")
    assert r.task_type == "write_flutter_widget"


def test_empty_task_type_raises() -> None:
    with pytest.raises(ValidationError, match="task_type"):
        make_record(task_type="   ")


def test_empty_pattern_raises() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        make_record(pattern="  ")


def test_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        make_record(confidence=1.5)
    with pytest.raises(ValidationError):
        make_record(confidence=-0.1)


def test_naive_datetime_assumed_utc() -> None:
    naive = datetime(2026, 3, 14, 10, 0, 0)
    r = make_record(last_used=naive)
    assert r.last_used.tzinfo == timezone.utc


def test_is_reliable_above_threshold() -> None:
    r = make_record(confidence=CONFIDENCE_PRUNE_THRESHOLD + 0.01)
    assert r.is_reliable is True


def test_is_reliable_below_threshold() -> None:
    r = make_record(confidence=CONFIDENCE_PRUNE_THRESHOLD - 0.01)
    assert r.is_reliable is False


def test_days_since_used_recent() -> None:
    r = make_record(last_used=datetime.now(timezone.utc) - timedelta(hours=12))
    assert r.days_since_used < 1.0


def test_days_since_used_old() -> None:
    r = make_record(last_used=datetime.now(timezone.utc) - timedelta(days=7))
    assert 6.9 < r.days_since_used < 7.1


# ── with_observation ──────────────────────────────────────────────────────────

def test_with_observation_boosts_confidence() -> None:
    r = make_record(confidence=0.5)
    updated = r.with_observation()
    assert updated.confidence == pytest.approx(0.5 + CONFIDENCE_BOOST, abs=1e-5)


def test_with_observation_caps_at_max() -> None:
    r = make_record(confidence=0.99)
    updated = r.with_observation()
    assert updated.confidence == CONFIDENCE_MAX


def test_with_observation_increments_count() -> None:
    r = make_record(observation_count=3)
    updated = r.with_observation()
    assert updated.observation_count == 4


def test_with_observation_appends_event_id() -> None:
    r = make_record(source_event_ids=["evt-001"])
    updated = r.with_observation(event_id="evt-002")
    assert "evt-002" in updated.source_event_ids
    assert "evt-001" in updated.source_event_ids


def test_with_observation_no_duplicate_event_id() -> None:
    r = make_record(source_event_ids=["evt-001"])
    updated = r.with_observation(event_id="evt-001")
    assert updated.source_event_ids.count("evt-001") == 1


def test_with_observation_returns_new_instance() -> None:
    r = make_record(confidence=0.5)
    updated = r.with_observation()
    assert r is not updated
    assert r.confidence == 0.5  # original unchanged


# ── with_decay ────────────────────────────────────────────────────────────────

def test_with_decay_reduces_confidence() -> None:
    r = make_record(confidence=0.8)
    decayed = r.with_decay()
    assert decayed.confidence == pytest.approx(0.8 * CONFIDENCE_DECAY_FACTOR, abs=1e-5)


def test_with_decay_never_below_zero() -> None:
    r = make_record(confidence=0.001)
    for _ in range(100):
        r = r.with_decay()
    assert r.confidence >= 0.0


def test_with_decay_does_not_change_last_used() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r = make_record(last_used=ts)
    decayed = r.with_decay()
    assert decayed.last_used == ts


def test_with_decay_returns_new_instance() -> None:
    r = make_record(confidence=0.8)
    decayed = r.with_decay()
    assert r is not decayed
    assert r.confidence == 0.8


# ── Serialisation ─────────────────────────────────────────────────────────────

def test_round_trip_json() -> None:
    original = make_record(
        task_type="export_3d_asset",
        pattern="Export as GLB with draco compression, check ngons first",
        confidence=0.75,
        source_event_ids=["evt-001", "evt-002"],
    )
    restored = SkillRecord.from_json(original.to_json())
    assert restored.task_type == original.task_type
    assert restored.pattern == original.pattern
    assert restored.confidence == original.confidence
    assert restored.source_event_ids == original.source_event_ids


# ── get_skill / save_skill ────────────────────────────────────────────────────

def test_get_skill_returns_none_when_absent(tmp_path: Path) -> None:
    result = get_skill("nonexistent_task", tmp_path)
    assert result is None


def test_save_and_get_roundtrip(tmp_path: Path) -> None:
    record = make_record(task_type="debug_fastapi", confidence=0.7)
    save_skill(record, tmp_path)
    loaded = get_skill("debug_fastapi", tmp_path)
    assert loaded is not None
    assert loaded.task_type == "debug_fastapi"
    assert loaded.confidence == pytest.approx(0.7)


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested"
    record = make_record()
    save_skill(record, nested)
    assert (nested / "create_api_endpoint.json").exists()


def test_save_overwrites_existing(tmp_path: Path) -> None:
    record = make_record(confidence=0.5)
    save_skill(record, tmp_path)
    updated = make_record(confidence=0.9)
    save_skill(updated, tmp_path)
    loaded = get_skill("create_api_endpoint", tmp_path)
    assert loaded is not None
    assert loaded.confidence == pytest.approx(0.9)


# ── list_skills ───────────────────────────────────────────────────────────────

def test_list_skills_returns_all(tmp_path: Path) -> None:
    save_skill(make_record(task_type="task_a", confidence=0.8), tmp_path)
    save_skill(make_record(task_type="task_b", confidence=0.4), tmp_path)
    save_skill(make_record(task_type="task_c", confidence=0.6), tmp_path)
    results = list_skills(tmp_path)
    assert len(results) == 3


def test_list_skills_sorted_by_confidence_descending(tmp_path: Path) -> None:
    save_skill(make_record(task_type="low",  confidence=0.2), tmp_path)
    save_skill(make_record(task_type="high", confidence=0.9), tmp_path)
    save_skill(make_record(task_type="mid",  confidence=0.5), tmp_path)
    results = list_skills(tmp_path)
    confs = [r.confidence for r in results]
    assert confs == sorted(confs, reverse=True)


def test_list_skills_min_confidence_filter(tmp_path: Path) -> None:
    save_skill(make_record(task_type="low",  confidence=0.2), tmp_path)
    save_skill(make_record(task_type="high", confidence=0.9), tmp_path)
    results = list_skills(tmp_path, min_confidence=0.5)
    assert len(results) == 1
    assert results[0].task_type == "high"


def test_list_skills_empty_dir(tmp_path: Path) -> None:
    results = list_skills(tmp_path / "nonexistent")
    assert results == []


# ── upsert_skill ──────────────────────────────────────────────────────────────

def test_upsert_creates_new_record(tmp_path: Path) -> None:
    record = upsert_skill("new_task", "always use async", base_dir=tmp_path)
    assert record.task_type == "new_task"
    assert record.confidence == 0.5
    assert record.observation_count == 1


def test_upsert_boosts_existing_same_pattern(tmp_path: Path) -> None:
    pattern = "use FastAPI"
    upsert_skill("api_task", pattern, base_dir=tmp_path)
    upsert_skill("api_task", pattern, base_dir=tmp_path)
    record = get_skill("api_task", tmp_path)
    assert record is not None
    assert record.confidence > 0.5
    assert record.observation_count == 2


def test_upsert_resets_on_pattern_change(tmp_path: Path) -> None:
    upsert_skill("api_task", "old pattern", base_dir=tmp_path)
    # Force high confidence first
    save_skill(make_record(task_type="api_task", confidence=0.95), tmp_path)
    # Now upsert with different pattern
    record = upsert_skill("api_task", "new pattern", base_dir=tmp_path)
    assert record.confidence == 0.5
    assert record.observation_count == 1
    assert record.pattern == "new pattern"


def test_upsert_records_event_id(tmp_path: Path) -> None:
    record = upsert_skill(
        "task_with_event", "some pattern",
        event_id="evt-12345",
        base_dir=tmp_path,
    )
    assert "evt-12345" in record.source_event_ids


def test_upsert_accumulates_event_ids(tmp_path: Path) -> None:
    pattern = "consistent pattern"
    upsert_skill("tracked_task", pattern, event_id="evt-001", base_dir=tmp_path)
    upsert_skill("tracked_task", pattern, event_id="evt-002", base_dir=tmp_path)
    record = get_skill("tracked_task", tmp_path)
    assert record is not None
    assert "evt-001" in record.source_event_ids
    assert "evt-002" in record.source_event_ids


# ── decay_pass ────────────────────────────────────────────────────────────────

def test_decay_pass_reduces_all_confidences(tmp_path: Path) -> None:
    save_skill(make_record(task_type="task_a", confidence=0.8), tmp_path)
    save_skill(make_record(task_type="task_b", confidence=0.6), tmp_path)
    decay_pass(tmp_path, prune=False)
    a = get_skill("task_a", tmp_path)
    b = get_skill("task_b", tmp_path)
    assert a is not None and a.confidence < 0.8
    assert b is not None and b.confidence < 0.6


def test_decay_pass_prunes_low_confidence(tmp_path: Path) -> None:
    # Set confidence just above threshold then decay it below
    # (e.g. 0.10 + 0.001 = 0.101. After 0.98 decay, it becomes 0.09898, which is < 0.10)
    low_conf = CONFIDENCE_PRUNE_THRESHOLD + 0.001
    save_skill(make_record(task_type="dying_task", confidence=low_conf), tmp_path)
    save_skill(make_record(task_type="healthy_task", confidence=0.9), tmp_path)

    result = decay_pass(tmp_path, prune=True)

    assert get_skill("dying_task", tmp_path) is None
    assert get_skill("healthy_task", tmp_path) is not None
    assert result["pruned"] == 1
    assert result["remaining"] == 1
    

def test_decay_pass_no_prune_keeps_all(tmp_path: Path) -> None:
    save_skill(make_record(task_type="task_a", confidence=0.05), tmp_path)
    result = decay_pass(tmp_path, prune=False)
    assert result["pruned"] == 0
    assert get_skill("task_a", tmp_path) is not None


def test_decay_pass_returns_summary(tmp_path: Path) -> None:
    save_skill(make_record(task_type="t1", confidence=0.8), tmp_path)
    save_skill(make_record(task_type="t2", confidence=0.7), tmp_path)
    result = decay_pass(tmp_path, prune=False)
    assert "decayed" in result
    assert "pruned" in result
    assert "remaining" in result
    assert result["decayed"] == 2


def test_decay_pass_empty_dir_returns_zeros(tmp_path: Path) -> None:
    result = decay_pass(tmp_path / "nonexistent")
    assert result == {"decayed": 0, "pruned": 0, "remaining": 0}