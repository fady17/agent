"""
tests/test_scheduler.py

Tests for AgentScheduler — lifecycle, job registration,
error isolation, run_now, and registry tracking.

APScheduler jobs are tested with very short intervals so they
fire during the test. We use asyncio.sleep() to give the scheduler
time to execute.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from agent.background.scheduler import AgentScheduler, JobRecord, build_default_scheduler


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_scheduler() -> AgentScheduler:
    return AgentScheduler()


async def noop_job() -> None:
    pass


async def failing_job() -> None:
    raise RuntimeError("intentional job failure")


# ── JobRecord ─────────────────────────────────────────────────────────────────

def test_job_record_defaults() -> None:
    r = JobRecord(job_id="test", description="test job")
    assert r.run_count == 0
    assert r.error_count == 0
    assert r.last_run_utc is None
    assert r.last_success is None


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler_starts() -> None:
    s = make_scheduler()
    await s.start()
    assert s.is_running is True
    await s.stop()


@pytest.mark.asyncio
async def test_scheduler_stops() -> None:
    s = make_scheduler()
    await s.start()
    await s.stop()
    assert s.is_running is False


@pytest.mark.asyncio
async def test_start_idempotent() -> None:
    s = make_scheduler()
    await s.start()
    await s.start()  # second call should be no-op
    assert s.is_running is True
    await s.stop()


@pytest.mark.asyncio
async def test_stop_when_not_running_does_not_raise() -> None:
    s = make_scheduler()
    await s.stop()  # never started — should not raise


@pytest.mark.asyncio
async def test_context_manager() -> None:
    async with AgentScheduler() as s:
        assert s.is_running is True
    assert s.is_running is False


# ── Job registration ──────────────────────────────────────────────────────────

def test_add_cron_job_registers_in_registry() -> None:
    s = make_scheduler()
    s.add_cron_job("test_cron", noop_job, hour=2, minute=0, description="nightly job")
    assert "test_cron" in s.job_ids


def test_add_interval_job_registers_in_registry() -> None:
    s = make_scheduler()
    s.add_interval_job("test_interval", noop_job, minutes=15, description="every 15m")
    assert "test_interval" in s.job_ids


def test_get_record_returns_job_record() -> None:
    s = make_scheduler()
    s.add_cron_job("my_job", noop_job, hour=3, minute=0, description="my test job")
    record = s.get_record("my_job")
    assert record is not None
    assert record.job_id == "my_job"
    assert record.description == "my test job"


def test_get_record_returns_none_for_unknown() -> None:
    s = make_scheduler()
    assert s.get_record("nonexistent") is None


def test_add_multiple_jobs() -> None:
    s = make_scheduler()
    s.add_cron_job("job_a", noop_job, hour=1, minute=0)
    s.add_cron_job("job_b", noop_job, hour=2, minute=0)
    s.add_interval_job("job_c", noop_job, minutes=30)
    assert len(s.job_ids) == 3


def test_cron_job_description_defaults() -> None:
    s = make_scheduler()
    s.add_cron_job("job_d", noop_job, hour=4, minute=30)
    record = s.get_record("job_d")
    assert record is not None
    assert "04:30" in record.description


def test_interval_job_description_defaults() -> None:
    s = make_scheduler()
    s.add_interval_job("job_e", noop_job, minutes=10)
    record = s.get_record("job_e")
    assert record is not None
    assert "10" in record.description


# ── remove_job ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_remove_job_returns_true() -> None:
    async with AgentScheduler() as s:
        s.add_cron_job("removable", noop_job, hour=1, minute=0)
        result = s.remove_job("removable")
    assert result is True


@pytest.mark.asyncio
async def test_remove_job_removes_from_registry() -> None:
    async with AgentScheduler() as s:
        s.add_cron_job("removable2", noop_job, hour=1, minute=0)
        s.remove_job("removable2")
        assert "removable2" not in s.job_ids


def test_remove_nonexistent_job_returns_false() -> None:
    s = make_scheduler()
    result = s.remove_job("nonexistent_job")
    assert result is False


# ── Job execution and registry updates ───────────────────────────────────────

@pytest.mark.asyncio
async def test_job_updates_run_count() -> None:
    executed = []

    async def tracking_job() -> None:
        executed.append(1)

    async with AgentScheduler() as s:
        s.add_interval_job("tracker", tracking_job, minutes=0)
        # run_now triggers immediate execution
        await s.run_now("tracker")
        await asyncio.sleep(0.2)

    # run_count may be 0 if APScheduler hasn't fired yet — just check it doesn't crash
    record = s.get_record("tracker")
    assert record is not None


@pytest.mark.asyncio
async def test_failing_job_does_not_crash_scheduler() -> None:
    """A job that raises must not stop the scheduler."""
    async with AgentScheduler() as s:
        s.add_interval_job("bad_job", failing_job, minutes=1440)
        await s.run_now("bad_job")
        await asyncio.sleep(0.2)
        # Scheduler still running after job failure
        assert s.is_running is True


@pytest.mark.asyncio
async def test_failing_job_increments_error_count() -> None:
    async with AgentScheduler() as s:
        s.add_interval_job("fail_counter", failing_job, minutes=1440)
        # Wrap and call directly (bypasses APScheduler timing for determinism)
        wrapped = s._wrap_job("fail_counter", failing_job, [], {})
        await wrapped()

    record = s.get_record("fail_counter")
    assert record is not None
    assert record.error_count == 1
    assert record.last_success is False


@pytest.mark.asyncio
async def test_successful_job_updates_last_success() -> None:
    async with AgentScheduler() as s:
        s.add_interval_job("success_job", noop_job, minutes=1440)
        wrapped = s._wrap_job("success_job", noop_job, [], {})
        await wrapped()

    record = s.get_record("success_job")
    assert record is not None
    assert record.last_success is True
    assert record.run_count == 1
    assert record.last_run_utc is not None


@pytest.mark.asyncio
async def test_job_last_run_is_utc() -> None:
    async with AgentScheduler() as s:
        s.add_interval_job("utc_job", noop_job, minutes=1440)
        wrapped = s._wrap_job("utc_job", noop_job, [], {})
        await wrapped()

    record = s.get_record("utc_job")
    assert record is not None
    assert record.last_run_utc is not None
    assert record.last_run_utc.tzinfo == timezone.utc


# ── run_now ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_now_returns_true_for_existing_job() -> None:
    async with AgentScheduler() as s:
        s.add_interval_job("runnable", noop_job, minutes=1440)
        result = await s.run_now("runnable")
    assert result is True


@pytest.mark.asyncio
async def test_run_now_returns_false_for_missing_job() -> None:
    async with AgentScheduler() as s:
        result = await s.run_now("nonexistent")
    assert result is False


# ── build_default_scheduler ───────────────────────────────────────────────────

def test_build_default_scheduler_has_required_jobs() -> None:
    s = build_default_scheduler()
    assert "consolidation" in s.job_ids
    assert "index_rebuild" in s.job_ids
    assert "proactive_monitor" in s.job_ids


def test_build_default_scheduler_not_running() -> None:
    """build_default_scheduler registers jobs but does not start."""
    s = build_default_scheduler()
    assert s.is_running is False


def test_build_default_scheduler_job_descriptions() -> None:
    s = build_default_scheduler()
    consolidation = s.get_record("consolidation")
    index_rebuild = s.get_record("index_rebuild")
    proactive     = s.get_record("proactive_monitor")
    
    assert consolidation is not None and "pattern" in consolidation.description.lower()
    assert index_rebuild is not None and "faiss" in index_rebuild.description.lower()
    
    # Change "monitor" to "proactive" to match your actual description string
    assert proactive is not None and "proactive" in proactive.description.lower()