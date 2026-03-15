# type: ignore
"""
agent/background/scheduler.py

Background scheduler — runs nightly maintenance jobs and proactive
monitoring alongside the main agent event loop.

Built on APScheduler's AsyncIOScheduler which integrates natively
with asyncio — no separate thread pool needed for async jobs.

Jobs registered at startup:
    consolidation_job    — nightly at cfg.consolidation_hour:consolidation_minute
                           reads episodic events, extracts patterns, upserts skill store
    index_rebuild_job    — nightly 30 min after consolidation
                           rebuilds FAISS index from updated semantic graph
    proactive_monitor    — every N minutes, checks calendar/git/filesystem
                           for events that warrant unprompted agent action

Design:
    - The scheduler is a singleton owned by the orchestrator.
    - Jobs are registered as coroutine functions.
    - All jobs catch their own exceptions — a failed job never crashes the scheduler.
    - Job results are written as episodic events so they appear in memory.
    - The scheduler is started after the session is loaded and stopped on shutdown.

Usage:
    scheduler = AgentScheduler(orchestrator=orch)
    await scheduler.start()
    # ... agent runs ...
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)


# ── JobRecord ─────────────────────────────────────────────────────────────────

@dataclass
class JobRecord:
    """Metadata about a registered background job."""
    job_id:       str
    description:  str
    last_run_utc: datetime | None = None
    last_success: bool | None     = None
    run_count:    int              = 0
    error_count:  int              = 0


# ── AgentScheduler ────────────────────────────────────────────────────────────

class AgentScheduler:
    """
    Manages all background jobs for the agent session.

    Wraps APScheduler's AsyncIOScheduler and adds:
    - A job registry with run history
    - Structured logging for every job execution
    - Episodic event writing on job completion
    - Safe error isolation — one job failing never affects others
    """

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._registry: dict[str, JobRecord] = {}
        self._running = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def job_ids(self) -> list[str]:
        return list(self._registry.keys())

    def get_record(self, job_id: str) -> JobRecord | None:
        return self._registry.get(job_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the scheduler. Safe to call multiple times — no-op if running.
        """
        if self._running:
            log.debug("scheduler.already_running")
            return

        self._scheduler.start()
        self._running = True
        log.info("scheduler.started", jobs=len(self._registry))

    async def stop(self) -> None:
        """
        Gracefully stop the scheduler.
        Running jobs are allowed to complete before shutdown.
        """
        if not self._running:
            return

        self._scheduler.shutdown(wait=True)
        self._running = False
        log.info("scheduler.stopped")

    async def __aenter__(self) -> "AgentScheduler":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ── Job registration ──────────────────────────────────────────────────────

    def add_cron_job(
        self,
        job_id: str,
        coro_func: Callable[..., Coroutine[Any, Any, None]],
        *,
        hour: int,
        minute: int,
        description: str = "",
        args: list | None = None,
        kwargs: dict | None = None,
    ) -> None:
        """
        Register an async job to run on a daily cron schedule.

        Args:
            job_id:      Unique identifier for this job.
            coro_func:   Async function to call.
            hour:        Hour of day (0-23 UTC).
            minute:      Minute of hour (0-59).
            description: Human-readable description for the job registry.
        """
        wrapped = self._wrap_job(job_id, coro_func, args or [], kwargs or {})

        self._scheduler.add_job(
            wrapped,
            trigger=CronTrigger(hour=hour, minute=minute, timezone="UTC"),
            id=job_id,
            replace_existing=True,
            misfire_grace_time=3600,  # allow up to 1h late start (e.g. machine was asleep)
        )

        self._registry[job_id] = JobRecord(
            job_id=job_id,
            description=description or f"cron {hour:02d}:{minute:02d} UTC",
        )
        log.info(
            "scheduler.job_registered",
            job_id=job_id,
            trigger=f"cron {hour:02d}:{minute:02d} UTC",
        )

    def add_interval_job(
        self,
        job_id: str,
        coro_func: Callable[..., Coroutine[Any, Any, None]],
        *,
        minutes: int,
        description: str = "",
        args: list | None = None,
        kwargs: dict | None = None,
    ) -> None:
        """
        Register an async job to run every N minutes.

        Args:
            job_id:      Unique identifier.
            coro_func:   Async function to call.
            minutes:     Interval in minutes.
            description: Human-readable description.
        """
        wrapped = self._wrap_job(job_id, coro_func, args or [], kwargs or {})

        self._scheduler.add_job(
            wrapped,
            trigger=IntervalTrigger(minutes=minutes),
            id=job_id,
            replace_existing=True,
        )

        self._registry[job_id] = JobRecord(
            job_id=job_id,
            description=description or f"every {minutes}m",
        )
        log.info(
            "scheduler.job_registered",
            job_id=job_id,
            trigger=f"interval {minutes}m",
        )

    def remove_job(self, job_id: str) -> bool:
        """Remove a registered job. Returns True if it existed."""
        try:
            self._scheduler.remove_job(job_id)
            self._registry.pop(job_id, None)
            log.info("scheduler.job_removed", job_id=job_id)
            return True
        except Exception:
            return False

    async def run_now(self, job_id: str) -> bool:
        """
        Trigger a registered job to run immediately (outside its schedule).
        Returns True if the job was found and triggered.
        """
        job = self._scheduler.get_job(job_id)
        if job is None:
            log.warning("scheduler.job_not_found", job_id=job_id)
            return False

        log.info("scheduler.run_now", job_id=job_id)
        # Reschedule to run at the next opportunity (now)
        job.modify(next_run_time=datetime.now(timezone.utc))
        return True

    # ── Internal ──────────────────────────────────────────────────────────────

    def _wrap_job(
        self,
        job_id: str,
        coro_func: Callable,
        args: list,
        kwargs: dict,
    ) -> Callable:
        """
        Wrap a coroutine function with logging, error isolation,
        and registry update.
        """
        scheduler = self  # capture for closure

        async def _wrapped() -> None:
            record = scheduler._registry.get(job_id)
            log.info("scheduler.job_start", job_id=job_id)

            try:
                await coro_func(*args, **kwargs)

                if record:
                    record.last_run_utc = datetime.now(timezone.utc)
                    record.last_success = True
                    record.run_count   += 1

                log.info("scheduler.job_complete", job_id=job_id)

            except Exception as exc:
                if record:
                    record.last_run_utc = datetime.now(timezone.utc)
                    record.last_success = False
                    record.run_count   += 1
                    record.error_count += 1

                log.error(
                    "scheduler.job_failed",
                    job_id=job_id,
                    error=str(exc),
                )
                # Never propagate — a failing job must not crash the scheduler

        return _wrapped


# ── Default job factory ───────────────────────────────────────────────────────

def build_default_scheduler() -> AgentScheduler:
    """
    Build a scheduler with the standard nightly jobs pre-registered.

    Jobs are registered but NOT started — call scheduler.start() separately
    after the session and memory components are loaded.

    The actual job implementations (consolidation, index rebuild, proactive
    monitor) are imported lazily inside the job functions to avoid circular
    imports at module load time.
    """
    scheduler = AgentScheduler()

    # ── Consolidation job ─────────────────────────────────────────────────────
    async def _consolidation_job() -> None:
        from agent.background.consolidation import run_consolidation
        await run_consolidation()

    scheduler.add_cron_job(
        job_id="consolidation",
        coro_func=_consolidation_job,
        hour=cfg.consolidation_hour,
        minute=cfg.consolidation_minute,
        description="Extract patterns from episodic events → update skill store + semantic graph",
    )

    # ── Index rebuild job (30 min after consolidation) ────────────────────────
    rebuild_minute = (cfg.consolidation_minute + 30) % 60
    rebuild_hour   = (cfg.consolidation_hour + (cfg.consolidation_minute + 30) // 60) % 24

    async def _index_rebuild_job() -> None:
        from agent.background.index_rebuild import run_index_rebuild 
        await run_index_rebuild()

    scheduler.add_cron_job(
        job_id="index_rebuild",
        coro_func=_index_rebuild_job,
        hour=rebuild_hour,
        minute=rebuild_minute,
        description="Rebuild FAISS index from updated semantic graph",
    )

    # ── Proactive monitor (every 15 minutes) ──────────────────────────────────
    async def _proactive_monitor() -> None:
        from agent.background.monitor import run_proactive_monitor
        await run_proactive_monitor()

    scheduler.add_interval_job(
        job_id="proactive_monitor",
        coro_func=_proactive_monitor,
        minutes=15,
        description="Check calendar, git log, filesystem for proactive actions",
    )

    return scheduler