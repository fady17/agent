"""
tests/test_logger.py

Verifies that:
  - setup_logging() is idempotent
  - get_logger() returns a usable logger
  - log calls produce valid JSON records in the trace file
  - required fields (timestamp, level, logger, event) are always present
  - context variables are merged into records
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import structlog

from agent.core import logger as logger_module

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    """Parse a .jsonl file into a list of dicts."""
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_logger(tmp_path: Path):
    """
    Reset logger state and redirect log output to a temp directory
    so tests are fully isolated from ~/.agent/logs/.
    """
    # Reset module state
    logger_module._configured = False
    structlog.reset_defaults()

    # Clear all handlers from root logger
    root = logging.getLogger()
    root.handlers.clear()

    # Patch cfg.logs_dir to point to tmp
    with patch.object(
        type(logger_module.cfg), "logs_dir",
        new_callable=lambda: property(lambda self: tmp_path)
    ):
        yield tmp_path

    # Cleanup after test
    logger_module._configured = False
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_setup_logging_is_idempotent(reset_logger: Path) -> None:
    logger_module.setup_logging()
    logger_module.setup_logging()
    logger_module.setup_logging()
    # Should not raise and should not add duplicate handlers
    root = logging.getLogger()
    assert len(root.handlers) == 2  # file + terminal


def test_get_logger_returns_bound_logger(reset_logger: Path) -> None:
    log = logger_module.get_logger("test.module")
    assert log is not None
    assert hasattr(log, "info")
    assert hasattr(log, "debug")
    assert hasattr(log, "warning")
    assert hasattr(log, "error")


def test_log_writes_to_trace_file(reset_logger: Path) -> None:
    log = logger_module.get_logger("test.module")
    log.info("test_event", key="value", count=42)

    trace_path = reset_logger / "trace.jsonl"
    assert trace_path.exists(), "trace.jsonl was not created"

    records = read_jsonl(trace_path)
    assert len(records) >= 1


def test_log_record_has_required_fields(reset_logger: Path) -> None:
    log = logger_module.get_logger("test.required_fields")
    log.info("something_happened", project="agrivision")

    records = read_jsonl(reset_logger / "trace.jsonl")
    record = records[-1]

    assert "timestamp" in record, "missing timestamp"
    assert "level" in record,     "missing level"
    assert "event" in record,     "missing event"
    assert record["event"] == "something_happened"
    assert record["level"] == "info"
    assert record["project"] == "agrivision"


def test_log_record_includes_logger_name(reset_logger: Path) -> None:
    log = logger_module.get_logger("agent.memory.episodic")
    log.debug("wrote_event", event_id="ep_001")

    records = read_jsonl(reset_logger / "trace.jsonl")
    record = records[-1]
    assert record.get("logger") == "agent.memory.episodic"


def test_multiple_log_levels_written(reset_logger: Path) -> None:
    log = logger_module.get_logger("test.levels")
    log.debug("debug_msg")
    log.info("info_msg")
    log.warning("warning_msg")
    log.error("error_msg")

    records = read_jsonl(reset_logger / "trace.jsonl")
    levels = {r["level"] for r in records}
    # debug may be filtered by terminal but all go to file
    assert {"info", "warning", "error"}.issubset(levels)


def test_structured_fields_preserved(reset_logger: Path) -> None:
    log = logger_module.get_logger("test.fields")
    log.info("llm.request", model="qwen3", tokens=512, latency_ms=340.5)

    records = read_jsonl(reset_logger / "trace.jsonl")
    record = next(r for r in records if r.get("event") == "llm.request")
    assert record["model"] == "qwen3"
    assert record["tokens"] == 512
    assert record["latency_ms"] == 340.5


def test_context_vars_merged(reset_logger: Path) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(session_id="sess-abc", project="agrivision")

    log = logger_module.get_logger("test.context")
    log.info("task_started")

    records = read_jsonl(reset_logger / "trace.jsonl")
    record = next(r for r in records if r.get("event") == "task_started")
    assert record.get("session_id") == "sess-abc"
    assert record.get("project") == "agrivision"

    structlog.contextvars.clear_contextvars()