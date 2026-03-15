"""
tests/test_replay_trace.py

Tests for the replay_trace script — record parsing, filtering,
level/event filters, stats, and raw output mode.
Uses tmp_path for all file I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.replay_trace import (
    _DECISION_EVENTS,
    _filter_records,
    _format_extras,
    _read_records,
    app,
)

runner = CliRunner()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_record(
    event: str = "test.event",
    level: str = "info",
    ts: str = "2026-03-14T10:00:00Z",
    **extras,
) -> dict:
    return {"timestamp": ts, "level": level, "event": event, "logger": "agent.test", **extras}


def write_trace(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# ── _read_records ─────────────────────────────────────────────────────────────

def test_read_records_parses_jsonl(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [make_record("event.a"), make_record("event.b")])
    records = _read_records(trace)
    assert len(records) == 2
    assert records[0]["event"] == "event.a"


def test_read_records_skips_blank_lines(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"event": "a"}\n\n{"event": "b"}\n')
    records = _read_records(trace)
    assert len(records) == 2


def test_read_records_skips_malformed_lines(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"event": "a"}\nnot json at all\n{"event": "b"}\n')
    records = _read_records(trace)
    assert len(records) == 2


def test_read_records_missing_file_exits(tmp_path: Path) -> None:
    import typer
    with pytest.raises(typer.Exit):
        _read_records(tmp_path / "nonexistent.jsonl")

# ── _filter_records ───────────────────────────────────────────────────────────

def test_filter_last() -> None:
    records = [make_record(f"event.{i}") for i in range(10)]
    result = _filter_records(records, level=None, event=None, last=3)
    assert len(result) == 3
    assert result[-1]["event"] == "event.9"


def test_filter_level_excludes_debug() -> None:
    records = [
        make_record(level="debug"),
        make_record(level="info"),
        make_record(level="warning"),
        make_record(level="error"),
    ]
    result = _filter_records(records, level="warning", event=None, last=100)
    levels = {r["level"] for r in result}
    assert "debug" not in levels
    assert "info" not in levels
    assert "warning" in levels
    assert "error" in levels


def test_filter_level_case_insensitive() -> None:
    records =[make_record(level="INFO"), make_record(level="WARNING")]
    result = _filter_records(records, level="info", event=None, last=100)
    assert len(result) == 2   # Both are info or higher
    

def test_filter_event_substring_match() -> None:
    records = [
        make_record("classifier.classified"),
        make_record("orchestrator.turn_complete"),
        make_record("router.route"),
    ]
    result = _filter_records(records, level=None, event="classifier", last=100)
    assert len(result) == 1
    assert result[0]["event"] == "classifier.classified"


def test_filter_event_case_insensitive() -> None:
    records = [make_record("Classifier.Classified"), make_record("other.event")]
    result = _filter_records(records, level=None, event="classifier", last=100)
    assert len(result) == 1


def test_filter_combined_level_and_event() -> None:
    records = [
        make_record("classifier.classified", level="info"),
        make_record("classifier.classified", level="debug"),
        make_record("other.event", level="info"),
    ]
    result = _filter_records(records, level="info", event="classifier", last=100)
    assert len(result) == 1
    assert result[0]["level"] == "info"


# ── _format_extras ────────────────────────────────────────────────────────────

def test_format_extras_excludes_standard_fields() -> None:
    record = make_record(domain="code", action="debug")
    extras = _format_extras(record)
    assert "timestamp" not in extras
    assert "level" not in extras
    assert "event" not in extras
    assert "logger" not in extras


def test_format_extras_includes_custom_fields() -> None:
    record = make_record(tokens=512, latency_ms=340.5)
    extras = _format_extras(record)
    assert "tokens" in extras
    assert "512" in extras
    assert "latency_ms" in extras


def test_format_extras_truncates_long_strings() -> None:
    record = make_record(long_field="x" * 100)
    extras = _format_extras(record)
    assert len(extras) < 200   # truncated


def test_format_extras_formats_float() -> None:
    record = make_record(cost=0.00123)
    extras = _format_extras(record)
    assert "cost=" in extras


# ── CLI — main command ────────────────────────────────────────────────────────

def test_cli_shows_table(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [make_record("test.event", tokens=5) for _ in range(5)])
    result = runner.invoke(app, ["--path", str(trace)])
    assert result.exit_code == 0
    assert "test.event" in result.output


def test_cli_empty_file_exits_cleanly(tmp_path: Path) -> None:
    trace = tmp_path / "empty.jsonl"
    trace.write_text("")
    result = runner.invoke(app, ["--path", str(trace)])
    assert result.exit_code == 0
    assert "empty" in result.output.lower()


def test_cli_missing_file_exits_with_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--path", str(tmp_path / "missing.jsonl")])
    assert result.exit_code != 0


def test_cli_last_flag(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [make_record(f"event.{i}") for i in range(20)])
    result = runner.invoke(app, ["--path", str(trace), "--last", "5"])
    assert result.exit_code == 0
    # Should show "5 records" in table title
    assert "5" in result.output


def test_cli_level_filter(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [
        make_record("debug.event", level="debug"),
        make_record("warn.event",  level="warning"),
    ])
    result = runner.invoke(app, ["--path", str(trace), "--level", "warning"])
    assert result.exit_code == 0
    assert "warn.event" in result.output
    assert "debug.event" not in result.output


def test_cli_event_filter(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [
        make_record("classifier.classified"),
        make_record("orchestrator.turn_complete"),
    ])
    result = runner.invoke(app, ["--path", str(trace), "--event", "classifier"])
    assert result.exit_code == 0
    assert "classifier" in result.output
    assert "orchestrator" not in result.output


def test_cli_raw_mode(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [make_record("raw.event", key="value")])
    result = runner.invoke(app, ["--path", str(trace), "--raw"])
    assert result.exit_code == 0
    # Raw mode prints JSON — must be parseable
    for line in result.output.splitlines():
        line = line.strip()
        if line and line.startswith("{"):
            parsed = json.loads(line)
            assert "event" in parsed


def test_cli_stats_mode(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [
        make_record("event.a", level="info"),
        make_record("event.a", level="info"),
        make_record("event.b", level="warning"),
    ])
    result = runner.invoke(app, ["--path", str(trace), "--stats"])
    assert result.exit_code == 0
    assert "event.a" in result.output
    assert "2" in result.output   # count for event.a


def test_cli_decisions_mode(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    decision_event = next(iter(_DECISION_EVENTS))
    write_trace(trace, [
        make_record(decision_event),
        make_record("non.decision.event"),
    ])
    result = runner.invoke(app, ["--path", str(trace), "--decisions"])
    assert result.exit_code == 0
    assert decision_event in result.output
    assert "non.decision.event" not in result.output


def test_cli_decisions_mode_no_decisions(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [make_record("ordinary.event")])
    result = runner.invoke(app, ["--path", str(trace), "--decisions"])
    assert result.exit_code == 0
    assert "No decision events" in result.output


def test_cli_shows_total_record_count(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, [make_record() for _ in range(7)])
    result = runner.invoke(app, ["--path", str(trace), "--last", "100"])
    assert result.exit_code == 0
    assert "7" in result.output