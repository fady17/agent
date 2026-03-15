"""
scripts/replay_trace.py

Trace log replayer — reads ~/.agent/logs/trace.jsonl and prints
the last N log records as a human-readable Rich table.

Useful for auditing what the agent decided and why after a session,
or for debugging unexpected behaviour by replaying the decision trail.

Usage:
    uv run python scripts/replay_trace.py
    uv run python scripts/replay_trace.py --last 50
    uv run python scripts/replay_trace.py --level INFO
    uv run python scripts/replay_trace.py --event orchestrator.turn_complete
    uv run python scripts/replay_trace.py --raw           # print raw JSON lines
    uv run python scripts/replay_trace.py --path /custom/path/trace.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

app     = typer.Typer(add_completion=False)
console = Console()

# Log levels in severity order for filtering
_LEVEL_ORDER = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}

# Events that represent significant orchestrator decisions — shown highlighted
_DECISION_EVENTS = {
    "orchestrator.turn_complete",
    "orchestrator.route",
    "router.route",
    "router.stream_route",
    "classifier.classified",
    "consolidation.complete",
    "index_rebuild.complete",
    "monitor.actions_queued",
    "scheduler.job_complete",
    "scheduler.job_failed",
}

# Colour per log level
_LEVEL_COLOURS = {
    "debug":    "dim",
    "info":     "green",
    "warning":  "yellow",
    "error":    "red",
    "critical": "bold red",
}


def _default_trace_path() -> Path:
    import os
    root = Path(os.getenv("AGENT_MEMORY_ROOT", "~/.agent")).expanduser().resolve()
    return root / "logs" / "trace.jsonl"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        console.print(f"[red]Trace file not found:[/red] {path}")
        console.print("[dim]Start a chat session first to generate logs.[/dim]")
        raise typer.Exit(1)

    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # Skip malformed lines silently

    return records


def _filter_records(
    records: list[dict],
    *,
    level: str | None,
    event: str | None,
    last: int,
) -> list[dict]:
    if level:
        min_order = _LEVEL_ORDER.get(level.lower(), 0)
        records = [
            r for r in records
            if _LEVEL_ORDER.get(r.get("level", "debug").lower(), 0) >= min_order
        ]

    if event:
        records = [r for r in records if event.lower() in r.get("event", "").lower()]

    return records[-last:]


def _format_extras(record: dict) -> str:
    """Extract extra structured fields beyond the standard ones."""
    skip = {"timestamp", "level", "logger", "event", "_record"}
    parts = []
    for k, v in record.items():
        if k in skip:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.1f}")
        elif isinstance(v, str) and len(v) > 60:
            parts.append(f"{k}={v[:57]}...")
        else:
            parts.append(f"{k}={v!r}")
    return "  ".join(parts)


def _print_table(records: list[dict], show_logger: bool) -> None:
    table = Table(
        title=f"Trace log — {len(records)} records",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Time",    style="dim",  width=12, no_wrap=True)
    table.add_column("Level",   width=8,      no_wrap=True)
    table.add_column("Event",   width=32,     no_wrap=True)
    if show_logger:
        table.add_column("Logger", style="dim", width=24, no_wrap=True)
    table.add_column("Details")

    for record in records:
        ts       = record.get("timestamp", "")
        level    = record.get("level", "?").lower()
        event    = record.get("event", "")
        logger   = record.get("logger", "")
        extras   = _format_extras(record)

        # Timestamp — show only time portion (HH:MM:SS)
        ts_short = ts[11:19] if len(ts) >= 19 else ts

        # Level badge
        colour      = _LEVEL_COLOURS.get(level, "white")
        level_text  = Text(level.upper()[:4], style=colour)

        # Highlight decision events
        event_style = "bold cyan" if event in _DECISION_EVENTS else ""
        event_text  = Text(event, style=event_style) if event_style else event

        row = [ts_short, level_text, event_text]
        if show_logger:
            row.append(logger.split(".")[-1])   # show only the last component
        row.append(extras[:120])

        table.add_row(*row)

    console.print(table)


def _print_decisions_only(records: list[dict]) -> None:
    """Print a condensed view of only the significant decision events."""
    decision_records = [r for r in records if r.get("event") in _DECISION_EVENTS]

    if not decision_records:
        console.print("[dim]No decision events found in the selected records.[/dim]")
        return

    table = Table(
        title=f"Decision trail — {len(decision_records)} events",
        border_style="dim",
    )
    table.add_column("Time",    style="dim",   width=12)
    table.add_column("Event",   style="cyan",  width=34)
    table.add_column("Details")

    for record in decision_records:
        ts      = record.get("timestamp", "")[11:19]
        event   = record.get("event", "")
        extras  = _format_extras(record)
        table.add_row(ts, event, extras[:100])

    console.print(table)


@app.command()
def main(
    last:         int            = typer.Option(20,   "--last",  "-n", help="Number of records to show"),
    level:        str | None     = typer.Option(None, "--level", "-l", help="Minimum log level (debug/info/warning/error)"),
    event:        str | None     = typer.Option(None, "--event", "-e", help="Filter by event name (substring match)"),
    decisions:    bool           = typer.Option(False,"--decisions", "-d", help="Show only significant decision events"),
    raw:          bool           = typer.Option(False,"--raw",       "-r", help="Print raw JSON lines"),
    show_logger:  bool           = typer.Option(False,"--logger",         help="Show logger column"),
    path:         Path | None    = typer.Option(None, "--path",           help="Custom trace file path"),
    stats:        bool           = typer.Option(False,"--stats",          help="Show summary statistics"),
) -> None:
    """
    Replay and inspect the agent trace log.

    Examples:
        replay_trace.py                     # last 20 records
        replay_trace.py --last 100          # last 100
        replay_trace.py --level warning     # warnings and above only
        replay_trace.py --event classifier  # classifier events only
        replay_trace.py --decisions         # decision trail only
        replay_trace.py --stats             # summary counts by event
    """
    trace_path = path or _default_trace_path()
    all_records = _read_records(trace_path)

    if not all_records:
        console.print("[dim]Trace file is empty.[/dim]")
        raise typer.Exit(0)

    console.print(
        f"[dim]Trace file:[/dim] {trace_path}  "
        f"[dim]({len(all_records)} total records)[/dim]"
    )

    # Stats mode
    if stats:
        _print_stats(all_records)
        raise typer.Exit(0)

    records = _filter_records(all_records, level=level, event=event, last=last)

    if not records:
        console.print("[dim]No records match the filters.[/dim]")
        raise typer.Exit(0)

    # Raw JSON mode
    if raw:
        for record in records:
            print(json.dumps(record))
        raise typer.Exit(0)

    if decisions:
        _print_decisions_only(records)
    else:
        _print_table(records, show_logger=show_logger)


def _print_stats(records: list[dict]) -> None:
    """Print summary statistics — event counts, level distribution."""
    from collections import Counter

    level_counts: Counter = Counter()
    event_counts: Counter = Counter()

    for r in records:
        level_counts[r.get("level", "unknown").lower()] += 1
        event_counts[r.get("event", "unknown")] += 1

    # Level distribution
    level_table = Table(title="Level distribution", border_style="dim", show_header=False)
    level_table.add_column("Level", style="dim")
    level_table.add_column("Count", justify="right")
    for level in ("debug", "info", "warning", "error", "critical"):
        count = level_counts.get(level, 0)
        if count:
            colour = _LEVEL_COLOURS.get(level, "white")
            level_table.add_row(f"[{colour}]{level}[/{colour}]", str(count))
    console.print(level_table)

    # Top 20 events
    event_table = Table(title="Top events", border_style="dim")
    event_table.add_column("Event",  style="cyan")
    event_table.add_column("Count",  justify="right")
    for event, count in event_counts.most_common(20):
        style = "bold" if event in _DECISION_EVENTS else ""
        event_table.add_row(f"[{style}]{event}[/{style}]" if style else event, str(count))
    console.print(event_table)


if __name__ == "__main__":
    app()