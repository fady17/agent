"""
scripts/session_report.py

Session cost and token report — reads episodic events and the session
state file to produce a clean per-session summary.

Shows:
    - Total tokens consumed (in + out, local vs cloud)
    - Estimated USD cost per session and cumulative
    - LLM call breakdown (local / cloud / total)
    - Top projects by activity
    - Event type distribution
    - Consolidation runs and patterns extracted

Usage:
    uv run python scripts/session_report.py
    uv run python scripts/session_report.py --sessions 5
    uv run python scripts/session_report.py --project agrivision
    uv run python scripts/session_report.py --csv > report.csv
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app     = typer.Typer(add_completion=False)
console = Console()


def _agent_root() -> Path:
    return Path(os.getenv("AGENT_MEMORY_ROOT", "~/.agent")).expanduser().resolve()


# ── SessionSummary ────────────────────────────────────────────────────────────

@dataclass
class SessionSummary:
    session_id:          str
    started_at:          datetime | None = None
    ended_at:            datetime | None = None
    turn_count:          int = 0
    total_llm_calls:     int = 0
    local_calls:         int = 0
    cloud_calls:         int = 0
    tokens_in:           int = 0
    tokens_out:          int = 0
    cost_usd:            float = 0.0
    projects:            list[str] = field(default_factory=list)
    event_type_counts:   dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def duration_minutes(self) -> float:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() / 60
        return 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_session_state(root: Path) -> dict | None:
    path = root / "memory" / "working" / "session.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_consolidation_state(root: Path) -> dict:
    path = root / "consolidation" / "state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_episodic_events(root: Path, project: str | None, limit: int) -> list[dict]:
    """Load raw episodic event dicts from the filesystem."""
    episodic_dir = root / "memory" / "episodic"
    if not episodic_dir.exists():
        return []

    event_files: list[Path] = sorted(
        episodic_dir.rglob("evt-*.json"),
        key=lambda p: p.name,
        reverse=True,
    )

    events: list[dict] = []
    for path in event_files:
        if len(events) >= limit:
            break
        try:
            data = json.loads(path.read_text())
            if project and data.get("project") != project.lower():
                continue
            events.append(data)
        except Exception:
            continue

    return events


# ── Summary computation ───────────────────────────────────────────────────────

def _build_current_session_summary(session_state: dict) -> SessionSummary:
    """Build a SessionSummary from the live session.json."""
    metrics = session_state.get("metrics", {})

    started_raw = session_state.get("started_at")
    last_raw    = session_state.get("last_active_at")

    def _parse_dt(v: str | None) -> datetime | None:
        if not v:
            return None
        try:
            dt = datetime.fromisoformat(v)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    return SessionSummary(
        session_id=session_state.get("session_id", "unknown"),
        started_at=_parse_dt(started_raw),
        ended_at=_parse_dt(last_raw),
        turn_count=session_state.get("turn_count", 0),
        total_llm_calls=metrics.get("total_llm_calls", 0),
        local_calls=metrics.get("local_calls", 0),
        cloud_calls=metrics.get("cloud_calls", 0),
        tokens_in=metrics.get("total_tokens_in", 0),
        tokens_out=metrics.get("total_tokens_out", 0),
        cost_usd=metrics.get("total_cost_usd", 0.0),
    )


def _build_event_stats(events: list[dict]) -> dict[str, Any]:
    """Aggregate statistics across a list of raw event dicts."""
    type_counts: Counter = Counter()
    project_counts: Counter = Counter()
    total_tokens_in  = 0
    total_tokens_out = 0
    total_cost       = 0.0
    local_calls      = 0
    cloud_calls      = 0

    for evt in events:
        etype   = evt.get("event_type", "unknown")
        project = evt.get("project") or "(none)"
        data    = evt.get("data", {}) or {}

        type_counts[etype]       += 1
        project_counts[project]  += 1

        total_tokens_in  += data.get("tokens_in", 0)
        total_tokens_out += data.get("tokens_out", 0)
        total_cost       += data.get("cost_usd", 0.0)

        if data.get("is_local", True):
            local_calls += 1
        elif data.get("cost_usd", 0) > 0:
            cloud_calls += 1

    return {
        "type_counts":     dict(type_counts.most_common(15)),
        "project_counts":  dict(project_counts.most_common(10)),
        "total_tokens_in":  total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_cost":       round(total_cost, 6),
        "local_calls":      local_calls,
        "cloud_calls":      cloud_calls,
    }


# ── Rich output ───────────────────────────────────────────────────────────────

def _print_session_panel(summary: SessionSummary) -> None:
    table = Table(
        title=f"Current session — {summary.session_id}",
        show_header=False,
        border_style="green",
    )
    table.add_column("Key",   style="dim", width=20)
    table.add_column("Value", style="bold")

    started = (
        summary.started_at.strftime("%Y-%m-%d %H:%M UTC")
        if summary.started_at else "(unknown)"
    )
    duration = (
        f"{summary.duration_minutes:.1f} min"
        if summary.duration_minutes else "(ongoing)"
    )

    table.add_row("Session ID",   summary.session_id)
    table.add_row("Started",      started)
    table.add_row("Duration",     duration)
    table.add_row("Turns",        str(summary.turn_count))
    table.add_row("LLM calls",    str(summary.total_llm_calls))
    table.add_row("  Local",      str(summary.local_calls))
    table.add_row("  Cloud",      str(summary.cloud_calls))
    table.add_row("Tokens in",    f"{summary.tokens_in:,}")
    table.add_row("Tokens out",   f"{summary.tokens_out:,}")
    table.add_row("Total tokens", f"{summary.total_tokens:,}")
    table.add_row(
        "Estimated cost",
        f"[{'green' if summary.cost_usd == 0 else 'yellow'}]"
        f"${summary.cost_usd:.4f}[/{'green' if summary.cost_usd == 0 else 'yellow'}]",
    )
    console.print(table)


def _print_event_stats(stats: dict, event_limit: int) -> None:
    # Project activity
    project_table = Table(title="Activity by project", border_style="dim")
    project_table.add_column("Project", style="cyan")
    project_table.add_column("Events",  justify="right")
    for project, count in list(stats["project_counts"].items())[:event_limit]:
        project_table.add_row(project, str(count))
    console.print(project_table)

    # Event type distribution
    type_table = Table(title="Event type distribution", border_style="dim")
    type_table.add_column("Event type", style="cyan")
    type_table.add_column("Count",      justify="right")
    for etype, count in list(stats["type_counts"].items())[:event_limit]:
        type_table.add_row(etype, str(count))
    console.print(type_table)


def _print_consolidation(con_state: dict) -> None:
    if not con_state:
        return
    table = Table(title="Consolidation", show_header=False, border_style="dim")
    table.add_column("Key",   style="dim")
    table.add_column("Value")
    table.add_row("Total runs",         str(con_state.get("total_runs", 0)))
    table.add_row("Patterns extracted", str(con_state.get("total_patterns_extracted", 0)))
    last_run = con_state.get("last_run_utc")
    table.add_row("Last run", last_run[:16] if last_run else "(never)")
    if con_state.get("last_error"):
        table.add_row("Last error", f"[red]{con_state['last_error'][:60]}[/red]")
    console.print(table)


def _print_token_cost_summary(stats: dict, session: SessionSummary | None) -> None:
    """Combined token + cost breakdown across episodic store."""
    table = Table(title="Token & cost summary (episodic store)", border_style="dim")
    table.add_column("Metric",  style="dim")
    table.add_column("Episodic store", justify="right")
    if session:
        table.add_column("Current session", justify="right")

    def row(label: str, ep_val: str, sess_val: str = "") -> None:
        if session:
            table.add_row(label, ep_val, sess_val)
        else:
            table.add_row(label, ep_val)

    row("Tokens in",    f"{stats['total_tokens_in']:,}",
        f"{session.tokens_in:,}" if session else "")
    row("Tokens out",   f"{stats['total_tokens_out']:,}",
        f"{session.tokens_out:,}" if session else "")
    row("Total tokens", f"{stats['total_tokens_in'] + stats['total_tokens_out']:,}",
        f"{session.total_tokens:,}" if session else "")
    row("Local calls",  str(stats["local_calls"]),
        str(session.local_calls) if session else "")
    row("Cloud calls",  str(stats["cloud_calls"]),
        str(session.cloud_calls) if session else "")
    cost_str = f"${stats['total_cost']:.4f}"
    sess_cost = f"${session.cost_usd:.4f}" if session else ""
    row("Est. cost USD", cost_str, sess_cost)

    console.print(table)


# ── CSV output ────────────────────────────────────────────────────────────────

def _print_csv(session: SessionSummary | None, stats: dict, con_state: dict) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "session_id", "started_at", "turn_count",
        "total_llm_calls", "local_calls", "cloud_calls",
        "tokens_in", "tokens_out", "cost_usd",
        "consolidation_runs", "patterns_extracted",
        "episodic_tokens_in", "episodic_tokens_out", "episodic_cost",
    ])
    writer.writerow([
        session.session_id if session else "",
        session.started_at.isoformat() if session and session.started_at else "",
        session.turn_count if session else "",
        session.total_llm_calls if session else "",
        session.local_calls if session else "",
        session.cloud_calls if session else "",
        session.tokens_in if session else "",
        session.tokens_out if session else "",
        session.cost_usd if session else "",
        con_state.get("total_runs", ""),
        con_state.get("total_patterns_extracted", ""),
        stats["total_tokens_in"],
        stats["total_tokens_out"],
        stats["total_cost"],
    ])
    print(buf.getvalue(), end="")


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    project:  str | None = typer.Option(None,  "--project",  "-p",
        help="Filter events by project name"),
    limit:    int        = typer.Option(1000,  "--limit",    "-n",
        help="Max episodic events to scan"),
    top:      int        = typer.Option(10,    "--top",      "-t",
        help="How many top items to show per table"),
    csv_out:  bool       = typer.Option(False, "--csv",
        help="Output as CSV (for piping to files)"),
) -> None:
    """
    Print a cost and token report for the current agent session.

    Combines live session state with the episodic event log to give
    a complete picture of activity, costs, and memory growth.
    """
    root = _agent_root()

    # Load data
    session_state = _load_session_state(root)
    con_state     = _load_consolidation_state(root)
    events        = _load_episodic_events(root, project=project, limit=limit)
    stats         = _build_event_stats(events)
    session       = _build_current_session_summary(session_state) if session_state else None

    if csv_out:
        _print_csv(session, stats, con_state)
        return

    # Rich output
    console.rule("[bold]Agent Session Report[/bold]")
    console.print(
        f"[dim]Agent root:[/dim] {root}  "
        f"[dim]|  Episodic events scanned:[/dim] {len(events)}"
        + (f"  [dim]| Project filter:[/dim] {project}" if project else "")
    )
    console.print()

    if session:
        _print_session_panel(session)
        console.print()

    _print_token_cost_summary(stats, session)
    console.print()

    _print_event_stats(stats, event_limit=top)
    console.print()

    _print_consolidation(con_state)


if __name__ == "__main__":
    app()