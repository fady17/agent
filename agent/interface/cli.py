"""
agent/interface/cli.py

CLI entry point for the autonomous personal agent.

Commands:
    agent chat              — interactive REPL, main usage
    agent memory show       — list recent episodic events
    agent memory search     — full-text search across episodic store
    agent memory skills     — list skill records by confidence
    agent memory delete     — delete a specific event by ID
    agent status            — session metrics, scheduler, consolidation state

Run:
    uv run agent chat
    uv run agent memory show --project agrivision --limit 10
    uv run agent status

The chat command wires all components (orchestrator, scheduler, file watcher)
into a running session and loops until the user exits.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Module-level imports so tests can patch these names on agent.interface.cli
from agent.core.config import cfg
from agent.core.session import SessionManager
from agent.background.consolidation import _load_state as load_consolidation_state
from agent.memory.episodic import EventType, list_events
from agent.memory.skills import list_skills
from agent.memory.search import search_and_load 

app     = typer.Typer(help="Autonomous personal agent — developer / 3D designer edition")
memory  = typer.Typer(help="Inspect and manage agent memory")
app.add_typer(memory, name="memory")

console = Console()


# ── chat ──────────────────────────────────────────────────────────────────────

@app.command()
def chat(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Set active project context"),
    no_stream: bool        = typer.Option(False, "--no-stream", help="Disable streaming output"),
    force_cloud: bool      = typer.Option(False, "--cloud", help="Force cloud LLM for all turns"),
    force_local: bool      = typer.Option(False, "--local", help="Force local LLM for all turns"),
) -> None:
    """Start an interactive chat session with the agent."""
    asyncio.run(_chat_session(
        project=project,
        stream=not no_stream,
        force_cloud=force_cloud,
        force_local=force_local,
    ))


async def _chat_session(
    *,
    project: str | None,
    stream: bool,
    force_cloud: bool,
    force_local: bool,
) -> None:
    from agent.background.scheduler import build_default_scheduler
    from agent.core.orchestrator import Orchestrator
    from agent.memory.embedder import get_embedder
    from agent.memory.graph import SemanticGraph
    from agent.memory.index import FaissIndex
    from agent.core.config import cfg

    console.print(Panel(
        "[bold]Autonomous Personal Agent[/bold]\n"
        "Type your request, or:\n"
        "  [dim]exit / quit / Ctrl+C[/dim] to stop\n"
        "  [dim]/status[/dim] to show session stats\n"
        "  [dim]/clear[/dim] to reset conversation history",
        title="[green]Agent[/green]",
        border_style="green",
    ))

    # ── Bootstrap components ──────────────────────────────────────────────────
    embedder = await get_embedder()

    graph = SemanticGraph()
    if cfg.graph_path.exists():
        graph.load()

    index = FaissIndex()
    index.try_load()

    orchestrator = Orchestrator.create(
        graph=graph,
        index=index,
        embedder=embedder,
    )

    # Load or create session
    loaded = orchestrator._session.load()
    if loaded is None:
        orchestrator._session.new_session()

    if project:
        orchestrator._session.update(
            orchestrator._session.state.with_project(project)
        )

    # Start scheduler in background
    scheduler = build_default_scheduler()
    await scheduler.start()

    # Register Ctrl+C handler for graceful shutdown
    shutdown_requested = False

    def _handle_sigint(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_sigint)

    conversation_history = []

    try:
        while not shutdown_requested:
            # Check for proactive actions
            from agent.background.monitor import drain_pending_actions
            pending = drain_pending_actions()
            for action in sorted(pending, key=lambda a: a.priority):
                console.print(
                    Panel(
                        f"[bold]{action.summary}[/bold]\n{action.suggestion}",
                        title=f"[yellow]⚡ {action.action_type}[/yellow]",
                        border_style="yellow",
                    )
                )

            # Prompt
            try:
                user_input = console.input("[bold green]You:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "bye"):
                break

            if user_input == "/status":
                _print_status(orchestrator)
                continue

            if user_input == "/clear":
                conversation_history = []
                console.print("[dim]Conversation history cleared.[/dim]")
                continue

            # Run turn
            console.print("[bold blue]Agent:[/bold blue] ", end="")

            if stream:
                full_response = ""
                async for chunk in orchestrator.run_turn_stream(
                    user_input,
                    conversation_history=conversation_history,
                    force_cloud=force_cloud,
                    force_local=force_local,
                ):
                    console.print(chunk, end="")
                    full_response += chunk
                console.print()  # newline after stream
                response_text = full_response
            else:
                result = await orchestrator.run_turn(
                    user_input,
                    conversation_history=conversation_history,
                    force_cloud=force_cloud,
                    force_local=force_local,
                )
                console.print(Markdown(result.content))
                response_text = result.content

                if not result.succeeded:
                    console.print(f"[dim red]Error: {result.error}[/dim red]")

            # Append to conversation history for multi-turn context
            from agent.llm.lm_studio import Message
            conversation_history.append(Message(role="user",      content=user_input))
            conversation_history.append(Message(role="assistant", content=response_text))

            # Keep history bounded to last 10 turns (20 messages)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

    finally:
        console.print("\n[dim]Shutting down...[/dim]")
        await scheduler.stop()
        await orchestrator.shutdown()
        _print_session_summary(orchestrator)


def _print_status(orchestrator) -> None:
    """Print current session stats inline during chat."""
    state = orchestrator._session.state
    metrics = state.metrics
    table = Table(title="Session status", show_header=False, border_style="dim")
    table.add_column("Key",   style="dim")
    table.add_column("Value")
    table.add_row("Session",   state.session_id)
    table.add_row("Turns",     str(state.turn_count))
    table.add_row("Project",   state.active_project or "(none)")
    table.add_row("LLM calls", str(metrics.total_llm_calls))
    table.add_row("Local",     str(metrics.local_calls))
    table.add_row("Cloud",     str(metrics.cloud_calls))
    table.add_row("Tokens in", str(metrics.total_tokens_in))
    table.add_row("Tokens out",str(metrics.total_tokens_out))
    table.add_row("Cost USD",  f"${metrics.total_cost_usd:.4f}")
    console.print(table)


def _print_session_summary(orchestrator) -> None:
    if not orchestrator._session.is_loaded:
        return
    state = orchestrator._session.state
    console.print(
        f"\n[dim]Session ended · {state.turn_count} turns · "
        f"${state.metrics.total_cost_usd:.4f} cost[/dim]"
    )


# ── memory show ───────────────────────────────────────────────────────────────

@memory.command("show")
def memory_show(
    project: Optional[str] = typer.Option(None,  "--project", "-p"),
    limit:   int            = typer.Option(20,    "--limit",   "-n"),
    event_type: Optional[str] = typer.Option(None, "--type",  "-t",
        help="Filter by event type (e.g. code.write, llm.call)"),
) -> None:
    """List recent episodic events."""
    etype = None
    if event_type:
        try:
            etype = EventType(event_type)
        except ValueError:
            console.print(f"[red]Unknown event type: {event_type!r}[/red]")
            raise typer.Exit(1)

    events = list_events(project=project, event_type=etype, limit=limit)

    if not events:
        console.print("[dim]No events found.[/dim]")
        return

    table = Table(title=f"Episodic events ({len(events)})", border_style="dim")
    table.add_column("Date",       style="dim", width=12)
    table.add_column("Type",       style="cyan", width=18)
    table.add_column("Project",    style="green", width=14)
    table.add_column("Summary")

    for evt in events:
        table.add_row(
            evt.timestamp.strftime("%m-%d %H:%M"),
            evt.event_type,
            evt.project or "",
            evt.summary[:80],
        )

    console.print(table)


# ── memory search ─────────────────────────────────────────────────────────────

@memory.command("search")
def memory_search(
    query:   str            = typer.Argument(..., help="Search term"),
    project: Optional[str]  = typer.Option(None, "--project", "-p"),
    limit:   int            = typer.Option(20, "--limit", "-n"),
) -> None:
    """Full-text search across the episodic store using ripgrep."""
    asyncio.run(_memory_search(query=query, project=project, limit=limit))


async def _memory_search(*, query: str, project: str | None, limit: int) -> None:
    
    console.print(f"[dim]Searching for:[/dim] {query!r}")
    events = await search_and_load(query, project=project, max_results=limit)

    if not events:
        console.print("[dim]No matching events found.[/dim]")
        return

    table = Table(title=f"Search results ({len(events)})", border_style="dim")
    table.add_column("Date",    style="dim",   width=12)
    table.add_column("Type",    style="cyan",  width=18)
    table.add_column("Project", style="green", width=14)
    table.add_column("Summary")

    for evt in events:
        table.add_row(
            evt.timestamp.strftime("%m-%d %H:%M"),
            evt.event_type,
            evt.project or "",
            evt.summary[:80],
        )

    console.print(table)


# ── memory skills ─────────────────────────────────────────────────────────────

@memory.command("skills")
def memory_skills(
    min_confidence: float = typer.Option(0.0, "--min", "-m",
        help="Minimum confidence threshold (0.0 - 1.0)"),
) -> None:
    """List skill records ordered by confidence."""
    skills = list_skills(min_confidence=min_confidence)

    if not skills:
        console.print("[dim]No skill records found.[/dim]")
        return

    table = Table(title=f"Skill records ({len(skills)})", border_style="dim")
    table.add_column("Task type",     style="cyan",  width=28)
    table.add_column("Confidence",    style="green", width=10)
    table.add_column("Observations",  width=12)
    table.add_column("Pattern")

    for skill in skills:
        conf_color = (
            "green"  if skill.confidence >= 0.8 else
            "yellow" if skill.confidence >= 0.5 else
            "red"
        )
        table.add_row(
            skill.task_type,
            f"[{conf_color}]{skill.confidence:.2f}[/{conf_color}]",
            str(skill.observation_count),
            skill.pattern[:70],
        )

    console.print(table)


# ── memory delete ─────────────────────────────────────────────────────────────

@memory.command("delete")
def memory_delete(
    event_id: str = typer.Argument(..., help="Event ID to delete (evt-...)"),
    yes:      bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a specific episodic event by ID."""
    # Find the event file
    matches = list(cfg.episodic_dir.rglob(f"{event_id}.json"))
    if not matches:
        console.print(f"[red]Event not found: {event_id}[/red]")
        raise typer.Exit(1)

    path = matches[0]

    if not yes:
        confirm = typer.confirm(f"Delete {path}?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    path.unlink()
    console.print(f"[green]Deleted:[/green] {event_id}")


# ── status ────────────────────────────────────────────────────────────────────

@app.command()
def status() -> None:
    """Show agent status — session, scheduler, memory, consolidation."""
    console.print(Panel("[bold]Agent Status[/bold]", border_style="blue"))

    # Session
    mgr   = SessionManager()
    state = mgr.load()

    session_table = Table(title="Session", show_header=False, border_style="dim")
    session_table.add_column("Key", style="dim")
    session_table.add_column("Value")

    if state:
        session_table.add_row("Session ID",  state.session_id)
        session_table.add_row("Turns",       str(state.turn_count))
        session_table.add_row("Project",     state.active_project or "(none)")
        session_table.add_row("LLM calls",   str(state.metrics.total_llm_calls))
        session_table.add_row("Cloud calls", str(state.metrics.cloud_calls))
        session_table.add_row("Total cost",  f"${state.metrics.total_cost_usd:.4f}")
    else:
        session_table.add_row("Session", "(no active session)")

    console.print(session_table)

    # Memory stats
    memory_table = Table(title="Memory", show_header=False, border_style="dim")
    memory_table.add_column("Key", style="dim")
    memory_table.add_column("Value")

    recent_events = list_events(limit=1000)
    skills        = list_skills()
    graph_path    = cfg.graph_path
    faiss_path    = cfg.faiss_index_path

    memory_table.add_row("Episodic events",  str(len(recent_events)))
    memory_table.add_row("Skill records",    str(len(skills)))
    memory_table.add_row(
        "Semantic graph",
        "[green]exists[/green]" if graph_path.exists() else "[red]missing[/red]"
    )
    memory_table.add_row(
        "FAISS index",
        "[green]exists[/green]" if faiss_path.exists() else "[red]missing[/red]"
    )

    console.print(memory_table)

    # Consolidation
    con_state = load_consolidation_state()
    con_table = Table(title="Consolidation", show_header=False, border_style="dim")
    con_table.add_column("Key", style="dim")
    con_table.add_column("Value")
    con_table.add_row("Total runs",          str(con_state.total_runs))
    con_table.add_row("Patterns extracted",  str(con_state.total_patterns_extracted))
    con_table.add_row(
        "Last run",
        con_state.last_run_utc.strftime("%Y-%m-%d %H:%M UTC")
        if con_state.last_run_utc else "(never)"
    )
    if con_state.last_error:
        con_table.add_row("Last error", f"[red]{con_state.last_error[:60]}[/red]")

    console.print(con_table)