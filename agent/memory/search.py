"""
agent/memory/search.py

Fast full-text search across the episodic event store using ripgrep.

Why ripgrep and not Python glob + string scan:
  - rg is 5-10x faster than pure Python on large file trees
  - searches file contents in parallel across CPU cores
  - handles thousands of JSON files in under 200ms
  - already on every developer machine

The search layer sits BELOW the retrieval pipeline. It is the raw
text-match stage — it returns event IDs and paths, not full events.
The retrieval pipeline deserialises and re-ranks from there.

Requires ripgrep (rg) to be installed:
    macOS:   brew install ripgrep
    Ubuntu:  apt install ripgrep
    Windows: winget install BurntSushi.ripgrep.MSVC
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.memory.episodic import EpisodicEvent

log = get_logger(__name__)

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SearchHit:
    """
    A single search result — the minimal info needed to load the full event.
    Keeps the search layer decoupled from the episodic schema.
    """
    event_id: str       # e.g. "evt-1741910400123-a3f2c1b0"
    path: Path          # absolute path to the .json file
    line_number: int    # line in the file where the match occurred
    matched_text: str   # the line content that matched (trimmed)


# ── ripgrep availability ──────────────────────────────────────────────────────

def _rg_binary() -> str:
    """Return the ripgrep binary path, raise if not installed."""
    binary = shutil.which("rg")
    if binary is None:
        raise RuntimeError(
            "ripgrep (rg) not found on PATH.\n"
            "Install it:  brew install ripgrep  |  apt install ripgrep"
        )
    return binary


# ── Core search ───────────────────────────────────────────────────────────────

async def search_events(
    query: str,
    *,
    base_dir: Path | None = None,
    project: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> list[SearchHit]:
    """
    Search episodic event files for a keyword or phrase using ripgrep.

    Returns up to `max_results` SearchHit objects ordered by ripgrep's
    default output (directory traversal order — approximately chronological
    within each day partition).

    Args:
        query:          Text to search for. Treated as a regex by rg.
                        Pass a plain string for literal keyword search.
        base_dir:       Root of the episodic store. Defaults to cfg.episodic_dir.
        project:        If set, restricts search to the project's subfolder
                        by filtering on the "project" JSON field.
        case_sensitive: Default False — case-insensitive is better for
                        natural language summaries.
        max_results:    Cap on returned results. ripgrep stops early once
                        this many matches are found (-m flag).
    """
    if not query.strip():
        return []

    base = base_dir or cfg.episodic_dir

    if not base.exists():
        log.debug("search.no_store", base=str(base))
        return []

    rg = _rg_binary()

    # ── Build rg command ──────────────────────────────────────────────────────
    # --json         structured JSON output, one object per line
    # --type json    only search .json files
    # -m N           stop after N matches total
    # -i / (none)    case insensitive / sensitive
    cmd: list[str] = [
        rg,
        "--json",
        "--type", "json",
        "-m", str(max_results),
    ]

    if not case_sensitive:
        cmd.append("-i")

    # If project filter given, wrap query to require the project field nearby.
    # Simplest reliable approach: search for query AND separately filter hits
    # that come from files containing the project value. We do a two-pass:
    # first rg for the query, then filter by project in Python.
    # This keeps the rg invocation simple and avoids multiline regex complexity.
    cmd.extend([query, str(base)])

    log.debug("search.rg_start", query=query, project=project, base=str(base))

    # ── Run subprocess ────────────────────────────────────────────────────────
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except asyncio.TimeoutError:  # noqa: UP041
        log.error("search.timeout", query=query)
        return []
    except Exception as exc:
        log.error("search.subprocess_error", error=str(exc))
        return []

    # rg exits 1 when no matches found — not an error
    if proc.returncode not in (0, 1):
        log.warning("search.rg_error", stderr=stderr.decode().strip())
        return []

    # ── Parse rg JSON output ──────────────────────────────────────────────────
    hits: list[SearchHit] = []

    for raw_line in stdout.splitlines():
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # rg --json emits objects with type: "begin"|"match"|"end"|"summary"
        if obj.get("type") != "match":
            continue

        data = obj.get("data", {})
        file_path = Path(data.get("path", {}).get("text", ""))
        line_number = data.get("line_number", 0)

        # Extract event_id from filename (stem is the event id)
        event_id = file_path.stem

        # Collect matched lines text
        lines_data = data.get("lines", {}).get("text", "")
        matched_text = lines_data.strip()

        if not event_id.startswith("evt-"):
            continue  # not an event file, skip

        hit = SearchHit(
            event_id=event_id,
            path=file_path,
            line_number=line_number,
            matched_text=matched_text,
        )
        hits.append(hit)

    # ── Project filter (post-process) ─────────────────────────────────────────
    if project is not None:
        normalised = project.strip().lower()
        hits = [h for h in hits if _hit_matches_project(h, normalised)]

    log.debug(
        "search.rg_done",
        query=query,
        project=project,
        hits=len(hits),
    )

    return hits[:max_results]


def _hit_matches_project(hit: SearchHit, project: str) -> bool:
    """
    Read the event file and check whether its project field matches.
    Used for post-search project filtering.
    Returns True if the file cannot be read (fail open — don't drop results
    due to transient IO issues).
    """
    try:
        raw = hit.path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj.get("project", "").lower() == project
    except Exception:
        return True  # fail open


# ── Convenience: search and load full events ─────────────────────────────────

async def search_and_load(
    query: str,
    *,
    base_dir: Path | None = None,
    project: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 50,
) -> list["EpisodicEvent"]:  # noqa: UP037
    """
    Search for events and return fully deserialised EpisodicEvent objects.
    Convenience wrapper — imports episodic lazily to avoid circular imports.
    """
    from agent.memory.episodic import read_event  # local import

    hits = await search_events(
        query,
        base_dir=base_dir,
        project=project,
        case_sensitive=case_sensitive,
        max_results=max_results,
    )

    events = []
    for hit in hits:
        try:
            events.append(read_event(hit.path))
        except Exception as exc:
            log.warning("search.load_error", path=str(hit.path), error=str(exc))

    return events