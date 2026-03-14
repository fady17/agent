"""
scripts/init_memory_tree.py

One-time setup — creates the full ~/.agent/ memory filesystem tree.
Safe to re-run: never overwrites existing data, only creates missing dirs/files.

Run:
    uv run python scripts/init_memory_tree.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Schema version — bump this when the tree structure changes ───────────────
SCHEMA_VERSION = "1.0.0"

# ── Tree definition ──────────────────────────────────────────────────────────
#
# ~/.agent/
#   memory/
#     episodic/          ← date-partitioned event JSON files
#     semantic/          ← knowledge graph + FAISS index
#     skills/            ← skill records keyed by task_type
#     working/           ← current session state (overwritten each run)
#   projects/            ← per-project context docs
#   consolidation/       ← consolidation engine state
#   logs/                ← structured trace logs (.jsonl)
#   config/              ← runtime config overrides (not secrets)
#
DIRECTORIES = [
    "memory/episodic",
    "memory/semantic",
    "memory/skills",
    "memory/working",
    "projects",
    "consolidation",
    "logs",
    "config",
]

# Files that must exist with initial content (only written if absent)
INITIAL_FILES: dict[str, object] = {
    # Semantic graph — empty directed graph in node-link format
    "memory/semantic/graph.json": {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [],
        "links": [],
    },
    # Consolidation state — tracks last run so the engine knows where to resume
    "consolidation/state.json": {
        "last_run_utc": None,
        "last_processed_event_id": None,
        "total_runs": 0,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def resolve_root() -> Path:
    raw = os.getenv("AGENT_MEMORY_ROOT", "~/.agent")
    return Path(raw).expanduser().resolve()


def write_if_absent(path: Path, content: object) -> bool:
    """Write JSON content only if the file does not already exist."""
    if path.exists():
        return False
    path.write_text(json.dumps(content, indent=2))
    return True


def write_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "description": "Autonomous personal agent — memory root",
        "directories": DIRECTORIES,
    }
    # Always rewrite manifest so schema_version stays current
    manifest_path.write_text(json.dumps(manifest, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    root = resolve_root()
    print(f"\nInitialising memory tree at: {root}")
    print("─" * 50)

    created_dirs: list[str] = []
    existing_dirs: list[str] = []
    created_files: list[str] = []

    # 1. Create directories
    for rel in DIRECTORIES:
        path = root / rel
        if path.exists():
            existing_dirs.append(rel)
        else:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(rel)

    # 2. Write initial files (skip if already present)
    for rel, content in INITIAL_FILES.items():
        path = root / rel
        if write_if_absent(path, content):
            created_files.append(rel)

    # 3. Write manifest (always current)
    write_manifest(root)

    # 4. Report
    if created_dirs:
        print(f"\n  Created {len(created_dirs)} director{'y' if len(created_dirs)==1 else 'ies'}:")
        for d in created_dirs:
            print(f"    + {d}/")

    if existing_dirs:
        print(f"\n  Already present ({len(existing_dirs)}):")
        for d in existing_dirs:
            print(f"    = {d}/")

    if created_files:
        print(f"\n  Initialised {len(created_files)} file(s):")
        for f in created_files:
            print(f"    + {f}")

    print(f"\n  manifest.json written (schema {SCHEMA_VERSION})")

    # 5. Verify — every expected path must exist
    print("\nVerification")
    print("─" * 50)
    all_ok = True
    for rel in DIRECTORIES:
        path = root / rel
        if path.is_dir():
            print(f"  [ok]  {rel}/")
        else:
            print(f"  [!!]  MISSING: {rel}/")
            all_ok = False

    for rel in INITIAL_FILES:
        path = root / rel
        if path.is_file():
            print(f"  [ok]  {rel}")
        else:
            print(f"  [!!]  MISSING: {rel}")
            all_ok = False

    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        print(f"  [ok]  manifest.json")
    else:
        print(f"  [!!]  MISSING: manifest.json")
        all_ok = False

    print()
    if all_ok:
        print(f"  Memory tree ready at {root}\n")
    else:
        print("  Tree incomplete — check errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()