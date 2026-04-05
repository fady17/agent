# Autonomous Personal Agent

A self-improving, locally-first AI agent for developers and 3D designers.

The agent runs on your machine, learns from your work, and acts without
being asked. It knows your projects, remembers your preferences, scripts
Blender, writes and tests code, and gets smarter every night.

---

## What it does

- **Answers questions and writes code** using a local LLM (LM Studio) with
  automatic fallback to the Anthropic API for large context or complex tasks
- **Remembers everything** — every LLM call, file change, and Blender export
  is stored as an episodic event and indexed for retrieval
- **Gets smarter overnight** — a nightly consolidation job extracts reusable
  patterns from your episodic history and stores them as skills
- **Acts without prompting** — a background monitor checks your git log,
  filesystem, and calendar every 15 minutes and surfaces relevant actions
  at your next interaction
- **Scripts Blender autonomously** — exports GLB files, fixes n-gons,
  generates LOD variants, all over a socket bridge without bpy in the agent venv

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| uv | latest |
| LM Studio | 0.3+ (for local inference) |
| Blender | 4.x (optional, for 3D tools) |
| ripgrep (`rg`) | any (for memory search) |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/fady17/agent.git
cd agent
uv sync --all-extras

# 2. Configure
cp .env.example .env
# Edit .env — set your LM Studio URL and (optionally) Anthropic API key

# 3. Initialise memory store
uv run python scripts/init_memory_tree.py

# 4. Verify the ML stack (requires LM Studio running)
uv run python scripts/verify_ml.py

# 5. Start chatting
uv run agent chat
```

---

## Installation details

```bash
# Core dependencies only
uv sync

# Core + dev tools (pytest, ruff, mypy)
uv sync --all-extras

# Add blender-mcp for interactive Claude Desktop sessions (optional)
uv sync --extra blender
```

---

## Usage

### Chat

```bash
uv run agent chat                        # local LLM, streaming
uv run agent chat --project agrivision   # set active project context
uv run agent chat --cloud                # force Anthropic API
uv run agent chat --no-stream            # disable streaming
```

Inside chat:
- `/status` — show session metrics and cost
- `/clear` — reset conversation history
- `exit` / `quit` / Ctrl+C — graceful shutdown

### Memory inspection

```bash
uv run agent memory show                          # last 20 events
uv run agent memory show --project agrivision     # filter by project
uv run agent memory show --type code.write        # filter by event type
uv run agent memory search "FastAPI endpoint"     # full-text search
uv run agent memory skills                        # list learned skills
uv run agent memory skills --min 0.7              # high-confidence only
uv run agent memory delete evt-<id> --yes         # delete an event
```

### Status

```bash
uv run agent status
```

Shows: session metrics, token/cost totals, memory counts, FAISS index
status, and consolidation history.

### Observability

```bash
# Replay the last 20 trace log entries
uv run python scripts/replay_trace.py

# Show only significant agent decisions
uv run python scripts/replay_trace.py --decisions

# Show warnings and above
uv run python scripts/replay_trace.py --level warning --last 50

# Event frequency statistics
uv run python scripts/replay_trace.py --stats

# Session cost report
uv run python scripts/session_report.py

# Export session data to CSV
uv run python scripts/session_report.py --csv >> ~/agent_costs.csv
```

### Blender bridge

```bash
# Start the server inside Blender
blender my_scene.blend --python agent/tools/blender_server.py

# Verify the bridge is reachable
uv run python scripts/verify_blender.py
```

---

## Running tests

```bash
# All tests
uv run pytest

# Specific module
uv run pytest tests/test_orchestrator.py -v

# Skip slow integration tests (filesystem watcher)
uv run pytest -m "not integration"

# With coverage
uv run pytest --cov=agent --cov-report=term-missing
```

The test suite covers ~450 tests across all components. No live LLM,
Blender installation, or network connection is required to run the suite.

---

## Memory layout

All agent memory lives outside the repo at `~/.agent/` (configurable
via `AGENT_MEMORY_ROOT` in `.env`). It is never committed to git.

```
~/.agent/
├── memory/
│   ├── episodic/          # YYYY/MM/DD/evt-*.json — append-only event log
│   ├── semantic/          # graph.json, index.faiss, node_index.json
│   ├── skills/            # {task_type}.json — learned skill records
│   └── working/           # session.json — current session state
├── consolidation/
│   └── state.json         # nightly consolidation watermark
├── logs/
│   └── trace.jsonl        # structured log (newline-delimited JSON)
├── config/
│   └── calendar.json      # optional upcoming events (proactive monitor)
└── projects/              # watched project directories
```

---

## LLM routing

The agent routes between local and cloud inference automatically:

| Condition | Backend |
|---|---|
| Input tokens > `LLM_CLOUD_THRESHOLD_TOKENS` | Anthropic API |
| LM Studio unreachable | Anthropic API |
| `--cloud` flag | Anthropic API |
| All other cases | LM Studio (local) |

Classification and consolidation always use the local LLM.
Repair retries follow the same routing as the original call.

---

## Nightly jobs

The scheduler runs three background jobs:

| Job | Schedule | Purpose |
|---|---|---|
| `consolidation` | 02:00 UTC | Extract patterns from episodic events → skill store |
| `index_rebuild` | 02:30 UTC | Rebuild FAISS index from updated semantic graph |
| `proactive_monitor` | Every 15 min | Check git, filesystem, calendar for proactive actions |

Jobs are fire-and-forget — a failing job never crashes the scheduler.

---

## Project structure

```
agent/
├── core/           config, logger, classifier, orchestrator, session
├── memory/         episodic, skills, graph, embedder, FAISS, retrieval, context builder
├── llm/            LM Studio client, Anthropic client, router, prompt engine, validator
├── tools/          shell, watcher, code (read/write/test/git), Blender bridge + tools
├── background/     scheduler, consolidation, index rebuild, proactive monitor
└── interface/      CLI (chat, memory, status)

scripts/            init_memory_tree, verify_*, replay_trace, session_report
tests/              ~450 tests, one file per module
```

---

## Configuration reference

See `.env.example` for all available options with documentation.

Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio endpoint |
| `LM_STUDIO_CHAT_MODEL` | — | Model name as shown in LM Studio |
| `ANTHROPIC_API_KEY` | — | Optional cloud fallback |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Cloud model |
| `AGENT_MEMORY_ROOT` | `~/.agent` | Memory store location |
| `LLM_CLOUD_THRESHOLD_TOKENS` | `2000` | Route to cloud above this count |
| `EMBEDDING_DIM` | `2560` | Must match embedding model output |
| `CONSOLIDATION_HOUR` | `2` | UTC hour for nightly consolidation |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Development

```bash
# Lint
uv run ruff check agent/ scripts/

# Type check
uv run mypy agent/

# Format
uv run ruff format agent/ scripts/
```


```bash
# macOS
brew install ripgrep
brew install koekeishiya/formulae/skhd
cp config/skhdrc ~/.skhdrc
skhd --start-service

```



```bash
# After any session
uv run python scripts/session_report.py

# Filter to one project
uv run python scripts/session_report.py --project agrivision

# Export to CSV for tracking over time
uv run python scripts/session_report.py --csv >> ~/agent_costs.csv

```