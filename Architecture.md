# Architecture

Technical reference for the autonomous personal agent.
Covers every major system boundary, data flow, and design decision.

---

## System overview

```
┌─────────────────────────────────────────────────────────────────┐
│  User                                                           │
│  uv run agent chat                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │ stdin / stdout
┌────────────────────────▼────────────────────────────────────────┐
│  CLI  (agent/interface/cli.py)                                  │
│  typer app · Rich output · asyncio.run()                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ run_turn() / run_turn_stream()
┌────────────────────────▼────────────────────────────────────────┐
│  Orchestrator  (agent/core/orchestrator.py)                     │
│  classify → retrieve → build context → prompt → LLM → validate │
│  → write event → update session → return TurnResult            │
└──┬─────────┬──────────┬──────────────┬──────────────────────────┘
   │         │          │              │
   ▼         ▼          ▼              ▼
Classifier  Memory   LLM Router    Session
            Pipeline               Manager
```

---

## Component inventory

### Phase 0 — Scaffold
| Module | Purpose |
|---|---|
| `agent/core/config.py` | `AgentConfig(BaseSettings)` singleton via `@lru_cache`. Single source of truth for all paths and env vars. |
| `agent/core/logger.py` | `structlog` dual-output: JSON to `~/.agent/logs/trace.jsonl` + coloured terminal. |
| `scripts/init_memory_tree.py` | Idempotent setup of `~/.agent/` directory tree. |

### Phase 1-3 — Memory
| Module | Purpose |
|---|---|
| `agent/memory/episodic.py` | `EpisodicEvent` model, `write_event()`, `list_events()`. Daily-partitioned JSON files. |
| `agent/memory/search.py` | Async ripgrep wrapper for full-text search across episodic store. |
| `agent/memory/skills.py` | `SkillRecord` with confidence decay. `upsert_skill()` handles create/boost/pattern-change. |
| `agent/memory/graph.py` | `SemanticGraph` wrapping `networkx.DiGraph`. Typed node/edge API. Persists as `graph.json`. |
| `agent/memory/embedder.py` | Async embedder. LM Studio primary, sentence-transformers fallback. Auto-switches on failure. |
| `agent/memory/index.py` | `FaissIndex` wrapping `faiss.IndexFlatL2`. Build, save, load, search. |

### Phase 4 — Retrieval
| Module | Purpose |
|---|---|
| `agent/memory/retrieval.py` | `RetrievalPipeline` — three-stage: metadata filter → ANN search + graph traversal → skill lookup. |
| `agent/memory/context_builder.py` | Token-budgeted formatter. Priority: skills > events > graph nodes. |
| `agent/core/session.py` | `SessionState` + `SessionManager`. Immutable update pattern, atomic persistence. |

### Phase 5 — LLM Gateway
| Module | Purpose |
|---|---|
| `agent/llm/lm_studio.py` | Async httpx client for LM Studio. Streaming, retry, `LLMResponse`. |
| `agent/llm/anthropic_client.py` | Same interface. Anthropic Messages API, SSE streaming, cost tracking. |
| `agent/llm/router.py` | Routes between local/cloud based on token count and liveness. TTL-cached liveness check. |
| `agent/llm/prompt_engine.py` | Template registry with few-shot injection. `_SafeFormatMap` for graceful substitution. |
| `agent/llm/validator.py` | JSON / text validation with repair-retry loop. Shows LLM its bad output + error. |

### Phase 6 — Orchestration
| Module | Purpose |
|---|---|
| `agent/core/classifier.py` | `TaskIntent` schema. `TaskClassifier` → always `force_local=True`, `temperature=0.0`. |
| `agent/core/orchestrator.py` | `run_turn()` pipeline. `run_turn_stream()` async generator. `shutdown()` writes SESSION_END event. |

### Phase 7 — Tools
| Module | Purpose |
|---|---|
| `agent/tools/shell.py` | `ShellRunner` — `asyncio.create_subprocess_exec`, timeout + SIGTERM/SIGKILL, `ShellResult`. |
| `agent/tools/watcher.py` | `FileWatcher` — watchdog observer in background thread, bridged to asyncio queue via `call_soon_threadsafe`. |
| `agent/tools/code.py` | `read_file`, `write_file` (atomic), `run_tests` (pytest parser), `git_log`. |

### Phase 8 — Blender
| Module | Purpose |
|---|---|
| `agent/tools/blender_server.py` | Runs *inside Blender*. Socket server, scripts scheduled on main thread via `bpy.app.timers`. |
| `agent/tools/blender_bridge.py` | Agent-side async TCP client. Zero bpy imports in agent venv. |
| `agent/tools/blender_tools.py` | `get_scene_info`, `export_glb`, `fix_ngons`, `generate_lod`. Script strings → bridge → parsed results. |

### Phase 9 — Background
| Module | Purpose |
|---|---|
| `agent/background/scheduler.py` | `AgentScheduler` wrapping APScheduler `AsyncIOScheduler`. Job registry, error isolation. |
| `agent/background/consolidation.py` | Nightly: events → LLM (`CONSOLIDATE` template) → `ExtractedPattern` list → `upsert_skill`. |
| `agent/background/index_rebuild.py` | Nightly (30 min after consolidation): rebuild FAISS from updated graph. |
| `agent/background/monitor.py` | Every 15 min: git log, filesystem scan, calendar. Writes `ProactiveAction` to pending queue. |

### Phase 10 — Interface
| Module | Purpose |
|---|---|
| `agent/interface/cli.py` | `typer` app: `chat`, `memory show/search/skills/delete`, `status`. |

### Phase 11 — Observability
| Script | Purpose |
|---|---|
| `scripts/replay_trace.py` | Reads `trace.jsonl`, filters by level/event, renders as Rich table. |
| `scripts/session_report.py` | Cost/token summary from session state + episodic log. CSV export. |

---

## Data flow — one turn

```
user input
    │
    ▼
TaskClassifier.classify()
    │  CLASSIFY template (few-shot, temperature=0.0, force_local)
    │  LLM returns JSON → ResponseValidator → TaskIntent
    ▼
RetrievalPipeline.retrieve(input, project, task_type)
    │  Stage 1: list_events(since, project, event_type) → recent_events
    │  Stage 2: embed(input) → FAISS ANN → GraphNode neighbors (depth=2)
    │  Stage 3: get_skill(task_type) → SkillRecord
    ▼
ContextBuilder.build(ContextBlock)
    │  Token budget: skills > events > nodes
    │  Returns ContextPayload(text, tokens_used, was_truncated)
    ▼
PromptEngine.build(RESPOND, input=..., context=payload)
    │  Injects context into {context} placeholder in system prompt
    │  Returns list[Message]
    ▼
LLMRouter.complete(messages)
    │  Decision: token count / liveness / force flags
    │  Local → LMStudioClient.complete()
    │  Cloud → AnthropicClient.complete()
    ▼
ResponseValidator.validate_text(response)
    │  min_length=1, repair retry on failure
    ▼
write_event(EpisodicEvent(LLM_CALL, summary, project, data))
    │  ~/.agent/memory/episodic/YYYY/MM/DD/evt-{ms13}-{hex8}.json
    ▼
SessionManager.update(state.with_turn().with_metrics(...))
    │  Atomic write to ~/.agent/memory/working/session.json
    ▼
TurnResult(content, intent, session, turn_number, latency_ms, cost_usd)
```

---

## Memory architecture

### Episodic store

Append-only, never rewritten. Partitioned by date for efficient listing.

```
~/.agent/memory/episodic/
└── 2026/
    └── 03/
        └── 14/
            ├── evt-1741910400123-a3f2c1.json
            ├── evt-1741910401456-b4e7d2.json
            └── ...
```

Event IDs are sortable by design: `evt-{13-digit-ms}-{8-hex-chars}`.
`list_events()` sorts by filename descending — no index needed for recency queries.
Full-text search uses ripgrep (`rg`) rather than a database — fast, zero deps.

### Semantic graph

`graph.json` stores the node-link structure (no embeddings).
`index.faiss` and `node_index.json` are derived artifacts rebuilt nightly.
The graph and index are intentionally decoupled: the graph is the source of
truth, the FAISS index is a search accelerator. Loss of the index is
non-fatal — the next `index_rebuild` job recreates it.

### Skill store

One JSON file per task type: `~/.agent/memory/skills/create_api_endpoint.json`.
Confidence decays multiplicatively each night (`0.98` per decay pass).
Skills below `0.10` are pruned automatically.
`upsert_skill()` is idempotent — re-observing a known task type boosts
confidence rather than creating a duplicate.

### Consolidation watermark

`~/.agent/consolidation/state.json` stores the last processed event ID.
Each nightly run only processes events newer than the watermark, keeping
the LLM call cost proportional to actual activity rather than total history.

---

## LLM routing logic

```
input token estimate (chars / 4)
    │
    ├─ > LLM_CLOUD_THRESHOLD_TOKENS ──────────────→ cloud
    │
    ├─ force_local=True ──────────────────────────→ local (bypass all)
    ├─ force_cloud=True ──────────────────────────→ cloud (bypass all)
    │
    ├─ LM Studio liveness (TTL=5min cache)
    │   ├─ unreachable + no API key ──────────────→ local (best effort)
    │   └─ unreachable + API key ─────────────────→ cloud
    │
    └─ default ───────────────────────────────────→ local
```

Classification and consolidation **always** use `force_local=True` because:
- They fire on every single turn / every night
- They are small, structured tasks local models handle well
- Cloud cost for classification would be non-trivial at scale

---

## Blender integration

### Why socket IPC

Blender embeds Python 3.11. The agent runs Python 3.12. They cannot share
a venv. Socket IPC is the only reliable inter-process communication path
that works across Python version boundaries.

### Thread safety

`bpy.ops` and most `bpy` operations are **not thread-safe** in Blender.
The socket server runs in a background thread (safe: only I/O), but
all script execution is scheduled on Blender's main thread via
`bpy.app.timers.register()`. This is the same approach used by
[blender-mcp](https://github.com/ahujasid/blender-mcp).

### Two integration modes

| Mode | Use case |
|---|---|
| `blender_server.py` socket bridge | Autonomous background jobs — the agent initiates without a human in the loop. Proactive monitor detects a new `.blend` file and triggers the LOD pipeline. |
| blender-mcp + Claude Desktop | Interactive sessions — a human asks Claude to manipulate a scene in real time. Rich tool palette including Poly Haven, Sketchfab, Hyper3D. |

The two modes are complementary. The agent uses the socket bridge.
A human using Claude Desktop can use blender-mcp in parallel.

---

## Background scheduler

```
APScheduler AsyncIOScheduler
    │
    ├─ consolidation         cron 02:00 UTC
    │   list_events(since=watermark)
    │   → batch(50) → LLM(CONSOLIDATE) → [ExtractedPattern]
    │   → upsert_skill() for each pattern
    │   → write MEMORY_CONSOLIDATE event
    │   → save watermark
    │
    ├─ index_rebuild         cron 02:30 UTC
    │   load SemanticGraph
    │   → embed all nodes
    │   → build FaissIndex
    │   → save index.faiss + node_index.json
    │
    └─ proactive_monitor     interval 15 min
        check git log (new commits)
        check filesystem (new .blend, .py, .dart)
        check calendar (~/.agent/config/calendar.json)
        → ProactiveAction queue (persisted)
        → drained at start of next chat turn
```

Job failures are logged and recorded as episodic `ERROR` events but never
propagate to the scheduler — one broken job cannot stop the others.

---

## Prompt templates

Five templates in `agent/llm/prompt_engine.py`:

| ID | Purpose | Key variables |
|---|---|---|
| `CLASSIFY` | → JSON `TaskIntent` | `{input}` |
| `RESPOND` | General assistant response | `{input}`, `{context}` |
| `TOOL_PLAN` | → JSON steps array | `{input}`, `{tools}` |
| `CONSOLIDATE` | → JSON patterns array | `{events}` |
| `SUMMARISE` | → Markdown summary | `{input}`, `{max_sentences}` |

`{context}` in `RESPOND` is replaced with the `ContextPayload.text` from
the retrieval pipeline. If context is empty the placeholder becomes an
empty string — the prompt still works cleanly.

`_SafeFormatMap` is a `dict` subclass that preserves unknown `{placeholders}`
rather than raising `KeyError`. This allows partial substitution:
system prompt unknowns are preserved, user template `{input}` is required.

---

## Token budget

```
ContextBuilder(max_tokens=1500)
    │
    ├─ reserve: ## Memory context header
    │
    ├─ Priority 1: skills (highest signal — encode learned preferences)
    │   each skill: ~20 tokens
    │
    ├─ Priority 2: recent events
    │   each event: ~30 tokens
    │
    └─ Priority 3: graph nodes
        each node: ~10 tokens

Token estimation: len(text) // 4   (conservative: 1 token ≈ 4 chars)
```

The budget is intentionally conservative. A 1500-token context block
leaves ~2500 tokens in a 4096-token window for the system prompt,
user message, and response.

---

## Episodic event types

```python
class EventType(StrEnum):
    CODE_WRITE        = "code.write"
    CODE_READ         = "code.read"
    CODE_DEBUG        = "code.debug"
    CODE_REVIEW       = "code.review"
    CODE_TEST         = "code.test"
    CODE_REFACTOR     = "code.refactor"
    DESIGN_EXPORT     = "design.export"
    DESIGN_RENDER     = "design.render"
    DESIGN_SCULPT     = "design.sculpt"
    DESIGN_RIG        = "design.rig"
    DESIGN_TEXTURE    = "design.texture"
    LLM_CALL          = "llm.call"
    SHELL_RUN         = "shell.run"
    FILE_CHANGE       = "file.change"
    SEARCH_QUERY      = "search.query"
    SESSION_START     = "session.start"
    SESSION_END       = "session.end"
    MEMORY_CONSOLIDATE = "memory.consolidate"
    SKILL_LEARN       = "skill.learn"
    ERROR             = "error"
```

---

## Design decisions

### Filesystem-first memory
No database, no vector store server, no Docker. The entire memory
store is flat JSON files on disk. This means:
- Zero infrastructure to maintain
- `rsync` / `git` / `cp` for backup
- Any text editor can inspect or edit memory
- `rg` (ripgrep) for full-text search without an index

The trade-off is O(n) listing for large episodic stores. At personal
scale (thousands of events, not millions), this is never a bottleneck.

### Immutable state mutations
`SessionState`, `SkillRecord`, and `GraphNode` all follow the immutable
update pattern — mutations return new instances. This prevents partial-write
corruption and makes state transitions explicit and testable.

### Local LLM by default
The agent routes to local inference wherever possible. Cloud is used only
when the input is too large for the context window or LM Studio is
unreachable. Classification always runs locally — it fires on every turn
and is a small, deterministic task that local models handle well.

### No shared state between components
The orchestrator owns no business logic. The classifier, retrieval pipeline,
context builder, router, and validator are all stateless utilities. The
scheduler never calls the orchestrator directly — it only writes to the
episodic store and skill store. The file watcher never calls anything —
it only writes to the asyncio queue. This keeps the dependency graph acyclic
and makes every component independently testable.

### bpy boundary
The agent venv never imports `bpy`. The Blender server script (`blender_server.py`)
runs inside Blender's embedded Python 3.11 interpreter. All bpy code lives
exclusively in that file. The bridge communicates via script strings over TCP.
This means:
- The agent works even when Blender is not installed
- No ABI mismatch between Python versions
- The server can be updated without touching the agent

---

## Test strategy

~450 tests across 30 test files. Every component has its own test file.

| Category | Approach |
|---|---|
| Pure logic | Direct function calls, no mocking |
| File I/O | `tmp_path` fixture, never touches real `~/.agent` |
| LLM clients | `httpx.MockTransport`, no live server needed |
| Orchestrator | All deps injected, mocked router + mocked classifier |
| CLI | `typer.testing.CliRunner` |
| Blender bridge | Real local TCP server in a background thread |
| E2E smoke | Full stack with mocked router + real memory filesystem |
| Integration (watcher) | `@pytest.mark.integration` — skipped in fast CI |