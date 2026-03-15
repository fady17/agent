"""
agent/background/index_rebuild.py

FAISS index nightly rebuild — runs 30 minutes after consolidation.

The consolidation engine may have added new nodes to the semantic graph
or updated existing ones. The FAISS index is a derived artifact —
it must be rebuilt whenever the graph changes to keep ANN search accurate.

Rebuild pipeline:
    1. Load SemanticGraph from graph.json
    2. Check node count — skip if graph is empty
    3. Get Embedder (LM Studio primary, sentence-transformers fallback)
    4. Embed all nodes via graph.nodes_for_indexing()
    5. Build fresh FaissIndex
    6. Save index.faiss + node_index.json (atomic — old files replaced)
    7. Write episodic event with rebuild stats

Atomicity:
    FaissIndex.save() writes to the configured paths directly.
    Both files are written sequentially — there is a brief window where
    index.faiss is new but node_index.json is old. The retrieval pipeline
    handles this by catching load errors and returning empty results
    rather than crashing. A full atomic swap (write to tmp, rename both)
    is overkill for personal-scale use where the agent is the only reader.

Design:
    - Force local embedding (LM Studio) — nightly rebuild is not latency
      sensitive and we don't want cloud API calls for batch embedding.
    - The job is safe to run manually at any time via:
        await run_index_rebuild()
    - Detailed stats are logged and written as an episodic event for
      the operator to inspect if something goes wrong.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.memory.embedder import Embedder, get_embedder
from agent.memory.episodic import EpisodicEvent, EventType, write_event
from agent.memory.graph import SemanticGraph
from agent.memory.index import FaissIndex

log = get_logger(__name__)


# ── Result ────────────────────────────────────────────────────────────────────

class IndexRebuildResult:
    """Result of one index rebuild run."""

    def __init__(
        self,
        *,
        success: bool,
        node_count: int = 0,
        dim: int = 0,
        elapsed_ms: float = 0.0,
        error: str = "",
        skipped: bool = False,
        skip_reason: str = "",
    ) -> None:
        self.success     = success
        self.node_count  = node_count
        self.dim         = dim
        self.elapsed_ms  = elapsed_ms
        self.error       = error
        self.skipped     = skipped
        self.skip_reason = skip_reason

    def __repr__(self) -> str:
        if self.skipped:
            return f"IndexRebuildResult(skipped={self.skip_reason!r})"
        if self.success:
            return (
                f"IndexRebuildResult(ok, nodes={self.node_count}, "
                f"dim={self.dim}, {self.elapsed_ms:.0f}ms)"
            )
        return f"IndexRebuildResult(FAILED, error={self.error!r})"


# ── Core rebuild logic ────────────────────────────────────────────────────────

async def run_index_rebuild(
    graph: SemanticGraph | None = None,
    embedder: Embedder | None = None,
    index: FaissIndex | None = None,
) -> IndexRebuildResult:
    """
    Rebuild the FAISS index from the current semantic graph.

    Args:
        graph:    SemanticGraph to index (loaded from disk if None).
        embedder: Embedder to use (session singleton if None).
        index:    FaissIndex to build into (default paths if None).

    Returns:
        IndexRebuildResult — always returns, never raises.
    """
    t0 = time.perf_counter()

    log.info("index_rebuild.start")

    try:
        # ── Load graph ────────────────────────────────────────────────────────
        _graph = graph
        if _graph is None:
            _graph = SemanticGraph()
            if cfg.graph_path.exists():
                _graph.load()
            else:
                log.info("index_rebuild.no_graph", path=str(cfg.graph_path))

        node_count = _graph.node_count

        if node_count == 0:
            result = IndexRebuildResult(
                success=True,
                skipped=True,
                skip_reason="graph has no nodes",
            )
            log.info("index_rebuild.skipped", reason=result.skip_reason)
            return result

        log.info("index_rebuild.embedding", node_count=node_count)

        # ── Get embedder ──────────────────────────────────────────────────────
        _embedder = embedder or await get_embedder()

        # ── Build index ───────────────────────────────────────────────────────
        _index = index or FaissIndex()
        await _index.build(_graph, _embedder)

        # ── Save atomically ───────────────────────────────────────────────────
        _index.save()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        dim        = _index.dim

        # ── Write episodic event ──────────────────────────────────────────────
        try:
            write_event(EpisodicEvent(
                event_type=EventType.MEMORY_CONSOLIDATE,
                summary=(
                    f"FAISS index rebuilt: {node_count} nodes, "
                    f"dim={dim}, {elapsed_ms:.0f}ms"
                ),
                data={
                    "node_count": node_count,
                    "dim":        dim,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            ))
        except Exception as exc:
            # Event write failure is non-fatal
            log.warning("index_rebuild.event_write_failed", error=str(exc))

        result = IndexRebuildResult(
            success=True,
            node_count=node_count,
            dim=dim,
            elapsed_ms=round(elapsed_ms, 1),
        )

        log.info(
            "index_rebuild.complete",
            nodes=node_count,
            dim=dim,
            elapsed_ms=result.elapsed_ms,
        )
        return result

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        error_msg  = str(exc)

        log.error("index_rebuild.failed", error=error_msg, elapsed_ms=elapsed_ms)

        # Write failure event
        try:
            write_event(EpisodicEvent(
                event_type=EventType.ERROR,
                summary=f"FAISS index rebuild failed: {error_msg[:120]}",
                data={"error": error_msg},
            ))
        except Exception:
            pass

        return IndexRebuildResult(
            success=False,
            elapsed_ms=round(elapsed_ms, 1),
            error=error_msg,
        )