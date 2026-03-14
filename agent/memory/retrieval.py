"""
agent/memory/retrieval.py

Three-stage retrieval pipeline — the read path of the memory system.

Given a natural language query and optional filters, returns a ContextBlock
ready to inject into an LLM prompt.

Stages:
    1. Metadata filter    — episodic events filtered by project/type/date
                            fast, no embedding needed, filesystem only
    2. Graph traversal    — ANN search over FAISS → top-K node IDs
                            → traverse outward N hops to pull related context
    3. ANN rerank         — embed the query, score all candidates,
                            return the top results ranked by relevance

Output:
    ContextBlock — structured object containing:
        - relevant episodic events (recent + matched)
        - relevant graph nodes (concepts, patterns, libraries)
        - relevant skill records (high-confidence patterns for this task type)
        - token budget metadata so the LLM gateway can truncate if needed

Design note:
    The pipeline is deliberately additive — each stage enriches the candidate
    set rather than filtering it down hard. The rerank step at the end is
    where relevance ordering is decided. This means a relevant event that
    doesn't match the keyword search can still surface via graph traversal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.memory.embedder import Embedder
from agent.memory.episodic import EpisodicEvent, EventType, list_events
from agent.memory.graph import GraphNode, SemanticGraph
from agent.memory.index import FaissIndex
from agent.memory.skills import SkillRecord, list_skills

log = get_logger(__name__)


# ── ContextBlock ──────────────────────────────────────────────────────────────

@dataclass
class ContextBlock:
    """
    Structured retrieval output, ready for LLM context injection.

    The context_builder (task 20) consumes this and formats it into
    the prompt string. Keeping retrieval and formatting separate means
    retrieval can be tested without any LLM involvement.
    """

    query: str
    project: str | None

    # Episodic memory — recent events and keyword-matched events
    recent_events: list[EpisodicEvent] = field(default_factory=list)

    # Semantic memory — graph nodes relevant to the query
    relevant_nodes: list[tuple[GraphNode, float]] = field(default_factory=list)
    # (node, distance) — distance=0.0 means exact match

    # Skill memory — high-confidence patterns applicable to this task type
    relevant_skills: list[SkillRecord] = field(default_factory=list)

    # Pipeline metadata — for logging and token budget decisions
    retrieval_ms: float = 0.0
    stage_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return (
            not self.recent_events
            and not self.relevant_nodes
            and not self.relevant_skills
        )

    def summary(self) -> str:
        return (
            f"ContextBlock("
            f"events={len(self.recent_events)}, "
            f"nodes={len(self.relevant_nodes)}, "
            f"skills={len(self.relevant_skills)}, "
            f"{self.retrieval_ms:.0f}ms)"
        )


# ── RetrievalPipeline ─────────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    Wires episodic store, semantic graph, FAISS index, and skill store
    into a single .retrieve() call.

    Instantiate once per session and reuse — loading the graph and index
    from disk is expensive; keeping them in memory is cheap.
    """

    def __init__(
        self,
        graph: SemanticGraph,
        index: FaissIndex,
        embedder: Embedder,
    ) -> None:
        self._graph    = graph
        self._index    = index
        self._embedder = embedder

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        project: str | None = None,
        task_type: str | None = None,
        since: datetime | None = None,
        max_events: int = 10,
        max_nodes: int = 10,
        max_skills: int = 5,
        graph_depth: int = 2,
        ann_k: int = 20,
    ) -> ContextBlock:
        """
        Run the three-stage retrieval pipeline and return a ContextBlock.

        Args:
            query:      Natural language query — embedded for ANN search.
            project:    Optional project slug to restrict episodic filter.
            task_type:  Optional task type to fetch targeted skill records.
            since:      Only include episodic events after this UTC datetime.
            max_events: Cap on returned episodic events.
            max_nodes:  Cap on returned graph nodes.
            max_skills: Cap on returned skill records.
            graph_depth: Traversal depth from each ANN-matched node.
            ann_k:      Number of ANN candidates before graph expansion.
        """
        t0 = time.perf_counter()
        block = ContextBlock(query=query, project=project)

        # ── Stage 1: Metadata filter (episodic) ──────────────────────────────
        recent = self._stage_episodic(
            project=project,
            since=since,
            limit=max_events,
        )
        block.recent_events = recent
        block.stage_counts["episodic"] = len(recent)

        log.debug("retrieval.stage1_done", events=len(recent))

        # ── Stage 2: Graph traversal via ANN ─────────────────────────────────
        nodes_with_dist = await self._stage_graph(
            query=query,
            ann_k=ann_k,
            depth=graph_depth,
            max_nodes=max_nodes,
        )
        block.relevant_nodes = nodes_with_dist
        block.stage_counts["graph_nodes"] = len(nodes_with_dist)

        log.debug("retrieval.stage2_done", nodes=len(nodes_with_dist))

        # ── Stage 3: ANN rerank — sort nodes by embedding distance ───────────
        # Already ordered by distance from _stage_graph (ANN results come
        # ordered; traversal nodes appended after, sorted by node_type weight)
        # Skill retrieval is post-rerank — targeted by task_type directly
        skills = self._stage_skills(
            task_type=task_type,
            limit=max_skills,
        )
        block.relevant_skills = skills
        block.stage_counts["skills"] = len(skills)

        log.debug("retrieval.stage3_done", skills=len(skills))

        block.retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        log.info(
            "retrieval.complete",
            query=query[:80],
            project=project,
            **block.stage_counts,
            retrieval_ms=block.retrieval_ms,
        )

        return block

    # ── Stage 1: Episodic filter ──────────────────────────────────────────────

    def _stage_episodic(
        self,
        *,
        project: str | None,
        since: datetime | None,
        limit: int,
    ) -> list[EpisodicEvent]:
        """
        Fetch recent episodic events matching the project and time filters.
        Always returns newest-first.
        """
        return list_events(
            project=project,
            since=since,
            limit=limit,
        )

    # ── Stage 2: Graph traversal ──────────────────────────────────────────────

    async def _stage_graph(
        self,
        *,
        query: str,
        ann_k: int,
        depth: int,
        max_nodes: int,
    ) -> list[tuple[GraphNode, float]]:
        """
        Embed the query, run ANN search, expand via graph traversal.

        Returns (GraphNode, distance) pairs ordered by:
            1. ANN distance (direct matches first, distance=actual)
            2. Traversal neighbours (appended after, distance=inf as placeholder)

        If the index is not built, returns empty — never raises.
        """
        if not self._index.is_built:
            log.debug("retrieval.index_not_built")
            return []

        # Embed query
        try:
            query_vec = await self._embedder.embed_one(query)
        except Exception as exc:
            log.warning("retrieval.embed_failed", error=str(exc))
            return []

        # ANN search
        ann_results = self._index.search(query_vec, k=ann_k)
        # ann_results: [(node_id, distance), ...]

        seen_ids: set[str] = set()
        result: list[tuple[GraphNode, float]] = []

        # Load direct ANN matches
        for node_id, dist in ann_results:
            node = self._graph.get_node(node_id)
            if node is None:
                continue
            if node_id not in seen_ids:
                result.append((node, dist))
                seen_ids.add(node_id)

        # Graph traversal — expand each direct match outward
        for node_id, _ in ann_results:
            if not self._graph.get_node(node_id):
                continue
            neighbours = self._graph.neighbors(node_id, depth=depth)
            for neighbour in neighbours:
                if neighbour.id not in seen_ids:
                    # Distance is unknown for traversal nodes — use infinity
                    # so they sort after direct ANN matches
                    result.append((neighbour, float("inf")))
                    seen_ids.add(neighbour.id)

        # Sort: finite distances first (ANN matches), then traversal nodes
        result.sort(key=lambda x: x[1])

        return result[:max_nodes]

    # ── Stage 3: Skill retrieval ──────────────────────────────────────────────

    def _stage_skills(
        self,
        *,
        task_type: str | None,
        limit: int,
    ) -> list[SkillRecord]:
        """
        Return relevant skill records.

        If task_type is given, attempt an exact lookup first — this is the
        fast path for the orchestrator which always knows the task type.
        Falls back to returning the top-N by confidence if no exact match.
        """
        from agent.memory.skills import get_skill

        if task_type:
            exact = get_skill(task_type)
            if exact and exact.is_reliable:
                return [exact]

        # Top-N by confidence, reliable only
        return list_skills(min_confidence=0.3)[:limit]