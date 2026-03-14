"""
tests/test_retrieval.py

Tests for the three-stage retrieval pipeline.

All external dependencies (embedder, filesystem) are either mocked or
redirected to tmp_path. Tests verify stage behaviour independently
and then the full pipeline end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from agent.memory.embedder import Embedder, reset_embedder
from agent.memory.episodic import EpisodicEvent, EventType, write_event
from agent.memory.graph import EdgeRelation, GraphNode, NodeType, SemanticGraph
from agent.memory.index import FaissIndex
from agent.memory.retrieval import ContextBlock, RetrievalPipeline
from agent.memory.skills import SkillRecord, save_skill


# ── Constants ─────────────────────────────────────────────────────────────────

DIM = 8


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_embedder_fixture():
    reset_embedder()
    yield
    reset_embedder()


def make_embedder(dim: int = DIM) -> Embedder:
    """Embedder that returns a zero vector — sufficient for non-ANN tests."""
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore[assignment]

    async def _embed(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), dim), dtype=np.float32)

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


def make_embedder_with_spike(node_ids: list[str], dim: int = DIM) -> Embedder:
    """
    Embedder that produces a unique spike vector per node position.
    The first call (during build) assigns vectors in order.
    Subsequent calls (query) accept a text and return the spike for position 0.
    """
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore[assignment]

    call_count = [0]

    async def _embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for i, _ in enumerate(texts):
            global_i = call_count[0] + i
            v = np.zeros(dim, dtype=np.float32)
            v[global_i % dim] = 1.0
            vecs.append(v)
        call_count[0] += len(texts)
        return np.array(vecs, dtype=np.float32)

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


def make_graph(node_specs: list[tuple[str, str, NodeType]]) -> SemanticGraph:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    for node_id, label, ntype in node_specs:
        g.upsert_node(GraphNode(id=node_id, label=label, node_type=ntype))
    return g


def make_index(graph: SemanticGraph, dim: int = DIM) -> FaissIndex:
    """Build a real FAISS index with zero vectors (fine for shape/plumbing tests)."""
    import faiss as _faiss

    index_data = graph.nodes_for_indexing()
    n = len(index_data)

    raw_index = _faiss.IndexFlatL2(dim)
    if n > 0:
        vecs = np.zeros((n, dim), dtype=np.float32)
        # Give each node a unique spike so ANN returns distinct results
        for i in range(n):
            vecs[i, i % dim] = 1.0
        raw_index.add(vecs) # type: ignore

    idx = FaissIndex(
        index_path=Path("/dev/null/index.faiss"),
        node_index_path=Path("/dev/null/node_index.json"),
    )
    idx._index    = raw_index
    idx._dim      = dim
    idx._node_map = {i: item[1] for i, item in enumerate(index_data)}
    return idx


def seed_episodic(base: Path, n: int = 3, project: str | None = "agrivision") -> list[EpisodicEvent]:
    events = []
    base_ts = datetime(2026, 3, 14, tzinfo=timezone.utc)
    for i in range(n):
        evt = EpisodicEvent(
            event_type=EventType.CODE_WRITE,
            summary=f"Wrote file number {i}",
            project=project,
            timestamp=base_ts + timedelta(minutes=i),
        )
        write_event(evt, base_dir=base)
        events.append(evt)
    return events


def seed_skills(base: Path) -> list[SkillRecord]:
    records = [
        SkillRecord(
            task_type="create_api_endpoint",
            pattern="Use FastAPI with async def and Pydantic",
            confidence=0.9,
        ),
        SkillRecord(
            task_type="export_3d_asset",
            pattern="Export as GLB, check ngons first",
            confidence=0.75,
        ),
    ]
    for r in records:
        save_skill(r, base_dir=base)
    return records


# ── ContextBlock ──────────────────────────────────────────────────────────────

def test_context_block_is_empty_when_no_results() -> None:
    block = ContextBlock(query="test", project=None)
    assert block.is_empty is True


def test_context_block_not_empty_with_events() -> None:
    evt = EpisodicEvent(event_type=EventType.CODE_WRITE, summary="wrote something")
    block = ContextBlock(query="test", project=None, recent_events=[evt])
    assert block.is_empty is False


def test_context_block_summary_format() -> None:
    block = ContextBlock(query="test", project="agrivision", retrieval_ms=42.5)
    s = block.summary()
    assert "events=0" in s
    assert "nodes=0" in s
    assert "skills=0" in s
    assert "42ms" in s or "43ms" in s


# ── Stage 1: Episodic ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stage1_returns_episodic_events(tmp_path: Path) -> None:
    seed_episodic(tmp_path, n=5, project="agrivision")

    graph = make_graph([])
    index = FaissIndex(index_path=Path("/dev/null"), node_index_path=Path("/dev/null"))
    embedder = make_embedder()

    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events") as mock_list:
        mock_list.return_value = [
            EpisodicEvent(event_type=EventType.CODE_WRITE, summary=f"event {i}")
            for i in range(3)
        ]
        block = await pipeline.retrieve("fastapi endpoint", project="agrivision", max_events=3)

    assert len(block.recent_events) == 3
    assert block.stage_counts["episodic"] == 3


@pytest.mark.asyncio
async def test_stage1_empty_when_no_events(tmp_path: Path) -> None:
    graph = make_graph([])
    index = FaissIndex(index_path=Path("/dev/null"), node_index_path=Path("/dev/null"))
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        block = await pipeline.retrieve("anything")

    assert block.recent_events == []


# ── Stage 2: Graph + ANN ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stage2_returns_graph_nodes() -> None:
    specs = [
        ("agrivision", "AgriVision", NodeType.PROJECT),
        ("fastapi",    "FastAPI",    NodeType.LIBRARY),
        ("flutter",    "Flutter",    NodeType.LIBRARY),
    ]
    graph = make_graph(specs)
    index = make_index(graph)
    embedder = make_embedder()

    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        with patch("agent.memory.retrieval.list_skills", return_value=[]):
            block = await pipeline.retrieve("agrivision project", max_nodes=10)

    assert len(block.relevant_nodes) > 0
    assert block.stage_counts["graph_nodes"] > 0


@pytest.mark.asyncio
async def test_stage2_skips_when_index_not_built() -> None:
    graph = make_graph([("agrivision", "AgriVision", NodeType.PROJECT)])
    index = FaissIndex(index_path=Path("/dev/null"), node_index_path=Path("/dev/null"))
    # index is empty — is_built is False
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        with patch("agent.memory.retrieval.list_skills", return_value=[]):
            block = await pipeline.retrieve("anything")

    assert block.relevant_nodes == []


@pytest.mark.asyncio
async def test_stage2_no_duplicate_nodes() -> None:
    specs = [
        ("a", "Node A", NodeType.CONCEPT),
        ("b", "Node B", NodeType.CONCEPT),
        ("c", "Node C", NodeType.CONCEPT),
    ]
    graph = make_graph(specs)
    # Add edges so traversal could produce duplicates
    from agent.memory.graph import GraphEdge
    graph.upsert_edge(GraphEdge(source_id="a", target_id="b", relation=EdgeRelation.RELATED_TO))
    graph.upsert_edge(GraphEdge(source_id="b", target_id="c", relation=EdgeRelation.RELATED_TO))
    index = make_index(graph)
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        with patch("agent.memory.retrieval.list_skills", return_value=[]):
            block = await pipeline.retrieve("concept query", ann_k=10, graph_depth=2, max_nodes=20)

    node_ids = [n.id for n, _ in block.relevant_nodes]
    assert len(node_ids) == len(set(node_ids)), "Duplicate nodes in results"


@pytest.mark.asyncio
async def test_stage2_ann_nodes_before_traversal_nodes() -> None:
    """Direct ANN matches (finite distance) must sort before traversal nodes (inf distance)."""
    specs = [
        ("ann_node",       "ANN Node",       NodeType.CONCEPT),
        ("traversal_node", "Traversal Node", NodeType.CONCEPT),
    ]
    graph = make_graph(specs)
    from agent.memory.graph import GraphEdge
    graph.upsert_edge(GraphEdge(
        source_id="ann_node",
        target_id="traversal_node",
        relation=EdgeRelation.RELATED_TO,
    ))
    index = make_index(graph)
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        with patch("agent.memory.retrieval.list_skills", return_value=[]):
            block = await pipeline.retrieve("query", ann_k=1, graph_depth=1, max_nodes=10)

    if len(block.relevant_nodes) >= 2:
        distances = [d for _, d in block.relevant_nodes]
        # All finite distances must appear before infinite ones
        finite = [d for d in distances if d != float("inf")]
        infinite = [d for d in distances if d == float("inf")]
        assert distances == finite + infinite


# ── Stage 3: Skills ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stage3_returns_skills_by_task_type(tmp_path: Path) -> None:
    skill = SkillRecord(
        task_type="create_api_endpoint",
        pattern="Use FastAPI with async",
        confidence=0.9,
    )
    graph = make_graph([])
    index = FaissIndex(index_path=Path("/dev/null"), node_index_path=Path("/dev/null"))
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        with patch("agent.memory.skills.get_skill", return_value=skill):
            with patch("agent.memory.retrieval.list_skills", return_value=[]):
                block = await pipeline.retrieve(
                    "create an api endpoint",
                    task_type="create_api_endpoint",
                )

    assert len(block.relevant_skills) == 1
    assert block.relevant_skills[0].task_type == "create_api_endpoint"


@pytest.mark.asyncio
async def test_stage3_falls_back_to_top_skills_when_no_task_type() -> None:
    skills = [
        SkillRecord(task_type="task_a", pattern="pattern a", confidence=0.8),
        SkillRecord(task_type="task_b", pattern="pattern b", confidence=0.6),
    ]
    graph = make_graph([])
    index = FaissIndex(index_path=Path("/dev/null"), node_index_path=Path("/dev/null"))
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        with patch("agent.memory.retrieval.list_skills", return_value=skills):
            block = await pipeline.retrieve("generic query")

    assert len(block.relevant_skills) == 2


# ── Full pipeline ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_pipeline_returns_context_block() -> None:
    specs = [
        ("agrivision", "AgriVision", NodeType.PROJECT),
        ("fastapi",    "FastAPI",    NodeType.LIBRARY),
    ]
    graph = make_graph(specs)
    index = make_index(graph)
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    events = [EpisodicEvent(event_type=EventType.CODE_WRITE, summary="wrote endpoint")]
    skills = [SkillRecord(task_type="api", pattern="use FastAPI", confidence=0.8)]

    with patch("agent.memory.retrieval.list_events", return_value=events):
        with patch("agent.memory.retrieval.list_skills", return_value=skills):
            block = await pipeline.retrieve(
                "create a FastAPI endpoint for agrivision",
                project="agrivision",
                task_type="create_api_endpoint",
            )

    assert isinstance(block, ContextBlock)
    assert block.query == "create a FastAPI endpoint for agrivision"
    assert block.project == "agrivision"
    assert not block.is_empty
    assert block.retrieval_ms > 0
    assert "episodic" in block.stage_counts
    assert "graph_nodes" in block.stage_counts
    assert "skills" in block.stage_counts


@pytest.mark.asyncio
async def test_full_pipeline_under_300ms() -> None:
    """Pipeline should complete well under the 300ms SLA with mocked deps."""
    import time
    specs = [
        ("node_a", "Node A", NodeType.CONCEPT),
        ("node_b", "Node B", NodeType.CONCEPT),
    ]
    graph = make_graph(specs)
    index = make_index(graph)
    embedder = make_embedder()
    pipeline = RetrievalPipeline(graph, index, embedder)

    with patch("agent.memory.retrieval.list_events", return_value=[]):
        with patch("agent.memory.retrieval.list_skills", return_value=[]):
            t0 = time.perf_counter()
            block = await pipeline.retrieve("performance test query")
            elapsed = time.perf_counter() - t0

    assert elapsed < 0.3, f"Pipeline took {elapsed:.3f}s — expected <0.3s"