"""
tests/test_index.py

Tests for the FaissIndex class.

The Embedder is mocked throughout — tests control the exact vectors
so ANN search results are deterministic and verifiable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import faiss
import numpy as np
import pytest

from agent.memory.embedder import Embedder, reset_embedder
from agent.memory.graph import EdgeRelation, GraphNode, NodeType, SemanticGraph
from agent.memory.index import FaissIndex


# ── Fixtures ──────────────────────────────────────────────────────────────────

DIM = 8  # small dim for fast tests — real dim is 2560


@pytest.fixture(autouse=True)
def clean_embedder():
    reset_embedder()
    yield
    reset_embedder()


def make_graph_with_nodes(n: int) -> SemanticGraph:
    """Return a SemanticGraph with n distinct nodes."""
    g = SemanticGraph(graph_path=Path("/dev/null"))
    for i in range(n):
        g.upsert_node(GraphNode(
            id=f"node_{i}",
            label=f"Node {i}",
            node_type=NodeType.CONCEPT,
        ))
    return g


def make_mock_embedder(n: int, dim: int = DIM) -> Embedder:
    """
    Return a mock Embedder that produces deterministic orthogonal-ish vectors.
    Vector i has a spike at position (i % dim) so nearest-neighbour is exact.
    """
    def _make_vectors(texts: list[str]) -> np.ndarray:
        # Use the text's index hint embedded in the label ("Node 0" → 0)
        vecs =[]
        for t in texts:
            # Each text is "concept: Node {i} | ..." — extract i from position
            idx = len(vecs)
            v = np.zeros(dim, dtype=np.float32)
            v[idx % dim] = 1.0 + (idx // dim) * 0.1
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    # Use a native async function for the side effect
    async def _mock_embed(texts: list[str]) -> np.ndarray:
        return _make_vectors(texts)

    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore[assignment]
    embedder.embed = AsyncMock(side_effect=_mock_embed)
    return embedder


def make_query_vector(position: int, dim: int = DIM) -> np.ndarray:
    """Make a query vector that should match node at `position`."""
    v = np.zeros(dim, dtype=np.float32)
    v[position % dim] = 1.0 + (position // dim) * 0.1
    return v


# ── Build ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_populates_index(tmp_path: Path) -> None:
    g = make_graph_with_nodes(5)
    embedder = make_mock_embedder(5)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    assert idx.is_built
    assert idx.total_vectors == 5
    assert idx.dim == DIM


@pytest.mark.asyncio
async def test_build_empty_graph_raises(tmp_path: Path) -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    embedder = make_mock_embedder(0)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    with pytest.raises(ValueError, match="no nodes"):
        await idx.build(g, embedder)


@pytest.mark.asyncio
async def test_build_node_map_covers_all_nodes(tmp_path: Path) -> None:
    n = 7
    g = make_graph_with_nodes(n)
    embedder = make_mock_embedder(n)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    assert len(idx._node_map) == n
    node_ids = set(idx._node_map.values())
    for i in range(n):
        assert f"node_{i}" in node_ids


# ── Save / Load roundtrip ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_creates_both_files(tmp_path: Path) -> None:
    g = make_graph_with_nodes(3)
    embedder = make_mock_embedder(3)
    idx_path  = tmp_path / "index.faiss"
    nidx_path = tmp_path / "node_index.json"

    idx = FaissIndex(index_path=idx_path, node_index_path=nidx_path)
    await idx.build(g, embedder)
    idx.save()

    assert idx_path.exists()
    assert nidx_path.exists()


@pytest.mark.asyncio
async def test_save_before_build_raises(tmp_path: Path) -> None:
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    with pytest.raises(RuntimeError, match="build()"):
        idx.save()


@pytest.mark.asyncio
async def test_save_load_roundtrip(tmp_path: Path) -> None:
    n = 4
    g = make_graph_with_nodes(n)
    embedder = make_mock_embedder(n)
    idx_path  = tmp_path / "index.faiss"
    nidx_path = tmp_path / "node_index.json"

    # Build and save
    idx1 = FaissIndex(index_path=idx_path, node_index_path=nidx_path)
    await idx1.build(g, embedder)
    idx1.save()

    # Load fresh instance
    idx2 = FaissIndex(index_path=idx_path, node_index_path=nidx_path)
    idx2.load()

    assert idx2.total_vectors == n
    assert idx2.dim == DIM
    assert idx2._node_map == idx1._node_map


def test_load_missing_index_raises(tmp_path: Path) -> None:
    idx = FaissIndex(
        index_path=tmp_path / "missing.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        idx.load()


def test_load_missing_node_index_raises(tmp_path: Path) -> None:
    # Create index.faiss but not node_index.json
    idx_path = tmp_path / "index.faiss"
    index = faiss.IndexFlatL2(DIM)
    index.add(np.zeros((1, DIM), dtype=np.float32)) # type: ignore
    faiss.write_index(index, str(idx_path))

    idx = FaissIndex(
        index_path=idx_path,
        node_index_path=tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError, match="Node index not found"):
        idx.load()


def test_try_load_returns_false_when_missing(tmp_path: Path) -> None:
    idx = FaissIndex(
        index_path=tmp_path / "missing.faiss",
        node_index_path=tmp_path / "missing.json",
    )
    assert idx.try_load() is False


@pytest.mark.asyncio
async def test_try_load_returns_true_when_present(tmp_path: Path) -> None:
    g = make_graph_with_nodes(2)
    embedder = make_mock_embedder(2)
    idx_path  = tmp_path / "index.faiss"
    nidx_path = tmp_path / "node_index.json"

    idx1 = FaissIndex(index_path=idx_path, node_index_path=nidx_path)
    await idx1.build(g, embedder)
    idx1.save()

    idx2 = FaissIndex(index_path=idx_path, node_index_path=nidx_path)
    assert idx2.try_load() is True
    assert idx2.is_built


# ── Search ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_correct_nearest_node(tmp_path: Path) -> None:
    n = 5
    g = make_graph_with_nodes(n)
    embedder = make_mock_embedder(n)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    # Query for node_0's vector — should return node_0 as top result
    query = make_query_vector(0)
    results = idx.search(query, k=1)

    assert len(results) == 1
    top_node_id, top_dist = results[0]
    assert top_node_id == "node_0"
    assert top_dist == pytest.approx(0.0, abs=1e-4)


@pytest.mark.asyncio
async def test_search_returns_k_results(tmp_path: Path) -> None:
    n = 8
    g = make_graph_with_nodes(n)
    embedder = make_mock_embedder(n)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    query = make_query_vector(0)
    results = idx.search(query, k=5)

    assert len(results) == 5


@pytest.mark.asyncio
async def test_search_caps_k_at_total_vectors(tmp_path: Path) -> None:
    n = 3
    g = make_graph_with_nodes(n)
    embedder = make_mock_embedder(n)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    # k=100 with only 3 vectors — should not raise
    results = idx.search(make_query_vector(0), k=100)
    assert len(results) == n


@pytest.mark.asyncio
async def test_search_results_ordered_by_distance(tmp_path: Path) -> None:
    n = 5
    g = make_graph_with_nodes(n)
    embedder = make_mock_embedder(n)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    results = idx.search(make_query_vector(2), k=5)
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_search_on_empty_index_returns_empty() -> None:
    idx = FaissIndex(
        index_path=Path("/dev/null"),
        node_index_path=Path("/dev/null"),
    )
    query = np.zeros(DIM, dtype=np.float32)
    results = idx.search(query, k=5)
    assert results == []


@pytest.mark.asyncio
async def test_search_wrong_dim_raises(tmp_path: Path) -> None:
    g = make_graph_with_nodes(3)
    embedder = make_mock_embedder(3)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    wrong_dim_query = np.zeros(DIM + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="dim"):
        idx.search(wrong_dim_query, k=3)


@pytest.mark.asyncio
async def test_search_accepts_2d_query(tmp_path: Path) -> None:
    g = make_graph_with_nodes(3)
    embedder = make_mock_embedder(3)
    idx = FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )
    await idx.build(g, embedder)

    query_2d = make_query_vector(0).reshape(1, -1)
    results = idx.search(query_2d, k=1)
    assert len(results) == 1
    assert results[0][0] == "node_0"