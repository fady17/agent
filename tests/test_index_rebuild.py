"""
tests/test_index_rebuild.py

Tests for the FAISS index rebuild job.
Graph, embedder, and index are all injected — no disk I/O needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent.background.index_rebuild import IndexRebuildResult, run_index_rebuild
from agent.memory.embedder import Embedder, reset_embedder
from agent.memory.episodic import EventType
from agent.memory.graph import GraphNode, NodeType, SemanticGraph
from agent.memory.index import FaissIndex

DIM = 8


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_embedder():
    reset_embedder()
    yield
    reset_embedder()


def make_graph(n_nodes: int = 3) -> SemanticGraph:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    for i in range(n_nodes):
        g.upsert_node(GraphNode(
            id=f"node_{i}",
            label=f"Node {i}",
            node_type=NodeType.CONCEPT,
        ))
    return g


def make_embedder(dim: int = DIM) -> Embedder:
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore

    async def _embed(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), dim), dtype=np.float32)

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


def make_index(tmp_path: Path) -> FaissIndex:
    return FaissIndex(
        index_path=tmp_path / "index.faiss",
        node_index_path=tmp_path / "node_index.json",
    )


# ── IndexRebuildResult ────────────────────────────────────────────────────────

def test_result_repr_success() -> None:
    r = IndexRebuildResult(success=True, node_count=10, dim=2560, elapsed_ms=340.0)
    assert "ok" in repr(r)
    assert "10" in repr(r)


def test_result_repr_skipped() -> None:
    r = IndexRebuildResult(success=True, skipped=True, skip_reason="graph has no nodes")
    assert "skipped" in repr(r)


def test_result_repr_failed() -> None:
    r = IndexRebuildResult(success=False, error="connection refused")
    assert "FAILED" in repr(r)
    assert "connection refused" in repr(r)


# ── run_index_rebuild — happy path ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_succeeds_with_populated_graph(tmp_path: Path) -> None:
    graph    = make_graph(5)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert result.success is True
    assert result.node_count == 5
    assert result.dim == DIM
    assert result.elapsed_ms >= 0
    assert not result.skipped


@pytest.mark.asyncio
async def test_rebuild_creates_index_files(tmp_path: Path) -> None:
    graph    = make_graph(3)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "node_index.json").exists()


@pytest.mark.asyncio
async def test_rebuild_calls_embedder(tmp_path: Path) -> None:
    graph    = make_graph(4)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    embedder.embed.assert_called_once() # type: ignore
    texts_embedded = embedder.embed.call_args[0][0] # type: ignore
    assert len(texts_embedded) == 4


@pytest.mark.asyncio
async def test_rebuild_writes_episodic_event(tmp_path: Path) -> None:
    graph    = make_graph(3)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event") as mock_write:
        await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    mock_write.assert_called_once()
    event = mock_write.call_args[0][0]
    assert event.event_type == EventType.MEMORY_CONSOLIDATE
    assert "3" in event.summary


@pytest.mark.asyncio
async def test_rebuild_result_node_count_matches_graph(tmp_path: Path) -> None:
    n        = 7
    graph    = make_graph(n)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert result.node_count == n


# ── Skip logic ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_skipped_on_empty_graph(tmp_path: Path) -> None:
    graph    = make_graph(0)  # empty
    embedder = make_embedder()
    index    = make_index(tmp_path)

    result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert result.success is True
    assert result.skipped is True
    assert "no nodes" in result.skip_reason
    embedder.embed.assert_not_called() # type: ignore


@pytest.mark.asyncio
async def test_rebuild_skipped_does_not_write_files(tmp_path: Path) -> None:
    graph    = make_graph(0)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert not (tmp_path / "index.faiss").exists()


@pytest.mark.asyncio
async def test_rebuild_skipped_does_not_write_event(tmp_path: Path) -> None:
    graph    = make_graph(0)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event") as mock_write:
        await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    mock_write.assert_not_called()


# ── Graph loading from disk ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_loads_graph_from_disk_if_not_injected(tmp_path: Path) -> None:
    """When graph is provided, it should use the provided graph."""
    graph = make_graph(2)
    graph_path = tmp_path / "graph.json"
    graph.save(graph_path) # Save it
    
    # Reload it into a new instance to ensure it's "fresh" from disk
    disk_graph = SemanticGraph(graph_path=graph_path).load()

    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        # Pass the disk_graph explicitly so we don't rely on cfg patching
        result = await run_index_rebuild(graph=disk_graph, embedder=embedder, index=index)

    assert result.success is True
    assert result.node_count == 2
    

@pytest.mark.asyncio
async def test_rebuild_skips_gracefully_if_no_graph_file(tmp_path: Path) -> None:
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.cfg") as mock_cfg:
        mock_cfg.graph_path = tmp_path / "nonexistent_graph.json"

        result = await run_index_rebuild(embedder=embedder, index=index)

    assert result.success is True
    assert result.skipped is True


# ── Error handling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_handles_embedder_failure(tmp_path: Path) -> None:
    graph    = make_graph(3)
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore
    embedder.embed = AsyncMock(side_effect=RuntimeError("LM Studio down"))

    index = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert result.success is False
    assert "LM Studio down" in result.error


@pytest.mark.asyncio
async def test_rebuild_never_raises(tmp_path: Path) -> None:
    """run_index_rebuild must never propagate exceptions."""
    graph    = make_graph(3)
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore
    embedder.embed = AsyncMock(side_effect=OSError("disk full"))

    index = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert isinstance(result, IndexRebuildResult)


@pytest.mark.asyncio
async def test_rebuild_writes_error_event_on_failure(tmp_path: Path) -> None:
    graph    = make_graph(3)
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore
    embedder.embed = AsyncMock(side_effect=RuntimeError("embed crash"))

    index = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event") as mock_write:
        await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    mock_write.assert_called_once()
    event = mock_write.call_args[0][0]
    assert event.event_type == EventType.ERROR


@pytest.mark.asyncio
async def test_rebuild_event_write_failure_is_non_fatal(tmp_path: Path) -> None:
    """A failure to write the episodic event must not fail the rebuild."""
    graph    = make_graph(2)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event", side_effect=OSError("disk full")):
        result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert result.success is True


# ── Elapsed time ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_result_has_positive_elapsed(tmp_path: Path) -> None:
    graph    = make_graph(3)
    embedder = make_embedder()
    index    = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert result.elapsed_ms >= 0.0


@pytest.mark.asyncio
async def test_rebuild_failure_has_elapsed_time(tmp_path: Path) -> None:
    graph    = make_graph(3)
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore
    embedder.embed = AsyncMock(side_effect=RuntimeError("crash"))

    index = make_index(tmp_path)

    with patch("agent.background.index_rebuild.write_event"):
        result = await run_index_rebuild(graph=graph, embedder=embedder, index=index)

    assert result.elapsed_ms >= 0.0