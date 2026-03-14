"""
tests/test_graph.py

Tests for the semantic graph schema and SemanticGraph operations.
All filesystem I/O uses pytest's tmp_path.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.memory.graph import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    NodeType,
    SemanticGraph,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_node(
    node_id: str = "agrivision",
    label: str = "AgriVision",
    node_type: NodeType = NodeType.PROJECT,
    **kwargs: object,
) -> GraphNode:
    return GraphNode(id=node_id, label=label, node_type=node_type, **kwargs) # type: ignore


def make_edge(
    source: str = "agrivision",
    target: str = "fastapi",
    relation: EdgeRelation = EdgeRelation.USES,
    **kwargs: object,
) -> GraphEdge:
    return GraphEdge(source_id=source, target_id=target, relation=relation, **kwargs) # type: ignore


def populated_graph() -> SemanticGraph:
    """Return a SemanticGraph with a few nodes and edges for traversal tests."""
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node("agrivision", "AgriVision", NodeType.PROJECT))
    g.upsert_node(make_node("fastapi", "FastAPI", NodeType.LIBRARY))
    g.upsert_node(make_node("plant_disease", "Plant Disease Detection", NodeType.CONCEPT))
    g.upsert_node(make_node("detect_py", "api/routes/detect.py", NodeType.FILE))
    g.upsert_edge(make_edge("agrivision", "fastapi", EdgeRelation.USES))
    g.upsert_edge(make_edge("agrivision", "plant_disease", EdgeRelation.CONTAINS))
    g.upsert_edge(make_edge("detect_py", "plant_disease", EdgeRelation.IMPLEMENTS))
    return g


# ── GraphNode schema ──────────────────────────────────────────────────────────

def test_node_minimal_constructs() -> None:
    node = make_node()
    assert node.id == "agrivision"
    assert node.node_type == NodeType.PROJECT
    assert node.attributes == {}
    assert node.embedding is None


def test_node_id_whitespace_stripped() -> None:
    node = make_node(node_id="  agrivision  ")
    assert node.id == "agrivision"


def test_node_id_empty_raises() -> None:
    with pytest.raises(ValidationError, match="id"):
        make_node(node_id="")


def test_node_id_with_whitespace_raises() -> None:
    with pytest.raises(ValidationError, match="whitespace"):
        make_node(node_id="bad id")


def test_node_label_empty_raises() -> None:
    with pytest.raises(ValidationError, match="label"):
        make_node(label="")


def test_node_timestamps_default_utc() -> None:
    node = make_node()
    assert node.created_at.tzinfo == timezone.utc
    assert node.updated_at.tzinfo == timezone.utc


def test_node_touch_updates_timestamp() -> None:
    node = make_node()
    touched = node.touch()
    assert touched is not node
    assert touched.updated_at >= node.updated_at


def test_node_embedding_excluded_from_serialisation() -> None:
    node = make_node()
    node_with_embed = node.model_copy(update={"embedding": [0.1, 0.2, 0.3]})
    dumped = json.loads(node_with_embed.model_dump_json())
    assert "embedding" not in dumped


def test_node_to_embed_text_basic() -> None:
    node = make_node("agrivision", "AgriVision", NodeType.PROJECT)
    text = node.to_embed_text()
    assert "project" in text.lower()
    assert "AgriVision" in text


def test_node_to_embed_text_includes_description() -> None:
    node = make_node(attributes={"description": "plant disease detection app"})
    text = node.to_embed_text()
    assert "plant disease detection app" in text


def test_node_to_embed_text_includes_pattern() -> None:
    node = GraphNode(
        id="pattern_fastapi",
        label="FastAPI async pattern",
        node_type=NodeType.PATTERN,
        attributes={"pattern": "Always use async def with Pydantic models"},
    )
    text = node.to_embed_text()
    assert "async def" in text


# ── GraphEdge schema ──────────────────────────────────────────────────────────

def test_edge_minimal_constructs() -> None:
    edge = make_edge()
    assert edge.source_id == "agrivision"
    assert edge.target_id == "fastapi"
    assert edge.relation == EdgeRelation.USES
    assert edge.weight == 1.0


def test_edge_source_empty_raises() -> None:
    with pytest.raises(ValidationError):
        make_edge(source="")


def test_edge_target_empty_raises() -> None:
    with pytest.raises(ValidationError):
        make_edge(target="")


def test_edge_weight_negative_raises() -> None:
    with pytest.raises(ValidationError):
        make_edge(weight=-0.1)


def test_edge_reinforced_increases_weight() -> None:
    edge = make_edge(weight=1.0)
    reinforced = edge.reinforced(by=1.0)
    assert reinforced.weight == pytest.approx(2.0)


def test_edge_reinforced_returns_new_instance() -> None:
    edge = make_edge()
    reinforced = edge.reinforced()
    assert edge is not reinforced
    assert edge.weight == 1.0


# ── SemanticGraph — node ops ──────────────────────────────────────────────────

def test_upsert_node_adds_node() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node())
    assert g.node_count == 1


def test_get_node_returns_correct_node() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node("fastapi", "FastAPI", NodeType.LIBRARY))
    node = g.get_node("fastapi")
    assert node is not None
    assert node.label == "FastAPI"
    assert node.node_type == NodeType.LIBRARY


def test_get_node_returns_none_when_absent() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    assert g.get_node("nonexistent") is None


def test_upsert_node_merges_attributes() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node(attributes={"lang": "python"}))
    g.upsert_node(make_node(attributes={"framework": "fastapi"}))
    node = g.get_node("agrivision")
    assert node is not None
    assert node.attributes["lang"] == "python"
    assert node.attributes["framework"] == "fastapi"


def test_upsert_node_new_attributes_win() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node(attributes={"version": "1.0"}))
    g.upsert_node(make_node(attributes={"version": "2.0"}))
    node = g.get_node("agrivision")
    assert node is not None
    assert node.attributes["version"] == "2.0"


def test_remove_node_returns_true() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node())
    assert g.remove_node("agrivision") is True
    assert g.node_count == 0


def test_remove_node_returns_false_when_absent() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    assert g.remove_node("nonexistent") is False


def test_all_nodes_returns_all() -> None:
    g = populated_graph()
    nodes = g.all_nodes()
    assert len(nodes) == 4


def test_all_nodes_filters_by_type() -> None:
    g = populated_graph()
    libs = g.all_nodes(node_type=NodeType.LIBRARY)
    assert len(libs) == 1
    assert libs[0].id == "fastapi"


# ── SemanticGraph — edge ops ──────────────────────────────────────────────────

def test_upsert_edge_adds_edge() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node("agrivision", "AgriVision", NodeType.PROJECT))
    g.upsert_node(make_node("fastapi", "FastAPI", NodeType.LIBRARY))
    g.upsert_edge(make_edge())
    assert g.edge_count == 1


def test_upsert_edge_missing_source_raises() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node("fastapi", "FastAPI", NodeType.LIBRARY))
    with pytest.raises(ValueError, match="Source node not found"):
        g.upsert_edge(make_edge("missing", "fastapi"))


def test_upsert_edge_missing_target_raises() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node("agrivision", "AgriVision", NodeType.PROJECT))
    with pytest.raises(ValueError, match="Target node not found"):
        g.upsert_edge(make_edge("agrivision", "missing"))


def test_upsert_edge_reinforces_existing() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node("agrivision", "AgriVision", NodeType.PROJECT))
    g.upsert_node(make_node("fastapi", "FastAPI", NodeType.LIBRARY))
    g.upsert_edge(make_edge(weight=1.0))
    g.upsert_edge(make_edge(weight=1.0))  # second observation
    edge = g.get_edge("agrivision", "fastapi")
    assert edge is not None
    assert edge.weight == pytest.approx(2.0)


def test_remove_edge_works() -> None:
    g = SemanticGraph(graph_path=Path("/dev/null"))
    g.upsert_node(make_node("agrivision", "AgriVision", NodeType.PROJECT))
    g.upsert_node(make_node("fastapi", "FastAPI", NodeType.LIBRARY))
    g.upsert_edge(make_edge())
    assert g.remove_edge("agrivision", "fastapi") is True
    assert g.edge_count == 0


def test_remove_node_also_removes_edges() -> None:
    g = populated_graph()
    edge_count_before = g.edge_count
    g.remove_node("agrivision")
    assert g.edge_count < edge_count_before


# ── Traversal ─────────────────────────────────────────────────────────────────

def test_neighbors_depth_1() -> None:
    g = populated_graph()
    neighbors = g.neighbors("agrivision", depth=1)
    neighbor_ids = {n.id for n in neighbors}
    assert "fastapi" in neighbor_ids
    assert "plant_disease" in neighbor_ids
    # detect_py is 2 hops away
    assert "detect_py" not in neighbor_ids


def test_neighbors_depth_2() -> None:
    g = populated_graph()
    neighbors = g.neighbors("agrivision", depth=2)
    neighbor_ids = {n.id for n in neighbors}
    # plant_disease → detect_py is within 2 hops
    assert "detect_py" in neighbor_ids


def test_neighbors_excludes_source() -> None:
    g = populated_graph()
    neighbors = g.neighbors("agrivision", depth=2)
    assert all(n.id != "agrivision" for n in neighbors)


def test_neighbors_with_relation_filter() -> None:
    g = populated_graph()
    neighbors = g.neighbors("agrivision", depth=1, relation=EdgeRelation.USES)
    assert len(neighbors) == 1
    assert neighbors[0].id == "fastapi"


def test_neighbors_nonexistent_node_returns_empty() -> None:
    g = populated_graph()
    assert g.neighbors("nonexistent") == []


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_creates_file(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    g = SemanticGraph(graph_path=graph_path)
    g.upsert_node(make_node())
    g.save()
    assert graph_path.exists()


def test_save_produces_valid_json(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    g = SemanticGraph(graph_path=graph_path)
    g.upsert_node(make_node("agrivision", "AgriVision", NodeType.PROJECT))
    g.save()
    data = json.loads(graph_path.read_text())
    assert "nodes" in data
    assert "links" in data


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"

    g1 = SemanticGraph(graph_path=graph_path)
    g1.upsert_node(make_node("agrivision", "AgriVision", NodeType.PROJECT))
    g1.upsert_node(make_node("fastapi", "FastAPI", NodeType.LIBRARY))
    g1.upsert_edge(make_edge())
    g1.save()

    g2 = SemanticGraph(graph_path=graph_path).load()
    assert g2.node_count == 2
    assert g2.edge_count == 1
    node = g2.get_node("agrivision")
    assert node is not None
    assert node.label == "AgriVision"


def test_load_nonexistent_returns_empty_graph(tmp_path: Path) -> None:
    g = SemanticGraph(graph_path=tmp_path / "nonexistent.json").load()
    assert g.node_count == 0
    assert g.edge_count == 0


def test_nodes_for_indexing_returns_positions(tmp_path: Path) -> None:
    g = populated_graph()
    index_data = g.nodes_for_indexing()
    assert len(index_data) == g.node_count
    positions = [item[0] for item in index_data]
    node_ids = [item[1] for item in index_data]
    assert positions == list(range(g.node_count))
    assert "agrivision" in node_ids
    assert all(isinstance(item[2], str) and len(item[2]) > 0 for item in index_data)