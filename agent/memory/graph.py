"""
agent/memory/graph.py

Semantic memory — the agent's knowledge graph.

Structure on disk:
    ~/.agent/memory/semantic/graph.json      — node/edge structure, no embeddings
    ~/.agent/memory/semantic/node_index.json — {faiss_position: node_id} mapping
    ~/.agent/memory/semantic/index.faiss     — ANN index (built separately)

Design:
    - graph.json is the source of truth for structure. FAISS is derived.
    - Nodes carry metadata and optional embedding (stored separately in FAISS).
    - Edges are directed and typed. Weight reflects observation frequency.
    - networkx DiGraph is the in-memory representation.
    - The graph is loaded once per session and mutated in memory.
      Persistence is explicit — call save_graph() after mutations.

Typical query path:
    1. Embed the query string via LM Studio
    2. ANN search in FAISS → top-K node positions
    3. Translate positions → node IDs via node_index.json
    4. Load those nodes from the graph
    5. Traverse outward 1-2 hops to pull in related context
    6. Return the candidate set to the retrieval pipeline for re-ranking
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field, field_validator

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)


# ── NodeType ──────────────────────────────────────────────────────────────────

class NodeType(StrEnum):
    PROJECT  = "project"   # a codebase or work context (agrivision, plant-chat)
    LIBRARY  = "library"   # a dependency or framework (FastAPI, Flutter, bpy)
    TOOL     = "tool"      # a CLI tool or external program (ripgrep, Blender)
    FILE     = "file"      # a specific source file or asset
    CONCEPT  = "concept"   # a stable domain idea (plant disease detection)
    PATTERN  = "pattern"   # an extracted behavioural preference from skill store
    PERSON   = "person"    # a collaborator or stakeholder (future use)


# ── EdgeRelation ──────────────────────────────────────────────────────────────

class EdgeRelation(StrEnum):
    USES        = "uses"          # project uses library/tool
    CONTAINS    = "contains"      # project contains file
    DEPENDS_ON  = "depends_on"    # project depends_on project
    IMPLEMENTS  = "implements"    # file implements concept
    APPLIES_TO  = "applies_to"    # pattern applies_to concept
    RELATED_TO  = "related_to"    # concept related_to concept (undirected semantics)
    PRODUCED_BY = "produced_by"   # pattern/concept produced_by episodic event
    PART_OF     = "part_of"       # file part_of project (alias for containment)


# ── GraphNode ─────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    """
    A node in the semantic knowledge graph.

    `id` is the stable unique identifier used as the networkx node key
    and as the lookup key in node_index.json.

    `embedding` is NOT stored in graph.json — it lives in the FAISS index.
    This field is only populated transiently when embeddings are needed
    (e.g. during index rebuild).
    """

    id: str
    label: str
    node_type: NodeType
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Transient — not persisted to graph.json, populated during index ops only
    embedding: list[float] | None = Field(default=None, exclude=True)

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("node id must not be empty")
        if len(v) > 200:
            raise ValueError(f"node id too long ({len(v)} chars, max 200)")
        # No whitespace — IDs are used as filenames and FAISS keys
        if any(c in v for c in (" ", "\t", "\n")):
            raise ValueError("node id must not contain whitespace")
        return v

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("node label must not be empty")
        return v

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse datetime: {v!r}")

    def touch(self) -> "GraphNode":
        """Return a copy with updated_at set to now."""
        return self.model_copy(update={"updated_at": datetime.now(timezone.utc)})

    # ── Text representation for embedding ─────────────────────────────────────

    def to_embed_text(self) -> str:
        """
        Produce the string that gets embedded into the FAISS index.
        Combines label, type, and key attributes for rich semantic content.
        """
        parts = [f"{self.node_type}: {self.label}"]
        # Include high-signal attribute values if present
        for key in ("description", "summary", "pattern", "language", "framework"):
            val = self.attributes.get(key)
            if val and isinstance(val, str):
                parts.append(val)
        return " | ".join(parts)


# ── GraphEdge ─────────────────────────────────────────────────────────────────

class GraphEdge(BaseModel):
    """
    A directed edge between two nodes.

    Weight starts at 1.0 and increases each time the same relationship
    is re-observed by the consolidation engine. Higher weight = stronger
    association = higher priority in weighted traversal.
    """

    source_id: str
    target_id: str
    relation: EdgeRelation
    weight: float = Field(default=1.0, ge=0.0)
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("source_id", "target_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("node id must not be empty")
        return v

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse datetime: {v!r}")

    def reinforced(self, by: float = 1.0) -> "GraphEdge":
        """Return a copy with weight increased by `by`."""
        return self.model_copy(update={"weight": round(self.weight + by, 6)})


# ── SemanticGraph ─────────────────────────────────────────────────────────────

class SemanticGraph:
    """
    In-memory semantic graph backed by networkx DiGraph.

    Wraps the networkx graph with typed operations so the rest of the
    codebase never touches nx directly. All mutations go through this class.
    Persistence is explicit — call .save() after mutations.
    """

    def __init__(self, graph_path: Path | None = None) -> None:
        self._path = graph_path or cfg.graph_path
        self._g: nx.DiGraph = nx.DiGraph()

    # ── Node operations ───────────────────────────────────────────────────────

    def upsert_node(self, node: GraphNode) -> GraphNode:
        """
        Insert or update a node.
        If the node already exists, merges attributes and updates updated_at.
        Returns the final node state.
        """
        if self._g.has_node(node.id):
            existing_data = self._g.nodes[node.id]
            # Merge attributes — new values win
            merged_attrs = {**existing_data.get("attributes", {}), **node.attributes}
            updated = node.model_copy(update={
                "attributes": merged_attrs,
                "created_at": existing_data.get("created_at", node.created_at),
                "updated_at": datetime.now(timezone.utc),
            })
            self._g.nodes[node.id].update(updated.model_dump(exclude={"embedding"}))
            log.debug("graph.node_updated", node_id=node.id, node_type=node.node_type)
            return updated
        else:
            self._g.add_node(node.id, **node.model_dump(exclude={"embedding"}))
            log.debug("graph.node_added", node_id=node.id, node_type=node.node_type)
            return node

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a node by ID, or None if not found."""
        if not self._g.has_node(node_id):
            return None
        data = dict(self._g.nodes[node_id])
        return GraphNode(**data)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges. Returns True if it existed."""
        if self._g.has_node(node_id):
            self._g.remove_node(node_id)
            log.debug("graph.node_removed", node_id=node_id)
            return True
        return False

    def all_nodes(self, node_type: NodeType | None = None) -> list[GraphNode]:
        """Return all nodes, optionally filtered by type."""
        nodes = []
        for node_id, data in self._g.nodes(data=True):
            try:
                node = GraphNode(**data)
                if node_type is None or node.node_type == node_type:
                    nodes.append(node)
            except Exception as exc:
                log.warning("graph.node_parse_error", node_id=node_id, error=str(exc))
        return nodes

    # ── Edge operations ───────────────────────────────────────────────────────

    def upsert_edge(self, edge: GraphEdge) -> GraphEdge:
        """
        Insert or reinforce an edge.
        If the same source→target→relation already exists, increases weight.
        Both nodes must exist — raises ValueError if either is missing.
        """
        if not self._g.has_node(edge.source_id):
            raise ValueError(f"Source node not found: {edge.source_id!r}")
        if not self._g.has_node(edge.target_id):
            raise ValueError(f"Target node not found: {edge.target_id!r}")

        if self._g.has_edge(edge.source_id, edge.target_id):
            existing = self._g.edges[edge.source_id, edge.target_id]
            if existing.get("relation") == edge.relation:
                # Reinforce existing edge
                reinforced = edge.reinforced(by=1.0)
                self._g.edges[edge.source_id, edge.target_id]["weight"] = reinforced.weight
                log.debug(
                    "graph.edge_reinforced",
                    source=edge.source_id,
                    target=edge.target_id,
                    relation=edge.relation,
                    weight=reinforced.weight,
                )
                return reinforced

        self._g.add_edge(
            edge.source_id,
            edge.target_id,
            **edge.model_dump(),
        )
        log.debug(
            "graph.edge_added",
            source=edge.source_id,
            target=edge.target_id,
            relation=edge.relation,
        )
        return edge

    def get_edge(self, source_id: str, target_id: str) -> GraphEdge | None:
        if not self._g.has_edge(source_id, target_id):
            return None
        return GraphEdge(**self._g.edges[source_id, target_id])

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        if self._g.has_edge(source_id, target_id):
            self._g.remove_edge(source_id, target_id)
            return True
        return False

    # ── Traversal ─────────────────────────────────────────────────────────────

    def neighbors(
        self,
        node_id: str,
        *,
        depth: int = 1,
        relation: EdgeRelation | None = None,
    ) -> list[GraphNode]:
        """
        Return all nodes reachable from node_id within `depth` hops.
        Optionally filter edges by relation type.
        The source node itself is excluded from results.
        """
        if not self._g.has_node(node_id):
            return []

        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for current in frontier:
                for successor in self._g.successors(current):
                    if successor in visited:
                        continue
                    if relation is not None:
                        edge_data = self._g.edges[current, successor]
                        if edge_data.get("relation") != relation:
                            continue
                    next_frontier.add(successor)
                    visited.add(successor)
            frontier = next_frontier

        result = []
        for nid in visited - {node_id}:
            node = self.get_node(nid)
            if node:
                result.append(node)
        return result

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """
        Serialise the graph to graph.json using networkx node-link format.
        Creates parent directories if needed.
        """
        target = path or self._path
        target.parent.mkdir(parents=True, exist_ok=True)

        data = nx.node_link_data(self._g, edges="links")
        target.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        log.debug(
            "graph.saved",
            path=str(target),
            nodes=self.node_count,
            edges=self.edge_count,
        )
        return target

    def load(self, path: Path | None = None) -> "SemanticGraph":
        """
        Load graph structure from graph.json.
        Returns self for chaining: graph = SemanticGraph().load()
        """
        source = path or self._path
        if not source.exists():
            log.debug("graph.load_empty", path=str(source))
            return self

        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            self._g = nx.node_link_graph(data, directed=True, edges="links")
            log.info(
                "graph.loaded",
                path=str(source),
                nodes=self.node_count,
                edges=self.edge_count,
            )
        except Exception as exc:
            log.error("graph.load_error", path=str(source), error=str(exc))
            raise

        return self

    # ── Node list for FAISS index rebuild ────────────────────────────────────

    def nodes_for_indexing(self) -> list[tuple[int, str, str]]:
        """
        Return (position, node_id, embed_text) for all nodes.
        Used by the FAISS index rebuilder.
        Position is the row index in the FAISS matrix.
        """
        result = []
        for position, (node_id, data) in enumerate(self._g.nodes(data=True)):
            try:
                node = GraphNode(**data)
                result.append((position, node_id, node.to_embed_text()))
            except Exception:
                continue
        return result