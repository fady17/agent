# type: ignore
"""
agent/memory/index.py

FAISS index over semantic graph node embeddings.

Files on disk:
    ~/.agent/memory/semantic/index.faiss      — the binary ANN index
    ~/.agent/memory/semantic/node_index.json  — {str(position): node_id} mapping

Relationship to graph.json:
    index.faiss and node_index.json are DERIVED from graph.json.
    They are never the source of truth. The rebuild process is:

        1. Load SemanticGraph from graph.json
        2. Call graph.nodes_for_indexing() → [(position, node_id, embed_text)]
        3. Embed all embed_text strings via Embedder
        4. Build IndexFlatL2 over the vectors
        5. Write index.faiss + node_index.json

    Position in the FAISS matrix == position in nodes_for_indexing() output.
    node_index.json maps that position back to a node_id so ANN results
    can be resolved to graph nodes.

Why IndexFlatL2:
    - Exact search — no approximation error on a personal-scale graph
      (hundreds to low thousands of nodes, not millions)
    - No training required — add vectors and search immediately
    - Rebuild is fast enough to run nightly even with 10k nodes

Usage:
    from agent.memory.index import FaissIndex
    idx = FaissIndex()
    await idx.build(graph, embedder)   # first time or nightly rebuild
    idx.save()
    idx.load()                         # load existing index at startup
    results = idx.search(query_vec, k=5)  # → [(node_id, distance), ...]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.memory.embedder import Embedder
from agent.memory.graph import SemanticGraph

log = get_logger(__name__)


class FaissIndex:
    """
    ANN index over semantic graph node embeddings.

    State:
        _index:      faiss.Index — in-memory index, None until built or loaded
        _node_map:   dict[int, str] — position → node_id
        _dim:        int — embedding dimension (locked on first build/load)
    """

    def __init__(
        self,
        index_path: Path | None = None,
        node_index_path: Path | None = None,
    ) -> None:
        self._index_path = index_path or cfg.faiss_index_path
        self._node_index_path = (
            node_index_path
            or cfg.semantic_dir / "node_index.json"
        )
        self._index: faiss.IndexFlatL2 | None = None
        self._node_map: dict[int, str] = {}
        self._dim: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    @property
    def dim(self) -> int:
        return self._dim

    # ── Build ─────────────────────────────────────────────────────────────────

    async def build(
        self,
        graph: SemanticGraph,
        embedder: Embedder,
    ) -> "FaissIndex":
        """
        Embed all nodes in the graph and build a fresh IndexFlatL2.

        Steps:
            1. Get (position, node_id, embed_text) tuples from the graph
            2. Embed all texts in one batched call
            3. Construct IndexFlatL2 with the correct dimension
            4. Add all vectors
            5. Build the node_map for position → node_id resolution

        Returns self for chaining.
        Raises ValueError if the graph has no nodes.
        """
        index_data = graph.nodes_for_indexing()

        if not index_data:
            raise ValueError(
                "Cannot build FAISS index — graph has no nodes. "
                "Seed the graph before calling build()."
            )

        positions = [item[0] for item in index_data]
        node_ids  = [item[1] for item in index_data]
        texts     = [item[2] for item in index_data]

        log.info("index.build_start", node_count=len(texts))
        t0 = time.perf_counter()

        # Embed all node texts
        vectors = await embedder.embed(texts)  # shape: (N, dim)

        if vectors.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedder returned {vectors.shape[0]} vectors for {len(texts)} texts"
            )

        dim = vectors.shape[1]

        # Build index
        index = faiss.IndexFlatL2(dim)
        index.add(vectors) 

        # Build node map
        node_map: dict[int, str] = {
            pos: nid for pos, nid in zip(positions, node_ids)
        }

        self._index    = index
        self._node_map = node_map
        self._dim      = dim

        elapsed = time.perf_counter() - t0
        log.info(
            "index.build_complete",
            vectors=index.ntotal,
            dim=dim,
            elapsed_ms=round(elapsed * 1000, 1),
        )

        return self

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(
        self,
        index_path: Path | None = None,
        node_index_path: Path | None = None,
    ) -> "FaissIndex":
        """
        Write index.faiss and node_index.json to disk.
        Raises RuntimeError if the index has not been built yet.
        """
        if self._index is None:
            raise RuntimeError("Cannot save — index has not been built. Call build() first.")

        idx_path  = index_path  or self._index_path
        nidx_path = node_index_path or self._node_index_path

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        nidx_path.parent.mkdir(parents=True, exist_ok=True)

        # Write binary FAISS index
        faiss.write_index(self._index, str(idx_path))

        # Write node map — JSON keys must be strings
        node_index_serialisable = {str(k): v for k, v in self._node_map.items()}
        nidx_path.write_text(
            json.dumps(node_index_serialisable, indent=2),
            encoding="utf-8",
        )

        log.info(
            "index.saved",
            index_path=str(idx_path),
            node_index_path=str(nidx_path),
            vectors=self._index.ntotal,
        )

        return self

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(
        self,
        index_path: Path | None = None,
        node_index_path: Path | None = None,
    ) -> "FaissIndex":
        """
        Load index.faiss and node_index.json from disk.
        Returns self for chaining: idx = FaissIndex().load()
        Raises FileNotFoundError if either file is absent.
        """
        idx_path  = index_path  or self._index_path
        nidx_path = node_index_path or self._node_index_path

        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")
        if not nidx_path.exists():
            raise FileNotFoundError(f"Node index not found: {nidx_path}")

        self._index = faiss.read_index(str(idx_path))
        self._dim   = self._index.d

        raw = json.loads(nidx_path.read_text(encoding="utf-8"))
        self._node_map = {int(k): v for k, v in raw.items()}

        log.info(
            "index.loaded",
            index_path=str(idx_path),
            vectors=self._index.ntotal,
            dim=self._dim,
        )

        return self

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find the k nearest nodes to the query vector.

        Args:
            query_vector: 1D or 2D float32 array.
                          If 1D (shape: (dim,)) it is reshaped to (1, dim).
            k:            Number of nearest neighbours to return.
                          Capped at total_vectors to avoid FAISS errors.

        Returns:
            List of (node_id, distance) tuples, ordered by distance ascending.
            Empty list if the index is not built.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Normalise query shape to (1, dim)
        vec = np.array(query_vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        if vec.shape[1] != self._dim:
            raise ValueError(
                f"Query vector dim {vec.shape[1]} does not match index dim {self._dim}"
            )

        # Cap k at actual total — faiss raises if k > ntotal
        k_actual = min(k, self._index.ntotal)

        distances, indices = self._index.search(vec, k_actual)

        results: list[tuple[str, float]] = []
        for dist, pos in zip(distances[0], indices[0]):
            if pos == -1:
                # FAISS returns -1 for padding when fewer results exist
                continue
            node_id = self._node_map.get(int(pos))
            if node_id is not None:
                results.append((node_id, float(dist)))

        return results

    # ── Try-load helper ───────────────────────────────────────────────────────

    def try_load(
        self,
        index_path: Path | None = None,
        node_index_path: Path | None = None,
    ) -> bool:
        """
        Attempt to load the index from disk.
        Returns True on success, False if files are missing.
        Does not raise — safe to call at startup.
        """
        try:
            self.load(index_path, node_index_path)
            return True
        except FileNotFoundError:
            log.debug("index.not_found_on_disk")
            return False
        except Exception as exc:
            log.warning("index.load_failed", error=str(exc))
            return False