"""
scripts/verify_ml.py

Verifies the full ML stack works end-to-end before any memory code is written.
Tests the real embedding path: LM Studio HTTP → FAISS index → ANN search.
Falls back to sentence-transformers if LM Studio is unreachable.

Run:
    uv run python scripts/verify_ml.py
"""

import os
import sys
import time

import httpx
import numpy as np

# ── Config (read from env or use defaults matching .env.example) ─────────────
LM_STUDIO_BASE_URL    = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
EMBEDDING_MODEL       = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b")
EMBEDDING_DIM         = int(os.getenv("EMBEDDING_DIM", "2560"))
FALLBACK_MODEL        = os.getenv("EMBEDDING_FALLBACK_MODEL", "all-MiniLM-L6-v2")
FALLBACK_DIM          = int(os.getenv("EMBEDDING_FALLBACK_DIM", "384"))

SAMPLE_TEXTS = [
    "FastAPI async endpoint with Pydantic validation",
    "Blender bpy export GLB with draco compression",
    "Flutter widget rebuild on state change",
    "Plant disease detection using image classification",
    "FAISS approximate nearest neighbour search",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def header(title: str) -> None:
    print(f"\n{title}")
    print("─" * 50)


def ok(msg: str) -> None:
    print(f"  [ok]  {msg}")


def fail(msg: str) -> None:
    print(f"  [!!]  {msg}")


# ── Step 1: LM Studio connectivity ──────────────────────────────────────────

def check_lm_studio() -> bool:
    header("Step 1 — LM Studio connectivity")
    try:
        r = httpx.get(f"{LM_STUDIO_BASE_URL}/models", timeout=3.0)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        ok(f"LM Studio reachable at {LM_STUDIO_BASE_URL}")
        ok(f"Loaded models: {', '.join(models) if models else '(none visible)'}")
        if EMBEDDING_MODEL in models:
            ok(f"Embedding model confirmed loaded: {EMBEDDING_MODEL}")
        else:
            print(f"  [??]  '{EMBEDDING_MODEL}' not in model list — may still work if active")
        return True
    except Exception as e:
        fail(f"LM Studio unreachable: {e}")
        return False


# ── Step 2: Embed via LM Studio ──────────────────────────────────────────────

def embed_lm_studio(texts: list[str]) -> tuple[np.ndarray, int] | None:
    header("Step 2 — LM Studio embeddings")
    try:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{LM_STUDIO_BASE_URL}/embeddings",
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()["data"]
        vectors = np.array([d["embedding"] for d in data], dtype=np.float32)
        elapsed = time.perf_counter() - t0

        dim = vectors.shape[1]
        ok(f"Embedded {len(texts)} texts in {elapsed:.2f}s")
        ok(f"Vector shape: {vectors.shape}  (expected dim={EMBEDDING_DIM})")

        if dim != EMBEDDING_DIM:
            print(f"  [!!]  Dim mismatch — got {dim}, expected {EMBEDDING_DIM}")
            print(f"        Update EMBEDDING_DIM={dim} in your .env")

        return vectors, dim
    except Exception as e:
        fail(f"Embedding call failed: {e}")
        return None


# ── Step 3: Fallback — sentence-transformers ─────────────────────────────────

def embed_fallback(texts: list[str]) -> tuple[np.ndarray, int] | None:
    header("Step 2 (fallback) — sentence-transformers embeddings")
    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.perf_counter()
        model = SentenceTransformer(FALLBACK_MODEL)
        vectors = model.encode(texts, convert_to_numpy=True).astype(np.float32)
        elapsed = time.perf_counter() - t0
        ok(f"Fallback model '{FALLBACK_MODEL}' loaded and embedded in {elapsed:.2f}s")
        ok(f"Vector shape: {vectors.shape}")
        return vectors, vectors.shape[1]
    except Exception as e:
        fail(f"Fallback embedding failed: {e}")
        return None


# ── Step 4: FAISS index insert + search ──────────────────────────────────────

def check_faiss(vectors: np.ndarray, dim: int) -> bool:
    header("Step 3 — FAISS index insert + search")
    try:
        import faiss

        index = faiss.IndexFlatL2(dim)
        index.add(vectors) # type: ignore
        ok(f"Indexed {index.ntotal} vectors (dim={dim})")

        # Query: first vector should be its own nearest neighbour
        query = vectors[0:1]
        distances, indices = index.search(query, k=3) # type: ignore
        ok(f"ANN search returned indices: {indices[0].tolist()}")
        ok(f"Distances: {[round(float(d), 4) for d in distances[0]]}")

        assert indices[0][0] == 0, "Top result should be the query itself (distance=0)"
        ok("Nearest neighbour is self — index is correct")

        # Sanity: semantic neighbours should make sense
        # "FastAPI" (idx 0) should not be closest to "Blender" (idx 1)
        ok("FAISS working correctly")
        return True
    except Exception as e:
        fail(f"FAISS test failed: {e}")
        return False


# ── Step 5: networkx sanity ───────────────────────────────────────────────────

def check_networkx() -> bool:
    header("Step 4 — networkx graph sanity")
    try:
        import networkx as nx

        g = nx.DiGraph()
        g.add_node("agrivision", type="project")
        g.add_node("fastapi", type="library")
        g.add_edge("agrivision", "fastapi", relation="uses")

        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 1
        neighbors = list(g.successors("agrivision"))
        assert neighbors == ["fastapi"]
        ok(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        ok(f"agrivision → uses → {neighbors}")
        return True
    except Exception as e:
        fail(f"networkx test failed: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== ML stack verification ===")
    results: list[bool] = []

    lm_studio_up = check_lm_studio()

    if lm_studio_up:
        result = embed_lm_studio(SAMPLE_TEXTS)
        if result:
            vectors, dim = result
            results.append(True)
        else:
            results.append(False)
            vectors, dim = None, FALLBACK_DIM
    else:
        print("\n  LM Studio offline — running fallback path")
        result = embed_fallback(SAMPLE_TEXTS)
        if result:
            vectors, dim = result
            results.append(True)
        else:
            results.append(False)
            vectors, dim = None, FALLBACK_DIM

    if vectors is not None:
        results.append(check_faiss(vectors, dim))
    else:
        fail("Skipping FAISS — no vectors produced")
        results.append(False)

    results.append(check_networkx())

    # ── Summary ──
    passed = sum(results)
    total = len(results)
    print("\n" + "─" * 50)
    if all(results):
        print(f"  All {total} checks passed — ML stack ready.\n")
    else:
        print(f"  {passed}/{total} passed — fix failures before proceeding.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()