"""
agent/memory/embedder.py

Async embedding client for the memory system.

Primary path:   LM Studio /v1/embeddings (Qwen3-Embedding-4B)
Fallback path:  sentence-transformers (offline, lower quality)

The fallback activates automatically when LM Studio is unreachable.
Once the primary path fails, the embedder stays in fallback mode for
the rest of the session to avoid per-call latency from repeated timeouts.

Batching:
    Qwen3-Embedding-4B has a context window — large inputs must be chunked.
    Default batch size is 32 texts per request, configurable via EMBEDDING_BATCH_SIZE.
    The embed() method handles chunking transparently.

Usage:
    from agent.memory.embedder import get_embedder
    embedder = await get_embedder()
    vectors = await embedder.embed(["FastAPI async pattern", "bpy export GLB"])
"""

from __future__ import annotations

import asyncio
import time
from typing import ClassVar

import httpx
import numpy as np

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)

# Max texts per single HTTP request to LM Studio
EMBEDDING_BATCH_SIZE = 32


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    """
    Async embedding client. Do not instantiate directly — use get_embedder().

    State:
        _using_fallback: once True, stays True for the session.
                         Avoids timeout overhead on every call after a failure.
        _fallback_model: lazy-loaded sentence-transformers model.
                         Only imported and loaded if LM Studio is unreachable.
    """

    _using_fallback: bool = False
    _fallback_model: ClassVar[object | None] = None  # SentenceTransformer instance

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts and return a float32 numpy array of shape (N, dim).

        Handles batching internally — safe to call with any number of texts.
        Raises RuntimeError only if both primary and fallback paths fail.
        """
        if not texts:
            return np.empty((0, cfg.embedding_dim), dtype=np.float32)

        # Chunk into batches
        batches = [
            texts[i : i + EMBEDDING_BATCH_SIZE]
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)
        ]

        results: list[np.ndarray] = []
        for batch in batches:
            vectors = await self._embed_batch(batch)
            results.append(vectors)

        return np.vstack(results).astype(np.float32)

    async def embed_one(self, text: str) -> np.ndarray:
        """
        Convenience wrapper — embed a single string.
        Returns a 1D float32 array of shape (dim,).
        """
        vectors = await self.embed([text])
        return vectors[0]

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def dim(self) -> int:
        """Expected embedding dimension for the active model."""
        if self._using_fallback:
            return cfg.embedding_fallback_dim
        return cfg.embedding_dim

    async def close(self) -> None:
        await self._client.aclose()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed one batch — tries primary first, falls back on failure."""
        if not self._using_fallback:
            try:
                return await self._embed_lm_studio(texts)
            except Exception as exc:
                log.warning(
                    "embedder.primary_failed",
                    error=str(exc),
                    switching_to="fallback",
                )
                self._using_fallback = True

        return await self._embed_fallback(texts)

    async def _embed_lm_studio(self, texts: list[str]) -> np.ndarray:
        """Call LM Studio /v1/embeddings."""
        t0 = time.perf_counter()

        response = await self._client.post(
            f"{cfg.lm_studio_base_url}/embeddings",
            json={
                "model": cfg.lm_studio_embedding_model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()

        # LM Studio returns data sorted by index
        vectors = np.array(
            [item["embedding"] for item in data["data"]],
            dtype=np.float32,
        )

        elapsed = time.perf_counter() - t0
        log.debug(
            "embedder.lm_studio",
            texts=len(texts),
            shape=list(vectors.shape),
            elapsed_ms=round(elapsed * 1000, 1),
        )

        return vectors

    async def _embed_fallback(self, texts: list[str]) -> np.ndarray:
        """
        Embed using sentence-transformers (offline fallback).
        Model is loaded lazily on first use — avoid importing at module level
        since it pulls in torch and slows startup for users who don't need it.
        """
        if Embedder._fallback_model is None:
            log.info(
                "embedder.fallback_loading",
                model=cfg.embedding_fallback_model,
            )
            # Run in thread pool — SentenceTransformer.__init__ is blocking
            loop = asyncio.get_event_loop()
            Embedder._fallback_model = await loop.run_in_executor(
                None, self._load_fallback_model
            )

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None,
            lambda: Embedder._fallback_model.encode(  # type: ignore[union-attr]
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        elapsed = time.perf_counter() - t0
        log.debug(
            "embedder.fallback",
            texts=len(texts),
            shape=list(vectors.shape),
            elapsed_ms=round(elapsed * 1000, 1),
        )

        return vectors.astype(np.float32)

    @staticmethod
    def _load_fallback_model() -> object:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(cfg.embedding_fallback_model)


# ── Singleton ─────────────────────────────────────────────────────────────────

_embedder_instance: Embedder | None = None
_embedder_lock = asyncio.Lock()


async def get_embedder() -> Embedder:
    """
    Return the session-scoped Embedder singleton.
    Creates the instance on first call. Thread-safe via asyncio lock.
    """
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    async with _embedder_lock:
        # Double-checked locking
        if _embedder_instance is None:
            _embedder_instance = Embedder()
            log.info(
                "embedder.initialised",
                primary_model=cfg.lm_studio_embedding_model,
                primary_dim=cfg.embedding_dim,
                fallback_model=cfg.embedding_fallback_model,
                fallback_dim=cfg.embedding_fallback_dim,
            )
    return _embedder_instance


def reset_embedder() -> None:
    """
    Reset the singleton. Used in tests to force re-initialisation
    and to simulate fallback switching between test cases.
    """
    global _embedder_instance
    Embedder._fallback_model = None
    _embedder_instance = None