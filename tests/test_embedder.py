"""
tests/test_embedder.py

Tests for the async Embedder.

Strategy:
  - LM Studio tests use httpx MockTransport to avoid needing a live server.
  - Fallback tests mock SentenceTransformer to avoid loading torch.
  - Singleton and reset tests verify session-scoped behaviour.

All tests are async. reset_embedder() is called between tests to
prevent singleton state leaking across cases.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from agent.memory.embedder import (
    EMBEDDING_BATCH_SIZE,
    Embedder,
    get_embedder,
    reset_embedder,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_embedder():
    """Reset the singleton and fallback model before every test."""
    reset_embedder()
    yield
    reset_embedder()


def make_lm_response(texts: list[str], dim: int = 2560) -> httpx.Response:
    """Build a fake LM Studio /v1/embeddings response."""
    data = [
        {"index": i, "embedding": [float(i) * 0.01] * dim}
        for i in range(len(texts))
    ]
    return httpx.Response(
        200,
        json={"object": "list", "data": data},
        request=httpx.Request("POST", "http://127.0.0.1:1234/v1/embeddings"),
    )


def mock_transport(texts: list[str], dim: int = 2560) -> httpx.MockTransport:
    """Return a MockTransport that serves a valid embedding response."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return make_lm_response(body["input"], dim=dim)

    return httpx.MockTransport(handler)


# ── Basic embedding — LM Studio path ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_returns_correct_shape() -> None:
    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=mock_transport([], dim=2560))

    texts = ["FastAPI pattern", "Blender export", "Flutter widget"]
    result = await embedder.embed(texts)

    assert result.shape == (3, 2560)
    assert result.dtype == np.float32


@pytest.mark.asyncio
async def test_embed_single_text() -> None:
    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=mock_transport([], dim=2560))

    result = await embedder.embed(["single text"])
    assert result.shape == (1, 2560)


@pytest.mark.asyncio
async def test_embed_one_returns_1d_array() -> None:
    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=mock_transport([], dim=2560))

    result = await embedder.embed_one("single text")
    assert result.ndim == 1
    assert result.shape == (2560,)


@pytest.mark.asyncio
async def test_embed_empty_list_returns_empty_array() -> None:
    embedder = Embedder()
    result = await embedder.embed([])
    assert result.shape == (0, 2560)


@pytest.mark.asyncio
async def test_embed_output_is_float32() -> None:
    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=mock_transport([], dim=2560))
    result = await embedder.embed(["test"])
    assert result.dtype == np.float32


# ── Batching ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_batches_large_inputs() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        return make_lm_response(body["input"], dim=2560)

    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    # 70 texts should produce 3 batches (32 + 32 + 6)
    texts = [f"text {i}" for i in range(70)]
    result = await embedder.embed(texts)

    assert result.shape == (70, 2560)
    assert call_count == 3


@pytest.mark.asyncio
async def test_embed_exactly_one_batch() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        return make_lm_response(body["input"], dim=2560)

    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    texts = [f"text {i}" for i in range(EMBEDDING_BATCH_SIZE)]
    result = await embedder.embed(texts)

    assert result.shape == (EMBEDDING_BATCH_SIZE, 2560)
    assert call_count == 1


# ── Fallback path ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_switches_to_fallback_on_lm_studio_failure() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            request=httpx.Request("POST", "http://127.0.0.1:1234/v1/embeddings"),
        )

    mock_st = MagicMock()
    mock_st.encode.return_value = np.ones((1, 384), dtype=np.float32)

    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    Embedder._fallback_model = mock_st

    result = await embedder.embed(["test text"])

    assert embedder.using_fallback is True
    assert result.shape == (1, 384)


@pytest.mark.asyncio
async def test_fallback_stays_active_after_first_failure() -> None:
    """Once fallback is active, LM Studio is never contacted again."""
    lm_call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal lm_call_count
        lm_call_count += 1
        raise httpx.ConnectError("refused")

    mock_st = MagicMock()
    mock_st.encode.return_value = np.ones((1, 384), dtype=np.float32)

    embedder = Embedder()
    embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    Embedder._fallback_model = mock_st

    await embedder.embed(["first call"])   # triggers fallback switch
    await embedder.embed(["second call"])  # should not hit LM Studio
    await embedder.embed(["third call"])

    # LM Studio was only contacted once (the first failed attempt)
    assert lm_call_count == 1
    assert embedder.using_fallback is True


@pytest.mark.asyncio
async def test_fallback_dim_property() -> None:
    embedder = Embedder()
    embedder._using_fallback = True
    assert embedder.dim == 384  # cfg.embedding_fallback_dim default


@pytest.mark.asyncio
async def test_primary_dim_property() -> None:
    embedder = Embedder()
    assert embedder.dim == 2560  # cfg.embedding_dim default


# ── Singleton ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_embedder_returns_same_instance() -> None:
    a = await get_embedder()
    b = await get_embedder()
    assert a is b


@pytest.mark.asyncio
async def test_reset_embedder_clears_singleton() -> None:
    a = await get_embedder()
    reset_embedder()
    b = await get_embedder()
    assert a is not b


@pytest.mark.asyncio
async def test_reset_clears_fallback_model() -> None:
    Embedder._fallback_model = MagicMock()
    reset_embedder()
    assert Embedder._fallback_model is None


# ── close ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_does_not_raise() -> None:
    embedder = Embedder()
    await embedder.close()  # should not raise