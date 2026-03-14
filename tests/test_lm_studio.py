"""
tests/test_lm_studio.py

Tests for LMStudioClient using httpx.MockTransport.
No live LM Studio instance required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx
import pytest

from agent.llm.lm_studio import (
    LLMResponse,
    LMStudioClient,
    Message,
    _estimate_cloud_cost,
)


# ── Mock transport helpers ────────────────────────────────────────────────────

def make_complete_response(
    content: str = "Hello from LM Studio",
    model: str = "test-model",
    tokens_in: int = 20,
    tokens_out: int = 10,
) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
    }


def make_models_response(model_ids: list[str] = ["test-model"]) -> dict:
    return {"data": [{"id": m} for m in model_ids]}


def complete_transport(
    content: str = "Hello",
    status: int = 200,
    model: str = "test-model",
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=make_models_response())
        if status != 200:
            return httpx.Response(
                status,
                request=request,
                json={"error": "test error"},
            )
        return httpx.Response(200, json=make_complete_response(content=content, model=model))
    return httpx.MockTransport(handler)


def stream_transport(chunks: list[str]) -> httpx.MockTransport:
    """Build SSE-format streaming response."""
    def _make_sse(text: str | None, done: bool = False) -> str:
        if done:
            return "data: [DONE]\n\n"
        payload = json.dumps({
            "choices": [{"delta": {"content": text}, "finish_reason": None}]
        })
        return f"data: {payload}\n\n"

    lines = [_make_sse(c) for c in chunks] + [_make_sse(None, done=True)]
    body = "".join(lines)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.MockTransport(handler)


def make_client(transport: httpx.MockTransport) -> LMStudioClient:
    client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", model="test-model")
    client._client = httpx.AsyncClient(transport=transport)
    return client


# ── LLMResponse model ─────────────────────────────────────────────────────────

def test_llm_response_constructs() -> None:
    r = LLMResponse(content="hello", model="test-model")
    assert r.content == "hello"
    assert r.is_local is True
    assert r.cost_usd == 0.0


def test_llm_response_cloud_cost_nonzero() -> None:
    r = LLMResponse(
        content="hi", model="claude-sonnet-4-6",
        tokens_in=1000, tokens_out=500, is_local=False,
    )
    assert r.cost_usd > 0.0


def test_llm_response_raw_excluded_from_repr() -> None:
    r = LLMResponse(content="hi", model="m", raw={"big": "payload"})
    dumped = r.model_dump()
    assert "raw" not in dumped


# ── Message model ─────────────────────────────────────────────────────────────

def test_message_constructs() -> None:
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


# ── is_available ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_is_available_true_when_models_loaded() -> None:
    client = make_client(complete_transport())
    assert await client.is_available() is True
    await client.close()


@pytest.mark.asyncio
async def test_is_available_false_when_unreachable() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = make_client(httpx.MockTransport(failing))
    assert await client.is_available() is False
    await client.close()


@pytest.mark.asyncio
async def test_is_available_false_when_no_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = make_client(httpx.MockTransport(handler))
    assert await client.is_available() is False
    await client.close()


# ── complete ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_returns_llm_response() -> None:
    client = make_client(complete_transport("Test response"))
    messages = [Message(role="user", content="say hello")]
    response = await client.complete(messages)

    assert isinstance(response, LLMResponse)
    assert response.content == "Test response"
    assert response.model == "test-model"
    assert response.is_local is True
    await client.close()


@pytest.mark.asyncio
async def test_complete_records_token_counts() -> None:
    transport = complete_transport()
    # Patch the transport to include specific token counts
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=make_complete_response(tokens_in=150, tokens_out=75))

    client = make_client(httpx.MockTransport(handler))
    response = await client.complete([{"role": "user", "content": "test"}])
    assert response.tokens_in == 150
    assert response.tokens_out == 75
    await client.close()


@pytest.mark.asyncio
async def test_complete_records_latency() -> None:
    client = make_client(complete_transport())
    response = await client.complete([Message(role="user", content="hi")])
    assert response.latency_ms >= 0.0
    await client.close()


@pytest.mark.asyncio
async def test_complete_accepts_dict_messages() -> None:
    client = make_client(complete_transport("dict response"))
    response = await client.complete([{"role": "user", "content": "hello"}])
    assert response.content == "dict response"
    await client.close()


@pytest.mark.asyncio
async def test_complete_accepts_mixed_messages() -> None:
    client = make_client(complete_transport("mixed"))
    messages: list = [
        Message(role="system", content="you are a dev assistant"),
        {"role": "user", "content": "hello"},
    ]
    response = await client.complete(messages)
    assert response.content == "mixed"
    await client.close()


@pytest.mark.asyncio
async def test_complete_raises_on_4xx() -> None:
    client = make_client(complete_transport(status=400))
    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([Message(role="user", content="test")])
    await client.close()


# ── Retry logic ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_retries_on_503_then_succeeds() -> None:
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] < 3:
            return httpx.Response(503, json={"error": "unavailable"}, request=request)
        return httpx.Response(200, json=make_complete_response("retry success"))

    client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", model="test-model")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    # Patch sleep to avoid waiting in tests
    import unittest.mock as mock
    with mock.patch("agent.llm.lm_studio.asyncio.sleep", return_value=None):
        response = await client.complete([Message(role="user", content="test")])

    assert response.content == "retry success"
    assert call_count[0] == 3
    await client.close()


@pytest.mark.asyncio
async def test_complete_raises_after_max_retries() -> None:
    def always_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"}, request=request)

    client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", model="test-model")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(always_503))

    import unittest.mock as mock
    with mock.patch("agent.llm.lm_studio.asyncio.sleep", return_value=None):
        with pytest.raises(RuntimeError, match="failed after"):
            await client.complete([Message(role="user", content="test")])
    await client.close()


@pytest.mark.asyncio
async def test_complete_does_not_retry_on_400() -> None:
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(400, json={"error": "bad request"}, request=request)

    client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", model="test-model")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([Message(role="user", content="test")])

    assert call_count[0] == 1  # no retries
    await client.close()


# ── stream ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_yields_chunks() -> None:
    chunks = ["Hello", ", ", "world", "!"]
    client = make_client(stream_transport(chunks))

    received: list[str] = []
    async for chunk in client.stream([Message(role="user", content="hi")]):
        received.append(chunk)

    assert received == chunks
    await client.close()


@pytest.mark.asyncio
async def test_stream_full_text_reconstructed() -> None:
    chunks = ["The", " quick", " brown", " fox"]
    client = make_client(stream_transport(chunks))

    text = ""
    async for chunk in client.stream([Message(role="user", content="tell me something")]):
        text += chunk

    assert text == "The quick brown fox"
    await client.close()


@pytest.mark.asyncio
async def test_stream_empty_chunks_not_yielded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Include empty delta content lines
        lines = [
            'data: {"choices": [{"delta": {"content": ""}, "finish_reason": null}]}\n\n',
            'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(200, content="".join(lines).encode())

    client = make_client(httpx.MockTransport(handler))
    chunks: list[str] = []
    async for c in client.stream([Message(role="user", content="test")]):
        chunks.append(c)

    # Empty string chunk should not be yielded
    assert chunks == ["hi"]
    await client.close()


# ── Context manager ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_manager() -> None:
    async with LMStudioClient(
        base_url="http://127.0.0.1:1234/v1",
        model="test-model",
    ) as client:
        client._client = httpx.AsyncClient(transport=complete_transport("ctx ok"))
        response = await client.complete([Message(role="user", content="hi")])
        assert response.content == "ctx ok"


# ── Cost estimation ───────────────────────────────────────────────────────────

def test_cost_estimate_sonnet() -> None:
    cost = _estimate_cloud_cost("claude-sonnet-4-6", 1_000_000, 0)
    assert cost == pytest.approx(3.0)


def test_cost_estimate_haiku() -> None:
    cost = _estimate_cloud_cost("claude-haiku-4-5", 1_000_000, 0)
    assert cost == pytest.approx(0.25)


def test_cost_estimate_unknown_uses_sonnet_rate() -> None:
    cost_unknown = _estimate_cloud_cost("unknown-model", 1_000_000, 0)
    cost_sonnet  = _estimate_cloud_cost("claude-sonnet-4-6", 1_000_000, 0)
    assert cost_unknown == cost_sonnet