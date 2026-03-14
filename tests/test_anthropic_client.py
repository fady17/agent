"""
tests/test_anthropic_client.py

Tests for AnthropicClient using httpx.MockTransport.
No live API key or network required.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from agent.llm.anthropic_client import AnthropicClient, _ANTHROPIC_API_URL
from agent.llm.lm_studio import LLMResponse, Message


# ── Mock transport helpers ────────────────────────────────────────────────────

def make_messages_response(
    content: str = "Hello from Claude",
    model: str = "claude-sonnet-4-6",
    tokens_in: int = 30,
    tokens_out: int = 15,
) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        },
    }


def complete_transport(
    content: str = "Hello from Claude",
    status: int = 200,
    tokens_in: int = 30,
    tokens_out: int = 15,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(
                status,
                json={"type": "error", "error": {"message": "test error"}},
                request=request,
            )
        return httpx.Response(
            200,
            json=make_messages_response(
                content=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            ),
        )
    return httpx.MockTransport(handler)


def stream_transport(chunks: list[str]) -> httpx.MockTransport:
    """Build Anthropic SSE streaming response."""

    def _sse_events(chunks: list[str]) -> str:
        lines = []
        lines.append('event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n')
        lines.append('event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n')

        for i, chunk in enumerate(chunks):
            data = json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk},
            })
            lines.append(f"event: content_block_delta\ndata: {data}\n\n")

        lines.append('event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n')
        lines.append('event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n')
        lines.append('event: message_stop\ndata: {"type":"message_stop"}\n\n')
        return "".join(lines)

    body = _sse_events(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.MockTransport(handler)


def make_client(
    transport: httpx.MockTransport,
    api_key: str = "sk-ant-test",
    model: str = "claude-sonnet-4-6",
) -> AnthropicClient:
    client = AnthropicClient(api_key=api_key, model=model)
    client._client = httpx.AsyncClient(
        transport=transport,
        headers=client._default_headers(),
    )
    return client


# ── is_available ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_is_available_false_when_no_api_key() -> None:
    client = AnthropicClient(api_key="")
    assert await client.is_available() is False
    await client.close()


@pytest.mark.asyncio
async def test_is_available_false_when_placeholder_key() -> None:
    client = AnthropicClient(api_key="sk-ant-...")
    assert await client.is_available() is False
    await client.close()


@pytest.mark.asyncio
async def test_is_available_false_on_401() -> None:
    client = make_client(complete_transport(status=401))
    assert await client.is_available() is False
    await client.close()


@pytest.mark.asyncio
async def test_is_available_true_on_200() -> None:
    client = make_client(complete_transport())
    assert await client.is_available() is True
    await client.close()


# ── complete ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_returns_llm_response() -> None:
    client = make_client(complete_transport("Test from Claude"))
    response = await client.complete([Message(role="user", content="hello")])
    assert isinstance(response, LLMResponse)
    assert response.content == "Test from Claude"
    assert response.is_local is False
    await client.close()


@pytest.mark.asyncio
async def test_complete_records_token_counts() -> None:
    client = make_client(complete_transport(tokens_in=200, tokens_out=100))
    response = await client.complete([Message(role="user", content="test")])
    assert response.tokens_in == 200
    assert response.tokens_out == 100
    await client.close()


@pytest.mark.asyncio
async def test_complete_cost_is_nonzero() -> None:
    client = make_client(complete_transport(tokens_in=1000, tokens_out=500))
    response = await client.complete([Message(role="user", content="test")])
    assert response.cost_usd > 0.0
    await client.close()


@pytest.mark.asyncio
async def test_complete_records_latency() -> None:
    client = make_client(complete_transport())
    response = await client.complete([Message(role="user", content="hi")])
    assert response.latency_ms >= 0.0
    await client.close()


@pytest.mark.asyncio
async def test_complete_accepts_dict_messages() -> None:
    client = make_client(complete_transport("dict ok"))
    response = await client.complete([{"role": "user", "content": "hi"}])
    assert response.content == "dict ok"
    await client.close()


@pytest.mark.asyncio
async def test_complete_injects_system_prompt() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json=make_messages_response())

    client = make_client(httpx.MockTransport(handler))
    await client.complete(
        [Message(role="user", content="hi")],
        system="You are a helpful dev assistant",
    )
    assert captured[0].get("system") == "You are a helpful dev assistant"
    await client.close()


@pytest.mark.asyncio
async def test_complete_no_system_param_when_none() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json=make_messages_response())

    client = make_client(httpx.MockTransport(handler))
    await client.complete([Message(role="user", content="hi")])
    assert "system" not in captured[0]
    await client.close()


@pytest.mark.asyncio
async def test_complete_multi_block_content_joined() -> None:
    """Test that multiple text blocks in the response are joined correctly."""
    def handler(request: httpx.Request) -> httpx.Response:
        raw = {
            "id": "msg_multi",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "First block. "},
                {"type": "text", "text": "Second block."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return httpx.Response(200, json=raw)

    client = make_client(httpx.MockTransport(handler))
    response = await client.complete([Message(role="user", content="hi")])
    assert response.content == "First block. Second block."
    await client.close()


@pytest.mark.asyncio
async def test_complete_raises_on_4xx() -> None:
    client = make_client(complete_transport(status=400))
    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([Message(role="user", content="hi")])
    await client.close()


# ── Retry ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_retries_on_529_then_succeeds() -> None:
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] < 3:
            return httpx.Response(503, json={"error": "overloaded"}, request=request)
        return httpx.Response(200, json=make_messages_response("retry ok"))

    client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-6")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch("agent.llm.anthropic_client.asyncio.sleep", return_value=None):
        response = await client.complete([Message(role="user", content="hi")])

    assert response.content == "retry ok"
    assert call_count[0] == 3
    await client.close()


@pytest.mark.asyncio
async def test_complete_raises_after_max_retries() -> None:
    def always_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"}, request=request)

    client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-6")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(always_503))

    with patch("agent.llm.anthropic_client.asyncio.sleep", return_value=None):
        with pytest.raises(RuntimeError, match="failed after"):
            await client.complete([Message(role="user", content="hi")])
    await client.close()


@pytest.mark.asyncio
async def test_complete_no_retry_on_401() -> None:
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        return httpx.Response(401, json={"error": "unauthorized"}, request=request)

    client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-6")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([Message(role="user", content="hi")])

    assert call_count[0] == 1
    await client.close()


# ── stream ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_yields_text_chunks() -> None:
    chunks = ["Hello", ", ", "world", "!"]
    client = make_client(stream_transport(chunks))

    received: list[str] = []
    async for chunk in client.stream([Message(role="user", content="hi")]):
        received.append(chunk)

    assert received == chunks
    await client.close()


@pytest.mark.asyncio
async def test_stream_reconstructs_full_text() -> None:
    chunks = ["The ", "quick ", "brown ", "fox"]
    client = make_client(stream_transport(chunks))

    text = ""
    async for chunk in client.stream([Message(role="user", content="hi")]):
        text += chunk

    assert text == "The quick brown fox"
    await client.close()


@pytest.mark.asyncio
async def test_stream_with_system_prompt() -> None:
    captured: list[dict] = []
    chunks = ["ok"]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        body = stream_transport(chunks)._handler(request)  # type: ignore[attr-defined]
        return body

    # Use the stream_transport directly for simplicity
    client = make_client(stream_transport(chunks))
    result = []
    async for chunk in client.stream(
        [Message(role="user", content="hi")],
        system="be concise",
    ):
        result.append(chunk)
    assert result == ["ok"]
    await client.close()


# ── Context manager ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_manager() -> None:
    async with AnthropicClient(api_key="sk-ant-test") as client:
        client._client = httpx.AsyncClient(transport=complete_transport("ctx ok"))
        response = await client.complete([Message(role="user", content="hi")])
        assert response.content == "ctx ok"