# type: ignore
"""
tests/test_router.py

Tests for LLMRouter routing logic.
Both LMStudioClient and AnthropicClient are replaced with mocks
so no network calls are made.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm.lm_studio import LLMResponse, Message
from agent.llm.router import LIVENESS_TTL_S, LLMRouter, RoutingDecision, _estimate_tokens


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def make_response(content: str = "ok", is_local: bool = True) -> LLMResponse:
    return LLMResponse(content=content, model="test-model", is_local=is_local)


def make_local_client(
    available: bool = True,
    response: str = "local response",
) -> MagicMock:
    client = MagicMock()
    client.is_available = AsyncMock(return_value=available)
    client.complete = AsyncMock(return_value=make_response(response, is_local=True))

    async def _stream(*args, **kwargs) -> AsyncGenerator[str, None]:
        yield "local "
        yield "stream"

    client.stream = MagicMock(return_value=_stream())
    client.close = AsyncMock()
    return client


def make_cloud_client(
    response: str = "cloud response",
) -> MagicMock:
    client = MagicMock()
    client.complete = AsyncMock(return_value=make_response(response, is_local=False))

    async def _stream(*args, **kwargs) -> AsyncGenerator[str, None]:
        yield "cloud "
        yield "stream"

    client.stream = MagicMock(return_value=_stream())
    client.close = AsyncMock()
    return client


def make_router(
    local_available: bool = True,
    local_response: str = "local response",
    cloud_response: str = "cloud response",
) -> LLMRouter:
    router = LLMRouter(
        local_client=make_local_client(local_available, local_response),
        cloud_client=make_cloud_client(cloud_response),
    )
    return router


SHORT_MESSAGES = [Message(role="user", content="hello")]
LONG_CONTENT   = "word " * 3000   # ~15000 chars → ~3750 tokens


# ── _estimate_tokens ──────────────────────────────────────────────────────────

def test_estimate_tokens_message_objects() -> None:
    msgs = [Message(role="user", content="abcd")]  # 4 chars → 1 token
    assert _estimate_tokens(msgs) == 1


def test_estimate_tokens_dict_messages() -> None:
    msgs = [{"role": "user", "content": "abcdefgh"}]  # 8 chars → 2 tokens
    assert _estimate_tokens(msgs) == 2


def test_estimate_tokens_multiple_messages() -> None:
    msgs = [
        Message(role="system",    content="a" * 40),
        Message(role="user",      content="a" * 40),
        Message(role="assistant", content="a" * 40),
    ]
    # 120 chars / 4 = 30 tokens
    assert _estimate_tokens(msgs) == 30


def test_estimate_tokens_minimum_one() -> None:
    msgs = [Message(role="user", content="")]
    assert _estimate_tokens(msgs) >= 1


# ── RoutingDecision ───────────────────────────────────────────────────────────

def test_routing_decision_is_frozen() -> None:
    d = RoutingDecision(use_cloud=True, reason="test", estimated_tokens=10)
    with pytest.raises(Exception):
        d.use_cloud = False  # type: ignore[misc]


# ── Default routing (local) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_default_routes_to_local() -> None:
    router = make_router(local_available=True)
    
    # Just run the completion
    response = await router.complete(SHORT_MESSAGES)
    
    # Assertions
    assert response.is_local is True
    router._local.complete.assert_called_once() 
    
    # Verify cloud was NOT called
    router._cloud.complete.assert_not_called()
    
    await router.close()
@pytest.mark.asyncio
async def test_default_response_content() -> None:
    router = make_router(local_response="from local")
    response = await router.complete(SHORT_MESSAGES)
    assert response.content == "from local"
    await router.close()


# ── Token threshold routing ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_routes_to_cloud_above_token_threshold() -> None:
    router = make_router(local_available=True)
    long_messages = [Message(role="user", content=LONG_CONTENT)]

    with patch("agent.llm.router.cfg") as mock_cfg:
        mock_cfg.llm_cloud_threshold_tokens = 100
        mock_cfg.anthropic_api_key = "sk-ant-real-key"
        response = await router.complete(long_messages)

    assert response.is_local is False
    router._cloud.complete.assert_called_once()
    await router.close()


@pytest.mark.asyncio
async def test_stays_local_below_token_threshold() -> None:
    router = make_router(local_available=True)
    with patch("agent.llm.router.cfg") as mock_cfg:
        mock_cfg.llm_cloud_threshold_tokens = 99999
        mock_cfg.anthropic_api_key = "sk-ant-real-key"
        response = await router.complete(SHORT_MESSAGES)

    assert response.is_local is True
    await router.close()


# ── Liveness-based routing ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_routes_to_cloud_when_local_unavailable() -> None:
    router = make_router(local_available=False)

    with patch("agent.llm.router.cfg") as mock_cfg:
        mock_cfg.llm_cloud_threshold_tokens = 99999
        mock_cfg.anthropic_api_key = "sk-ant-real-key"
        response = await router.complete(SHORT_MESSAGES)

    assert response.is_local is False
    router._cloud.complete.assert_called_once()
    await router.close()


@pytest.mark.asyncio
async def test_stays_local_when_no_api_key_and_local_down() -> None:
    """No API key + local down = try local anyway (best effort)."""
    router = make_router(local_available=False)

    with patch("agent.llm.router.cfg") as mock_cfg:
        mock_cfg.llm_cloud_threshold_tokens = 99999
        mock_cfg.anthropic_api_key = ""
        response = await router.complete(SHORT_MESSAGES)

    # Routed to local despite being down (best effort)
    assert response.is_local is True
    await router.close()


# ── Force overrides ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_force_local_overrides_threshold() -> None:
    router = make_router(local_available=True)
    long_messages = [Message(role="user", content=LONG_CONTENT)]

    with patch("agent.llm.router.cfg") as mock_cfg:
        mock_cfg.llm_cloud_threshold_tokens = 1
        mock_cfg.anthropic_api_key = "sk-ant-real"
        response = await router.complete(long_messages, force_local=True)

    assert response.is_local is True
    router._local.complete.assert_called_once()
    await router.close()


@pytest.mark.asyncio
async def test_force_cloud_overrides_local_availability() -> None:
    router = make_router(local_available=True)
    response = await router.complete(SHORT_MESSAGES, force_cloud=True)
    assert response.is_local is False
    router._cloud.complete.assert_called_once()
    await router.close()


@pytest.mark.asyncio
async def test_force_local_skips_liveness_check() -> None:
    router = make_router(local_available=False)
    # force_local should not call is_available at all
    await router.complete(SHORT_MESSAGES, force_local=True)
    router._local.is_available.assert_not_called()
    await router.close()


# ── System prompt injection ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_prompt_passed_to_cloud() -> None:
    router = make_router()
    await router.complete(SHORT_MESSAGES, system="be concise", force_cloud=True)
    _, kwargs = router._cloud.complete.call_args
    assert kwargs.get("system") == "be concise"
    await router.close()


@pytest.mark.asyncio
async def test_system_prompt_prepended_for_local() -> None:
    router = make_router(local_available=True)
    await router.complete(SHORT_MESSAGES, system="you are a dev assistant")
    args, _ = router._local.complete.call_args
    messages_sent = args[0]
    assert messages_sent[0]["role"] == "system"
    assert messages_sent[0]["content"] == "you are a dev assistant"
    await router.close()


@pytest.mark.asyncio
async def test_system_not_duplicated_if_already_present() -> None:
    router = make_router(local_available=True)
    messages_with_system = [
        {"role": "system", "content": "existing system"},
        {"role": "user", "content": "hello"},
    ]
    await router.complete(messages_with_system, system="new system")
    args, _ = router._local.complete.call_args
    sent = args[0]
    system_msgs = [m for m in sent if m.get("role") == "system"]
    assert len(system_msgs) == 1  # not duplicated
    await router.close()


# ── Liveness cache ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_liveness_cached_across_calls() -> None:
    router = make_router(local_available=True)
    await router.complete(SHORT_MESSAGES)
    await router.complete(SHORT_MESSAGES)
    await router.complete(SHORT_MESSAGES)
    # is_available should only be called once (cached after first check)
    assert router._local.is_available.call_count == 1
    await router.close()


@pytest.mark.asyncio
async def test_invalidate_liveness_forces_recheck() -> None:
    router = make_router(local_available=True)
    await router.complete(SHORT_MESSAGES)
    router.invalidate_liveness_cache()
    await router.complete(SHORT_MESSAGES)
    assert router._local.is_available.call_count == 2
    await router.close()


@pytest.mark.asyncio
async def test_liveness_rechecked_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    import time as time_module
    router = make_router(local_available=True)

    # First call — populates cache
    await router.complete(SHORT_MESSAGES)
    assert router._local.is_available.call_count == 1

    # Simulate TTL expiry by backdating the check time
    router._liveness_checked_at -= LIVENESS_TTL_S + 1

    # Second call — should re-check
    await router.complete(SHORT_MESSAGES)
    assert router._local.is_available.call_count == 2
    await router.close()


# ── Streaming ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_routes_to_local_by_default() -> None:
    router = make_router(local_available=True)
    gen = await router.stream(SHORT_MESSAGES)
    chunks = [c async for c in gen]
    assert "".join(chunks) == "local stream"
    router._local.stream.assert_called_once()
    await router.close()


@pytest.mark.asyncio
async def test_stream_routes_to_cloud_when_forced() -> None:
    router = make_router(local_available=True)
    gen = await router.stream(SHORT_MESSAGES, force_cloud=True)
    chunks = [c async for c in gen]
    assert "".join(chunks) == "cloud stream"
    router._cloud.stream.assert_called_once()
    await router.close()


# ── Context manager ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_manager_closes_both_clients() -> None:
    local  = make_local_client()
    cloud  = make_cloud_client()
    router = LLMRouter(local_client=local, cloud_client=cloud)

    async with router:
        pass

    local.close.assert_called_once()
    cloud.close.assert_called_once()