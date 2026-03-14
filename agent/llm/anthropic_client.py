"""
agent/llm/anthropic_client.py

Async client for the Anthropic Messages API.

Same public interface as LMStudioClient — both return LLMResponse and
both expose a stream() async generator. The router never needs to know
which client it is talking to.

Key differences from LM Studio:
    - Auth:     x-api-key header (from cfg.anthropic_api_key)
    - Shape:    request uses "messages" + optional "system" param (not in messages list)
                response content is a list of blocks, not a plain string
    - Cost:     every call costs money — tokens tracked and cost estimated
    - Streaming: Anthropic SSE uses event types (content_block_delta, message_delta)
                 rather than raw delta objects

Anthropic API reference:
    POST https://api.anthropic.com/v1/messages
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.llm.lm_studio import LLMResponse, Message

log = get_logger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

_MAX_RETRIES   = 3
_RETRY_BASE_S  = 1.0
_RETRY_BACKOFF = 2.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ── AnthropicClient ───────────────────────────────────────────────────────────

class AnthropicClient:
    """
    Async client for the Anthropic Messages API.

    Usage:
        client = AnthropicClient()
        response = await client.complete(messages)

        async for chunk in client.stream(messages):
            print(chunk, end="", flush=True)

        await client.close()
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key or cfg.anthropic_api_key
        self._model   = model  or cfg.anthropic_model
        self._client  = httpx.AsyncClient(
            timeout=timeout,
            headers=self._default_headers(),
        )

    # ── Health check ──────────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        """
        Check if the Anthropic API is reachable and the key is valid.
        Uses a minimal prompt to minimise cost (still costs a few tokens).
        Returns False if the key is missing or empty.
        """
        if not self._api_key or self._api_key.startswith("sk-ant-..."):
            log.debug("anthropic.no_api_key")
            return False
        try:
            r = await self._client.post(
                _ANTHROPIC_API_URL,
                json=self._build_payload(
                    [Message(role="user", content="hi")],
                    system=None,
                    temperature=0.0,
                    max_tokens=1,
                    stream=False,
                ),
                timeout=5.0,
            )
            available = r.status_code not in (401, 403)
            log.debug("anthropic.available", status=r.status_code)
            return available
        except Exception as exc:
            log.debug("anthropic.unavailable", error=str(exc))
            return False

    # ── Non-streaming completion ──────────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Send a Messages API request and return the full response.

        Args:
            messages:    Conversation history (user/assistant turns only —
                         system prompt goes in the `system` parameter).
            system:      Optional system prompt injected at the top.
            temperature: Sampling temperature 0.0–1.0.
            max_tokens:  Maximum tokens to generate.

        Returns:
            LLMResponse with content, token counts, latency, and cost.

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx errors.
            RuntimeError: If all retries are exhausted.
        """
        payload = self._build_payload(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        t0 = time.perf_counter()
        raw = await self._post_with_retry(payload)
        latency_ms = (time.perf_counter() - t0) * 1000

        content = self._extract_content(raw)
        usage   = raw.get("usage", {})
        tokens_in  = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        model      = raw.get("model", self._model)

        response = LLMResponse(
            content=content,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=round(latency_ms, 1),
            is_local=False,
            raw=raw,
        )

        log.info(
            "anthropic.complete",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(response.cost_usd, 6),
            latency_ms=response.latency_ms,
        )

        return response

    # ── Streaming completion ──────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a Messages API response, yielding text chunks as they arrive.

        Anthropic SSE event sequence:
            message_start → content_block_start → content_block_delta (×N)
            → content_block_stop → message_delta → message_stop
        We yield only the text from content_block_delta events.
        """
        payload = self._build_payload(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        log.debug("anthropic.stream_start", model=self._model)

        async with self._client.stream(
            "POST",
            _ANTHROPIC_API_URL,
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()

                # SSE format: "event: <type>" followed by "data: <json>"
                if line.startswith("data: "):
                    data_str = line[len("data: "):]
                    try:
                        event_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # content_block_delta carries the actual text chunks
                    if event_data.get("type") == "content_block_delta":
                        delta = event_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text

        log.debug("anthropic.stream_done")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _default_headers(self) -> dict[str, str]:
        return {
            "x-api-key":         self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type":      "application/json",
        }

    def _build_payload(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        normalised = [
            m if isinstance(m, dict) else m.model_dump()
            for m in messages
        ]
        payload: dict[str, Any] = {
            "model":       self._model,
            "messages":    normalised,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      stream,
        }
        if system:
            payload["system"] = system
        return payload

    @staticmethod
    def _extract_content(raw: dict[str, Any]) -> str:
        """
        Extract text from Anthropic's content block list.
        Response shape: {"content": [{"type": "text", "text": "..."}]}
        """
        blocks = raw.get("content", [])
        parts = [
            block.get("text", "")
            for block in blocks
            if block.get("type") == "text"
        ]
        return "".join(parts)

    async def _post_with_retry(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                r = await self._client.post(_ANTHROPIC_API_URL, json=payload)

                if r.status_code in _RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"Retryable status {r.status_code}",
                        request=r.request,
                        response=r,
                    )

                r.raise_for_status()
                return r.json()

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    raise
                last_exc = exc

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc

            if attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_S * (_RETRY_BACKOFF ** (attempt - 1))
                log.warning(
                    "anthropic.retry",
                    attempt=attempt,
                    wait_s=wait,
                    error=str(last_exc),
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Anthropic request failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AnthropicClient":  # noqa: UP037
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()