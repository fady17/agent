"""
agent/llm/lm_studio.py

Async client for LM Studio's OpenAI-compatible chat completions API.

Endpoints used:
    POST /v1/chat/completions   — chat inference, streaming + non-streaming
    GET  /v1/models             — health check / model listing

Design:
    - All public methods are async coroutines — no blocking I/O
    - Streaming returns an async generator of text chunks
    - Non-streaming returns a complete LLMResponse
    - Retries on transient errors (5xx, timeout) with exponential backoff
    - Never retries on 4xx — those are caller errors

LLMResponse is the shared response model used by both LM Studio and
the Anthropic client so the router and orchestrator never need to
know which backend produced the response.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)

# ── Shared response model ─────────────────────────────────────────────────────

class Message(BaseModel):
    """A single chat message."""
    role: str                  # "system" | "user" | "assistant"
    content: str


class LLMResponse(BaseModel):
    """
    Normalised response from any LLM backend.
    Both LMStudioClient and AnthropicClient return this type.
    """
    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    is_local: bool = True
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def cost_usd(self) -> float:
        """Local inference is always free."""
        return 0.0 if self.is_local else _estimate_cloud_cost(
            self.model, self.tokens_in, self.tokens_out
        )


def _estimate_cloud_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """
    Rough cost estimate for Anthropic models.
    Prices in USD per million tokens (as of early 2026 — update as needed).
    """
    # claude-sonnet-4-6 pricing
    if "sonnet" in model:
        return (tokens_in * 3.0 + tokens_out * 15.0) / 1_000_000
    # claude-haiku fallback
    if "haiku" in model:
        return (tokens_in * 0.25 + tokens_out * 1.25) / 1_000_000
    # Unknown model — use sonnet rate as conservative estimate
    return (tokens_in * 3.0 + tokens_out * 15.0) / 1_000_000


# ── Retry config ──────────────────────────────────────────────────────────────

_MAX_RETRIES    = 3
_RETRY_BASE_S   = 1.0   # first retry after 1s
_RETRY_BACKOFF  = 2.0   # multiply by 2 each retry: 1s, 2s, 4s
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ── LMStudioClient ────────────────────────────────────────────────────────────

class LMStudioClient:
    """
    Async client for LM Studio's OpenAI-compatible API.

    Instantiate once per session and reuse — the underlying httpx.AsyncClient
    maintains a connection pool.

    Usage:
        client = LMStudioClient()
        # Non-streaming
        response = await client.complete(messages)
        # Streaming
        async for chunk in client.stream(messages):
            print(chunk, end="", flush=True)
        await client.close()
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = (base_url or cfg.lm_studio_base_url).rstrip("/")
        self._model    = model or cfg.lm_studio_chat_model
        self._client   = httpx.AsyncClient(timeout=timeout)

    # ── Health check ──────────────────────────────────────────────────────────

    async def is_available(self) -> bool:
        """
        Quick liveness check — returns True if LM Studio is reachable
        and at least one model is loaded.
        """
        try:
            r = await self._client.get(
                f"{self._base_url}/models",
                timeout=3.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            available = len(models) > 0
            log.debug(
                "lm_studio.available",
                models=[m.get("id") for m in models],
            )
            return available
        except Exception as exc:
            log.debug("lm_studio.unavailable", error=str(exc))
            return False

    # ── Non-streaming completion ──────────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """
        Send a chat completion request and return the full response.

        Args:
            messages:    Conversation history. Accepts Message objects or
                         plain dicts with "role" and "content" keys.
            temperature: Sampling temperature 0.0–1.0.
            max_tokens:  Maximum tokens to generate.

        Returns:
            LLMResponse with content, token counts, and latency.

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx errors.
            RuntimeError: If all retries are exhausted.
        """
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        t0 = time.perf_counter()
        raw = await self._post_with_retry("/chat/completions", payload)
        latency_ms = (time.perf_counter() - t0) * 1000

        content = raw["choices"][0]["message"]["content"]
        usage   = raw.get("usage", {})

        response = LLMResponse(
            content=content,
            model=raw.get("model", self._model),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=round(latency_ms, 1),
            is_local=True,
            raw=raw,
        )

        log.info(
            "lm_studio.complete",
            model=response.model,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            latency_ms=response.latency_ms,
        )

        return response

    # ── Streaming completion ──────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion, yielding text chunks as they arrive.

        Usage:
            async for chunk in client.stream(messages):
                print(chunk, end="", flush=True)

        Note: Streaming does not retry on error — the response has already
        started by the time an error would occur mid-stream.
        """
        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        log.debug("lm_studio.stream_start", model=self._model)

        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data = line[len("data: "):]
                if data == "[DONE]":
                    break

                try:
                    import json
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    text  = delta.get("content", "")
                    if text:
                        yield text
                except Exception as exc:
                    log.warning("lm_studio.stream_parse_error", error=str(exc))
                    continue

        log.debug("lm_studio.stream_done")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        normalised = [
            m if isinstance(m, dict) else m.model_dump()
            for m in messages
        ]
        return {
            "model":       self._model,
            "messages":    normalised,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      stream,
        }

    async def _post_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        POST to the given path with exponential backoff retry.
        Only retries on transient server errors — never on 4xx.
        """
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                r = await self._client.post(url, json=payload)

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
                    raise  # 4xx — don't retry
                last_exc = exc

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc

            if attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_S * (_RETRY_BACKOFF ** (attempt - 1))
                log.warning(
                    "lm_studio.retry",
                    attempt=attempt,
                    wait_s=wait,
                    error=str(last_exc),
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"LM Studio request failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LMStudioClient":  # noqa: UP037
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()