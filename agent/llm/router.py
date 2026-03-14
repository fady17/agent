"""
agent/llm/router.py

LLM router — the single entry point for all inference in the agent.

The orchestrator and all sub-agents call the router exclusively.
Neither LMStudioClient nor AnthropicClient is imported anywhere else
in the application — routing is centralised here.

Routing decision (in priority order):
    1. Estimate input token count from messages
    2. If estimated tokens > cfg.llm_cloud_threshold_tokens  → cloud
    3. If LM Studio is unreachable (cached liveness)         → cloud
    4. If Anthropic API key is missing                       → local
    5. Default                                               → local

Liveness caching:
    LM Studio availability is checked once per session and cached.
    Re-checked automatically if the cached result is older than
    LIVENESS_TTL_S seconds. This avoids a health check on every call
    while still recovering if LM Studio comes back online mid-session.

Streaming:
    router.stream() returns an async generator — same as both clients.
    The caller never needs to know which backend is streaming.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.llm.anthropic_client import AnthropicClient
from agent.llm.lm_studio import LLMResponse, LMStudioClient, Message

log = get_logger(__name__)

# Re-check LM Studio liveness every 5 minutes
LIVENESS_TTL_S = 300.0

# Characters per token heuristic (same as context builder)
_CHARS_PER_TOKEN = 4


# ── RoutingDecision ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoutingDecision:
    """Records why a particular backend was chosen — logged for observability."""
    use_cloud: bool
    reason: str
    estimated_tokens: int


# ── LLMRouter ─────────────────────────────────────────────────────────────────

class LLMRouter:
    """
    Routes LLM requests between LM Studio (local) and Anthropic (cloud).

    Instantiate once per session. The router owns both client instances
    and is responsible for their lifecycle — call close() at shutdown.

    Usage:
        router = LLMRouter()
        response = await router.complete(messages)
        async for chunk in router.stream(messages):
            print(chunk, end="", flush=True)
        await router.close()
    """

    def __init__(
        self,
        local_client: LMStudioClient | None = None,
        cloud_client: AnthropicClient | None = None,
    ) -> None:
        self._local = local_client or LMStudioClient()
        self._cloud = cloud_client or AnthropicClient()

        # Liveness cache
        self._local_available: bool | None = None
        self._liveness_checked_at: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        force_local: bool = False,
        force_cloud: bool = False,
    ) -> LLMResponse:
        """
        Route and execute a non-streaming completion.

        Args:
            messages:     Conversation history.
            system:       Optional system prompt (passed to Anthropic;
                          prepended as system message for LM Studio).
            temperature:  Sampling temperature.
            max_tokens:   Maximum tokens to generate.
            force_local:  Skip routing logic, always use LM Studio.
            force_cloud:  Skip routing logic, always use Anthropic.

        Returns:
            LLMResponse from whichever backend handled the request.
        """
        decision = await self._decide(
            messages,
            force_local=force_local,
            force_cloud=force_cloud,
        )

        log.info(
            "router.route",
            backend="cloud" if decision.use_cloud else "local",
            reason=decision.reason,
            estimated_tokens=decision.estimated_tokens,
        )

        if decision.use_cloud:
            return await self._cloud.complete(
                messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            full_messages = self._inject_system(messages, system)
            return await self._local.complete(
                full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def stream(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        force_local: bool = False,
        force_cloud: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Route and execute a streaming completion.
        Returns an async generator of text chunks.
        """
        decision = await self._decide(
            messages,
            force_local=force_local,
            force_cloud=force_cloud,
        )

        log.info(
            "router.stream_route",
            backend="cloud" if decision.use_cloud else "local",
            reason=decision.reason,
        )

        if decision.use_cloud:
            return self._cloud.stream(
                messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            full_messages = self._inject_system(messages, system)
            return self._local.stream(
                full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def close(self) -> None:
        await self._local.close()
        await self._cloud.close()

    async def __aenter__(self) -> "LLMRouter":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── Routing logic ─────────────────────────────────────────────────────────

    async def _decide(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        force_local: bool,
        force_cloud: bool,
    ) -> RoutingDecision:
        """Evaluate routing rules and return a RoutingDecision."""

        estimated = _estimate_tokens(messages)

        # Force overrides — bypass all logic
        if force_local:
            return RoutingDecision(
                use_cloud=False,
                reason="force_local",
                estimated_tokens=estimated,
            )
        if force_cloud:
            return RoutingDecision(
                use_cloud=True,
                reason="force_cloud",
                estimated_tokens=estimated,
            )

        # Rule 1: token threshold — large context goes to cloud
        if estimated > cfg.llm_cloud_threshold_tokens:
            return RoutingDecision(
                use_cloud=True,
                reason=f"token_threshold ({estimated} > {cfg.llm_cloud_threshold_tokens})",
                estimated_tokens=estimated,
            )

        # Rule 2: LM Studio liveness
        local_up = await self._check_local_liveness()
        if not local_up:
            # Rule 3: no API key — can't fall back to cloud
            if not cfg.anthropic_api_key or cfg.anthropic_api_key.startswith("sk-ant-..."):
                log.warning(
                    "router.no_fallback",
                    detail="LM Studio down and no Anthropic API key configured",
                )
                # Best effort — try local anyway (will likely fail, but surfaces the error)
                return RoutingDecision(
                    use_cloud=False,
                    reason="local_down_no_api_key",
                    estimated_tokens=estimated,
                )
            return RoutingDecision(
                use_cloud=True,
                reason="local_unavailable",
                estimated_tokens=estimated,
            )

        # Default: use local
        return RoutingDecision(
            use_cloud=False,
            reason="default_local",
            estimated_tokens=estimated,
        )

    async def _check_local_liveness(self) -> bool:
        """
        Return cached LM Studio availability, refreshing if TTL has expired.
        """
        now = time.monotonic()
        if (
            self._local_available is None
            or (now - self._liveness_checked_at) > LIVENESS_TTL_S
        ):
            self._local_available = await self._local.is_available()
            self._liveness_checked_at = now
            log.debug(
                "router.liveness_checked",
                local_available=self._local_available,
            )
        return self._local_available  # type: ignore[return-value]

    def invalidate_liveness_cache(self) -> None:
        """Force a liveness re-check on the next call. Useful after errors."""
        self._local_available = None
        self._liveness_checked_at = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_system(
        messages: list[Message] | list[dict[str, str]],
        system: str | None,
    ) -> list[dict[str, str]]:
        """
        Prepend a system message to the list for LM Studio.
        LM Studio (OpenAI-compatible) accepts system role in the messages array.
        """
        normalised = [
            m if isinstance(m, dict) else m.model_dump()
            for m in messages
        ]
        if not system:
            return normalised
        # Only prepend if there isn't already a system message
        if normalised and normalised[0].get("role") == "system":
            return normalised
        return [{"role": "system", "content": system}] + normalised


# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(messages: list[Message] | list[dict[str, str]]) -> int:
    """
    Estimate total token count for a message list.
    Uses the 4 chars/token heuristic — conservative but consistent.
    """
    total_chars = sum(
        len(m.content if isinstance(m, Message) else m.get("content", ""))
        for m in messages
    )
    return max(1, total_chars // _CHARS_PER_TOKEN)