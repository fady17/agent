"""
agent/core/orchestrator.py

Orchestrator — the agent's central nervous system.

Every user interaction flows through run_turn(). One turn:

    1. Classify intent        → TaskIntent (domain, action, project, urgency)
    2. Retrieve memory        → ContextBlock (events, nodes, skills)
    3. Build context          → ContextPayload (token-budgeted string)
    4. Build prompt           → list[Message] via PromptEngine
    5. Call LLM               → LLMResponse via LLMRouter
    6. Validate response      → ValidationResult via ResponseValidator
    7. Write episodic event   → persist to ~/.agent/memory/episodic/
    8. Update session state   → SessionManager.update()
    9. Return TurnResult      → structured response for the interface layer

The orchestrator owns no business logic — it delegates everything.
Adding a new capability = adding a sub-agent, not touching this file.

Streaming:
    run_turn_stream() is the streaming variant. It yields text chunks
    from the LLM and finalises state after the stream completes.
    The interface layer calls one or the other — never both.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator

from agent.core.classifier import Domain, TaskClassifier, TaskIntent
from agent.core.config import cfg
from agent.core.logger import get_logger
from agent.core.session import SessionManager, SessionState
from agent.llm.prompt_engine import PromptEngine, TemplateID
from agent.llm.router import LLMRouter
from agent.llm.validator import ResponseValidator
from agent.memory.context_builder import ContextBuilder
from agent.memory.embedder import Embedder
from agent.memory.episodic import EpisodicEvent, EventType, write_event
from agent.memory.graph import SemanticGraph
from agent.memory.index import FaissIndex
from agent.memory.retrieval import RetrievalPipeline

log = get_logger(__name__)


# ── TurnResult ────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """
    Structured output of one orchestrator turn.

    Returned to the interface layer (CLI, web UI, hotkey daemon).
    The interface never touches raw LLM responses — it only sees TurnResult.
    """

    content: str                   # The assistant's response text
    intent: TaskIntent             # What the classifier decided
    session: SessionState          # Session state after this turn
    turn_number: int               # Which turn this is (1-indexed)
    latency_ms: float = 0.0        # Total wall time for this turn
    tokens_used: int = 0           # LLM tokens consumed
    cost_usd: float = 0.0          # Cost (0.0 for local calls)
    was_truncated: bool = False    # Context was truncated due to token budget
    error: str | None = None       # Set if the turn completed with a soft error

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Wires all agent components into a single run_turn() interface.

    Instantiate once per session. All components are injected —
    the orchestrator never constructs anything internally.

    Usage:
        orchestrator = Orchestrator.create()   # convenience factory
        result = await orchestrator.run_turn("Fix the CORS error")
        print(result.content)
    """

    def __init__(
        self,
        *,
        router: LLMRouter,
        classifier: TaskClassifier,
        retrieval: RetrievalPipeline,
        context_builder: ContextBuilder,
        prompt_engine: PromptEngine,
        validator: ResponseValidator,
        session_manager: SessionManager,
    ) -> None:
        self._router          = router
        self._classifier      = classifier
        self._retrieval       = retrieval
        self._context_builder = context_builder
        self._prompt_engine   = prompt_engine
        self._validator       = validator
        self._session         = session_manager

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        graph: SemanticGraph | None = None,
        index: FaissIndex | None = None,
        embedder: Embedder | None = None,
        session_path=None,
    ) -> "Orchestrator":
        """
        Convenience factory — constructs all dependencies with defaults.
        Override any component by passing it explicitly.
        """
        from agent.memory.embedder import get_embedder as _get_embedder

        # Components with no external deps
        router          = LLMRouter()
        prompt_engine   = PromptEngine()
        validator       = ResponseValidator(max_retries=3)
        context_builder = ContextBuilder(max_tokens=1500)
        session_manager = SessionManager(state_path=session_path)

        # Memory components
        _graph   = graph   or SemanticGraph()
        _index   = index   or FaissIndex()
        # Embedder is async — caller must await get_embedder() separately
        # and pass it in, or use create_async() below.
        _embedder = embedder or Embedder()

        retrieval  = RetrievalPipeline(_graph, _index, _embedder)
        classifier = TaskClassifier(router, prompt_engine, validator)

        return cls(
            router=router,
            classifier=classifier,
            retrieval=retrieval,
            context_builder=context_builder,
            prompt_engine=prompt_engine,
            validator=validator,
            session_manager=session_manager,
        )

    # ── Main turn ─────────────────────────────────────────────────────────────

    async def run_turn(
        self,
        user_input: str,
        *,
        conversation_history=None,
        force_local: bool = False,
        force_cloud: bool = False,
    ) -> TurnResult:
        """
        Execute one complete agent turn and return a TurnResult.

        Args:
            user_input:            Raw text from the user.
            conversation_history:  Prior Message turns (for multi-turn context).
            force_local:           Override routing to always use local LLM.
            force_cloud:           Override routing to always use cloud LLM.

        Returns:
            TurnResult — always returns, never raises.
            On hard failure, TurnResult.error is set and content contains
            a user-facing error message.
        """
        import time
        t0 = time.perf_counter()

        # Ensure session is initialised
        if not self._session.is_loaded:
            self._session.new_session()

        session = self._session.state

        try:
            result = await self._execute_turn(
                user_input=user_input,
                session=session,
                conversation_history=conversation_history or [],
                force_local=force_local,
                force_cloud=force_cloud,
            )
        except Exception as exc:
            log.error("orchestrator.turn_failed", error=str(exc), turn=session.turn_count + 1)
            # Soft failure — return an error result rather than propagating
            elapsed = (time.perf_counter() - t0) * 1000
            updated = self._session.update(session.with_turn())
            return TurnResult(
                content="I encountered an error processing your request. Please try again.",
                intent=TaskIntent(domain=Domain.UNKNOWN, action="unknown"),
                session=updated,
                turn_number=updated.turn_count,
                latency_ms=round(elapsed, 1),
                error=str(exc),
            )

        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info(
            "orchestrator.turn_complete",
            turn=result.turn_number,
            domain=result.intent.domain,
            latency_ms=result.latency_ms,
            tokens=result.tokens_used,
            cost=result.cost_usd,
        )
        return result

    async def run_turn_stream(
        self,
        user_input: str,
        *,
        conversation_history=None,
        force_local: bool = False,
        force_cloud: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant of run_turn.

        Yields text chunks as they arrive from the LLM.
        Writes the episodic event and updates session after the stream ends.

        Usage:
            async for chunk in orchestrator.run_turn_stream("explain this"):
                print(chunk, end="", flush=True)
        """
        if not self._session.is_loaded:
            self._session.new_session()

        session = self._session.state

        # Steps 1-4 are identical to run_turn
        intent = await self._classifier.classify(user_input)
        context_block = await self._retrieval.retrieve(
            user_input,
            project=intent.project,
            task_type=intent.action,
        )
        context_payload = self._context_builder.build(context_block)
        messages = self._prompt_engine.build(
            TemplateID.RESPOND,
            variables={"input": user_input},
            context=context_payload,
            conversation_history=conversation_history,
        )

        full_content: list[str] = []

        stream = await self._router.stream(
            messages,
            force_local=force_local,
            force_cloud=force_cloud,
        )

        async for chunk in stream:
            full_content.append(chunk)
            yield chunk

        # Post-stream finalisation
        assembled = "".join(full_content)
        await self._finalise_turn(
            user_input=user_input,
            content=assembled,
            intent=intent,
            session=session,
            tokens_in=0,   # streaming doesn't return token counts
            tokens_out=0,
            cost_usd=0.0,
            is_local=not force_cloud,
            was_truncated=context_payload.was_truncated,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _execute_turn(
        self,
        *,
        user_input: str,
        session: SessionState,
        conversation_history,
        force_local: bool,
        force_cloud: bool,
    ) -> TurnResult:
        """Core turn logic — separated so run_turn can wrap it in try/except."""

        # ── Step 1: Classify ──────────────────────────────────────────────────
        intent = await self._classifier.classify(user_input)

        # Update session project if classifier found one
        if intent.project and intent.project != session.active_project:
            session = self._session.update(session.with_project(intent.project))

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        context_block = await self._retrieval.retrieve(
            user_input,
            project=intent.project or session.active_project,
            task_type=intent.action,
        )

        # ── Step 3: Build context ─────────────────────────────────────────────
        context_payload = self._context_builder.build(context_block)

        # ── Step 4: Build prompt ──────────────────────────────────────────────
        messages = self._prompt_engine.build(
            TemplateID.RESPOND,
            variables={"input": user_input},
            context=context_payload,
            conversation_history=conversation_history,
        )

        # ── Step 5: Call LLM ──────────────────────────────────────────────────
        llm_response = await self._router.complete(
            messages,
            force_local=force_local,
            force_cloud=force_cloud,
        )

        # ── Step 6: Validate ──────────────────────────────────────────────────
        validation = await self._validator.validate_text(
            llm_response,
            min_length=1,
        )

        content = validation.value if validation.success else llm_response.content

        # ── Steps 7-8: Persist + update session ───────────────────────────────
        updated_session = await self._finalise_turn(
            user_input=user_input,
            content=content,
            intent=intent,
            session=session,
            tokens_in=llm_response.tokens_in,
            tokens_out=llm_response.tokens_out,
            cost_usd=llm_response.cost_usd,
            is_local=llm_response.is_local,
            was_truncated=context_payload.was_truncated,
        )

        return TurnResult(
            content=content,
            intent=intent,
            session=updated_session,
            turn_number=updated_session.turn_count,
            tokens_used=llm_response.tokens_in + llm_response.tokens_out,
            cost_usd=llm_response.cost_usd,
            was_truncated=context_payload.was_truncated,
        )

    async def _finalise_turn(
        self,
        *,
        user_input: str,
        content: str,
        intent: TaskIntent,
        session: SessionState,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        is_local: bool,
        was_truncated: bool,
    ) -> SessionState:
        """
        Write episodic event and update session state.
        Runs after both streaming and non-streaming turns.
        """

        # Write episodic event
        try:
            event = EpisodicEvent(
                event_type=EventType.LLM_CALL,
                summary=f"[{intent.domain}/{intent.action}] {user_input[:120]}",
                project=intent.project or session.active_project,
                data={
                    "input_preview": user_input[:200],
                    "domain":        intent.domain,
                    "action":        intent.action,
                    "tokens_in":     tokens_in,
                    "tokens_out":    tokens_out,
                    "cost_usd":      cost_usd,
                    "is_local":      is_local,
                },
            )
            write_event(event)
        except Exception as exc:
            # Never let event writing crash a turn
            log.warning("orchestrator.event_write_failed", error=str(exc))

        # Update session metrics and turn count
        updated_metrics = session.metrics.add_call(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            is_local=is_local,
        )

        updated = (
            session
            .with_turn()
            .with_task(intent.action, user_input[:120])
            .with_metrics(updated_metrics)
        )

        return self._session.update(updated)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Graceful shutdown — write SESSION_END event and close LLM clients.
        Called by the interface layer on Ctrl+C or normal exit.
        """
        if self._session.is_loaded:
            try:
                event = EpisodicEvent(
                    event_type=EventType.SESSION_END,
                    summary=f"Session ended after {self._session.state.turn_count} turns",
                    data={
                        "turns":     self._session.state.turn_count,
                        "cost_usd":  self._session.state.metrics.total_cost_usd,
                        "tokens":    self._session.state.metrics.total_tokens_in
                                     + self._session.state.metrics.total_tokens_out,
                    },
                )
                write_event(event)
            except Exception as exc:
                log.warning("orchestrator.shutdown_event_failed", error=str(exc))

        await self._router.close()
        log.info("orchestrator.shutdown_complete")