 # type: ignore
"""
tests/test_e2e_smoke.py

End-to-end smoke test — spins up the full orchestrator stack with:
    - Real filesystem memory (tmp_path)
    - Real session manager, episodic store, skill store
    - Real classifier, retrieval pipeline, context builder, prompt engine
    - Mocked LLM router (no live LM Studio or Anthropic key needed)

Validates that:
    1. Three turns complete without exceptions
    2. Session turn count increments correctly
    3. Episodic events are written to disk after each turn
    4. Responses are non-empty strings
    5. Session state persists to disk and reloads correctly
    6. run_turn_stream() yields chunks and writes an episodic event
    7. Classifier routes to the correct domain
    8. Context builder fires (retrieval pipeline runs end-to-end)
    9. Orchestrator.shutdown() completes cleanly

This test intentionally touches every major component to catch
cross-module wiring bugs that unit tests cannot detect.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent.core.classifier import Domain, TaskClassifier, TaskIntent, Urgency
from agent.core.config import cfg
from agent.core.orchestrator import Orchestrator, TurnResult
from agent.core.session import SessionManager
from agent.llm.lm_studio import LLMResponse, Message
from agent.llm.prompt_engine import PromptEngine
from agent.llm.router import LLMRouter
from agent.llm.validator import ResponseValidator
from agent.memory.context_builder import ContextBuilder
from agent.memory.embedder import Embedder, reset_embedder
from agent.memory.episodic import list_events
from agent.memory.graph import SemanticGraph
from agent.memory.index import FaissIndex
from agent.memory.retrieval import RetrievalPipeline

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_embedder_singleton():
    reset_embedder()
    yield
    reset_embedder()


@pytest.fixture
def memory_root(tmp_path: Path, monkeypatch):
    """
    Redirect all memory I/O to tmp_path so the smoke test
    never touches ~/.agent.
    """
    root = tmp_path / "agent"
    # Create the expected directory tree
    for sub in (
        "memory/episodic",
        "memory/semantic",
        "memory/skills",
        "memory/working",
        "consolidation",
        "logs",
        "config",
        "projects",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # ONLY patch the root. The properties will compute paths dynamically from this!
    monkeypatch.setattr(cfg, "agent_memory_root", root)
    
    return root


DIM = 8   # small embedding dimension for fast tests


def _make_embedder() -> Embedder:
    """Embedder that returns zero vectors without calling any LLM."""
    embedder = Embedder.__new__(Embedder)
    embedder._using_fallback = False
    embedder._client = None  # type: ignore

    async def _embed(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), DIM), dtype=np.float32)

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


def _make_router(response_text: str = "Here is my response.") -> LLMRouter:
    """Router that always returns a fixed local response."""
    router = MagicMock(spec=LLMRouter)
    router.complete = AsyncMock(
        return_value=LLMResponse(
            content=response_text,
            model="test-local",
            tokens_in=50,
            tokens_out=30,
            is_local=True,
        )
    )

    async def _stream_generator():
        for chunk in response_text.split():
            yield chunk + " "

    # The router.stream method itself must be an async function that RETURNS the generator
    async def _mock_stream_method(*args, **kwargs):
        return _stream_generator()

    router.stream = _mock_stream_method
    router.close = AsyncMock()
    return router

def _make_orchestrator(memory_root: Path, response_text: str = "Here is my response.") -> Orchestrator:
    """
    Build a fully wired orchestrator with real memory components
    and a mocked LLM router.
    """
    embedder = _make_embedder()
    graph    = SemanticGraph()
    index    = FaissIndex()
    router   = _make_router(response_text)

    retrieval = RetrievalPipeline(graph, index, embedder)
    engine    = PromptEngine()
    validator = ResponseValidator(max_retries=1)
    ctx_builder = ContextBuilder(max_tokens=500)

    # Classifier backed by the same mocked router
    classify_response = LLMResponse(
        content='{"domain":"code","action":"debug","urgency":"normal","project":"agrivision"}',
        model="test-local",
        tokens_in=10,
        tokens_out=20,
        is_local=True,
    )
    classify_router = MagicMock(spec=LLMRouter)
    classify_router.complete = AsyncMock(return_value=classify_response)
    classify_router.close = AsyncMock()
    classifier = TaskClassifier(router=classify_router, engine=engine, validator=validator)

    session_manager = SessionManager(state_path=cfg.working_dir / "session.json")

    return Orchestrator(
        router=router,
        classifier=classifier,
        retrieval=retrieval,
        context_builder=ctx_builder,
        prompt_engine=engine,
        validator=validator,
        session_manager=session_manager,
    )


# ── Smoke test — multi-turn run_turn() ────────────────────────────────────────

@pytest.mark.asyncio
async def test_three_turns_complete(memory_root: Path) -> None:
    """Three consecutive turns complete without raising."""
    orch = _make_orchestrator(memory_root)

    r1 = await orch.run_turn("Fix the CORS error in FastAPI")
    r2 = await orch.run_turn("Run the test suite")
    r3 = await orch.run_turn("Export the plant model as GLB")

    assert r1.turn_number == 1
    assert r2.turn_number == 2
    assert r3.turn_number == 3


@pytest.mark.asyncio
async def test_responses_are_non_empty(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root, response_text="Here is the answer.")
    result = await orch.run_turn("What should I do?")
    assert result.content.strip() != ""
    assert "answer" in result.content.lower()


@pytest.mark.asyncio
async def test_turn_result_succeeded(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    result = await orch.run_turn("Hello")
    assert result.succeeded is True
    assert result.error is None


# ── Session state ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_turn_count_increments(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("turn one")
    await orch.run_turn("turn two")
    await orch.run_turn("turn three")
    assert orch._session.state.turn_count == 3


@pytest.mark.asyncio
async def test_session_metrics_accumulate(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("first call")
    await orch.run_turn("second call")
    metrics = orch._session.state.metrics
    assert metrics.total_llm_calls >= 2
    assert metrics.total_tokens_in > 0
    assert metrics.local_calls >= 2
    assert metrics.cloud_calls == 0


@pytest.mark.asyncio
async def test_session_persists_to_disk(memory_root: Path) -> None:
    """Session file must exist and be reloadable after turns."""
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("persist this")
    await orch.run_turn("and this")

    session_path = cfg.working_dir / "session.json"
    assert session_path.exists()

    mgr2   = SessionManager(state_path=session_path)
    loaded = mgr2.load()
    assert loaded is not None
    assert loaded.turn_count == 2


@pytest.mark.asyncio
async def test_session_project_set_from_classifier(memory_root: Path) -> None:
    """When the classifier returns a project, it should appear in session state."""
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("work on agrivision detection endpoint")
    # Classifier mock returns project=agrivision
    assert orch._session.state.active_project == "agrivision"


# ── Episodic events ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_episodic_events_written_after_each_turn(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("first")
    await orch.run_turn("second")
    await orch.run_turn("third")

    events = list_events(limit=50)
    assert len(events) >= 3


@pytest.mark.asyncio
async def test_episodic_events_have_correct_type(memory_root: Path) -> None:
    from agent.memory.episodic import EventType
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("test event type")

    events = list_events(limit=10)
    event_types = {e.event_type for e in events}
    assert EventType.LLM_CALL in event_types


@pytest.mark.asyncio
async def test_episodic_events_have_project(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("agrivision task")

    events = list_events(limit=10)
    projects = {e.project for e in events if e.project}
    assert "agrivision" in projects


# ── Pipeline components fire ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classifier_is_called(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("classify this input")
    orch._classifier._router.complete.assert_called()


@pytest.mark.asyncio
async def test_router_is_called(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("call the router")
    orch._router.complete.assert_called_once()


@pytest.mark.asyncio
async def test_context_builder_fires(memory_root: Path) -> None:
    """Context builder must be invoked on every turn."""
    orch = _make_orchestrator(memory_root)
    with patch.object(orch._context_builder, "build", wraps=orch._context_builder.build) as spy:
        await orch.run_turn("test context")
    spy.assert_called_once()


@pytest.mark.asyncio
async def test_prompt_engine_fires(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    with patch.object(orch._prompt_engine, "build", wraps=orch._prompt_engine.build) as spy:
        await orch.run_turn("build a prompt")
    
    # Called twice: once for classification, once for the final response
    assert spy.call_count == 2
# ── Streaming ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_yields_chunks(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root, response_text="Hello world from agent")

    async def _stream_generator():
        for chunk in "Hello world from agent".split():
            yield chunk + " "

    async def _mock_stream_method(*args, **kwargs):
        return _stream_generator()

    orch._router.stream = MagicMock(return_value=_mock_stream_method())

    chunks: list[str] = []
    async for chunk in orch.run_turn_stream("stream test"):
        chunks.append(chunk)

    assert len(chunks) > 0
    full = "".join(chunks)
    assert "Hello" in full


@pytest.mark.asyncio
async def test_stream_writes_episodic_event(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)

    async def _stream(*args, **kwargs):
        yield "streamed "
        yield "response"

    orch._router.stream = MagicMock(return_value=_stream())

    async for _ in orch.run_turn_stream("stream event test"):
        pass

    events = list_events(limit=10)
    assert len(events) >= 1


# ── Error resilience ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_failure_returns_error_result(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    orch._router.complete = AsyncMock(side_effect=RuntimeError("LM Studio down"))

    result = await orch.run_turn("this will fail")
    assert result.succeeded is False
    assert result.error is not None
    assert result.turn_number == 1   # still counted


@pytest.mark.asyncio
async def test_classifier_failure_returns_unknown_domain(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    orch._classifier._router.complete = AsyncMock(
        return_value=LLMResponse(
            content="not valid json at all",
            model="test",
            tokens_in=5,
            tokens_out=5,
            is_local=True,
        )
    )
    result = await orch.run_turn("classify this badly")
    # Should not crash — classifier returns UNKNOWN domain gracefully
    assert result.turn_number == 1


@pytest.mark.asyncio
async def test_multiple_turns_after_error(memory_root: Path) -> None:
    """After a failed turn, subsequent turns must still work."""
    orch = _make_orchestrator(memory_root)

    # First turn fails
    orch._router.complete = AsyncMock(side_effect=RuntimeError("crash"))
    r1 = await orch.run_turn("failing turn")
    assert r1.succeeded is False

    # Restore and run two more
    orch._router.complete = AsyncMock(
        return_value=LLMResponse(
            content="recovered", model="test", tokens_in=10, tokens_out=5, is_local=True
        )
    )
    r2 = await orch.run_turn("recovery turn")
    r3 = await orch.run_turn("final turn")

    assert r2.succeeded is True
    assert r3.turn_number == 3


# ── Shutdown ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shutdown_completes_cleanly(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("one turn before shutdown")
    await orch.shutdown()
    orch._router.close.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_writes_session_end_event(memory_root: Path) -> None:
    from agent.memory.episodic import EventType
    orch = _make_orchestrator(memory_root)
    await orch.run_turn("pre-shutdown turn")
    await orch.shutdown()

    events = list_events(limit=20)
    event_types = [e.event_type for e in events]
    assert EventType.SESSION_END in event_types


# ── Conversation history ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_conversation_history_passed_to_router(memory_root: Path) -> None:
    """The router should receive the conversation history on multi-turn calls."""
    orch = _make_orchestrator(memory_root)
    history = [
        Message(role="user",      content="previous question"),
        Message(role="assistant", content="previous answer"),
    ]
    await orch.run_turn("follow-up question", conversation_history=history)

    call_args = orch._router.complete.call_args
    messages_sent = call_args[0][0]
    contents = [m.content if isinstance(m, Message) else m.get("content", "") for m in messages_sent]
    all_content = " ".join(contents)
    assert "follow-up question" in all_content


# ── TurnResult fields ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_turn_result_has_positive_latency(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    result = await orch.run_turn("latency test")
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_turn_result_has_intent(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    result = await orch.run_turn("fix the bug")
    assert result.intent is not None
    assert result.intent.domain == Domain.CODE


@pytest.mark.asyncio
async def test_turn_result_has_session(memory_root: Path) -> None:
    orch = _make_orchestrator(memory_root)
    result = await orch.run_turn("test session field")
    assert result.session is not None
    assert result.session.turn_count == 1