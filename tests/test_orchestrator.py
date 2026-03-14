 # type: ignore
"""
tests/test_orchestrator.py

Tests for the Orchestrator — the full run_turn() pipeline with all
dependencies mocked. Tests verify coordination between components,
not the components themselves (those have their own test suites).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.core.classifier import Domain, TaskIntent, Urgency
from agent.core.orchestrator import Orchestrator, TurnResult
from agent.core.session import SessionManager, SessionState
from agent.llm.lm_studio import LLMResponse, Message
from agent.llm.validator import ValidationResult
from agent.memory.context_builder import ContextBuilder, ContextPayload
from agent.memory.retrieval import ContextBlock

# ── Factories / helpers ───────────────────────────────────────────────────────

def make_intent(
    domain: Domain = Domain.CODE,
    action: str = "debug",
    project: str | None = "agrivision",
    urgency: Urgency = Urgency.NORMAL,
) -> TaskIntent:
    return TaskIntent(domain=domain, action=action, project=project, urgency=urgency)


def make_llm_response(
    content: str = "Here is the fix.",
    tokens_in: int = 50,
    tokens_out: int = 80,
    is_local: bool = True,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        is_local=is_local,
    )


def make_context_block() -> ContextBlock:
    return ContextBlock(query="test", project="agrivision")


def make_context_payload(text: str = "", truncated: bool = False) -> ContextPayload:
    return ContextPayload(
        text=text,
        tokens_used=len(text) // 4,
        items_included={"skills": 0, "events": 0, "nodes": 0},
        was_truncated=truncated,
    )


def make_orchestrator(
    tmp_path: Path,
    *,
    intent: TaskIntent | None = None,
    llm_content: str = "Here is the fix.",
    tokens_in: int = 50,
    tokens_out: int = 80,
    is_local: bool = True,
    validation_success: bool = True,
    context_text: str = "",
    context_truncated: bool = False,
) -> Orchestrator:
    """Build a fully mocked Orchestrator for testing."""

    # Router
    router = MagicMock()
    router.complete = AsyncMock(return_value=make_llm_response(
        llm_content, tokens_in=tokens_in, tokens_out=tokens_out, is_local=is_local
    ))
    router.close = AsyncMock()

    # Classifier
    classifier = MagicMock()
    classifier.classify = AsyncMock(return_value=intent or make_intent())

    # Retrieval
    retrieval = MagicMock()
    retrieval.retrieve = AsyncMock(return_value=make_context_block())

    # Context builder
    ctx_builder = MagicMock()
    ctx_builder.build = MagicMock(
        return_value=make_context_payload(context_text, context_truncated)
    )

    # Prompt engine
    prompt_engine = MagicMock()
    prompt_engine.build = MagicMock(return_value=[
        Message(role="system", content="system"),
        Message(role="user",   content="user input"),
    ])

    # Validator
    validator = MagicMock()
    if validation_success:
        validator.validate_text = AsyncMock(
            return_value=ValidationResult(success=True, value=llm_content, attempts=1)
        )
    else:
        validator.validate_text = AsyncMock(
            return_value=ValidationResult(success=False, error="too short", attempts=3)
        )

    # Session manager
    session_manager = SessionManager(state_path=tmp_path / "session.json")
    session_manager.new_session(session_id="sess-test")

    return Orchestrator(
        router=router,
        classifier=classifier,
        retrieval=retrieval,
        context_builder=ctx_builder,
        prompt_engine=prompt_engine,
        validator=validator,
        session_manager=session_manager,
    )


# ── TurnResult ────────────────────────────────────────────────────────────────

def test_turn_result_succeeded_true_when_no_error() -> None:
    intent = make_intent()
    state = SessionState(session_id="s1")
    result = TurnResult(content="ok", intent=intent, session=state, turn_number=1)
    assert result.succeeded is True


def test_turn_result_succeeded_false_when_error() -> None:
    intent = make_intent()
    state = SessionState(session_id="s1")
    result = TurnResult(
        content="error", intent=intent, session=state, turn_number=1, error="oops"
    )
    assert result.succeeded is False


# ── run_turn — happy path ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_turn_returns_turn_result(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    result = await orch.run_turn("Fix the CORS error in FastAPI")
    assert isinstance(result, TurnResult)


@pytest.mark.asyncio
async def test_run_turn_content_matches_llm_response(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, llm_content="Add CORS middleware.")
    result = await orch.run_turn("Fix CORS")
    assert result.content == "Add CORS middleware."


@pytest.mark.asyncio
async def test_run_turn_intent_attached(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, intent=make_intent(domain=Domain.CODE, action="debug"))
    result = await orch.run_turn("fix the bug")
    assert result.intent.domain == Domain.CODE
    assert result.intent.action == "debug"


@pytest.mark.asyncio
async def test_run_turn_turn_number_increments(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    r1 = await orch.run_turn("first")
    r2 = await orch.run_turn("second")
    r3 = await orch.run_turn("third")
    assert r1.turn_number == 1
    assert r2.turn_number == 2
    assert r3.turn_number == 3


@pytest.mark.asyncio
async def test_run_turn_tokens_tracked(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, tokens_in=100, tokens_out=50)
    result = await orch.run_turn("test")
    assert result.tokens_used == 150


@pytest.mark.asyncio
async def test_run_turn_latency_ms_positive(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    result = await orch.run_turn("test")
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_run_turn_succeeded_true(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    result = await orch.run_turn("fix bug")
    assert result.succeeded is True
    assert result.error is None


# ── run_turn — session state ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_turn_updates_session_turn_count(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("first turn")
    await orch.run_turn("second turn")
    assert orch._session.state.turn_count == 2


@pytest.mark.asyncio
async def test_run_turn_updates_session_metrics(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, tokens_in=100, tokens_out=50, is_local=True)
    await orch.run_turn("test")
    metrics = orch._session.state.metrics
    assert metrics.total_tokens_in == 100
    assert metrics.total_tokens_out == 50
    assert metrics.local_calls == 1
    assert metrics.cloud_calls == 0


@pytest.mark.asyncio
async def test_run_turn_updates_project_when_intent_has_project(tmp_path: Path) -> None:
    intent = make_intent(project="plant-chat")
    orch = make_orchestrator(tmp_path, intent=intent)
    await orch.run_turn("work on plant-chat")
    assert orch._session.state.active_project == "plant-chat"


@pytest.mark.asyncio
async def test_run_turn_auto_inits_session_if_not_loaded(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    # Reset session so it's not loaded
    orch._session.reset()
    assert not orch._session.is_loaded
    result = await orch.run_turn("hello")
    assert orch._session.is_loaded
    assert result.turn_number == 1


# ── run_turn — context truncation ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_turn_reports_truncation(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, context_truncated=True)
    result = await orch.run_turn("test")
    assert result.was_truncated is True


@pytest.mark.asyncio
async def test_run_turn_no_truncation_by_default(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path, context_truncated=False)
    result = await orch.run_turn("test")
    assert result.was_truncated is False


# ── run_turn — validation fallback ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_turn_uses_raw_content_when_validation_fails(tmp_path: Path) -> None:
    """When validator fails, orchestrator falls back to raw LLM content."""
    orch = make_orchestrator(tmp_path, llm_content="raw response", validation_success=False)
    result = await orch.run_turn("test")
    # Should still succeed — validation failure is not a hard error
    assert result.content == "raw response"


# ── run_turn — pipeline calls ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_turn_calls_classifier(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("classify this please")
    orch._classifier.classify.assert_called_once()
    call_args = orch._classifier.classify.call_args
    assert "classify this please" in call_args[0]


@pytest.mark.asyncio
async def test_run_turn_calls_retrieval_with_project(tmp_path: Path) -> None:
    intent = make_intent(project="agrivision")
    orch = make_orchestrator(tmp_path, intent=intent)
    await orch.run_turn("run tests")
    orch._retrieval.retrieve.assert_called_once()
    _, kwargs = orch._retrieval.retrieve.call_args
    assert kwargs.get("project") == "agrivision"


@pytest.mark.asyncio
async def test_run_turn_calls_context_builder(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("test")
    orch._context_builder.build.assert_called_once()


@pytest.mark.asyncio
async def test_run_turn_calls_prompt_engine(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("test")
    orch._prompt_engine.build.assert_called_once()


@pytest.mark.asyncio
async def test_run_turn_calls_router(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("test")
    orch._router.complete.assert_called_once()


@pytest.mark.asyncio
async def test_run_turn_calls_validator(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("test")
    orch._validator.validate_text.assert_called_once()


@pytest.mark.asyncio
async def test_run_turn_passes_force_cloud_to_router(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("complex task", force_cloud=True)
    _, kwargs = orch._router.complete.call_args
    assert kwargs.get("force_cloud") is True


@pytest.mark.asyncio
async def test_run_turn_passes_force_local_to_router(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("simple task", force_local=True)
    _, kwargs = orch._router.complete.call_args
    assert kwargs.get("force_local") is True


# ── run_turn — error handling ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_turn_handles_classifier_exception(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    orch._classifier.classify = AsyncMock(side_effect=RuntimeError("classifier down"))
    result = await orch.run_turn("test input")
    assert result.succeeded is False
    assert result.error is not None
    assert "error" in result.content.lower()


@pytest.mark.asyncio
async def test_run_turn_handles_router_exception(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    orch._router.complete = AsyncMock(side_effect=RuntimeError("LM Studio down"))
    result = await orch.run_turn("test input")
    assert result.succeeded is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_run_turn_error_still_increments_turn(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    orch._router.complete = AsyncMock(side_effect=RuntimeError("crash"))
    result = await orch.run_turn("test")
    assert result.turn_number == 1  # Turn still counted despite error


@pytest.mark.asyncio
async def test_run_turn_episodic_write_failure_does_not_crash(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    with patch("agent.core.orchestrator.write_event", side_effect=OSError("disk full")):
        result = await orch.run_turn("test")
    # Should still succeed — event write failure is non-fatal
    assert result.succeeded is True


# ── shutdown ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shutdown_closes_router(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("one turn")
    await orch.shutdown()
    orch._router.close.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_without_session_does_not_raise(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    orch._session.reset()
    await orch.shutdown()  # Should not raise


@pytest.mark.asyncio
async def test_shutdown_writes_session_end_event(tmp_path: Path) -> None:
    orch = make_orchestrator(tmp_path)
    await orch.run_turn("one turn")
    with patch("agent.core.orchestrator.write_event") as mock_write:
        await orch.shutdown()
    mock_write.assert_called_once()
    event_written = mock_write.call_args[0][0]
    from agent.memory.episodic import EventType
    assert event_written.event_type == EventType.SESSION_END