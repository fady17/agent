"""
tests/test_classifier.py

Tests for TaskIntent (schema, validators, coercion) and
TaskClassifier (full pipeline with mocked router + validator).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from agent.core.classifier import (
    Domain,
    TaskClassifier,
    TaskIntent,
    Urgency,
    _fallback_intent,
)
from agent.llm.lm_studio import LLMResponse, Message
from agent.llm.validator import ValidationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_intent(**kwargs) -> TaskIntent:
    defaults = {"domain": Domain.CODE, "action": "debug"}
    return TaskIntent(**{**defaults, **kwargs})


def make_router(json_response: str = '{"domain":"code","action":"debug","urgency":"normal"}') -> MagicMock:
    router = MagicMock()
    router.complete = AsyncMock(
        return_value=LLMResponse(content=json_response, model="test-model")
    )
    return router


def make_classifier(router=None) -> TaskClassifier:
    return TaskClassifier(router=router or make_router())


# ── TaskIntent — construction ─────────────────────────────────────────────────

def test_minimal_intent_constructs() -> None:
    intent = make_intent()
    assert intent.domain == Domain.CODE
    assert intent.action == "debug"
    assert intent.urgency == Urgency.NORMAL
    assert intent.project is None


def test_intent_all_fields() -> None:
    intent = TaskIntent(
        domain=Domain.DESIGN,
        action="export GLB",
        project="agrivision",
        urgency=Urgency.HIGH,
    )
    assert intent.project == "agrivision"
    assert intent.urgency == Urgency.HIGH


def test_intent_empty_action_raises() -> None:
    with pytest.raises(ValidationError, match="action"):
        make_intent(action="   ")


# ── TaskIntent — field coercion ───────────────────────────────────────────────

def test_project_normalised_to_lowercase() -> None:
    intent = make_intent(project="AgriVision")
    assert intent.project == "agrivision"


def test_project_whitespace_becomes_none() -> None:
    intent = make_intent(project="  ")
    assert intent.project is None


def test_project_none_stays_none() -> None:
    intent = make_intent(project=None)
    assert intent.project is None


def test_unknown_domain_coerced_not_raised() -> None:
    intent = TaskIntent(domain="completely_unknown_domain", action="do something")  # type: ignore[arg-type]
    assert intent.domain == Domain.UNKNOWN


def test_unknown_urgency_coerced_to_normal() -> None:
    intent = TaskIntent(domain=Domain.CODE, action="debug", urgency="urgent_now")  # type: ignore[arg-type]
    assert intent.urgency == Urgency.NORMAL


def test_all_domains_valid() -> None:
    for domain in Domain:
        intent = TaskIntent(domain=domain, action="test")
        assert intent.domain == domain


def test_all_urgencies_valid() -> None:
    for urgency in Urgency:
        intent = TaskIntent(domain=Domain.CODE, action="test", urgency=urgency)
        assert intent.urgency == urgency


# ── TaskIntent — properties ───────────────────────────────────────────────────

def test_is_known_true_for_known_domain() -> None:
    intent = make_intent(domain=Domain.CODE)
    assert intent.is_known is True


def test_is_known_false_for_unknown() -> None:
    intent = make_intent(domain=Domain.UNKNOWN)
    assert intent.is_known is False


def test_needs_clarification_true_for_unknown() -> None:
    intent = make_intent(domain=Domain.UNKNOWN)
    assert intent.needs_clarification is True


def test_needs_clarification_false_for_known() -> None:
    intent = make_intent(domain=Domain.SHELL)
    assert intent.needs_clarification is False


def test_summary_with_project() -> None:
    intent = make_intent(domain=Domain.CODE, action="debug", project="agrivision")
    s = intent.summary()
    assert "code" in s
    assert "debug" in s
    assert "agrivision" in s


def test_summary_without_project() -> None:
    intent = make_intent(domain=Domain.SHELL, action="run linter")
    s = intent.summary()
    assert "shell" in s
    assert "[" not in s


def test_raw_input_excluded_from_serialisation() -> None:
    intent = make_intent(raw_input="original user text")
    dumped = intent.model_dump()
    assert "raw_input" not in dumped


# ── _fallback_intent ──────────────────────────────────────────────────────────

def test_fallback_intent_has_unknown_domain() -> None:
    intent = _fallback_intent("some input")
    assert intent.domain == Domain.UNKNOWN


def test_fallback_intent_has_unknown_action() -> None:
    intent = _fallback_intent("some input")
    assert intent.action == "unknown"


def test_fallback_intent_stores_raw_input() -> None:
    intent = _fallback_intent("what is the weather?")
    assert intent.raw_input == "what is the weather?"


# ── TaskClassifier — happy path ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_returns_task_intent() -> None:
    classifier = make_classifier()
    intent = await classifier.classify("Fix the CORS error in FastAPI")
    assert isinstance(intent, TaskIntent)


@pytest.mark.asyncio
async def test_classify_correct_domain() -> None:
    router = make_router('{"domain":"code","action":"debug","urgency":"high","project":"agrivision"}')
    classifier = make_classifier(router)
    intent = await classifier.classify("Fix the CORS error")
    assert intent.domain == Domain.CODE
    assert intent.urgency == Urgency.HIGH


@pytest.mark.asyncio
async def test_classify_uses_force_local_by_default() -> None:
    router = make_router()
    classifier = make_classifier(router)
    await classifier.classify("run the tests")
    _, kwargs = router.complete.call_args
    assert kwargs.get("force_local") is True


@pytest.mark.asyncio
async def test_classify_uses_zero_temperature() -> None:
    router = make_router()
    classifier = make_classifier(router)
    await classifier.classify("run tests")
    _, kwargs = router.complete.call_args
    assert kwargs.get("temperature") == 0.0


@pytest.mark.asyncio
async def test_classify_attaches_raw_input() -> None:
    router = make_router('{"domain":"code","action":"fix","urgency":"normal"}')
    classifier = make_classifier(router)
    intent = await classifier.classify("fix the bug please")
    assert intent.raw_input == "fix the bug please"


@pytest.mark.asyncio
async def test_classify_design_domain() -> None:
    router = make_router('{"domain":"design","action":"export GLB","urgency":"normal","project":"agrivision"}')
    classifier = make_classifier(router)
    intent = await classifier.classify("Export the plant model as GLB")
    assert intent.domain == Domain.DESIGN
    assert intent.project == "agrivision"


@pytest.mark.asyncio
async def test_classify_null_project_in_json() -> None:
    router = make_router('{"domain":"search","action":"lookup","urgency":"low","project":null}')
    classifier = make_classifier(router)
    intent = await classifier.classify("What is the weather in Cairo?")
    assert intent.project is None


# ── TaskClassifier — fallback paths ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_empty_input_returns_fallback() -> None:
    classifier = make_classifier()
    intent = await classifier.classify("   ")
    assert intent.domain == Domain.UNKNOWN
    assert intent.needs_clarification is True


@pytest.mark.asyncio
async def test_classify_llm_error_returns_fallback() -> None:
    router = MagicMock()
    router.complete = AsyncMock(side_effect=RuntimeError("LM Studio down"))
    classifier = make_classifier(router)
    intent = await classifier.classify("do something")
    assert intent.domain == Domain.UNKNOWN


@pytest.mark.asyncio
async def test_classify_bad_json_returns_fallback() -> None:
    """When validator exhausts retries, classifier returns fallback."""
    router = MagicMock()
    router.complete = AsyncMock(return_value=LLMResponse(
        content="definitely not json",
        model="test-model",
    ))
    # Validator with max_retries=1 and no repair capability
    validator = MagicMock()
    validator.validate_json = AsyncMock(
        return_value=ValidationResult(
            success=False,
            error="JSON parse error",
            attempts=3,
        )
    )
    classifier = TaskClassifier(router=router, validator=validator)
    intent = await classifier.classify("do something")
    assert intent.domain == Domain.UNKNOWN


@pytest.mark.asyncio
async def test_classify_unknown_domain_in_response_coerced() -> None:
    """Unknown domain from LLM is coerced to UNKNOWN, not an error."""
    router = make_router('{"domain":"quantum_computing","action":"optimize","urgency":"normal"}')
    classifier = make_classifier(router)
    intent = await classifier.classify("optimise the quantum circuit")
    assert intent.domain == Domain.UNKNOWN


# ── TaskClassifier — prompt construction ─────────────────────────────────────

@pytest.mark.asyncio
async def test_classify_passes_input_to_prompt() -> None:
    """Verify the user input reaches the router's message list."""
    router = make_router()
    classifier = make_classifier(router)
    await classifier.classify("run the ruff linter")

    call_args = router.complete.call_args
    messages_sent = call_args[0][0]
    all_content = " ".join(m.content for m in messages_sent)
    assert "run the ruff linter" in all_content


@pytest.mark.asyncio
async def test_classify_sends_multiple_messages_with_few_shot() -> None:
    """CLASSIFY template includes few-shot — more than 2 messages."""
    router = make_router()
    classifier = make_classifier(router)
    await classifier.classify("fix the bug")

    call_args = router.complete.call_args
    messages_sent = call_args[0][0]
    assert len(messages_sent) > 2  # system + few-shot pairs + final user