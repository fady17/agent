"""
tests/test_validator.py

Tests for ResponseValidator — JSON validation, text validation,
passthrough, retry behaviour, repair prompt construction, and
code fence stripping.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agent.llm.lm_studio import LLMResponse, Message
from agent.llm.validator import (
    ResponseValidator,
    ValidationResult,
    _build_repair_prompt,
    _strip_code_fences,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_response(content: str, is_local: bool = True) -> LLMResponse:
    return LLMResponse(content=content, model="test-model", is_local=is_local)


def make_router(responses: list[str]) -> MagicMock:
    """Mock router that returns responses in sequence."""
    router = MagicMock()
    router.complete = AsyncMock(
        side_effect=[make_response(r) for r in responses]
    )
    return router


MESSAGES = [Message(role="user", content="return JSON")]


# ── Pydantic model for testing ────────────────────────────────────────────────

class TaskIntent(BaseModel):
    domain: str
    action: str
    urgency: str = "normal"


# ── ValidationResult ──────────────────────────────────────────────────────────

def test_result_repr_success() -> None:
    r = ValidationResult(success=True, value="ok", attempts=1)
    assert "ok" in repr(r)


def test_result_repr_failure() -> None:
    r = ValidationResult(success=False, error="bad json", attempts=2)
    assert "FAILED" in repr(r)
    assert "bad json" in repr(r)


def test_result_unwrap_success() -> None:
    r = ValidationResult(success=True, value={"key": "val"}, attempts=1)
    assert r.unwrap() == {"key": "val"}


def test_result_unwrap_failure_raises() -> None:
    r = ValidationResult(success=False, error="parse error", attempts=1)
    with pytest.raises(RuntimeError, match="parse error"):
        r.unwrap()


# ── _strip_code_fences ────────────────────────────────────────────────────────

def test_strip_json_fence() -> None:
    raw = '```json\n{"key": "value"}\n```'
    assert _strip_code_fences(raw) == '{"key": "value"}'


def test_strip_plain_fence() -> None:
    raw = '```\n{"key": "value"}\n```'
    assert _strip_code_fences(raw) == '{"key": "value"}'


def test_strip_no_fence_unchanged() -> None:
    raw = '{"key": "value"}'
    assert _strip_code_fences(raw) == '{"key": "value"}'


def test_strip_handles_extra_whitespace() -> None:
    raw = '  ```json\n  {"a": 1}  \n```  '
    result = _strip_code_fences(raw)
    assert result == '{"a": 1}'


# ── _build_repair_prompt ──────────────────────────────────────────────────────

def test_repair_prompt_ends_with_user_message() -> None:
    msgs = _build_repair_prompt(
        original_messages=MESSAGES,
        bad_output="not json",
        error="JSON parse error",
        expected_format="valid JSON",
    )
    assert msgs[-1].role == "user"


def test_repair_prompt_includes_bad_output() -> None:
    msgs = _build_repair_prompt(
        original_messages=MESSAGES,
        bad_output="not json at all",
        error="parse error",
        expected_format="valid JSON",
    )
    assert "not json at all" in msgs[-1].content


def test_repair_prompt_includes_error() -> None:
    msgs = _build_repair_prompt(
        original_messages=MESSAGES,
        bad_output="{}",
        error="field required: domain",
        expected_format="valid JSON",
    )
    assert "field required" in msgs[-1].content


def test_repair_prompt_includes_assistant_with_bad_output() -> None:
    msgs = _build_repair_prompt(
        original_messages=MESSAGES,
        bad_output="bad output",
        error="error",
        expected_format="JSON",
    )
    # Second to last should be the assistant echoing bad output
    assert msgs[-2].role == "assistant"
    assert msgs[-2].content == "bad output"


def test_repair_prompt_preserves_original_messages() -> None:
    original = [
        Message(role="system", content="system prompt"),
        Message(role="user",   content="user input"),
    ]
    msgs = _build_repair_prompt(
        original_messages=original,
        bad_output="x",
        error="e",
        expected_format="JSON",
    )
    assert msgs[0].content == "system prompt"
    assert msgs[1].content == "user input"


# ── ResponseValidator construction ────────────────────────────────────────────

def test_validator_default_max_retries() -> None:
    v = ResponseValidator()
    assert v.max_retries == 3


def test_validator_custom_max_retries() -> None:
    v = ResponseValidator(max_retries=5)
    assert v.max_retries == 5


def test_validator_zero_retries_raises() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ResponseValidator(max_retries=0)


# ── passthrough ───────────────────────────────────────────────────────────────

def test_passthrough_always_succeeds() -> None:
    v = ResponseValidator()
    result = v.passthrough(make_response("anything at all"))
    assert result.success is True
    assert result.value == "anything at all"
    assert result.attempts == 1


def test_passthrough_empty_content_still_succeeds() -> None:
    v = ResponseValidator()
    result = v.passthrough(make_response(""))
    assert result.success is True


# ── validate_json — success paths ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_json_valid_dict() -> None:
    v = ResponseValidator()
    result = await v.validate_json(make_response('{"domain": "code", "action": "debug"}'))
    assert result.success is True
    assert result.value == {"domain": "code", "action": "debug"}
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_validate_json_with_pydantic_model() -> None:
    v = ResponseValidator()
    result = await v.validate_json(
        make_response('{"domain": "code", "action": "debug", "urgency": "high"}'),
        pydantic_model=TaskIntent,
    )
    assert result.success is True
    assert isinstance(result.value, TaskIntent)
    assert result.value.domain == "code"


@pytest.mark.asyncio
async def test_validate_json_strips_code_fences() -> None:
    v = ResponseValidator()
    result = await v.validate_json(
        make_response('```json\n{"domain": "design"}\n```'),
    )
    assert result.success is True
    assert result.value["domain"] == "design"


@pytest.mark.asyncio
async def test_validate_json_list_response() -> None:
    v = ResponseValidator()
    result = await v.validate_json(make_response('[1, 2, 3]'))
    assert result.success is True
    assert result.value == [1, 2, 3]


# ── validate_json — failure and retry ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_json_invalid_no_router_fails_immediately() -> None:
    v = ResponseValidator(max_retries=3)
    result = await v.validate_json(make_response("not json"))
    assert result.success is False
    assert "JSON parse error" in result.error
    assert result.attempts == 1  # No router = no retries


@pytest.mark.asyncio
async def test_validate_json_retries_until_valid() -> None:
    v = ResponseValidator(max_retries=3)
    router = make_router(['still bad', '{"domain": "code", "action": "fix"}'])

    result = await v.validate_json(
        make_response("not json"),
        router=router,
        messages=MESSAGES,
    )
    assert result.success is True
    assert result.value["domain"] == "code"
    assert result.attempts == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_validate_json_exhausts_retries() -> None:
    v = ResponseValidator(max_retries=3)
    router = make_router(["bad1", "bad2"])

    result = await v.validate_json(
        make_response("bad0"),
        router=router,
        messages=MESSAGES,
    )
    assert result.success is False
    assert router.complete.call_count == 2


@pytest.mark.asyncio
async def test_validate_json_pydantic_failure_retries() -> None:
    v = ResponseValidator(max_retries=3)
    # First response is valid JSON but missing required field
    # Second response is valid and complete
    router = make_router(['{"domain": "code"}'])  # missing "action"

    result = await v.validate_json(
        make_response('{"domain": "code"}'),
        router=router,
        messages=MESSAGES,
        pydantic_model=TaskIntent,
    )
    # Pydantic validation fails (action missing), retries once
    assert router.complete.call_count >= 1


@pytest.mark.asyncio
async def test_validate_json_repair_call_uses_original_messages() -> None:
    v = ResponseValidator(max_retries=2)
    router = make_router(['{"domain": "code", "action": "fix"}'])
    original = [Message(role="user", content="classify this")]

    await v.validate_json(
        make_response("bad json"),
        router=router,
        messages=original,
    )
    # Repair call was made — check it received the repair messages
    call_args = router.complete.call_args
    sent_messages = call_args[0][0]  # first positional arg
    # Repair messages contain the original + bad output + repair request
    assert any(m.content == "bad json" for m in sent_messages)


# ── validate_text — success paths ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_text_valid_content() -> None:
    v = ResponseValidator()
    result = await v.validate_text(make_response("A solid response here."))
    assert result.success is True
    assert result.value == "A solid response here."


@pytest.mark.asyncio
async def test_validate_text_strips_whitespace() -> None:
    v = ResponseValidator()
    result = await v.validate_text(make_response("  hello  "))
    assert result.success is True
    assert result.value == "hello"


@pytest.mark.asyncio
async def test_validate_text_exact_min_length() -> None:
    v = ResponseValidator()
    result = await v.validate_text(make_response("hi"), min_length=2)
    assert result.success is True


@pytest.mark.asyncio
async def test_validate_text_within_max_length() -> None:
    v = ResponseValidator()
    result = await v.validate_text(make_response("hello"), max_length=100)
    assert result.success is True


# ── validate_text — failure and retry ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_text_too_short_no_router_fails() -> None:
    v = ResponseValidator()
    result = await v.validate_text(make_response("hi"), min_length=100)
    assert result.success is False
    assert "too short" in result.error


@pytest.mark.asyncio
async def test_validate_text_too_long_fails() -> None:
    v = ResponseValidator()
    result = await v.validate_text(make_response("hello world"), max_length=5)
    assert result.success is False
    assert "too long" in result.error


@pytest.mark.asyncio
async def test_validate_text_retries_until_valid() -> None:
    v = ResponseValidator(max_retries=3)
    router = make_router(["short", "A proper and longer response that passes"])

    result = await v.validate_text(
        make_response("x"),
        min_length=10,
        router=router,
        messages=MESSAGES,
    )
    assert result.success is True
    assert result.attempts == 3


@pytest.mark.asyncio
async def test_validate_text_empty_string_fails() -> None:
    v = ResponseValidator()
    result = await v.validate_text(make_response("  "), min_length=1)
    assert result.success is False


# ── response attached to result ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_result_carries_final_response() -> None:
    v = ResponseValidator()
    response = make_response('{"a": 1}')
    result = await v.validate_json(response)
    assert result.response is response


@pytest.mark.asyncio
async def test_result_carries_repaired_response() -> None:
    v = ResponseValidator(max_retries=2)
    repaired = make_response('{"domain": "code", "action": "fix"}')
    router = MagicMock()
    router.complete = AsyncMock(return_value=repaired)

    result = await v.validate_json(
        make_response("bad"),
        router=router,
        messages=MESSAGES,
    )
    assert result.success is True
    assert result.response is repaired