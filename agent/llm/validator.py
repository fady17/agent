"""
agent/llm/validator.py

Response validator — ensures LLM output conforms to expected shape
before it reaches the orchestrator or any downstream code.

Three validation modes:
    json        — parse JSON, optionally validate against a Pydantic model
    text        — non-empty string, optionally within length bounds
    passthrough — no validation, return content as-is

Retry behaviour:
    When JSON parsing fails or Pydantic validation rejects the output,
    the validator can request a correction from the LLM using a targeted
    repair prompt. Up to `max_retries` attempts are made before raising.

    The repair prompt shows the LLM its previous output and the exact
    error — this is significantly more effective than a bare retry.

Usage:
    validator = ResponseValidator()

    # Validate JSON response
    result = await validator.validate_json(
        response,
        router=router,
        messages=messages,
        model=TaskIntent,       # optional Pydantic model
    )

    # Validate text response
    result = await validator.validate_text(response, min_length=10)

    # Passthrough (no-op)
    result = validator.passthrough(response)
"""

from __future__ import annotations

import json
from typing import Any, Type, TypeVar  # noqa: UP035

from pydantic import BaseModel, ValidationError

from agent.core.logger import get_logger
from agent.llm.lm_studio import LLMResponse, Message
from agent.llm.router import LLMRouter

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Maximum retries for repair attempts
DEFAULT_MAX_RETRIES = 3


# ── ValidationResult ──────────────────────────────────────────────────────────

class ValidationResult:
    """
    Outcome of a validation attempt.

    Attributes:
        success:    True if validation passed.
        value:      The validated value (str, dict, or Pydantic model instance).
        error:      Error message if success is False.
        attempts:   How many LLM calls were made (1 = no retries needed).
        response:   The final LLMResponse used (may differ from input if retried).
    """

    __slots__ = ("success", "value", "error", "attempts", "response")

    def __init__(
        self,
        *,
        success: bool,
        value: Any = None,
        error: str = "",
        attempts: int = 1,
        response: LLMResponse | None = None,
    ) -> None:
        self.success  = success
        self.value    = value
        self.error    = error
        self.attempts = attempts
        self.response = response

    def unwrap(self) -> Any:
        """Return value or raise RuntimeError if validation failed."""
        if not self.success:
            raise RuntimeError(f"Validation failed after {self.attempts} attempts: {self.error}")
        return self.value

    def __repr__(self) -> str:
        if self.success:
            return f"ValidationResult(ok, attempts={self.attempts})"
        return f"ValidationResult(FAILED, error={self.error!r}, attempts={self.attempts})"


# ── ResponseValidator ─────────────────────────────────────────────────────────

class ResponseValidator:
    """
    Validates LLM responses and retries with repair prompts on failure.

    The validator is stateless — it can be shared across calls.
    """

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.max_retries = max_retries

    # ── JSON validation ───────────────────────────────────────────────────────

    async def validate_json(
        self,
        response: LLMResponse,
        *,
        router: "LLMRouter | None" = None,  # noqa: F821, UP037
        messages: list[Message] | None = None,
        pydantic_model: Type[T] | None = None,  # noqa: UP006
        force_local: bool = False,
    ) -> ValidationResult:
        """
        Validate that the response content is valid JSON.
        Optionally validate the parsed dict against a Pydantic model.

        If validation fails and `router` + `messages` are provided,
        attempts to repair the output by sending a correction prompt.

        Args:
            response:       The LLM response to validate.
            router:         LLMRouter for retry calls (optional).
            messages:       Original message list for retry context (optional).
            pydantic_model: Optional Pydantic model to validate against.
            force_local:    Route repair calls to local LLM.

        Returns:
            ValidationResult with .value as dict or Pydantic model instance.
        """
        current_response = response
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            content = current_response.content.strip()

            # Strip markdown code fences if present
            content = _strip_code_fences(content)

            # Attempt JSON parse
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = f"JSON parse error: {exc}"
                log.warning(
                    "validator.json_parse_failed",
                    attempt=attempt,
                    error=last_error,
                )
            else:
                # JSON parsed — optionally validate against Pydantic model
                if pydantic_model is not None:
                    try:
                        validated = pydantic_model.model_validate(parsed)
                        log.debug(
                            "validator.json_ok",
                            attempt=attempt,
                            model=pydantic_model.__name__,
                        )
                        return ValidationResult(
                            success=True,
                            value=validated,
                            attempts=attempt,
                            response=current_response,
                        )
                    except ValidationError as exc:
                        last_error = f"Pydantic validation error: {exc.error_count()} error(s) — {exc.errors()[0]['msg']}"
                        log.warning(
                            "validator.pydantic_failed",
                            attempt=attempt,
                            error=last_error,
                        )
                else:
                    log.debug("validator.json_ok", attempt=attempt)
                    return ValidationResult(
                        success=True,
                        value=parsed,
                        attempts=attempt,
                        response=current_response,
                    )

            # Validation failed — attempt repair if router is available
            if attempt < self.max_retries and router is not None and messages is not None:
                repair_messages = _build_repair_prompt(
                    original_messages=messages,
                    bad_output=current_response.content,
                    error=last_error,
                    expected_format="valid JSON",
                )
                try:
                    current_response = await router.complete(
                        repair_messages,
                        force_local=force_local,
                    )
                    log.info(
                        "validator.repair_attempt",
                        attempt=attempt + 1,
                        max=self.max_retries,
                    )
                except Exception as exc:
                    last_error = f"Repair call failed: {exc}"
                    break
            elif router is None or messages is None:
                break  # No retry capability

        return ValidationResult(
            success=False,
            error=last_error,
            attempts=self.max_retries if router else 1,
            response=current_response,
        )

    # ── Text validation ───────────────────────────────────────────────────────

    async def validate_text(
        self,
        response: LLMResponse,
        *,
        min_length: int = 1,
        max_length: int | None = None,
        router: "LLMRouter | None" = None,  # noqa: F821, UP037
        messages: list[Message] | None = None,
        force_local: bool = False,
    ) -> ValidationResult:
        """
        Validate that the response content is non-empty text within length bounds.

        Args:
            response:   The LLM response to validate.
            min_length: Minimum character length (default 1).
            max_length: Optional maximum character length.
            router:     LLMRouter for retry calls.
            messages:   Original messages for retry context.
            force_local: Route repair calls to local LLM.
        """
        current_response = response
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            content = current_response.content.strip()

            if len(content) < min_length:
                last_error = (
                    f"Response too short: {len(content)} chars, "
                    f"minimum {min_length}"
                )
            elif max_length is not None and len(content) > max_length:
                last_error = (
                    f"Response too long: {len(content)} chars, "
                    f"maximum {max_length}"
                )
            else:
                log.debug("validator.text_ok", attempt=attempt, length=len(content))
                return ValidationResult(
                    success=True,
                    value=content,
                    attempts=attempt,
                    response=current_response,
                )

            log.warning(
                "validator.text_failed",
                attempt=attempt,
                error=last_error,
            )

            if attempt < self.max_retries and router is not None and messages is not None:
                repair_messages = _build_repair_prompt(
                    original_messages=messages,
                    bad_output=current_response.content,
                    error=last_error,
                    expected_format=f"text between {min_length} and {max_length or 'unlimited'} characters",
                )
                try:
                    current_response = await router.complete(
                        repair_messages,
                        force_local=force_local,
                    )
                except Exception as exc:
                    last_error = f"Repair call failed: {exc}"
                    break
            else:
                break

        return ValidationResult(
            success=False,
            error=last_error,
            attempts=attempt, # type: ignore
            response=current_response,
        )

    # ── Passthrough ───────────────────────────────────────────────────────────

    def passthrough(self, response: LLMResponse) -> ValidationResult:
        """No validation — return the content as-is."""
        return ValidationResult(
            success=True,
            value=response.content,
            attempts=1,
            response=response,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences that LLMs often wrap JSON in.
    Handles ```json ... ``` and ``` ... ``` variants.
    """
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _build_repair_prompt(
    *,
    original_messages: list[Message],
    bad_output: str,
    error: str,
    expected_format: str,
) -> list[Message]:
    """
    Build a targeted repair prompt showing the LLM its bad output and the error.

    The repair message is appended to the original conversation so the model
    has full context about what it was trying to do.
    """
    repair_content = (
        f"Your previous response was invalid.\n\n"
        f"Error: {error}\n\n"
        f"Your output was:\n{bad_output}\n\n"
        f"Please respond again with {expected_format} only. "
        f"No explanation, no markdown, no preamble."
    )

    # Append bad output as assistant turn, then the repair request as user turn
    repair_messages = list(original_messages) + [
        Message(role="assistant", content=bad_output),
        Message(role="user",      content=repair_content),
    ]
    return repair_messages