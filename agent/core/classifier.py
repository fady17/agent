"""
agent/core/classifier.py

Task classifier — the first step on every orchestrator turn.

Takes raw user input and returns a validated TaskIntent that tells
the orchestrator what domain, action, project, and urgency the user
intends. Every routing and dispatch decision flows from this.

Pipeline per classification call:
    1. Build CLASSIFY prompt via PromptEngine (includes few-shot examples)
    2. Route through LLMRouter (local for classification — it's cheap)
    3. Validate JSON output via ResponseValidator (retries on bad JSON)
    4. Return TaskIntent — a Pydantic model the orchestrator can trust

Design decisions:
    - Classification always uses force_local=True by default.
      It's a small, structured task that local LLMs handle well,
      and it fires on every single turn — cloud cost would add up fast.
    - The validator handles bad JSON automatically — the orchestrator
      never sees a raw string from this module.
    - Unknown domain falls back gracefully — TaskIntent.domain == "unknown"
      triggers a clarification flow in the orchestrator rather than an error.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent.core.logger import get_logger
from agent.llm.prompt_engine import PromptEngine, TemplateID
from agent.llm.router import LLMRouter
from agent.llm.validator import ResponseValidator

log = get_logger(__name__)


# ── Domain / Urgency enums ────────────────────────────────────────────────────

class Domain(StrEnum):
    CODE          = "code"
    DESIGN        = "design"
    SHELL         = "shell"
    SEARCH        = "search"
    COMMUNICATION = "communication"
    MEMORY        = "memory"
    UNKNOWN       = "unknown"


class Urgency(StrEnum):
    HIGH   = "high"
    NORMAL = "normal"
    LOW    = "low"


# ── TaskIntent ────────────────────────────────────────────────────────────────

class TaskIntent(BaseModel):
    """
    Structured result of classifying a user input.

    Produced by TaskClassifier and consumed by the Orchestrator
    to decide which sub-agent handles the request.

    Fields:
        domain:   Broad category of work (code, design, shell, …)
        action:   Short verb phrase describing the specific action
        project:  Project context inferred from input (may be None)
        urgency:  How time-sensitive the task appears
        raw_input: The original user string (preserved for logging/audit)
    """

    domain:    Domain
    action:    str
    project:   str | None = None
    urgency:   Urgency = Urgency.NORMAL
    raw_input: str = Field(default="", exclude=True)

    @field_validator("action")
    @classmethod
    def action_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("action must not be empty")
        return v

    @field_validator("project", mode="before")
    @classmethod
    def normalise_project(cls, v: Any) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return str(v).strip().lower()

    @field_validator("domain", mode="before")
    @classmethod
    def coerce_unknown_domain(cls, v: Any) -> Domain:
        """Coerce unrecognised domain strings to UNKNOWN instead of raising."""
        try:
            return Domain(str(v).lower())
        except ValueError:
            return Domain.UNKNOWN

    @field_validator("urgency", mode="before")
    @classmethod
    def coerce_unknown_urgency(cls, v: Any) -> Urgency:
        """Coerce unrecognised urgency strings to NORMAL instead of raising."""
        try:
            return Urgency(str(v).lower())
        except ValueError:
            return Urgency.NORMAL

    @property
    def is_known(self) -> bool:
        return self.domain != Domain.UNKNOWN

    @property
    def needs_clarification(self) -> bool:
        return self.domain == Domain.UNKNOWN

    def summary(self) -> str:
        project_tag = f" [{self.project}]" if self.project else ""
        return f"{self.domain}/{self.action}{project_tag} ({self.urgency})"


# ── TaskClassifier ────────────────────────────────────────────────────────────

class TaskClassifier:
    """
    Classifies raw user input into a TaskIntent.

    Stateless — safe to share across the session.
    All dependencies (router, engine, validator) are injected
    so the classifier is fully testable without live LLM calls.
    """

    def __init__(
        self,
        router: LLMRouter,
        engine: PromptEngine | None = None,
        validator: ResponseValidator | None = None,
    ) -> None:
        self._router    = router
        self._engine    = engine or PromptEngine()
        self._validator = validator or ResponseValidator(max_retries=3)

    # ── Public API ────────────────────────────────────────────────────────────

    async def classify(
        self,
        user_input: str,
        *,
        force_local: bool = True,
    ) -> TaskIntent:
        """
        Classify user input and return a TaskIntent.

        Args:
            user_input:   The raw string from the user.
            force_local:  Route to local LLM (default True — classification
                          is cheap and fires on every turn).

        Returns:
            TaskIntent — always returns a valid instance.
            If classification fails after all retries, returns a fallback
            TaskIntent with domain=UNKNOWN rather than raising.
        """
        if not user_input.strip():
            log.debug("classifier.empty_input")
            return _fallback_intent(user_input, reason="empty input")

        messages = self._engine.build(
            TemplateID.CLASSIFY,
            variables={"input": user_input},
        )

        log.debug("classifier.classify_start", input_preview=user_input[:80])

        try:
            response = await self._router.complete(
                messages,
                force_local=force_local,
                temperature=0.0,   # deterministic — classification is not creative
                max_tokens=128,    # JSON intent is small
            )
        except Exception as exc:
            log.error("classifier.llm_failed", error=str(exc))
            return _fallback_intent(user_input, reason=f"LLM call failed: {exc}")

        result = await self._validator.validate_json(
            response,
            router=self._router,
            messages=messages,
            pydantic_model=TaskIntent,
            force_local=force_local,
        )

        if result.success:
            intent: TaskIntent = result.value
            # Attach the raw input for audit trail
            intent = intent.model_copy(update={"raw_input": user_input})
            log.info(
                "classifier.classified",
                domain=intent.domain,
                action=intent.action,
                project=intent.project,
                urgency=intent.urgency,
                attempts=result.attempts,
            )
            return intent

        log.warning(
            "classifier.failed",
            error=result.error,
            attempts=result.attempts,
            input_preview=user_input[:80],
        )
        return _fallback_intent(user_input, reason=result.error)


# ── Fallback ──────────────────────────────────────────────────────────────────

def _fallback_intent(raw_input: str, reason: str = "") -> TaskIntent:
    """
    Return a safe fallback when classification fails.
    The orchestrator checks intent.needs_clarification and handles accordingly.
    """
    log.debug("classifier.fallback", reason=reason)
    return TaskIntent(
        domain=Domain.UNKNOWN,
        action="unknown",
        project=None,
        urgency=Urgency.NORMAL,
        raw_input=raw_input,
    )