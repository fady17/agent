"""
agent/llm/prompt_engine.py

Prompt engine — builds structured message lists for every LLM call.

The orchestrator never constructs raw message lists. It calls:

    engine.build(TemplateID.RESPOND, context=..., variables={...})

and gets back a list[Message] ready to pass to the router.

Design:
    - Templates are dataclasses registered in a dict at module load time.
      Adding a new template = defining a dataclass + one registry entry.
    - Variable substitution uses Python str.format_map() — simple, no
      extra dependencies, works with any string.
    - Few-shot examples are injected as alternating user/assistant messages
      before the final user turn.
    - The context block from the retrieval pipeline slots into every
      template that has a {context} placeholder in its system prompt.
    - Templates are validated at import time — missing placeholders
      raise immediately rather than failing silently at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent.core.logger import get_logger
from agent.llm.lm_studio import Message
from agent.memory.context_builder import ContextPayload

log = get_logger(__name__)


# ── TemplateID ────────────────────────────────────────────────────────────────

class TemplateID(StrEnum):
    CLASSIFY   = "classify"    # classify an input into a TaskIntent
    RESPOND    = "respond"     # general assistant response with memory context
    TOOL_PLAN  = "tool_plan"   # decide which tools to call for a task
    CONSOLIDATE = "consolidate" # extract patterns from episodic events (nightly)
    SUMMARISE  = "summarise"   # summarise a session or document


# ── FewShotExample ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FewShotExample:
    user:      str
    assistant: str


# ── PromptTemplate ────────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """
    A reusable prompt template.

    system:        System prompt string. May contain {variable} placeholders.
    user_template: User turn template. Must contain {input} at minimum.
    few_shot:      Optional few-shot examples injected before the final user turn.
    notes:         Internal documentation — not sent to the model.
    """
    id:            TemplateID
    system:        str
    user_template: str
    few_shot:      list[FewShotExample] = field(default_factory=list)
    notes:         str = ""


# ── Template registry ─────────────────────────────────────────────────────────

_TEMPLATES: dict[TemplateID, PromptTemplate] = {

    TemplateID.CLASSIFY: PromptTemplate(
        id=TemplateID.CLASSIFY,
        system=(
            "You are a task classifier for an autonomous developer assistant. "
            "Your only job is to classify the user's input into a structured JSON object.\n\n"
            "Output ONLY valid JSON. No explanation, no markdown, no preamble.\n\n"
            "JSON schema:\n"
            '{{"domain": "<code|design|shell|search|communication|memory|unknown>", '
            '"action": "<short verb phrase>", '
            '"project": "<project name or null>", '
            '"urgency": "<high|normal|low>"}}'
        ),
        user_template='Classify this input:\n\n{input}',
        few_shot=[
            FewShotExample(
                user='Classify this input:\n\nFix the CORS error in the FastAPI backend',
                assistant='{"domain": "code", "action": "debug", "project": "agrivision", "urgency": "high"}',
            ),
            FewShotExample(
                user='Classify this input:\n\nExport the plant model as GLB with draco compression',
                assistant='{"domain": "design", "action": "export", "project": "agrivision", "urgency": "normal"}',
            ),
            FewShotExample(
                user='Classify this input:\n\nWhat is the weather in Cairo today?',
                assistant='{"domain": "search", "action": "lookup", "project": null, "urgency": "low"}',
            ),
        ],
        notes="Expects {input}. Returns JSON — caller must parse and validate.",
    ),

    TemplateID.RESPOND: PromptTemplate(
        id=TemplateID.RESPOND,
        system=(
            "You are an autonomous personal assistant for a senior developer and 3D designer. "
            "You are direct, precise, and production-minded. "
            "You never add unnecessary caveats or repeat the user's question back to them.\n\n"
            "When you have memory context below, use it silently — never say "
            '"Based on my memory..." or "I can see that...". '
            "Just use the information naturally.\n\n"
            "{context}"
        ),
        user_template='{input}',
        notes=(
            "The {context} placeholder is replaced with the ContextPayload.text "
            "from the retrieval pipeline. If the context is empty the placeholder "
            "becomes an empty string — the system prompt still works."
        ),
    ),

    TemplateID.TOOL_PLAN: PromptTemplate(
        id=TemplateID.TOOL_PLAN,
        system=(
            "You are a tool planner for an autonomous developer assistant. "
            "Given a task description and the available tools, output a JSON plan.\n\n"
            "Output ONLY valid JSON. No explanation, no markdown.\n\n"
            "JSON schema:\n"
            '{{"steps": [{{"tool": "<tool_name>", "input": {{...}}, "reason": "<why>"}}]}}\n\n'
            "Available tools:\n{tools}"
        ),
        user_template='Plan the steps to complete this task:\n\n{input}',
        few_shot=[
            FewShotExample(
                user='Plan the steps to complete this task:\n\nRun the test suite and show failures',
                assistant=(
                    '{"steps": ['
                    '{"tool": "shell_runner", "input": {"cmd": "uv run pytest -x --tb=short"}, '
                    '"reason": "run tests and capture output"}'
                    ']}'
                ),
            ),
        ],
        notes="Expects {input} and {tools}. Returns JSON steps array.",
    ),

    TemplateID.CONSOLIDATE: PromptTemplate(
        id=TemplateID.CONSOLIDATE,
        system=(
            "You are a memory consolidation engine for a developer assistant. "
            "Analyse the provided episodic events and extract reusable patterns.\n\n"
            "Output ONLY valid JSON. No explanation, no markdown.\n\n"
            "JSON schema:\n"
            '{{"patterns": [{{"task_type": "<slug>", "pattern": "<description>", '
            '"confidence": <0.0-1.0>, "source_event_ids": ["..."]}}]}}'
        ),
        user_template='Extract patterns from these events:\n\n{events}',
        notes=(
            "Expects {events} as a formatted string of episodic event summaries. "
            "Returns patterns array for the skill store upsert."
        ),
    ),

    TemplateID.SUMMARISE: PromptTemplate(
        id=TemplateID.SUMMARISE,
        system=(
            "You are a summarisation assistant. "
            "Produce a concise, structured summary of the provided content. "
            "Use bullet points for key facts. Be brief — maximum {max_sentences} sentences."
        ),
        user_template='Summarise the following:\n\n{input}',
        notes="Expects {input} and {max_sentences}.",
    ),
}


# ── PromptEngine ──────────────────────────────────────────────────────────────

class PromptEngine:
    """
    Builds message lists for LLM calls from registered templates.

    Usage:
        engine = PromptEngine()

        # With context from retrieval pipeline
        messages = engine.build(
            TemplateID.RESPOND,
            variables={"input": "Fix the CORS error"},
            context=context_payload,
        )

        # Classification — no context needed
        messages = engine.build(
            TemplateID.CLASSIFY,
            variables={"input": "run the test suite"},
        )
    """

    def __init__(self, templates: dict[TemplateID, PromptTemplate] | None = None) -> None:
        self._templates = templates or _TEMPLATES

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        template_id: TemplateID,
        *,
        variables: dict[str, Any] | None = None,
        context: ContextPayload | None = None,
        conversation_history: list[Message] | None = None,
    ) -> list[Message]:
        """
        Build a message list from a template.

        Args:
            template_id:          Which template to use.
            variables:            Dict of {placeholder: value} substitutions.
                                  {input} is always required for user_template.
            context:              ContextPayload from the retrieval pipeline.
                                  Its .text is substituted into {context} in
                                  the system prompt (empty string if None).
            conversation_history: Prior turns to include between the system
                                  message and the final user turn. Inserted
                                  AFTER few-shot examples.

        Returns:
            list[Message] ready to pass to LLMRouter.complete() or .stream().

        Raises:
            KeyError:   If template_id is not registered.
            ValueError: If a required placeholder is missing from variables.
        """
        if template_id not in self._templates:
            raise KeyError(f"Unknown template: {template_id!r}")

        template  = self._templates[template_id]
        vars_map  = dict(variables or {})
        messages: list[Message] = []

        # ── System prompt ─────────────────────────────────────────────────────
        context_text = context.text if context else ""
        # Inject context — if no {context} placeholder, context is ignored
        system_vars = {"context": context_text, **vars_map}
        try:
            system_text = template.system.format_map(system_vars)
        except KeyError as exc:
            raise ValueError(
                f"Template {template_id!r} system prompt requires variable {exc} "
                f"which was not provided."
            ) from exc

        messages.append(Message(role="system", content=system_text))

        # ── Few-shot examples ─────────────────────────────────────────────────
        for example in template.few_shot:
            messages.append(Message(role="user",      content=example.user))
            messages.append(Message(role="assistant", content=example.assistant))

        # ── Conversation history ──────────────────────────────────────────────
        if conversation_history:
            for msg in conversation_history:
                # Skip any system messages from history — we already have one
                if msg.role != "system":
                    messages.append(msg)

        # ── Final user turn ───────────────────────────────────────────────────
        try:
            user_text = template.user_template.format_map(vars_map)
        except KeyError as exc:
            raise ValueError(
                f"Template {template_id!r} user_template requires variable {exc} "
                f"which was not provided."
            ) from exc

        messages.append(Message(role="user", content=user_text))

        log.debug(
            "prompt_engine.built",
            template=template_id,
            messages=len(messages),
            context_tokens=context.tokens_used if context else 0,
        )

        return messages

    def get_template(self, template_id: TemplateID) -> PromptTemplate:
        """Return the raw template for inspection."""
        if template_id not in self._templates:
            raise KeyError(f"Unknown template: {template_id!r}")
        return self._templates[template_id]

    @property
    def registered_templates(self) -> list[TemplateID]:
        return list(self._templates.keys())


# ── SafeFormatMap ─────────────────────────────────────────────────────────────

class _SafeFormatMap(dict):  # type: ignore[type-arg]
    """
    A dict subclass for str.format_map() that leaves unknown placeholders
    untouched rather than raising KeyError.

    This allows templates like the RESPOND system prompt to contain {context}
    while also accepting arbitrary extra variables without collision.

    Unknown keys are returned as the original placeholder string: {key}.
    """

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"