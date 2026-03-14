"""
tests/test_prompt_engine.py

Tests for PromptEngine — template building, variable substitution,
few-shot injection, context slot, conversation history, and error paths.
"""

from __future__ import annotations

import pytest

from agent.llm.lm_studio import Message
from agent.llm.prompt_engine import (
    FewShotExample,
    PromptEngine,
    PromptTemplate,
    TemplateID,
    _SafeFormatMap,
)
from agent.memory.context_builder import ContextPayload


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_engine() -> PromptEngine:
    return PromptEngine()


def make_context(text: str = "## Memory context\n- FastAPI preferred", tokens: int = 10) -> ContextPayload:
    return ContextPayload(
        text=text,
        tokens_used=tokens,
        items_included={"skills": 1, "events": 0, "nodes": 0},
        was_truncated=False,
    )


def make_custom_template(
    template_id: TemplateID = TemplateID.CLASSIFY,
    system: str = "System: {var1}",
    user_template: str = "User: {input}",
    few_shot: list[FewShotExample] | None = None,
) -> dict[TemplateID, PromptTemplate]:
    return {
        template_id: PromptTemplate(
            id=template_id,
            system=system,
            user_template=user_template,
            few_shot=few_shot or [],
        )
    }


# ── _SafeFormatMap ────────────────────────────────────────────────────────────

def test_safe_format_map_known_key() -> None:
    result = "Hello {name}".format_map(_SafeFormatMap({"name": "Fady"}))
    assert result == "Hello Fady"


def test_safe_format_map_unknown_key_preserved() -> None:
    result = "Hello {name} and {unknown}".format_map(_SafeFormatMap({"name": "Fady"}))
    assert result == "Hello Fady and {unknown}"


def test_safe_format_map_all_unknown() -> None:
    result = "{a} {b} {c}".format_map(_SafeFormatMap({}))
    assert result == "{a} {b} {c}"


# ── PromptEngine registration ─────────────────────────────────────────────────

def test_all_builtin_templates_registered() -> None:
    engine = make_engine()
    for tid in TemplateID:
        assert tid in engine.registered_templates


def test_get_template_returns_correct_template() -> None:
    engine = make_engine()
    template = engine.get_template(TemplateID.CLASSIFY)
    assert template.id == TemplateID.CLASSIFY


def test_get_template_unknown_raises() -> None:
    engine = make_engine()
    with pytest.raises(KeyError, match="Unknown template"):
        engine.get_template("nonexistent")  # type: ignore[arg-type]


def test_build_unknown_template_raises() -> None:
    engine = make_engine()
    with pytest.raises(KeyError, match="Unknown template"):
        engine.build("nonexistent", variables={"input": "test"})  # type: ignore[arg-type]


# ── Message structure ─────────────────────────────────────────────────────────

def test_build_returns_message_list() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.RESPOND, variables={"input": "hello"})
    assert isinstance(messages, list)
    assert all(isinstance(m, Message) for m in messages)


def test_build_first_message_is_system() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.RESPOND, variables={"input": "hi"})
    assert messages[0].role == "system"


def test_build_last_message_is_user() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.RESPOND, variables={"input": "hi"})
    assert messages[-1].role == "user"


def test_build_user_message_contains_input() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.RESPOND, variables={"input": "Fix the CORS error"})
    assert "Fix the CORS error" in messages[-1].content


# ── CLASSIFY template ─────────────────────────────────────────────────────────

def test_classify_has_few_shot_examples() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.CLASSIFY, variables={"input": "run tests"})
    # system + N×(user+assistant) few-shot + final user
    roles = [m.role for m in messages]
    # Should have interleaved user/assistant pairs between system and final user
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert "assistant" in roles


def test_classify_few_shot_order() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.CLASSIFY, variables={"input": "test"})
    # Verify user/assistant alternation in few-shot section
    few_shot_section = messages[1:-1]
    for i in range(0, len(few_shot_section) - 1, 2):
        assert few_shot_section[i].role == "user"
        assert few_shot_section[i + 1].role == "assistant"


def test_classify_final_user_contains_input() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.CLASSIFY, variables={"input": "build the docker image"})
    assert "build the docker image" in messages[-1].content


# ── RESPOND template ──────────────────────────────────────────────────────────

def test_respond_no_few_shot() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.RESPOND, variables={"input": "hi"})
    # Without history: system + user only
    assert len(messages) == 2


def test_respond_context_injected_into_system() -> None:
    engine = make_engine()
    ctx = make_context("## Memory context\nFastAPI preferred")
    messages = engine.build(TemplateID.RESPOND, variables={"input": "hi"}, context=ctx)
    assert "FastAPI preferred" in messages[0].content


def test_respond_empty_context_not_in_system() -> None:
    engine = make_engine()
    ctx = ContextPayload(
        text="",
        tokens_used=0,
        items_included={},
        was_truncated=False,
    )
    messages = engine.build(TemplateID.RESPOND, variables={"input": "hi"}, context=ctx)
    # The {context} placeholder replaced with empty string — no memory section
    assert "Memory context" not in messages[0].content


def test_respond_no_context_arg_produces_empty_placeholder() -> None:
    engine = make_engine()
    messages = engine.build(TemplateID.RESPOND, variables={"input": "hi"}, context=None)
    assert "Memory context" not in messages[0].content


# ── TOOL_PLAN template ────────────────────────────────────────────────────────

def test_tool_plan_requires_tools_variable() -> None:
    engine = make_engine()
    messages = engine.build(
        TemplateID.TOOL_PLAN,
        variables={"input": "run tests", "tools": "shell_runner, file_reader"},
    )
    assert "shell_runner" in messages[0].content


def test_tool_plan_has_few_shot() -> None:
    engine = make_engine()
    messages = engine.build(
        TemplateID.TOOL_PLAN,
        variables={"input": "run tests", "tools": "shell_runner"},
    )
    roles = [m.role for m in messages]
    assert "assistant" in roles


# ── CONSOLIDATE template ──────────────────────────────────────────────────────

def test_consolidate_substitutes_events() -> None:
    engine = make_engine()
    events_text = "- Wrote FastAPI endpoint\n- Ran pytest\n"
    messages = engine.build(
        TemplateID.CONSOLIDATE,
        variables={"events": events_text},
    )
    assert "Wrote FastAPI endpoint" in messages[-1].content


# ── SUMMARISE template ────────────────────────────────────────────────────────

def test_summarise_substitutes_max_sentences() -> None:
    engine = make_engine()
    messages = engine.build(
        TemplateID.SUMMARISE,
        variables={"input": "some long text", "max_sentences": "3"},
    )
    assert "3" in messages[0].content


# ── Variable substitution errors ──────────────────────────────────────────────

def test_missing_input_variable_raises() -> None:
    engine = make_engine()
    # If the system prompt has variables that aren't provided, 
    # it will raise for those BEFORE it hits the 'input' check.
    # Ensure your variables dict provides what the system prompt needs.
    with pytest.raises(ValueError, match="input"):
        # Assuming the default CLASSIFY/RESPOND templates require 'input'
        engine.build(TemplateID.RESPOND, variables={})
        

def test_missing_system_variable_raises() -> None:
    """Custom template with required system var."""
    engine = PromptEngine(templates=make_custom_template(
        system="Hello {required_var}",
        user_template="Input: {input}",
    ))
    
    # We now expect a ValueError for 'required_var' because system prompt 
    # is evaluated before user_template
    with pytest.raises(ValueError, match="required_var"):
        engine.build(TemplateID.CLASSIFY, variables={"input": "test_input"})

# ── Conversation history ──────────────────────────────────────────────────────

def test_conversation_history_inserted_before_final_user() -> None:
    engine = make_engine()
    history = [
        Message(role="user",      content="previous question"),
        Message(role="assistant", content="previous answer"),
    ]
    messages = engine.build(
        TemplateID.RESPOND,
        variables={"input": "follow-up question"},
        conversation_history=history,
    )
    # system, history[0], history[1], final_user
    assert messages[1].content == "previous question"
    assert messages[2].content == "previous answer"
    assert messages[-1].content == "follow-up question"


def test_conversation_history_system_messages_skipped() -> None:
    engine = make_engine()
    history = [
        Message(role="system",    content="old system"),
        Message(role="user",      content="user turn"),
        Message(role="assistant", content="assistant turn"),
    ]
    messages = engine.build(
        TemplateID.RESPOND,
        variables={"input": "new question"},
        conversation_history=history,
    )
    # Old system message should not appear — we already have our own
    system_messages = [m for m in messages if m.role == "system"]
    assert len(system_messages) == 1


def test_history_with_few_shot_ordering() -> None:
    """Few-shot appears before history which appears before final user."""
    engine = make_engine()
    history = [Message(role="user", content="history turn")]
    messages = engine.build(
        TemplateID.CLASSIFY,
        variables={"input": "final"},
        conversation_history=history,
    )
    # system, few-shot pairs..., history turn, final user
    assert messages[0].role == "system"
    assert messages[-1].content == "Classify this input:\n\nfinal"
    # history should be second-to-last before final user
    assert messages[-2].content == "history turn"


# ── No variables ─────────────────────────────────────────────────────────────

def test_build_with_none_variables_uses_empty_dict() -> None:
    """Passing variables=None should not crash — treated as empty."""
    engine = PromptEngine(templates=make_custom_template(
        system="Static system prompt",
        user_template="Static user: {input}",
    ))
    # Should raise ValueError about missing {input}, not a crash
    with pytest.raises(ValueError):
        engine.build(TemplateID.CLASSIFY, variables=None)


# ── Custom template registry ──────────────────────────────────────────────────

def test_custom_template_registry() -> None:
    custom = {
        TemplateID.CLASSIFY: PromptTemplate(
            id=TemplateID.CLASSIFY,
            system="Custom system",
            user_template="Custom: {input}",
        )
    }
    engine = PromptEngine(templates=custom)
    messages = engine.build(TemplateID.CLASSIFY, variables={"input": "test"})
    assert messages[0].content == "Custom system"
    assert "Custom: test" in messages[-1].content