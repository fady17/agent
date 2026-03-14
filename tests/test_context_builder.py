"""
tests/test_context_builder.py

Tests for the ContextBuilder — formatting, token budget enforcement,
priority ordering, and truncation behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.memory.context_builder import (
    ContextBuilder,
    ContextPayload,
    _estimate_tokens,
)
from agent.memory.episodic import EpisodicEvent, EventType
from agent.memory.graph import GraphNode, NodeType
from agent.memory.retrieval import ContextBlock
from agent.memory.skills import SkillRecord


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_skill(
    task_type: str = "create_api_endpoint",
    pattern: str = "Use FastAPI with async def and Pydantic models",
    confidence: float = 0.91,
) -> SkillRecord:
    return SkillRecord(task_type=task_type, pattern=pattern, confidence=confidence)


def make_event(
    summary: str = "Wrote FastAPI endpoint for plant detection",
    project: str | None = "agrivision",
    event_type: EventType = EventType.CODE_WRITE,
) -> EpisodicEvent:
    return EpisodicEvent(
        event_type=event_type,
        summary=summary,
        project=project,
        timestamp=datetime(2026, 3, 14, 10, 30, 0, tzinfo=timezone.utc),
    )


def make_node(
    node_id: str = "fastapi",
    label: str = "FastAPI",
    node_type: NodeType = NodeType.LIBRARY,
    description: str | None = None,
) -> GraphNode:
    attrs = {}
    if description:
        attrs["description"] = description
    return GraphNode(id=node_id, label=label, node_type=node_type, attributes=attrs)


def make_block(
    skills: list[SkillRecord] | None = None,
    events: list[EpisodicEvent] | None = None,
    nodes: list[GraphNode] | None = None,
    project: str | None = "agrivision",
) -> ContextBlock:
    return ContextBlock(
        query="test query",
        project=project,
        relevant_skills=skills or [],
        recent_events=events or [],
        relevant_nodes=[(n, 0.1) for n in (nodes or [])],
    )


# ── ContextBuilder construction ───────────────────────────────────────────────

def test_builder_default_max_tokens() -> None:
    builder = ContextBuilder()
    assert builder.max_tokens == 1500


def test_builder_custom_max_tokens() -> None:
    builder = ContextBuilder(max_tokens=500)
    assert builder.max_tokens == 500


def test_builder_too_small_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ContextBuilder(max_tokens=50)


# ── Empty block ───────────────────────────────────────────────────────────────

def test_empty_block_returns_empty_payload() -> None:
    builder = ContextBuilder()
    block = ContextBlock(query="test", project=None)
    payload = builder.build(block)
    assert payload.is_empty
    assert payload.tokens_used == 0
    assert payload.was_truncated is False


# ── Output structure ──────────────────────────────────────────────────────────

def test_output_contains_main_header() -> None:
    builder = ContextBuilder()
    block = make_block(skills=[make_skill()])
    payload = builder.build(block)
    assert "## Memory context" in payload.text


def test_output_contains_skills_header_when_skills_present() -> None:
    builder = ContextBuilder()
    block = make_block(skills=[make_skill()])
    payload = builder.build(block)
    assert "### Your learned patterns" in payload.text


def test_output_contains_events_header_when_events_present() -> None:
    builder = ContextBuilder()
    block = make_block(events=[make_event()])
    payload = builder.build(block)
    assert "### Recent activity" in payload.text


def test_output_contains_nodes_header_when_nodes_present() -> None:
    builder = ContextBuilder()
    block = make_block(nodes=[make_node()])
    payload = builder.build(block)
    assert "### Related concepts" in payload.text


# ── Skill formatting ──────────────────────────────────────────────────────────

def test_skill_content_in_output() -> None:
    builder = ContextBuilder()
    skill = make_skill(task_type="create_api_endpoint", confidence=0.91)
    block = make_block(skills=[skill])
    payload = builder.build(block)
    assert "create_api_endpoint" in payload.text
    assert "0.91" in payload.text
    assert "FastAPI" in payload.text


def test_skill_confidence_formatted_to_2dp() -> None:
    builder = ContextBuilder()
    skill = make_skill(confidence=0.8)
    block = make_block(skills=[skill])
    payload = builder.build(block)
    assert "0.80" in payload.text


# ── Event formatting ──────────────────────────────────────────────────────────

def test_event_content_in_output() -> None:
    builder = ContextBuilder()
    event = make_event(summary="Exported plant model GLB", project="agrivision")
    block = make_block(events=[event])
    payload = builder.build(block)
    assert "Exported plant model GLB" in payload.text
    assert "agrivision" in payload.text
    assert "2026-03-14" in payload.text


def test_event_without_project_still_formats() -> None:
    builder = ContextBuilder()
    event = make_event(project=None)
    block = make_block(events=[event])
    payload = builder.build(block)
    assert "Wrote FastAPI" in payload.text


# ── Node formatting ───────────────────────────────────────────────────────────

def test_node_content_in_output() -> None:
    builder = ContextBuilder()
    node = make_node(label="FastAPI", node_type=NodeType.LIBRARY)
    block = make_block(nodes=[node])
    payload = builder.build(block)
    assert "FastAPI" in payload.text
    assert "library" in payload.text


def test_node_description_included_when_present() -> None:
    builder = ContextBuilder()
    node = make_node(description="async Python web framework")
    block = make_block(nodes=[node])
    payload = builder.build(block)
    assert "async Python web framework" in payload.text


def test_node_without_description_still_formats() -> None:
    builder = ContextBuilder()
    node = make_node(description=None)
    block = make_block(nodes=[node])
    payload = builder.build(block)
    assert "FastAPI" in payload.text


# ── Priority order ────────────────────────────────────────────────────────────

def test_skills_appear_before_events() -> None:
    builder = ContextBuilder()
    block = make_block(skills=[make_skill()], events=[make_event()])
    payload = builder.build(block)
    skills_pos = payload.text.find("learned patterns")
    events_pos = payload.text.find("Recent activity")
    assert skills_pos < events_pos


def test_events_appear_before_nodes() -> None:
    builder = ContextBuilder()
    block = make_block(events=[make_event()], nodes=[make_node()])
    payload = builder.build(block)
    events_pos = payload.text.find("Recent activity")
    nodes_pos  = payload.text.find("Related concepts")
    assert events_pos < nodes_pos


def test_skills_appear_before_nodes() -> None:
    builder = ContextBuilder()
    block = make_block(skills=[make_skill()], nodes=[make_node()])
    payload = builder.build(block)
    skills_pos = payload.text.find("learned patterns")
    nodes_pos  = payload.text.find("Related concepts")
    assert skills_pos < nodes_pos


# ── Token budget ──────────────────────────────────────────────────────────────

def test_tokens_used_is_positive_when_content_present() -> None:
    builder = ContextBuilder()
    block = make_block(skills=[make_skill()])
    payload = builder.build(block)
    assert payload.tokens_used > 0


def test_tokens_used_does_not_exceed_max() -> None:
    builder = ContextBuilder(max_tokens=1500)
    block = make_block(
        skills=[make_skill(f"task_{i}") for i in range(20)],
        events=[make_event(f"event summary number {i}") for i in range(20)],
        nodes=[make_node(f"node_{i}", f"Node {i}") for i in range(20)],
    )
    payload = builder.build(block)
    assert payload.tokens_used <= builder.max_tokens


def test_truncation_drops_items_over_budget() -> None:
    # Very tight budget — only room for a few items
    builder = ContextBuilder(max_tokens=120)
    block = make_block(
        skills=[make_skill(f"task_{i}", f"pattern text for task number {i}") for i in range(10)],
    )
    payload = builder.build(block)
    assert payload.items_included["skills"] < 10
    assert payload.was_truncated is True


def test_no_truncation_flag_when_all_fits() -> None:
    builder = ContextBuilder(max_tokens=1500)
    block = make_block(
        skills=[make_skill()],
        events=[make_event()],
        nodes=[make_node()],
    )
    payload = builder.build(block)
    assert payload.was_truncated is False


def test_items_included_counts_are_correct() -> None:
    builder = ContextBuilder(max_tokens=1500)
    block = make_block(
        skills=[make_skill("t1"), make_skill("t2")],
        events=[make_event("event one"), make_event("event two")],
        nodes=[make_node("n1", "Node 1"), make_node("n2", "Node 2")],
    )
    payload = builder.build(block)
    assert payload.items_included["skills"] == 2
    assert payload.items_included["events"] == 2
    assert payload.items_included["nodes"] == 2


# ── Token estimator ───────────────────────────────────────────────────────────

def test_estimate_tokens_minimum_one() -> None:
    assert _estimate_tokens("a") == 1
    assert _estimate_tokens("") == 1


def test_estimate_tokens_scales_with_length() -> None:
    short = _estimate_tokens("hi")
    long  = _estimate_tokens("hi" * 100)
    assert long > short


def test_estimate_tokens_four_chars_per_token() -> None:
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("abcdefgh") == 2
    assert _estimate_tokens("a" * 400) == 100


# ── ContextPayload ────────────────────────────────────────────────────────────

def test_payload_is_empty_on_blank_text() -> None:
    payload = ContextPayload(
        text="   ",
        tokens_used=0,
        items_included={},
        was_truncated=False,
    )
    assert payload.is_empty is True


def test_payload_not_empty_with_content() -> None:
    payload = ContextPayload(
        text="## Memory context\n...",
        tokens_used=5,
        items_included={},
        was_truncated=False,
    )
    assert payload.is_empty is False