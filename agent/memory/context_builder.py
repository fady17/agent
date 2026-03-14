"""
agent/memory/context_builder.py

Formats a ContextBlock from the retrieval pipeline into a structured
string ready for injection into an LLM prompt.

Responsibilities:
    - Enforce a token budget — never exceed max_tokens
    - Priority order: skills > recent events > graph nodes
      (skills are highest signal — they encode learned preferences directly)
    - Produce deterministic output — same ContextBlock always gives same string
    - Remain LLM-agnostic — outputs plain text, no model-specific tokens

Token estimation:
    We use a simple heuristic: 1 token ≈ 4 characters.
    This is conservative (real tokenisers vary) but safe — we'd rather
    inject slightly less context than overflow the window.

Output format (injected into system prompt):

    ## Memory context

    ### Your learned patterns
    [task_type] (confidence: 0.91)
    Pattern: Use FastAPI with async def and Pydantic models

    ### Recent activity
    [2026-03-14 10:30 UTC] [code.write] agrivision
    Wrote FastAPI endpoint for plant disease detection

    ### Related concepts
    - [project] AgriVision — plant disease detection app
    - [library] FastAPI — async Python web framework

Usage:
    from agent.memory.context_builder import ContextBuilder
    builder = ContextBuilder(max_tokens=1500)
    payload = builder.build(context_block)
    # payload.text → inject into system prompt
    # payload.tokens_used → log for cost tracking
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.core.logger import get_logger
from agent.memory.retrieval import ContextBlock

log = get_logger(__name__)

# Conservative token estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4

# Section headers (counted toward budget)
_HEADER_MAIN    = "## Memory context\n"
_HEADER_SKILLS  = "\n### Your learned patterns\n"
_HEADER_EVENTS  = "\n### Recent activity\n"
_HEADER_NODES   = "\n### Related concepts\n"


# ── ContextPayload ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContextPayload:
    """
    Output of the context builder — ready to inject into a system prompt.

    text:          The formatted context string.
    tokens_used:   Estimated token count of text.
    items_included: How many items from each section survived the budget.
    was_truncated: True if any items were dropped due to token budget.
    """
    text: str
    tokens_used: int
    items_included: dict[str, int]
    was_truncated: bool

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


# ── ContextBuilder ────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Formats a ContextBlock into a token-budgeted context string.

    Args:
        max_tokens: Hard budget. Builder stops adding items once this is hit.
                    Default 1500 leaves comfortable room in a 4k context window
                    for the system prompt, user message, and response.
    """

    def __init__(self, max_tokens: int = 1500) -> None:
        if max_tokens < 100:
            raise ValueError(f"max_tokens must be >= 100, got {max_tokens}")
        self.max_tokens = max_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, block: ContextBlock) -> ContextPayload:
        """
        Format a ContextBlock into a ContextPayload.
        Returns an empty payload if the block has no content.
        """
        if block.is_empty:
            return ContextPayload(
                text="",
                tokens_used=0,
                items_included={"skills": 0, "events": 0, "nodes": 0},
                was_truncated=False,
            )

        budget = self.max_tokens
        parts: list[str] = []
        counts: dict[str, int] = {"skills": 0, "events": 0, "nodes": 0}
        any_dropped = False

        # Reserve tokens for the main header
        main_header_cost = _estimate_tokens(_HEADER_MAIN)
        if main_header_cost > budget:
            return _empty_payload()
        parts.append(_HEADER_MAIN)
        budget -= main_header_cost

        # ── Priority 1: Skills ────────────────────────────────────────────────
        if block.relevant_skills:
            header_cost = _estimate_tokens(_HEADER_SKILLS)
            if header_cost <= budget:
                skill_lines: list[str] = []
                for skill in block.relevant_skills:
                    line = _format_skill(skill)
                    cost = _estimate_tokens(line)
                    if cost <= budget - header_cost:
                        skill_lines.append(line)
                        budget -= cost
                        counts["skills"] += 1
                    else:
                        any_dropped = True
                        break

                if skill_lines:
                    parts.append(_HEADER_SKILLS)
                    budget -= header_cost
                    parts.extend(skill_lines)

        # ── Priority 2: Recent events ─────────────────────────────────────────
        if block.recent_events:
            header_cost = _estimate_tokens(_HEADER_EVENTS)
            if header_cost <= budget:
                event_lines: list[str] = []
                for event in block.recent_events:
                    line = _format_event(event)
                    cost = _estimate_tokens(line)
                    if cost <= budget - header_cost:
                        event_lines.append(line)
                        budget -= cost
                        counts["events"] += 1
                    else:
                        any_dropped = True
                        break

                if event_lines:
                    parts.append(_HEADER_EVENTS)
                    budget -= header_cost
                    parts.extend(event_lines)

        # ── Priority 3: Graph nodes ───────────────────────────────────────────
        if block.relevant_nodes:
            header_cost = _estimate_tokens(_HEADER_NODES)
            if header_cost <= budget:
                node_lines: list[str] = []
                for node, _dist in block.relevant_nodes:
                    line = _format_node(node)
                    cost = _estimate_tokens(line)
                    if cost <= budget - header_cost:
                        node_lines.append(line)
                        budget -= cost
                        counts["nodes"] += 1
                    else:
                        any_dropped = True
                        break

                if node_lines:
                    parts.append(_HEADER_NODES)
                    budget -= header_cost
                    parts.extend(node_lines)

        text = "".join(parts).rstrip()
        tokens_used = self.max_tokens - budget

        log.debug(
            "context_builder.built",
            query=block.query[:60],
            tokens_used=tokens_used,
            max_tokens=self.max_tokens,
            was_truncated=any_dropped,
            **counts,
        )

        return ContextPayload(
            text=text,
            tokens_used=tokens_used,
            items_included=counts,
            was_truncated=any_dropped,
        )


# ── Formatters ────────────────────────────────────────────────────────────────

def _format_skill(skill: "SkillRecord") -> str:  # noqa: F821
    lines = [
        f"[{skill.task_type}] (confidence: {skill.confidence:.2f})\n",
        f"Pattern: {skill.pattern}\n",
        "\n",
    ]
    return "".join(lines)


def _format_event(event: "EpisodicEvent") -> str:  # noqa: F821
    ts = event.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    project_tag = f" {event.project}" if event.project else ""
    lines = [
        f"[{ts}] [{event.event_type}]{project_tag}\n",
        f"{event.summary}\n",
        "\n",
    ]
    return "".join(lines)


def _format_node(node: "GraphNode") -> str:  # noqa: F821
    description = node.attributes.get("description", "")
    desc_part = f" — {description}" if description else ""
    return f"- [{node.node_type}] {node.label}{desc_part}\n"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Estimate token count using character heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _empty_payload() -> ContextPayload:
    return ContextPayload(
        text="",
        tokens_used=0,
        items_included={"skills": 0, "events": 0, "nodes": 0},
        was_truncated=False,
    )


# ── Type imports for formatters (avoid circular at module level) ───────────────
# These are only used inside functions — imported lazily at call time via
# the annotations. Python resolves them correctly with from __future__ annotations.
from agent.memory.episodic import EpisodicEvent  # noqa: E402
from agent.memory.graph import GraphNode          # noqa: E402
from agent.memory.skills import SkillRecord       # noqa: E402