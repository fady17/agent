"""
tests/test_config.py

Tests for the config system.
These run against a controlled env — no real .env file needed.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.core.config import AgentConfig

# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_config(**overrides: object) -> AgentConfig:
    """Build a config from minimal valid values + any overrides."""
    defaults = {
        "lm_studio_base_url": "http://127.0.0.1:1234/v1",
        "lm_studio_chat_model": "test-model",
        "anthropic_api_key": "sk-test",
        "anthropic_model": "claude-sonnet-4-6",
        "lm_studio_embedding_model": "text-embedding-qwen3-embedding-4b",
        "embedding_dim": 2560,
        "agent_memory_root": "/tmp/test-agent",
        "log_level": "INFO",
    }
    return AgentConfig(**{**defaults, **overrides})  # type: ignore[arg-type]


# ── Basic loading ─────────────────────────────────────────────────────────────

def test_config_loads_with_defaults() -> None:
    cfg = make_config()
    assert cfg.lm_studio_base_url == "http://127.0.0.1:1234/v1"
    assert cfg.embedding_dim == 2560
    assert cfg.llm_cloud_threshold_tokens == 2000


def test_config_is_typed() -> None:
    cfg = make_config()
    assert isinstance(cfg.agent_memory_root, Path)
    assert isinstance(cfg.embedding_dim, int)
    assert isinstance(cfg.blender_socket_port, int)


# ── Path resolution ───────────────────────────────────────────────────────────

def test_memory_root_resolves_tilde() -> None:
    cfg = make_config(agent_memory_root="~/.agent")
    assert not str(cfg.agent_memory_root).startswith("~")
    assert cfg.agent_memory_root.is_absolute()


def test_memory_root_resolves_absolute() -> None:
    cfg = make_config(agent_memory_root="/tmp/test-agent")
    # Call .resolve() on the expected path so it matches macOS's /private/tmp
    assert cfg.agent_memory_root == Path("/tmp/test-agent").resolve()


# ── Derived paths ─────────────────────────────────────────────────────────────

def test_derived_paths_are_under_memory_root() -> None:
    cfg = make_config(agent_memory_root="/tmp/test-agent")
    # Call .resolve() on the root path for accurate comparison
    root = Path("/tmp/test-agent").resolve()
    assert cfg.episodic_dir == root / "memory" / "episodic"
    assert cfg.semantic_dir == root / "memory" / "semantic"
    assert cfg.skills_dir   == root / "memory" / "skills"
    assert cfg.working_dir  == root / "memory" / "working"
    assert cfg.projects_dir == root / "projects"
    assert cfg.logs_dir     == root / "logs"
    assert cfg.graph_path   == root / "memory" / "semantic" / "graph.json"
    assert cfg.faiss_index_path == root / "memory" / "semantic" / "index.faiss"

# ── Validators ────────────────────────────────────────────────────────────────

def test_log_level_normalised_to_uppercase() -> None:
    cfg = make_config(log_level="debug")
    assert cfg.log_level == "DEBUG"


def test_log_level_invalid_raises() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        make_config(log_level="VERBOSE")


def test_consolidation_hour_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        make_config(consolidation_hour=25)


def test_consolidation_minute_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        make_config(consolidation_minute=61)


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_get_config_returns_same_instance() -> None:
    from agent.core.config import get_config
    # Clear the cache so test isolation is maintained
    get_config.cache_clear()
    a = get_config()
    b = get_config()
    assert a is b