"""
agent/core/config.py

Single source of truth for all runtime configuration.
Loaded once at startup, importable anywhere as:

    from agent.core.config import cfg

Pydantic-settings reads values from the environment / .env file,
validates types, and resolves paths to absolute form.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── LLM inference ────────────────────────────────────────────────────────
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_chat_model: str = "lmstudio-community/mistral-7b-instruct-v0.3"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # ── Embeddings ───────────────────────────────────────────────────────────
    lm_studio_embedding_model: str = "text-embedding-qwen3-embedding-4b"
    embedding_dim: int = 2560

    # Fallback used only when LM Studio is unreachable
    embedding_fallback_model: str = "all-MiniLM-L6-v2"
    embedding_fallback_dim: int = 384

    # ── Memory ───────────────────────────────────────────────────────────────
    agent_memory_root: Path = Path("~/.agent")

    # ── Routing ──────────────────────────────────────────────────────────────
    llm_cloud_threshold_tokens: int = 2000

    # ── Blender bridge ───────────────────────────────────────────────────────
    blender_socket_host: str = "127.0.0.1"
    blender_socket_port: int = 9999

    # ── Scheduling ───────────────────────────────────────────────────────────
    consolidation_hour: int = 2
    consolidation_minute: int = 0

    # ── Observability ────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── UI / Palette ─────────────────────────────────────────────────────────
    palette_window_width: int = 640
    palette_window_height: int = 72
    palette_max_height: int = 600

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("agent_memory_root", mode="before")
    @classmethod
    def resolve_memory_root(cls, v: str | Path) -> Path:
        """Expand ~ and resolve to absolute path at load time."""
        return Path(v).expanduser().resolve()

    @field_validator("log_level", mode="before")
    @classmethod
    def normalise_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("blender_socket_port", "consolidation_hour", "consolidation_minute", mode="before")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError(f"Value must be non-negative, got {v}")
        return int(v)

    @model_validator(mode="after")
    def validate_consolidation_time(self) -> "AgentConfig":
        if not (0 <= self.consolidation_hour <= 23):
            raise ValueError(f"consolidation_hour must be 0-23, got {self.consolidation_hour}")
        if not (0 <= self.consolidation_minute <= 59):
            raise ValueError(f"consolidation_minute must be 0-59, got {self.consolidation_minute}")
        return self

    # ── Derived paths (computed, not configurable) ───────────────────────────

    @property
    def episodic_dir(self) -> Path:
        return self.agent_memory_root / "memory" / "episodic"

    @property
    def semantic_dir(self) -> Path:
        return self.agent_memory_root / "memory" / "semantic"

    @property
    def skills_dir(self) -> Path:
        return self.agent_memory_root / "memory" / "skills"

    @property
    def working_dir(self) -> Path:
        return self.agent_memory_root / "memory" / "working"

    @property
    def projects_dir(self) -> Path:
        return self.agent_memory_root / "projects"

    @property
    def consolidation_dir(self) -> Path:
        return self.agent_memory_root / "consolidation"

    @property
    def logs_dir(self) -> Path:
        return self.agent_memory_root / "logs"

    @property
    def graph_path(self) -> Path:
        return self.semantic_dir / "graph.json"

    @property
    def faiss_index_path(self) -> Path:
        return self.semantic_dir / "index.faiss"
    
    # ── Derived paths (computed, not configurable) ───────────────────────────

    @property
    def palette_socket(self) -> Path:
        return self.agent_memory_root / "palette.sock"



@lru_cache(maxsize=1)
def get_config() -> AgentConfig:
    """
    Return the singleton config instance.
    Cached after first call — safe to call anywhere with zero overhead.
    """
    return AgentConfig()


# Module-level singleton — import this everywhere
cfg: AgentConfig = get_config()