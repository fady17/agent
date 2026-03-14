"""
agent/core/logger.py

Structured logging for the entire agent.

Two output streams configured on first call to setup_logging():
  1. Terminal  — human-readable, coloured, dev-friendly
  2. File      — newline-delimited JSON at ~/.agent/logs/trace.jsonl
                 one record per line, machine-parseable for replay/audit

Usage anywhere in the codebase:

    from agent.core.logger import get_logger
    log = get_logger(__name__)

    log.info("event_name", key="value", count=3)
    log.debug("retrieval.query", query="FastAPI", results=5)
    log.warning("llm.retry", attempt=2, reason="timeout")
    log.error("tool.failed", tool="shell_runner", error=str(e))

Every log call produces a structured record — never use bare print() in
application code. print() is only acceptable in CLI scripts and test output.
"""

import logging
import sys
from pathlib import Path

import structlog

from agent.core.config import cfg

# ── Internal state ────────────────────────────────────────────────────────────
_configured = False


def setup_logging() -> None:
    """
    Configure structlog once at agent startup.
    Subsequent calls are no-ops — safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    # Ensure log directory exists (memory tree may not be initialised yet)
    log_dir: Path = cfg.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_path = log_dir / "trace.jsonl"

    # ── Shared processors (run for every log record) ─────────────────────────
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    # ── File handler — JSON, one record per line ──────────────────────────────
    file_handler = logging.FileHandler(trace_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # ── Terminal handler — coloured if stdout is a tty ────────────────────────
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setLevel(getattr(logging, cfg.log_level, logging.INFO))

    # ── stdlib root logger ────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # handlers filter independently
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(terminal_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── structlog configuration ───────────────────────────────────────────────
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Terminal formatter — pretty coloured output
    terminal_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        foreign_pre_chain=shared_processors,
    )

    # File formatter — JSON
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    terminal_handler.setFormatter(terminal_formatter)
    file_handler.setFormatter(file_formatter)

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a named structlog logger.
    setup_logging() is called automatically on first use.
    """
    setup_logging()
    return structlog.get_logger(name)