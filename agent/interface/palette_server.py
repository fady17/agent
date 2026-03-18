"""
agent/interface/palette_server.py

Palette server — lightweight Unix socket server that the command palette
connects to. Runs alongside the agent process (started from cli.py chat
or as a standalone daemon).

Separating this from the CLI's chat loop means the palette and the
terminal chat can run simultaneously without competing for stdin.

Protocol (newline-delimited JSON):

    Request:
        {
            "id":      "req-<uuid8>",
            "type":    "query" | "context" | "ping" | "action",
            "input":   "<user text>",
            "context": {
                "app":       "Blender",
                "selection": "selected text or empty",
                "clipboard": "clipboard contents or empty"
            }
        }

    Response (streaming — one JSON line per chunk):
        {"id": "req-abc", "type": "chunk",  "text": "Here "}
        {"id": "req-abc", "type": "chunk",  "text": "is the "}
        {"id": "req-abc", "type": "done",   "text": "",
         "tokens_in": 50, "tokens_out": 30, "cost_usd": 0.0}

    Error:
        {"id": "req-abc", "type": "error",  "text": "<message>"}

    Ping/pong:
        request:  {"id": "ping-1", "type": "ping"}
        response: {"id": "ping-1", "type": "pong", "text": ""}
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)

_READ_LIMIT = 1_048_576   # 1 MB max request size


# ── PaletteServer ─────────────────────────────────────────────────────────────

class PaletteServer:
    """
    Unix socket server for the command palette.

    Owns a reference to the Orchestrator so it can run turns directly.
    One connection handled at a time (the palette is single-user by design).

    Usage:
        server = PaletteServer(orchestrator)
        await server.start()
        # runs until stop() is called
        await server.stop()
    """

    def __init__(
        self,
        orchestrator,             # Orchestrator — typed as Any to avoid circular import
        socket_path: Path | None = None,
    ) -> None:
        self._orchestrator  = orchestrator
        self._socket_path   = socket_path or cfg.palette_socket # type: ignore
        self._server: asyncio.AbstractServer | None = None
        self._running = False

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start listening on the Unix socket."""
        # Remove stale socket from a previous run
        if self._socket_path.exists():
            self._socket_path.unlink()

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._socket_path),
        )
        self._running = True
        log.info("palette_server.started", socket=str(self._socket_path))

    async def stop(self) -> None:
        if not self._running or self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        if self._socket_path.exists():
            self._socket_path.unlink(missing_ok=True)
        self._running = False
        log.info("palette_server.stopped")

    async def __aenter__(self) -> "PaletteServer":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ── Connection handler ────────────────────────────────────────────────────

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "palette")
        log.debug("palette_server.connected", peer=str(peer))

        try:
            while True:
                try:
                    raw = await reader.readline()
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    break

                if not raw:
                    break

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    await _send(writer, {
                        "id": "unknown", "type": "error",
                        "text": f"JSON parse error: {exc}",
                    })
                    continue

                await self._dispatch(request, writer)

        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            log.debug("palette_server.disconnected")

    async def _dispatch(
        self,
        request: dict,
        writer: asyncio.StreamWriter,
    ) -> None:
        req_id   = request.get("id", f"req-{uuid.uuid4().hex[:8]}")
        req_type = request.get("type", "query")

        # ── Ping ──────────────────────────────────────────────────────────────
        if req_type == "ping":
            await _send(writer, {"id": req_id, "type": "pong", "text": ""})
            return

        # ── Query (streaming) ─────────────────────────────────────────────────
        if req_type in ("query", "action"):
            user_input = request.get("input", "").strip()
            if not user_input:
                await _send(writer, {
                    "id": req_id, "type": "error", "text": "Empty input",
                })
                return

            context    = request.get("context", {})
            app_name   = context.get("app", "")
            selection  = context.get("selection", "")
            clipboard  = context.get("clipboard", "")

            # Build enriched input with context hints
            enriched = _build_enriched_input(user_input, app_name, selection, clipboard)

            log.info(
                "palette_server.query",
                req_id=req_id,
                preview=user_input[:60],
                app=app_name,
            )

            tokens_in = tokens_out = 0
            cost_usd  = 0.0

            try:
                stream = await self._orchestrator.run_turn_stream(enriched)
                async for chunk in stream:
                    await _send(writer, {
                        "id": req_id, "type": "chunk", "text": chunk,
                    })

                # Pull final metrics from session
                state = self._orchestrator._session.state
                m     = state.metrics
                # Estimate this turn's tokens from the delta
                tokens_out = max(0, m.total_tokens_out - getattr(self, "_last_tokens_out", 0))
                tokens_in  = max(0, m.total_tokens_in  - getattr(self, "_last_tokens_in",  0))
                cost_usd   = max(0, m.total_cost_usd   - getattr(self, "_last_cost",        0.0))
                self._last_tokens_out = m.total_tokens_out
                self._last_tokens_in  = m.total_tokens_in
                self._last_cost       = m.total_cost_usd

            except Exception as exc:
                log.error("palette_server.query_failed", error=str(exc))
                await _send(writer, {
                    "id": req_id, "type": "error", "text": str(exc),
                })
                return

            await _send(writer, {
                "id":        req_id,
                "type":      "done",
                "text":      "",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd":  round(cost_usd, 6),
            })

        # ── Context (probe — no response, just acknowledge) ───────────────────
        elif req_type == "context":
            await _send(writer, {"id": req_id, "type": "ack", "text": ""})

        else:
            await _send(writer, {
                "id": req_id, "type": "error",
                "text": f"Unknown request type: {req_type!r}",
            })


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _send(writer: asyncio.StreamWriter, payload: dict) -> None:
    """Write one newline-delimited JSON frame."""
    try:
        writer.write((json.dumps(payload) + "\n").encode("utf-8"))
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass


def _build_enriched_input(
    user_input: str,
    app_name: str,
    selection: str,
    clipboard: str,
) -> str:
    """
    Enrich the user's raw input with context captured at hotkey activation.
    The agent sees a single string — no schema changes needed.
    """
    parts = [user_input]

    if app_name:
        parts.append(f"[Active app: {app_name}]")

    # Prefer selected text over clipboard — it's more intentional
    context_text = selection or clipboard
    if context_text and len(context_text.strip()) > 3:
        # Truncate to avoid blowing the context window
        truncated = context_text.strip()[:800]
        label = "Selected text" if selection else "Clipboard"
        parts.append(f"[{label}:\n{truncated}]")

    return "\n".join(parts)