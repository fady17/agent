"""
agent/tools/blender_bridge.py

Blender bridge — the agent-side async client for blender_server.py.

Connects to the Blender socket server over TCP and sends Python script
strings, receiving structured JSON results.

The bridge is the ONLY way the rest of the agent interacts with Blender.
No other module imports bpy or assumes Blender is present.

Protocol:
    Request:  {"id": "req-<uuid>", "script": "..."}\n
    Response: {"id": "req-<uuid>", "success": bool, "result": str, "error": str|null}\n

Usage:
    bridge = BlenderBridge()
    result = await bridge.send_script("print(bpy.data.objects.keys())")
    if result.success:
        print(result.output)
    else:
        print(result.error)

    # Or use the context manager:
    async with BlenderBridge() as bridge:
        result = await bridge.send_script("import bpy; print(len(bpy.data.meshes))")
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)

# Maximum bytes to read per message (must match server's MAX_MESSAGE_SIZE)
_READ_LIMIT = 10_485_760  # 10 MB


# ── ScriptResult ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScriptResult:
    """
    Result of one script execution inside Blender.

    success: True if the script ran without exceptions.
    output:  stdout + stderr captured during execution (may be empty).
    error:   Exception traceback if success=False, else empty string.
    request_id: The ID sent with the request (for correlation/logging).
    """
    success:    bool
    output:     str
    error:      str
    request_id: str

    def __str__(self) -> str:
        if self.success:
            return f"ScriptResult(ok, output={self.output[:80]!r})"
        return f"ScriptResult(FAILED, error={self.error[:120]!r})"


# ── BlenderBridge ─────────────────────────────────────────────────────────────

class BlenderBridge:
    """
    Async client for the Blender socket server.

    Maintains a persistent TCP connection. Reconnects automatically
    if the connection drops between calls.

    Thread safety: designed for single asyncio event loop use only.
    Do not share across threads.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        connect_timeout: float = 5.0,
        call_timeout: float = 30.0,
    ) -> None:
        self._host            = host or cfg.blender_socket_host
        self._port            = port or cfg.blender_socket_port
        self._connect_timeout = connect_timeout
        self._call_timeout    = call_timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return (
            self._writer is not None
            and not self._writer.is_closing()
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Open a TCP connection to the Blender server.
        Safe to call multiple times — no-op if already connected.
        Raises ConnectionRefusedError if Blender is not running.
        """
        if self.is_connected:
            return

        log.debug(
            "blender_bridge.connecting",
            host=self._host,
            port=self._port,
        )

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._connect_timeout,
            )
            log.info("blender_bridge.connected", host=self._host, port=self._port)
        except asyncio.TimeoutError:
            raise ConnectionRefusedError(
                f"Blender server at {self._host}:{self._port} "
                f"did not respond within {self._connect_timeout}s"
            )

    async def disconnect(self) -> None:
        """Close the connection gracefully."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None
        log.debug("blender_bridge.disconnected")

    async def send_script(
        self,
        script: str,
        *,
        timeout: float | None = None,
    ) -> ScriptResult:
        """
        Send a Python script to Blender and return the result.

        Args:
            script:  Python code string. Will be exec()'d inside Blender.
            timeout: Per-call timeout in seconds. Defaults to self._call_timeout.

        Returns:
            ScriptResult — always returns, never raises.
            On connection failure: ScriptResult.success=False, error is set.
        """
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        effective_timeout = timeout if timeout is not None else self._call_timeout

        log.debug(
            "blender_bridge.send",
            request_id=request_id,
            script_preview=script[:80],
        )

        # Ensure connected
        try:
            await self.connect()
        except Exception as exc:
            return ScriptResult(
                success=False,
                output="",
                error=f"Connection failed: {exc}",
                request_id=request_id,
            )

        # Send request
        message = json.dumps({"id": request_id, "script": script}) + "\n"
        try:
            self._writer.write(message.encode("utf-8")) # type: ignore
            await self._writer.drain() # type: ignore
        except Exception as exc:
            await self._handle_disconnect()
            return ScriptResult(
                success=False,
                output="",
                error=f"Send failed: {exc}",
                request_id=request_id,
            )

        # Receive response
        try:
            raw = await asyncio.wait_for(
                self._reader.readline(), # type: ignore
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            await self._handle_disconnect()
            return ScriptResult(
                success=False,
                output="",
                error=f"Timed out waiting for Blender response after {effective_timeout}s",
                request_id=request_id,
            )
        except Exception as exc:
            await self._handle_disconnect()
            return ScriptResult(
                success=False,
                output="",
                error=f"Receive failed: {exc}",
                request_id=request_id,
            )

        # Parse response
        try:
            response = json.loads(raw.decode("utf-8").strip())
        except json.JSONDecodeError as exc:
            return ScriptResult(
                success=False,
                output="",
                error=f"Invalid JSON response: {exc} | raw={raw[:200]!r}",
                request_id=request_id,
            )

        result = ScriptResult(
            success=response.get("success", False),
            output=response.get("result", "") or "",
            error=response.get("error", "") or "",
            request_id=response.get("id", request_id),
        )

        log.debug(
            "blender_bridge.result",
            request_id=request_id,
            success=result.success,
            output_len=len(result.output),
        )

        if not result.success:
            log.warning(
                "blender_bridge.script_error",
                request_id=request_id,
                error_preview=result.error[:200],
            )

        return result

    async def is_available(self) -> bool:
        """
        Check if Blender is reachable by attempting a connection.
        Returns True if connected and a ping script succeeds.
        """
        try:
            result = await self.send_script("print('__pong__')", timeout=3.0)
            return result.success and "__pong__" in result.output
        except Exception:
            return False

    async def shutdown_server(self) -> ScriptResult:
        """
        Send the shutdown signal to the Blender server.
        The server will stop accepting new connections after this.
        """
        return await self.send_script("__shutdown__")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _handle_disconnect(self) -> None:
        """Reset connection state after an error."""
        self._writer = None
        self._reader = None
        log.debug("blender_bridge.connection_reset")

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "BlenderBridge":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()