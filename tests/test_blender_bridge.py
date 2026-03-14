"""
tests/test_blender_bridge.py

Tests for BlenderBridge (agent-side client) and blender_server internals.

No real Blender installation required. Tests spin up a minimal
asyncio TCP server that mimics the Blender server protocol, then
verify the bridge communicates correctly.

Also tests blender_server.py's execute_script and message parsing
functions directly — no socket needed for those.
"""

from __future__ import annotations

import asyncio
import json
import threading
import socket as socket_module
from pathlib import Path

import pytest

from agent.tools.blender_bridge import BlenderBridge, ScriptResult
from agent.tools.blender_server import execute_script, _make_namespace, recv_message


# ── ScriptResult ──────────────────────────────────────────────────────────────

def test_script_result_str_success() -> None:
    r = ScriptResult(success=True, output="hello", error="", request_id="req-1")
    assert "ok" in str(r)


def test_script_result_str_failure() -> None:
    r = ScriptResult(success=False, output="", error="NameError: x", request_id="req-1")
    assert "FAILED" in str(r)


def test_script_result_is_frozen() -> None:
    r = ScriptResult(success=True, output="", error="", request_id="req-1")
    with pytest.raises(Exception):
        r.success = False  # type: ignore[misc]


# ── execute_script (server internals) ────────────────────────────────────────

def test_execute_script_success() -> None:
    ns = _make_namespace()
    success, result, error = execute_script("print('hello from script')", ns)
    assert success is True
    assert "hello from script" in result
    assert error is None


def test_execute_script_captures_stdout() -> None:
    ns = _make_namespace()
    success, result, _ = execute_script("print('line1')\nprint('line2')", ns)
    assert success is True
    assert "line1" in result
    assert "line2" in result


def test_execute_script_captures_stderr() -> None:
    ns = _make_namespace()
    success, result, _ = execute_script(
        "import sys; sys.stderr.write('warning\\n')", ns
    )
    assert success is True
    assert "warning" in result


def test_execute_script_failure_returns_traceback() -> None:
    ns = _make_namespace()
    success, result, error = execute_script("raise ValueError('bad value')", ns)
    assert success is False
    assert error is not None
    assert "ValueError" in error
    assert "bad value" in error


def test_execute_script_syntax_error() -> None:
    ns = _make_namespace()
    success, result, error = execute_script("def bad(:", ns)
    assert success is False
    assert error is not None


def test_execute_script_multiline() -> None:
    ns = _make_namespace()
    script = "x = 1\ny = 2\nprint(x + y)"
    success, result, _ = execute_script(script, ns)
    assert success is True
    assert "3" in result


def test_execute_script_persists_namespace() -> None:
    ns = _make_namespace()
    execute_script("my_var = 42", ns)
    success, result, _ = execute_script("print(my_var)", ns)
    assert success is True
    assert "42" in result


def test_make_namespace_has_builtins() -> None:
    ns = _make_namespace()
    assert "__builtins__" in ns


def test_make_namespace_has_agent_name() -> None:
    ns = _make_namespace()
    assert ns["__name__"] == "__agent__"


# ── Mock server fixture ───────────────────────────────────────────────────────

class _MockBlenderServer:
    """
    Minimal TCP server that mimics blender_server.py's protocol.
    Runs in a background thread for the duration of each test.
    """

    def __init__(self, responses: list[dict] | None = None) -> None:
        self._responses = responses or []
        self._received: list[dict] = []
        self._server: socket_module.socket | None = None
        self._thread: threading.Thread | None = None
        self.host = "127.0.0.1"
        self.port = self._find_free_port()

    @staticmethod
    def _find_free_port() -> int:
        with socket_module.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def start(self) -> None:
        self._server = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
        self._server.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self._server.settimeout(2.0)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._server.accept() # type: ignore
        except socket_module.timeout:
            return
        except Exception:
            return

        with conn:
            for response in self._responses:
                try:
                    data = b""
                    while b"\n" not in data:
                        chunk = conn.recv(4096)
                        if not chunk:
                            return
                        data += chunk
                    line = data.split(b"\n")[0]
                    self._received.append(json.loads(line.decode()))
                    payload = json.dumps(response) + "\n"
                    conn.sendall(payload.encode())
                except Exception:
                    break

    def stop(self) -> None:
        if self._server:
            self._server.close()

    @property
    def received(self) -> list[dict]:
        return self._received


@pytest.fixture
def mock_server():
    server = _MockBlenderServer()
    server.start()
    yield server
    server.stop()


def server_with_responses(*responses: dict) -> _MockBlenderServer:
    s = _MockBlenderServer(responses=list(responses))
    s.start()
    return s


# ── BlenderBridge — connection ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bridge_connects_successfully() -> None:
    srv = server_with_responses(
        {"id": "req-1", "success": True, "result": "__pong__", "error": None}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0)
    try:
        await bridge.connect()
        assert bridge.is_connected is True
    finally:
        await bridge.disconnect()
        srv.stop()


@pytest.mark.asyncio
async def test_bridge_connection_refused() -> None:
    bridge = BlenderBridge(host="127.0.0.1", port=19999, connect_timeout=0.3)
    with pytest.raises(ConnectionRefusedError):
        await bridge.connect()


@pytest.mark.asyncio
async def test_bridge_is_not_connected_initially() -> None:
    bridge = BlenderBridge(host="127.0.0.1", port=9999)
    assert bridge.is_connected is False


@pytest.mark.asyncio
async def test_bridge_disconnect_when_not_connected_does_not_raise() -> None:
    bridge = BlenderBridge(host="127.0.0.1", port=9999)
    await bridge.disconnect()  # Should not raise


# ── BlenderBridge — send_script ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_script_success() -> None:
    srv = server_with_responses(
        {"id": "req-001", "success": True, "result": "scene_data", "error": None}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=5.0)
    try:
        result = await bridge.send_script("print(bpy.context.scene.name)")
        assert result.success is True
        assert result.output == "scene_data"
        assert result.error == ""
    finally:
        await bridge.disconnect()
        srv.stop()


@pytest.mark.asyncio
async def test_send_script_failure() -> None:
    srv = server_with_responses(
        {"id": "req-001", "success": False, "result": "", "error": "NameError: bpy not defined"}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=5.0)
    try:
        result = await bridge.send_script("bpy.undefined_thing()")
        assert result.success is False
        assert "NameError" in result.error
    finally:
        await bridge.disconnect()
        srv.stop()


@pytest.mark.asyncio
async def test_send_script_returns_script_result() -> None:
    srv = server_with_responses(
        {"id": "x", "success": True, "result": "ok", "error": None}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=5.0)
    try:
        result = await bridge.send_script("pass")
        assert isinstance(result, ScriptResult)
    finally:
        await bridge.disconnect()
        srv.stop()


@pytest.mark.asyncio
async def test_send_script_connection_refused_returns_error_result() -> None:
    bridge = BlenderBridge(host="127.0.0.1", port=19999, connect_timeout=0.3)
    result = await bridge.send_script("print('hi')")
    assert result.success is False
    assert "Connection failed" in result.error


@pytest.mark.asyncio
async def test_send_script_request_id_in_result() -> None:
    srv = server_with_responses(
        {"id": "req-abc123", "success": True, "result": "", "error": None}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=5.0)
    try:
        result = await bridge.send_script("pass")
        assert result.request_id is not None
        assert len(result.request_id) > 0
    finally:
        await bridge.disconnect()
        srv.stop()


@pytest.mark.asyncio
async def test_send_script_serialises_request_as_json() -> None:
    srv = server_with_responses(
        {"id": "x", "success": True, "result": "", "error": None}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=5.0)
    try:
        await bridge.send_script("my_script_content = True")
        received = srv.received
        assert len(received) == 1
        assert "script" in received[0]
        assert "my_script_content" in received[0]["script"]
        assert "id" in received[0]
    finally:
        await bridge.disconnect()
        srv.stop()


@pytest.mark.asyncio
async def test_send_multiple_scripts_sequentially() -> None:
    srv = server_with_responses(
        {"id": "r1", "success": True, "result": "first", "error": None},
        {"id": "r2", "success": True, "result": "second", "error": None},
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=5.0)
    try:
        r1 = await bridge.send_script("print('first')")
        r2 = await bridge.send_script("print('second')")
        assert r1.output == "first"
        assert r2.output == "second"
    finally:
        await bridge.disconnect()
        srv.stop()


# ── BlenderBridge — context manager ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_manager_connects_and_disconnects() -> None:
    srv = server_with_responses(
        {"id": "x", "success": True, "result": "ok", "error": None}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=5.0)
    async with bridge:
        assert bridge.is_connected is True
        await bridge.send_script("pass")
    assert bridge.is_connected is False
    srv.stop()


# ── BlenderBridge — is_available ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_is_available_true_when_server_responds() -> None:
    srv = server_with_responses(
        {"id": "x", "success": True, "result": "__pong__\n", "error": None}
    )
    bridge = BlenderBridge(host=srv.host, port=srv.port, connect_timeout=2.0, call_timeout=3.0)
    try:
        available = await bridge.is_available()
        assert available is True
    finally:
        await bridge.disconnect()
        srv.stop()


@pytest.mark.asyncio
async def test_is_available_false_when_server_unreachable() -> None:
    bridge = BlenderBridge(host="127.0.0.1", port=19998, connect_timeout=0.3)
    available = await bridge.is_available()
    assert available is False