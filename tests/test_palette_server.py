"""
tests/test_palette_server.py

Tests for PaletteServer — protocol handling, request dispatch,
streaming response, ping, and enriched input building.
No tkinter required. Uses real Unix sockets via asyncio.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.interface.palette_server import (
    PaletteServer,
    _build_enriched_input,
    _send,
)
from agent.llm.lm_studio import LLMResponse


# ── Socket path helper ────────────────────────────────────────────────────────
# macOS AF_UNIX path limit is 104 chars (BSD UNIX_PATH_MAX).
# pytest's tmp_path is deep under /private/var/folders/... and exceeds it.
# We use a short deterministic path under /tmp instead.

import hashlib
import os

def _short_sock(name: str) -> Path:
    """
    Return a short Unix socket path guaranteed to be under 104 chars.
    Uses a hash of the test name to avoid collisions across parallel runs.
    """
    h = hashlib.md5(name.encode()).hexdigest()[:8]
    p = Path(f"/tmp/agt_{h}.sock")
    # Clean up any leftover from a previous failed run
    p.unlink(missing_ok=True)
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_orchestrator(response: str = "Here is my answer.") -> MagicMock:
    orch = MagicMock()
    orch._session = MagicMock()
    orch._session.state.metrics.total_tokens_in  = 100
    orch._session.state.metrics.total_tokens_out = 50
    orch._session.state.metrics.total_cost_usd   = 0.0

     # 1. Define the actual generator
    async def _stream_generator():
        for word in response.split():
            yield word + " "

    # 2. Define the async method that returns it
    async def _mock_run_turn_stream(*args, **kwargs):
        return _stream_generator()

    # 3. Assign the method
    orch.run_turn_stream = _mock_run_turn_stream
    return orch


async def _open_client(socket_path: Path):
    return await asyncio.open_unix_connection(str(socket_path))


async def _send_recv(reader, writer, payload: dict) -> dict:
    writer.write((json.dumps(payload) + "\n").encode())
    await writer.drain()
    raw = await asyncio.wait_for(reader.readline(), timeout=3.0)
    return json.loads(raw.decode().strip())


async def _collect_stream(reader, timeout: float = 3.0) -> list[dict]:
    """Read all messages until a 'done' or 'error' frame."""
    messages = []
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if not raw:
            break
        msg = json.loads(raw.decode().strip())
        messages.append(msg)
        if msg.get("type") in ("done", "error"):
            break
    return messages


# ── _build_enriched_input ─────────────────────────────────────────────────────

def test_enriched_input_basic() -> None:
    result = _build_enriched_input("fix the bug", "", "", "")
    assert "fix the bug" in result


def test_enriched_input_includes_app() -> None:
    result = _build_enriched_input("fix it", "Blender", "", "")
    assert "Blender" in result


def test_enriched_input_prefers_selection_over_clipboard() -> None:
    result = _build_enriched_input("explain", "", "selected text here", "clipboard text")
    assert "Selected text" in result
    assert "Clipboard" not in result


def test_enriched_input_falls_back_to_clipboard() -> None:
    result = _build_enriched_input("explain", "", "", "clipboard content here")
    assert "Clipboard" in result
    assert "clipboard content here" in result


def test_enriched_input_ignores_tiny_context() -> None:
    result = _build_enriched_input("hi", "", "x", "")
    # Single character context — too short to be useful
    assert "Selected text" not in result


def test_enriched_input_truncates_long_context() -> None:
    long_text = "word " * 300   # ~1500 chars
    result = _build_enriched_input("summarise", "", long_text, "")
    assert len(result) < 1500


# ── PaletteServer lifecycle ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_server_starts_and_creates_socket() -> None:
    sock = _short_sock("starts")
    orch = make_orchestrator()
    server = PaletteServer(orch, socket_path=sock)
    await server.start()
    assert sock.exists()
    assert server.is_running
    await server.stop()


@pytest.mark.asyncio
async def test_server_stop_removes_socket() -> None:
    sock = _short_sock("stop_removes")
    orch = make_orchestrator()
    server = PaletteServer(orch, socket_path=sock)
    await server.start()
    await server.stop()
    assert not sock.exists()
    assert not server.is_running


@pytest.mark.asyncio
async def test_server_context_manager() -> None:
    sock = _short_sock("ctx_mgr")
    async with PaletteServer(make_orchestrator(), socket_path=sock) as server:
        assert server.is_running
    assert not server.is_running


@pytest.mark.asyncio
async def test_server_cleans_stale_socket() -> None:
    sock = _short_sock("stale")
    sock.write_text("stale")  # simulate leftover socket
    async with PaletteServer(make_orchestrator(), socket_path=sock):
        pass  # should not raise


# ── Ping / pong ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ping_returns_pong() -> None:
    sock = _short_sock("ping_pong")
    async with PaletteServer(make_orchestrator(), socket_path=sock):
        reader, writer = await _open_client(sock)
        resp = await _send_recv(reader, writer, {"id": "ping-1", "type": "ping"})
        assert resp["type"] == "pong"
        assert resp["id"] == "ping-1"
        writer.close()


@pytest.mark.asyncio
async def test_ping_without_id_still_responds() -> None:
    sock = _short_sock("ping_noid")
    async with PaletteServer(make_orchestrator(), socket_path=sock):
        reader, writer = await _open_client(sock)
        resp = await _send_recv(reader, writer, {"type": "ping"})
        assert resp["type"] == "pong"
        writer.close()


# ── Query streaming ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_streams_chunks() -> None:
    sock = _short_sock("streams_chunks")
    async with PaletteServer(make_orchestrator("hello world"), socket_path=sock):
        reader, writer = await _open_client(sock)
        writer.write((json.dumps({
            "id": "q1", "type": "query",
            "input": "test query",
            "context": {"app": "Terminal", "selection": "", "clipboard": ""},
        }) + "\n").encode())
        await writer.drain()

        messages = await _collect_stream(reader)
        writer.close()

    types = [m["type"] for m in messages]
    assert "chunk" in types
    assert "done" in types


@pytest.mark.asyncio
async def test_query_final_message_is_done() -> None:
    sock = _short_sock("final_done")
    async with PaletteServer(make_orchestrator("final response"), socket_path=sock):
        reader, writer = await _open_client(sock)
        writer.write((json.dumps({
            "id": "q2", "type": "query",
            "input": "give me a response",
            "context": {},
        }) + "\n").encode())
        await writer.drain()

        messages = await _collect_stream(reader)
        writer.close()

    last = messages[-1]
    assert last["type"] == "done"
    assert last["id"] == "q2"


@pytest.mark.asyncio
async def test_query_chunks_reconstruct_response() -> None:
    response_text = "The quick brown fox"
    sock = _short_sock("reconstruct")
    async with PaletteServer(make_orchestrator(response_text), socket_path=sock):
        reader, writer = await _open_client(sock)
        writer.write((json.dumps({
            "id": "q3", "type": "query",
            "input": "test",
            "context": {},
        }) + "\n").encode())
        await writer.drain()

        messages = await _collect_stream(reader)
        writer.close()

    chunks = "".join(m["text"] for m in messages if m["type"] == "chunk")
    assert "quick" in chunks
    assert "fox" in chunks


@pytest.mark.asyncio
async def test_empty_query_returns_error() -> None:
    sock = _short_sock("empty_query")
    async with PaletteServer(make_orchestrator(), socket_path=sock):
        reader, writer = await _open_client(sock)
        resp = await _send_recv(reader, writer, {
            "id": "q-empty", "type": "query", "input": "",
        })
        assert resp["type"] == "error"
        writer.close()


# ── Error handling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_exception_returns_error() -> None:
    sock = _short_sock("orch_exc")
    orch = MagicMock()
    orch._session = MagicMock()
    orch._session.state.metrics.total_tokens_in  = 0
    orch._session.state.metrics.total_tokens_out = 0
    orch._session.state.metrics.total_cost_usd   = 0.0
    orch.run_turn_stream = MagicMock(side_effect=RuntimeError("agent crashed"))

    async with PaletteServer(orch, socket_path=sock):
        reader, writer = await _open_client(sock)
        writer.write((json.dumps({
            "id": "q-fail", "type": "query",
            "input": "this will crash",
            "context": {},
        }) + "\n").encode())
        await writer.drain()

        messages = await _collect_stream(reader)
        writer.close()

    error_msgs = [m for m in messages if m["type"] == "error"]
    assert len(error_msgs) >= 1


@pytest.mark.asyncio
async def test_malformed_json_returns_error() -> None:
    sock = _short_sock("malformed")
    async with PaletteServer(make_orchestrator(), socket_path=sock):
        reader, writer = await _open_client(sock)
        writer.write(b"not json at all\n")
        await writer.drain()
        resp = await asyncio.wait_for(reader.readline(), timeout=2.0)
        msg = json.loads(resp.decode().strip())
        assert msg["type"] == "error"
        writer.close()


@pytest.mark.asyncio
async def test_unknown_request_type_returns_error() -> None:
    sock = _short_sock("unknown_type")
    async with PaletteServer(make_orchestrator(), socket_path=sock):
        reader, writer = await _open_client(sock)
        resp = await _send_recv(reader, writer, {
            "id": "q-unknown", "type": "future_feature",
        })
        assert resp["type"] == "error"
        writer.close()


# ── Context enrichment in queries ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_enrichment_reaches_orchestrator() -> None:
    """Verify that app name and clipboard are woven into the input the
    orchestrator receives."""
    sock = _short_sock("ctx_enrichment")
    captured_inputs: list[str] = []

    async def _stream_generator(input_text, **kwargs):
        captured_inputs.append(input_text)
        yield "ok"

    # 2. Define the async method that returns it
    async def _mock_run_turn_stream(*args, **kwargs):
        return _stream_generator(args[0], **kwargs)


    orch = MagicMock()
    orch._session = MagicMock()
    orch._session.state.metrics.total_tokens_in  = 0
    orch._session.state.metrics.total_tokens_out = 0
    orch._session.state.metrics.total_cost_usd   = 0.0
    orch.run_turn_stream = _mock_run_turn_stream

    async with PaletteServer(orch, socket_path=sock):
        reader, writer = await _open_client(sock)
        writer.write((json.dumps({
            "id": "ctx-test", "type": "query",
            "input": "explain this error",
            "context": {
                "app": "Xcode",
                "selection": "Thread 1: EXC_BAD_ACCESS",
                "clipboard": "",
            },
        }) + "\n").encode())
        await writer.drain()
        await _collect_stream(reader)
        writer.close()

    assert len(captured_inputs) == 1
    enriched = captured_inputs[0]
    assert "Xcode" in enriched
    assert "EXC_BAD_ACCESS" in enriched


@pytest.mark.asyncio
async def test_two_sequential_queries() -> None:
    """Server must handle multiple queries on the same connection."""
    responses = ["first answer", "second answer"]
    call_count = [0]

    async def _stream_generator(call_idx):
        r = responses[call_idx % len(responses)]
        for w in r.split():
            yield w + " "

    # 2. Define the async method that returns it
    async def _mock_run_turn_stream(*args, **kwargs):
        current_idx = call_count[0]
        call_count[0] += 1
        return _stream_generator(current_idx)

    sock = _short_sock("sequential")
    orch = MagicMock()
    orch._session = MagicMock()
    orch._session.state.metrics.total_tokens_in  = 0
    orch._session.state.metrics.total_tokens_out = 0
    orch._session.state.metrics.total_cost_usd   = 0.0
    orch.run_turn_stream = _mock_run_turn_stream
    
    async with PaletteServer(orch, socket_path=sock):
        reader, writer = await _open_client(sock)

        writer.write((json.dumps({"id": "q1", "type": "query", "input": "first", "context": {}}) + "\n").encode())
        await writer.drain()
        msgs1 = await _collect_stream(reader)

        writer.write((json.dumps({"id": "q2", "type": "query", "input": "second", "context": {}}) + "\n").encode())
        await writer.drain()
        msgs2 = await _collect_stream(reader)

        writer.close()

    assert any(m["type"] == "done" for m in msgs1)
    assert any(m["type"] == "done" for m in msgs2)
    assert call_count[0] == 2