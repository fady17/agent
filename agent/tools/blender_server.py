# type: ignore
"""
agent/tools/blender_server.py

Blender socket server — runs INSIDE Blender's Python interpreter.

This script is executed by Blender at startup via:
    blender --background --python agent/tools/blender_server.py
    blender my_scene.blend --python agent/tools/blender_server.py

Architecture:
    - Socket listener runs in a BACKGROUND THREAD (safe — only I/O)
    - Script execution is SCHEDULED ON BLENDER'S MAIN THREAD via
      bpy.app.timers.register() — this is critical. Any bpy.ops call,
      depsgraph update, or viewport operation will crash if called from
      a background thread. The timer approach is the same solution used
      by blender-mcp (ahujasid/blender-mcp) and is the correct pattern.
    - Results are passed back to the socket thread via threading.Event
      + a shared result dict.

Protocol (newline-delimited JSON):
    Request:  {"id": "req-001", "script": "print(bpy.data.objects.keys())"}
    Response: {"id": "req-001", "success": true, "result": "...", "error": null}
    Response: {"id": "req-001", "success": false, "result": null, "error": "NameError: ..."}

Security:
    Listens on 127.0.0.1 only — never exposed to the network.

To stop: send {"id": "x", "script": "__shutdown__"} or Ctrl+C in Blender.
"""

import io
import json
import socket
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout

# ── Configuration ─────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 9999
BUFFER_SIZE = 65536
MAX_MESSAGE_SIZE = 10_485_760   # 10 MB
MAIN_THREAD_TIMEOUT = 30.0      # seconds to wait for bpy.app.timers result


# ── Execution namespace ───────────────────────────────────────────────────────

def _make_namespace() -> dict:
    """
    Build the shared execution namespace for all agent scripts in a session.
    Pre-imports bpy and common modules so scripts don't have to.
    The namespace persists across calls — variables set in one script
    are available in the next (intentional: enables multi-step operations).
    """
    ns: dict = {
        "__builtins__": __builtins__,
        "__name__":     "__agent__",
    }
    try:
        import bpy 
        import bmesh
        import mathutils
        ns["bpy"]       = bpy
        ns["bmesh"]     = bmesh
        ns["mathutils"] = mathutils
    except ImportError:
        pass  # Running outside Blender in tests — bpy not available
    return ns


# ── Script execution (must run on Blender's main thread) ─────────────────────

def execute_script(script: str, namespace: dict) -> tuple[bool, str, str | None]:
    """
    Execute a Python script string in the given namespace.

    MUST be called on Blender's main thread when running inside Blender.
    Safe to call directly in tests (no bpy involved).

    Returns:
        (success, result_str, error_str)
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            code = compile(script, "<agent_script>", "exec")
            exec(code, namespace)  # noqa: S102

        stdout_out = stdout_buf.getvalue()
        stderr_out = stderr_buf.getvalue()
        result = stdout_out
        if stderr_out:
            result = result + ("\n" if result else "") + stderr_out

        return True, result, None

    except Exception:
        error = traceback.format_exc()
        stdout_out = stdout_buf.getvalue()
        return False, stdout_out, error


# ── Main-thread execution via bpy.app.timers ──────────────────────────────────

def _execute_on_main_thread(
    script: str,
    namespace: dict,
    result_holder: dict,
    done_event: threading.Event,
) -> None:
    """
    Timer callback — runs on Blender's main thread.
    Executes the script, stores result, signals the waiting socket thread.

    bpy.app.timers callbacks must return None (do not repeat)
    or a float (repeat after N seconds). We return None.
    """
    try:
        success, result, error = execute_script(script, namespace)
        result_holder["success"] = success
        result_holder["result"]  = result
        result_holder["error"]   = error
    except Exception as exc:
        result_holder["success"] = False
        result_holder["result"]  = ""
        result_holder["error"]   = traceback.format_exc()
    finally:
        done_event.set()


def schedule_on_main_thread(
    script: str,
    namespace: dict,
    timeout: float = MAIN_THREAD_TIMEOUT,
) -> tuple[bool, str, str | None]:
    """
    Schedule a script on Blender's main thread and wait for the result.

    Called from the background socket thread. Blocks until execution
    completes or timeout expires.

    Falls back to direct execution if bpy.app.timers is unavailable
    (i.e. running outside Blender in tests).
    """
    try:
        import bpy
        result_holder: dict = {}
        done_event = threading.Event()

        bpy.app.timers.register(
            lambda: _execute_on_main_thread(script, namespace, result_holder, done_event),
            first_interval=0.0,
        )

        if not done_event.wait(timeout=timeout):
            return False, "", f"Main thread execution timed out after {timeout}s"

        return (
            result_holder.get("success", False),
            result_holder.get("result", ""),
            result_holder.get("error"),
        )

    except ImportError:
        # bpy not available — running in tests, execute directly
        return execute_script(script, namespace)


# ── Message framing ───────────────────────────────────────────────────────────

def recv_message(conn: socket.socket) -> dict | None:
    """Receive a complete newline-terminated JSON message."""
    data = b""
    while True:
        try:
            chunk = conn.recv(BUFFER_SIZE)
        except (ConnectionResetError, OSError):
            return None
        if not chunk:
            return None
        data += chunk
        if len(data) > MAX_MESSAGE_SIZE:
            return {"error": "message too large"}
        if b"\n" in data:
            line, _ = data.split(b"\n", 1)
            try:
                return json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                return {"error": f"JSON decode error: {exc}"}


def send_response(conn: socket.socket, response: dict) -> None:
    """Send a newline-terminated JSON response."""
    try:
        payload = json.dumps(response) + "\n"
        conn.sendall(payload.encode("utf-8"))
    except (BrokenPipeError, OSError):
        pass


# ── Connection handler ────────────────────────────────────────────────────────

def handle_connection(
    conn: socket.socket,
    addr: tuple,
    namespace: dict,
    shutdown_event: threading.Event,
) -> None:
    """Handle one client connection in the background socket thread."""
    print(f"[blender_server] connected: {addr}", flush=True)

    try:
        while not shutdown_event.is_set():
            msg = recv_message(conn)
            if msg is None:
                break

            request_id = msg.get("id", "unknown")

            if "error" in msg and "script" not in msg:
                send_response(conn, {
                    "id": request_id, "success": False,
                    "result": None, "error": msg["error"],
                })
                continue

            script = msg.get("script", "")

            if script == "__shutdown__":
                send_response(conn, {
                    "id": request_id, "success": True,
                    "result": "shutting down", "error": None,
                })
                shutdown_event.set()
                break

            # ── KEY FIX: schedule on main thread ──────────────────────────
            success, result, error = schedule_on_main_thread(script, namespace)
            send_response(conn, {
                "id":      request_id,
                "success": success,
                "result":  result,
                "error":   error,
            })

    finally:
        conn.close()
        print(f"[blender_server] disconnected: {addr}", flush=True)


# ── Server ────────────────────────────────────────────────────────────────────

def run_server(host: str = HOST, port: int = PORT) -> None:
    """Start the socket server and block until shutdown."""
    shutdown_event = threading.Event()
    namespace = _make_namespace()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((host, port))
    except OSError as exc:
        print(f"[blender_server] ERROR: Cannot bind {host}:{port} — {exc}", flush=True)
        return

    server.listen(1)
    server.settimeout(1.0)
    print(f"[blender_server] listening on {host}:{port}", flush=True)

    try:
        while not shutdown_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            thread = threading.Thread(
                target=handle_connection,
                args=(conn, addr, namespace, shutdown_event),
                daemon=True,
            )
            thread.start()

    except KeyboardInterrupt:
        print("[blender_server] interrupted", flush=True)
    finally:
        server.close()
        print("[blender_server] stopped", flush=True)


# ── Blender integration ───────────────────────────────────────────────────────

def _start_in_blender_thread() -> None:
    """
    Start the server in a background thread inside Blender.
    The socket thread handles I/O; bpy.app.timers brings scripts
    back to the main thread for safe execution.
    """
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print(f"[blender_server] started on {HOST}:{PORT} (main-thread execution via timers)", flush=True)


if __name__ == "__main__":
    run_server()
else:
    # Executed by Blender's --python flag
    _start_in_blender_thread()