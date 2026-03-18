# type: ignore

"""
agent/interface/palette.py

Command palette — persistent floating overlay triggered by Option+Space.

Designed to evolve:
    Phase 1 (now)   — floating text input + streaming response
    Phase 2         — fuzzy action list, /command prefix, skill-sourced suggestions
    Phase 3         — proactive cards, monitor queue integration

Architecture:
    - Main thread:       tkinter mainloop (required — tk is not thread-safe)
    - Background thread: asyncio event loop for socket I/O
    - Bridge:            queue.Queue (the only thread-safe crossing point)

The palette is a PERSISTENT PROCESS — it starts hidden and toggles on
each hotkey trigger. Cold start latency is paid once at login, not on
every invocation.

Usage (standalone — started by launchd or manually):
    python -m agent.interface.palette

Usage (triggered by skhd):
    # ~/.skhdrc:
    # alt - space : python -m agent.interface.palette --toggle

The --toggle flag sends a SIGUSR1 to the running process to show/hide
the window, then exits. The running process is identified by its PID
file at ~/.agent/palette.pid.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# ── Graceful import of tkinter (not available in all CI environments) ─────────
try:
    import tkinter as tk
    from tkinter import font as tkfont
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PID_FILE      = Path("~/.agent/palette.pid").expanduser()
POLL_MS       = 16          # ~60fps UI polling
ANIM_STEP_PX  = 24          # pixels per animation frame for height expand
FONT_FAMILY   = "SF Pro Text"   # falls back to Helvetica on older macOS
FONT_SIZE_IN  = 15          # input field
FONT_SIZE_RESP= 13          # response area

# Colour tokens — will become a proper theme in Phase 2
COLOURS = {
    "bg":           "#1C1C1E",   # system background dark
    "bg_input":     "#2C2C2E",   # elevated surface
    "bg_response":  "#1C1C1E",
    "border":       "#3A3A3C",
    "text_primary": "#F2F2F7",
    "text_dim":     "#8E8E93",
    "text_code":    "#64D2FF",
    "accent":       "#0A84FF",
    "success":      "#30D158",
    "warning":      "#FFD60A",
    "separator":    "#3A3A3C",
}

# ── Context capture (macOS) ───────────────────────────────────────────────────

def _capture_context() -> dict[str, str]:
    """
    Capture the active app name and clipboard at hotkey activation time.
    Selection text requires Accessibility permissions — attempted but
    degrades gracefully if denied.
    """
    context: dict[str, str] = {"app": "", "selection": "", "clipboard": ""}

    # Active app via NSWorkspace
    try:
        script = (
            'tell application "System Events" to '
            'get name of first application process whose frontmost is true'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=1.0,
        )
        context["app"] = result.stdout.strip()
    except Exception:
        pass

    # Clipboard
    try:
        result = subprocess.run(
            ["pbpaste"], capture_output=True, text=True, timeout=0.5
        )
        context["clipboard"] = result.stdout[:800]
    except Exception:
        pass

    return context


# ── Async agent client ────────────────────────────────────────────────────────

class _AgentSocketClient:
    """
    Async client for the palette server Unix socket.
    Runs in the background thread's event loop.
    """

    def __init__(self, socket_path: Path) -> None:
        self._path   = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    @property
    def is_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> bool:
        if self.is_connected:
            return True
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._path)),
                timeout=2.0,
            )
            return True
        except Exception as exc:
            log.debug("palette_client.connect_failed", error=str(exc))
            return False

    async def disconnect(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

    async def ping(self) -> bool:
        if not await self.connect():
            return False
        req_id = f"ping-{uuid.uuid4().hex[:6]}"
        try:
            self._writer.write( 
                (json.dumps({"id": req_id, "type": "ping"}) + "\n").encode()
            )
            await self._writer.drain()
            raw = await asyncio.wait_for(self._reader.readline(), timeout=2.0)
            resp = json.loads(raw.decode().strip())
            return resp.get("type") == "pong"
        except Exception:
            await self.disconnect()
            return False

    async def query_stream(
        self,
        user_input: str,
        context: dict[str, str],
        on_chunk: Any,    # callable(str) — called in async context
        on_done: Any,     # callable(dict) — called with final metadata
        on_error: Any,    # callable(str) — called on error
    ) -> None:
        """
        Send a query and call on_chunk for each streaming token.
        Calls on_done when complete, on_error on failure.
        """
        if not await self.connect():
            on_error("Agent not running. Start with: uv run agent chat")
            return

        req_id  = f"req-{uuid.uuid4().hex[:8]}"
        payload = json.dumps({
            "id":      req_id,
            "type":    "query",
            "input":   user_input,
            "context": context,
        }) + "\n"

        try:
            self._writer.write(payload.encode())
            await self._writer.drain()

            while True:
                raw = await asyncio.wait_for(
                    self._reader.readline(), timeout=30.0
                )
                if not raw:
                    break

                msg = json.loads(raw.decode().strip())
                msg_type = msg.get("type")

                if msg_type == "chunk":
                    on_chunk(msg.get("text", ""))
                elif msg_type == "done":
                    on_done(msg)
                    break
                elif msg_type == "error":
                    on_error(msg.get("text", "Unknown error"))
                    break

        except asyncio.TimeoutError:
            on_error("Request timed out — is the agent processing?")
            await self.disconnect()
        except Exception as exc:
            on_error(str(exc))
            await self.disconnect()


# ── Background asyncio thread ─────────────────────────────────────────────────

class _BackgroundLoop:
    """
    Manages the asyncio event loop running in a daemon thread.
    The main thread submits coroutines via submit(); results
    come back through the shared queue.
    """

    def __init__(self, socket_path: Path, ui_queue: "queue.Queue[dict]") -> None:
        self._socket_path = socket_path
        self._queue       = ui_queue
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: _AgentSocketClient | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="palette-async"
        )
        self._thread.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = _AgentSocketClient(self._socket_path)
        self._loop.run_forever()

    def submit(self, coro) -> None:
        """Submit a coroutine to the background loop (thread-safe)."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def query(self, user_input: str, context: dict[str, str]) -> None:
        """Send a query; results arrive via the ui_queue."""
        q = self._queue

        def _chunk(text: str) -> None:
            q.put({"type": "chunk", "text": text})

        def _done(meta: dict) -> None:
            q.put({"type": "done", **meta})

        def _error(msg: str) -> None:
            q.put({"type": "error", "text": msg})

        async def _go() -> None:
            await self._client.query_stream(user_input, context, _chunk, _done, _error)

        self.submit(_go())


# ── Palette window ────────────────────────────────────────────────────────────

class PaletteWindow:
    """
    The command palette UI.

    Phase 1: text input + streaming response display.
    Phase 2 hooks: _action_list (stub), _handle_slash_command (stub).
    """

    def __init__(self) -> None:
        if not _TK_AVAILABLE:
            raise RuntimeError(
                "tkinter is not available. "
                "On macOS this is bundled with Python — "
                "try: brew install python-tk"
            )

        self._root       = tk.Tk()
        self._ui_queue: queue.Queue[dict] = queue.Queue()
        self._bg         = _BackgroundLoop(cfg.palette_socket, self._ui_queue)
        self._context: dict[str, str] = {}
        self._response_text = ""
        self._is_responding = False
        self._target_height = cfg.palette_window_height
        self._current_height = cfg.palette_window_height

        self._build_window()
        self._build_widgets()
        self._setup_bindings()
        self._bg.start()

        # Start UI polling loop
        self._root.after(POLL_MS, self._poll_queue)

        # Register SIGUSR1 handler for toggle-from-skhd
        signal.signal(signal.SIGUSR1, self._on_toggle_signal)

        log.info("palette.ready", socket=str(cfg.palette_socket))

    # ── Window construction ───────────────────────────────────────────────────

    def _build_window(self) -> None:
        root = self._root
        root.overrideredirect(True)       # borderless
        root.attributes("-topmost", True)  # always on top
        root.attributes("-alpha", 0.97)
        root.configure(bg=COLOURS["bg"])
        root.resizable(False, False)

        # Position: top-centre of the screen
        sw = root.winfo_screenwidth()
        w  = cfg.palette_window_width
        x  = (sw - w) // 2
        y  = 120    # below menu bar
        root.geometry(f"{w}x{self._current_height}+{x}+{y}")
        root.withdraw()   # start hidden

    def _build_widgets(self) -> None:
        root = self._root
        W    = cfg.palette_window_width

        # Outer frame with rounded-ish border
        self._frame = tk.Frame(
            root,
            bg=COLOURS["bg"],
            highlightbackground=COLOURS["border"],
            highlightthickness=1,
        )
        self._frame.place(x=0, y=0, width=W, height=cfg.palette_max_height)

        # ── Input row ─────────────────────────────────────────────────────────
        input_frame = tk.Frame(self._frame, bg=COLOURS["bg_input"], height=48)
        input_frame.pack(fill="x", side="top")
        input_frame.pack_propagate(False)

        # Search icon (Unicode)
        icon = tk.Label(
            input_frame,
            text="⌘",
            bg=COLOURS["bg_input"],
            fg=COLOURS["text_dim"],
            font=(FONT_FAMILY, 16),
        )
        icon.pack(side="left", padx=(14, 6), pady=0)

        self._input_var = tk.StringVar()
        self._input = tk.Entry(
            input_frame,
            textvariable=self._input_var,
            bg=COLOURS["bg_input"],
            fg=COLOURS["text_primary"],
            insertbackground=COLOURS["text_primary"],
            relief="flat",
            font=(FONT_FAMILY, FONT_SIZE_IN),
            bd=0,
        )
        self._input.pack(side="left", fill="both", expand=True, pady=12, padx=(0, 14))

        # ── Separator ─────────────────────────────────────────────────────────
        self._separator = tk.Frame(self._frame, bg=COLOURS["separator"], height=1)

        # ── Response area ──────────────────────────────────────────────────────
        self._response_frame = tk.Frame(self._frame, bg=COLOURS["bg_response"])

        self._response_text_widget = tk.Text(
            self._response_frame,
            bg=COLOURS["bg_response"],
            fg=COLOURS["text_primary"],
            font=(FONT_FAMILY, FONT_SIZE_RESP),
            relief="flat",
            bd=0,
            padx=16,
            pady=12,
            wrap="word",
            cursor="arrow",
            state="disabled",
        )
        self._response_text_widget.pack(fill="both", expand=True)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="")
        self._status = tk.Label(
            self._frame,
            textvariable=self._status_var,
            bg=COLOURS["bg"],
            fg=COLOURS["text_dim"],
            font=(FONT_FAMILY, 11),
            anchor="w",
        )

        # ── Phase 2 hook: action list placeholder ─────────────────────────────
        # self._action_list = ActionListWidget(self._frame)

    def _setup_bindings(self) -> None:
        root = self._root
        root.bind("<Escape>",  lambda _: self.hide())
        root.bind("<Return>",  lambda _: self._on_submit())
        root.bind("<FocusOut>", self._on_focus_out)

        # Phase 2: arrow key navigation for action list
        # root.bind("<Up>",   lambda _: self._action_list.move_up())
        # root.bind("<Down>", lambda _: self._action_list.move_down())

    # ── Show / hide ───────────────────────────────────────────────────────────

    def show(self) -> None:
        """Show the palette and focus the input field."""
        self._context = _capture_context()
        self._input_var.set("")
        self._clear_response()
        self._set_height(cfg.palette_window_height)

        self._root.deiconify()
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._input.focus_set()
        log.debug("palette.shown", app=self._context.get("app", ""))

    def hide(self) -> None:
        self._root.withdraw()
        log.debug("palette.hidden")

    def toggle(self) -> None:
        if self._root.state() == "withdrawn":
            self.show()
        else:
            self.hide()

    def _on_toggle_signal(self, *_) -> None:
        """Called when SIGUSR1 is received (from skhd via --toggle)."""
        # Schedule on main thread — signal handlers can't call tk directly
        self._root.after(0, self.toggle)

    def _on_focus_out(self, event) -> None:
        """Hide when focus leaves the window."""
        # Small delay to allow click-inside to register first
        self._root.after(100, self._check_focus)

    def _check_focus(self) -> None:
        try:
            if self._root.focus_displayof() is None:
                self.hide()
        except Exception:
            pass

    # ── Submit ────────────────────────────────────────────────────────────────

    def _on_submit(self) -> None:
        text = self._input_var.get().strip()
        if not text or self._is_responding:
            return

        # Phase 2 hook: /command prefix routing
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        self._start_response(text)

    def _start_response(self, user_input: str) -> None:
        self._is_responding = True
        self._response_text = ""
        self._clear_response()
        self._show_response_area()
        self._set_status("Thinking…")
        self._bg.query(user_input, self._context)

    def _handle_slash_command(self, command: str) -> None:
        """
        Phase 2 hook — /command prefix routing.
        For now, pass through as a regular query with the slash stripped.
        """
        self._start_response(command[1:])

    # ── Response rendering ────────────────────────────────────────────────────

    def _poll_queue(self) -> None:
        """Drain the ui_queue and update widgets. Called every POLL_MS ms."""
        try:
            while True:
                msg = self._ui_queue.get_nowait()
                self._handle_ui_message(msg)
        except queue.Empty:
            pass
        finally:
            self._root.after(POLL_MS, self._poll_queue)

    def _handle_ui_message(self, msg: dict) -> None:
        t = msg.get("type")

        if t == "chunk":
            self._append_response(msg.get("text", ""))

        elif t == "done":
            self._is_responding = False
            tokens_in  = msg.get("tokens_in", 0)
            tokens_out = msg.get("tokens_out", 0)
            cost       = msg.get("cost_usd", 0.0)
            if cost > 0:
                self._set_status(
                    f"{tokens_in + tokens_out} tokens · ${cost:.4f}  "
                    f"· Esc to dismiss"
                )
            else:
                self._set_status(
                    f"{tokens_in + tokens_out} tokens · local  "
                    f"· Esc to dismiss"
                )

        elif t == "error":
            self._is_responding = False
            self._append_response(f"\n⚠ {msg.get('text', 'Unknown error')}")
            self._set_status("Error · Esc to dismiss")

    def _append_response(self, text: str) -> None:
        w = self._response_text_widget
        w.configure(state="normal")
        w.insert("end", text)
        w.see("end")
        w.configure(state="disabled")
        self._response_text += text
        self._auto_expand()

    def _clear_response(self) -> None:
        w = self._response_text_widget
        w.configure(state="normal")
        w.delete("1.0", "end")
        w.configure(state="disabled")
        self._response_text = ""
        self._set_status("")

    def _show_response_area(self) -> None:
        self._separator.pack(fill="x", side="top")
        self._response_frame.pack(fill="both", expand=True, side="top")
        self._status.pack(fill="x", side="bottom", padx=14, pady=4)

    def _auto_expand(self) -> None:
        """Grow the window height as the response accumulates, up to max."""
        lines     = self._response_text.count("\n") + 1
        text_h    = lines * (FONT_SIZE_RESP + 6) + 24    # approx line height
        target    = min(
            cfg.palette_window_height + 16 + text_h + 28,   # input + sep + text + status
            cfg.palette_max_height,
        )
        if target > self._target_height:
            self._target_height = int(target)
            self._animate_height()

    def _animate_height(self) -> None:
        if self._current_height >= self._target_height:
            return
        self._current_height = min(
            self._current_height + ANIM_STEP_PX,
            self._target_height,
        )
        self._set_height(self._current_height)
        if self._current_height < self._target_height:
            self._root.after(8, self._animate_height)

    def _set_height(self, h: int) -> None:
        self._current_height = h
        sw = self._root.winfo_screenwidth()
        w  = cfg.palette_window_width
        x  = (sw - w) // 2
        y  = 120
        self._root.geometry(f"{w}x{h}+{x}+{y}")

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the tkinter main loop. Blocks until the window is destroyed."""
        _write_pid()
        try:
            self._root.mainloop()
        finally:
            _remove_pid()


# ── PID file management ───────────────────────────────────────────────────────

def _write_pid() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


def _get_running_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)   # check process is alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agent command palette")
    parser.add_argument(
        "--toggle",
        action="store_true",
        help="Toggle the running palette (sends SIGUSR1 to the daemon, then exits)",
    )
    args = parser.parse_args()

    if args.toggle:
        pid = _get_running_pid()
        if pid is None:
            print("Palette daemon is not running.", file=sys.stderr)
            sys.exit(1)
        os.kill(pid, signal.SIGUSR1)
        sys.exit(0)

    # Start the persistent palette process
    if not _TK_AVAILABLE:
        print(
            "tkinter is not available.\n"
            "Install with: brew install python-tk@3.12",
            file=sys.stderr,
        )
        sys.exit(1)

    window = PaletteWindow()
    window.run()


if __name__ == "__main__":
    main()