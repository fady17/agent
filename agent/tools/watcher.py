 # type: ignore
"""
agent/tools/watcher.py

Async file watcher — monitors directory trees for changes and pushes
structured FileEvent objects into an asyncio.Queue.

Architecture:
    watchdog runs its observer in a background thread (it's OS-native:
    FSEvents on macOS, inotify on Linux, ReadDirectoryChangesW on Windows).
    Events are bridged from the watchdog thread into the asyncio event loop
    via asyncio.Queue and loop.call_soon_threadsafe().

    The orchestrator's background loop drains the queue and decides whether
    a file change warrants a proactive action.

Decoupling:
    The watcher never calls the orchestrator directly. It only writes to
    the queue. This means:
      - Watcher can be started/stopped independently
      - Queue can be drained by any consumer (orchestrator, test, CLI)
      - No circular imports between tools and core

Filtering:
    By default, hidden files (.git, __pycache__, .venv) and editor
    temp files are ignored. Custom patterns can be added per-instance.

Usage:
    queue = asyncio.Queue()
    watcher = FileWatcher(paths=[Path("~/projects/agent")], queue=queue)
    await watcher.start()

    # In orchestrator background loop:
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    # handle event...

    await watcher.stop()
"""

from __future__ import annotations

import asyncio
import fnmatch
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable  # noqa: UP035

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from agent.core.logger import get_logger

log = get_logger(__name__)

# Patterns to always ignore — never surfaced to the queue
DEFAULT_IGNORE_PATTERNS: list[str] = [
    "*.pyc",
    "*/__pycache__/*",
    "*/.git/*",
    "*/.venv/*",
    "*/venv/*",
    "*/.mypy_cache/*",
    "*/.ruff_cache/*",
    "*/.pytest_cache/*",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "*/node_modules/*",
    "*/.agent/*",     # never watch the memory tree itself
]

# Max events to buffer before the consumer must drain the queue
DEFAULT_QUEUE_MAXSIZE = 500


# ── FileEventType ─────────────────────────────────────────────────────────────

class FileEventType(StrEnum):
    CREATED  = "created"
    MODIFIED = "modified"
    DELETED  = "deleted"
    MOVED    = "moved"


# ── FileEvent ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FileEvent:
    """
    A single file system change, normalised from watchdog's raw event.

    path:       Absolute path of the changed file.
    event_type: What happened (created/modified/deleted/moved).
    dest_path:  For MOVED events — the destination path.
    timestamp:  When the event was observed (UTC).
    extension:  File extension (lowercase, including dot) for quick filtering.
    """

    path:       Path
    event_type: FileEventType
    dest_path:  Path | None = None
    timestamp:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_python(self) -> bool:
        return self.extension == ".py"

    @property
    def is_blend(self) -> bool:
        return self.extension == ".blend"

    @property
    def is_dart(self) -> bool:
        return self.extension == ".dart"

    @property
    def is_asset(self) -> bool:
        return self.extension in {".glb", ".fbx", ".obj", ".gltf", ".usd", ".usda"}

    def __str__(self) -> str:
        return f"FileEvent({self.event_type} {self.path.name})"


# ── Bridge: watchdog thread → asyncio queue ───────────────────────────────────

class _AsyncBridgeHandler(FileSystemEventHandler):
    """
    Watchdog event handler that bridges from the watchdog thread into
    an asyncio queue via loop.call_soon_threadsafe().

    This is the only thread-safe crossing point in the whole system.
    Everything else runs in the asyncio event loop.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        ignore_patterns: list[str],
        callback: Callable[[FileEvent], None] | None = None,
    ) -> None:
        super().__init__()
        self._loop    = loop
        self._queue   = queue
        self._ignores = ignore_patterns
        self._callback = callback

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path, FileEventType.CREATED)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path, FileEventType.MODIFIED)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(event.src_path, FileEventType.DELETED)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch(
                event.src_path,
                FileEventType.MOVED,
                dest=event.dest_path,
            )

    def _dispatch(
        self,
        src_path: str,
        event_type: FileEventType,
        dest: str | None = None,
    ) -> None:
        path = Path(src_path).resolve()

        if self._should_ignore(path):
            return

        file_event = FileEvent(
            path=path,
            event_type=event_type,
            dest_path=Path(dest).resolve() if dest else None,
        )

        # Bridge from watchdog thread into asyncio
        self._loop.call_soon_threadsafe(self._enqueue, file_event)

    def _enqueue(self, event: FileEvent) -> None:
        """Called in the asyncio event loop thread."""
        try:
            self._queue.put_nowait(event)
            if self._callback:
                self._callback(event)
        except asyncio.QueueFull:
            log.warning(
                "watcher.queue_full",
                file_event=str(event),  # Renamed from 'event'
                hint="Consumer is not draining the queue fast enough",
            )
            
    def _should_ignore(self, path: Path) -> bool:
        path_str = str(path)
        for pattern in self._ignores:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
        return False


# ── FileWatcher ───────────────────────────────────────────────────────────────

class FileWatcher:
    """
    Watches one or more directory trees for file system events.

    Thread-safe: the watchdog observer runs in its own thread;
    all public methods of this class are called from the asyncio event loop.

    Usage:
        queue  = asyncio.Queue()
        watcher = FileWatcher(paths=[Path(".")], queue=queue)
        await watcher.start()
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                print(event)
        except asyncio.TimeoutError:
            pass  # no events — normal
        finally:
            await watcher.stop()
    """

    def __init__(
        self,
        paths: list[Path],
        queue: asyncio.Queue | None = None,
        *,
        recursive: bool = True,
        ignore_patterns: list[str] | None = None,
        extra_ignore_patterns: list[str] | None = None,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        on_event: Callable[[FileEvent], None] | None = None,
    ) -> None:
        if not paths:
            raise ValueError("At least one watch path must be provided")

        self._paths    = [Path(p).resolve() for p in paths]
        self._queue    = queue or asyncio.Queue(maxsize=queue_maxsize)
        self._recursive = recursive
        self._on_event  = on_event

        # Merge default + extra ignore patterns
        base = ignore_patterns if ignore_patterns is not None else DEFAULT_IGNORE_PATTERNS
        self._ignore_patterns = base + (extra_ignore_patterns or [])

        self._observer: Observer | None = None
        self._running  = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def queue(self) -> asyncio.Queue:
        return self._queue

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def watched_paths(self) -> list[Path]:
        return list(self._paths)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the file observer in a background thread.
        Safe to call multiple times — no-op if already running.
        """
        if self._running:
            log.debug("watcher.already_running")
            return

        loop = asyncio.get_event_loop()
        handler = _AsyncBridgeHandler(
            loop=loop,
            queue=self._queue,
            ignore_patterns=self._ignore_patterns,
            callback=self._on_event,
        )

        self._observer = Observer()
        for path in self._paths:
            if not path.exists():
                log.warning("watcher.path_not_found", path=str(path))
                continue
            self._observer.schedule(handler, str(path), recursive=self._recursive)
            log.info("watcher.watching", path=str(path), recursive=self._recursive)

        self._observer.start()
        self._running = True
        log.info("watcher.started", paths=[str(p) for p in self._paths])

    async def stop(self) -> None:
        """
        Stop the observer and clean up the background thread.
        Blocks briefly until the thread joins.
        """
        if not self._running or self._observer is None:
            return

        self._observer.stop()
        # Join in a thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._observer.join)
        self._running = False
        log.info("watcher.stopped")

    async def __aenter__(self) -> "FileWatcher":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ── Queue helpers ─────────────────────────────────────────────────────────

    async def get_event(self, timeout: float = 1.0) -> FileEvent | None:
        """
        Get the next event from the queue with a timeout.
        Returns None on timeout — not an error.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def drain(self, max_events: int = 100) -> list[FileEvent]:
        """
        Non-blocking drain — returns all currently queued events up to max_events.
        Used when the consumer wants to batch-process pending events.
        """
        events: list[FileEvent] = []
        while not self._queue.empty() and len(events) < max_events:
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events