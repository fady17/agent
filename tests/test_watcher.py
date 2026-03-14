"""
tests/test_watcher.py

Tests for FileWatcher and FileEvent.

Two test categories:
    1. Schema / pure logic tests — no filesystem, fast
    2. Integration tests — create real files on disk, verify queue receives events

Integration tests use a short asyncio.sleep to let watchdog's observer
thread process events. They are marked with pytest.mark.integration
so they can be skipped in fast CI runs:
    uv run pytest tests/test_watcher.py -m "not integration"
"""

from __future__ import annotations

import asyncio
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.watcher import (
    DEFAULT_IGNORE_PATTERNS,
    FileEvent,
    FileEventType,
    FileWatcher,
    _AsyncBridgeHandler,
)


# ── FileEvent schema ──────────────────────────────────────────────────────────

def make_event(
    path: str = "/project/main.py",
    event_type: FileEventType = FileEventType.MODIFIED,
    dest: str | None = None,
) -> FileEvent:
    return FileEvent(
        path=Path(path),
        event_type=event_type,
        dest_path=Path(dest) if dest else None,
    )


def test_file_event_extension_lowercase() -> None:
    evt = make_event("/project/App.PY")
    assert evt.extension == ".py"


def test_file_event_name() -> None:
    evt = make_event("/project/src/main.py")
    assert evt.name == "main.py"


def test_file_event_is_python_true() -> None:
    assert make_event("/project/main.py").is_python is True


def test_file_event_is_python_false() -> None:
    assert make_event("/project/model.blend").is_python is False


def test_file_event_is_blend_true() -> None:
    assert make_event("/project/scene.blend").is_blend is True


def test_file_event_is_dart_true() -> None:
    assert make_event("/project/lib/widget.dart").is_dart is True


def test_file_event_is_asset_glb() -> None:
    assert make_event("/project/model.glb").is_asset is True


def test_file_event_is_asset_fbx() -> None:
    assert make_event("/project/model.fbx").is_asset is True


def test_file_event_is_asset_false_for_py() -> None:
    assert make_event("/project/main.py").is_asset is False


def test_file_event_timestamp_utc() -> None:
    evt = make_event()
    assert evt.timestamp.tzinfo == timezone.utc


def test_file_event_str_representation() -> None:
    evt = make_event("/project/main.py", FileEventType.CREATED)
    s = str(evt)
    assert "created" in s
    assert "main.py" in s


def test_file_event_moved_has_dest_path() -> None:
    evt = make_event("/old.py", FileEventType.MOVED, dest="/new.py")
    assert evt.dest_path == Path("/new.py")


def test_file_event_is_frozen() -> None:
    evt = make_event()
    with pytest.raises(Exception):
        evt.event_type = FileEventType.DELETED  # type: ignore[misc]


# ── FileWatcher construction ──────────────────────────────────────────────────

def test_watcher_requires_at_least_one_path() -> None:
    with pytest.raises(ValueError, match="At least one"):
        FileWatcher(paths=[])


def test_watcher_accepts_multiple_paths(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    watcher = FileWatcher(paths=[dir_a, dir_b])
    assert len(watcher.watched_paths) == 2


def test_watcher_creates_queue_if_not_provided(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    assert watcher.queue is not None


def test_watcher_uses_provided_queue(tmp_path: Path) -> None:
    q: asyncio.Queue = asyncio.Queue()
    watcher = FileWatcher(paths=[tmp_path], queue=q)
    assert watcher.queue is q


def test_watcher_not_running_initially(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    assert watcher.is_running is False


# ── _AsyncBridgeHandler ignore patterns ──────────────────────────────────────

def make_handler(ignore_patterns: list[str] | None = None) -> _AsyncBridgeHandler:
    loop = MagicMock()
    queue: asyncio.Queue = asyncio.Queue()
    return _AsyncBridgeHandler(
        loop=loop,
        queue=queue,
        ignore_patterns=ignore_patterns or DEFAULT_IGNORE_PATTERNS,
    )


def test_handler_ignores_pyc_files() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/main.cpython-312.pyc")) is True


def test_handler_ignores_pycache() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/__pycache__/main.cpython-312.pyc")) is True


def test_handler_ignores_git_files() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/.git/COMMIT_EDITMSG")) is True


def test_handler_ignores_venv() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/.venv/lib/python3.12/site.py")) is True


def test_handler_ignores_swp_files() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/.main.py.swp")) is True


def test_handler_ignores_ds_store() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/.DS_Store")) is True


def test_handler_ignores_agent_memory() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/home/user/.agent/memory/episodic/2026/01/01/evt-001.json")) is True


def test_handler_does_not_ignore_python_source() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/main.py")) is False


def test_handler_does_not_ignore_blend_file() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/scene.blend")) is False


def test_handler_does_not_ignore_dart_file() -> None:
    handler = make_handler()
    assert handler._should_ignore(Path("/project/lib/widget.dart")) is False


def test_handler_custom_ignore_pattern() -> None:
    handler = make_handler(ignore_patterns=["*.log"])
    assert handler._should_ignore(Path("/project/debug.log")) is True
    assert handler._should_ignore(Path("/project/main.py")) is False


# ── _AsyncBridgeHandler enqueue ───────────────────────────────────────────────

def test_handler_enqueue_calls_queue_put() -> None:
    loop = MagicMock()
    queue: asyncio.Queue = asyncio.Queue()
    handler = _AsyncBridgeHandler(loop=loop, queue=queue, ignore_patterns=[])
    evt = make_event()
    handler._enqueue(evt)
    assert not queue.empty()
    assert queue.get_nowait() is evt


def test_handler_enqueue_full_queue_does_not_raise() -> None:
    loop = MagicMock()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait(make_event())  # fill it
    handler = _AsyncBridgeHandler(loop=loop, queue=queue, ignore_patterns=[])
    handler._enqueue(make_event())  # should not raise


def test_handler_enqueue_calls_callback() -> None:
    loop = MagicMock()
    queue: asyncio.Queue = asyncio.Queue()
    received: list[FileEvent] = []
    handler = _AsyncBridgeHandler(
        loop=loop,
        queue=queue,
        ignore_patterns=[],
        callback=received.append,
    )
    evt = make_event()
    handler._enqueue(evt)
    assert received == [evt]


# ── FileWatcher lifecycle ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_watcher_start_sets_running(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    await watcher.start()
    assert watcher.is_running is True
    await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_stop_clears_running(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    await watcher.start()
    await watcher.stop()
    assert watcher.is_running is False


@pytest.mark.asyncio
async def test_watcher_start_idempotent(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    await watcher.start()
    await watcher.start()  # second call should be no-op
    assert watcher.is_running is True
    await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_stop_when_not_running_does_not_raise(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    await watcher.stop()  # never started — should not raise


@pytest.mark.asyncio
async def test_watcher_context_manager(tmp_path: Path) -> None:
    async with FileWatcher(paths=[tmp_path]) as watcher:
        assert watcher.is_running is True
    assert watcher.is_running is False


@pytest.mark.asyncio
async def test_watcher_nonexistent_path_does_not_crash(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    watcher = FileWatcher(paths=[missing])
    await watcher.start()
    assert watcher.is_running is True
    await watcher.stop()


# ── Queue helpers ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_event_returns_none_on_timeout(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    result = await watcher.get_event(timeout=0.05)
    assert result is None


@pytest.mark.asyncio
async def test_get_event_returns_event_when_queued(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    evt = make_event()
    await watcher.queue.put(evt)
    result = await watcher.get_event(timeout=1.0)
    assert result is evt


def test_drain_returns_all_queued_events(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    for _ in range(5):
        watcher.queue.put_nowait(make_event())
    events = watcher.drain()
    assert len(events) == 5


def test_drain_respects_max_events(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    for _ in range(10):
        watcher.queue.put_nowait(make_event())
    events = watcher.drain(max_events=3)
    assert len(events) == 3
    assert watcher.queue.qsize() == 7  # remaining in queue


def test_drain_empty_queue_returns_empty_list(tmp_path: Path) -> None:
    watcher = FileWatcher(paths=[tmp_path])
    assert watcher.drain() == []


# ── Integration: real filesystem events ──────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_detects_file_creation(tmp_path: Path) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    async with FileWatcher(paths=[tmp_path], queue=queue):
        await asyncio.sleep(0.1)  # let observer start
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        await asyncio.sleep(0.5)  # let watchdog process the event

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert any(e.event_type == FileEventType.CREATED and e.name == "test.py" for e in events)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_detects_file_modification(tmp_path: Path) -> None:
    test_file = tmp_path / "watch.py"
    test_file.write_text("v1")

    queue: asyncio.Queue = asyncio.Queue()
    async with FileWatcher(paths=[tmp_path], queue=queue):
        await asyncio.sleep(0.1)
        test_file.write_text("v2")
        await asyncio.sleep(0.5)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert any(e.event_type == FileEventType.MODIFIED and e.name == "watch.py" for e in events)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_ignores_pyc(tmp_path: Path) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    async with FileWatcher(paths=[tmp_path], queue=queue):
        await asyncio.sleep(0.1)
        pyc_file = tmp_path / "main.pyc"
        pyc_file.write_text("bytecode")
        await asyncio.sleep(0.5)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert not any(e.name == "main.pyc" for e in events)