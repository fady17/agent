"""
tests/test_code.py

Tests for code agent tools: read_file, write_file, run_tests, git_log.

Uses real filesystem (tmp_path) and real subprocesses for run_tests and git_log.
git_log tests require git to be installed and available on PATH.
run_tests tests use a small synthetic test file written to tmp_path.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from agent.tools.code import (
    GitCommit,
    GitLogResult,
    ReadResult,
    TestResult,
    WriteResult,
    git_log,
    read_file,
    run_tests,
    write_file,
    _parse_git_log,
    _parse_pytest_output,
)
from agent.tools.shell import ShellResult

# Skip git tests if git is not installed
git_available = shutil.which("git") is not None
skip_no_git = pytest.mark.skipif(not git_available, reason="git not installed")


# ── ReadResult ────────────────────────────────────────────────────────────────

def test_read_result_line_count() -> None:
    r = ReadResult(
        path=Path("/f"), content="a\nb\nc", lines=["a", "b", "c"],
        numbered="", encoding="utf-8", size_bytes=5, success=True,
    )
    assert r.line_count == 3


# ── TestResult ────────────────────────────────────────────────────────────────

def test_test_result_total() -> None:
    r = TestResult(passed=3, failed=1, errors=0, skipped=1, warnings=0,
                   duration_s=0.5, success=False, output="")
    assert r.total == 5


def test_test_result_summary_all_passed() -> None:
    r = TestResult(passed=5, failed=0, errors=0, skipped=0, warnings=0,
                   duration_s=0.23, success=True, output="")
    assert "5 passed" in r.summary
    assert "0.23s" in r.summary


def test_test_result_summary_mixed() -> None:
    r = TestResult(passed=3, failed=2, errors=0, skipped=1, warnings=0,
                   duration_s=1.05, success=False, output="")
    assert "3 passed" in r.summary
    assert "2 failed" in r.summary
    assert "1 skipped" in r.summary


def test_test_result_summary_no_tests() -> None:
    r = TestResult(passed=0, failed=0, errors=0, skipped=0, warnings=0,
                   duration_s=0.01, success=False, output="")
    assert "no tests" in r.summary


# ── _parse_pytest_output ──────────────────────────────────────────────────────

def make_shell(exit_code: int = 0) -> ShellResult:
    return ShellResult(cmd=["pytest"], stdout="", stderr="", exit_code=exit_code)


def test_parse_all_passed() -> None:
    output = "5 passed in 0.23s"
    r = _parse_pytest_output(output, make_shell(0))
    assert r.passed == 5
    assert r.duration_s == pytest.approx(0.23)
    assert r.success is True


def test_parse_passed_and_failed() -> None:
    output = "3 passed, 2 failed in 1.05s"
    r = _parse_pytest_output(output, make_shell(1))
    assert r.passed == 3
    assert r.failed == 2
    assert r.success is False


def test_parse_with_errors() -> None:
    output = "1 passed, 1 error in 0.44s"
    r = _parse_pytest_output(output, make_shell(1))
    assert r.errors == 1
    assert r.success is False


def test_parse_with_skipped() -> None:
    output = "4 passed, 1 skipped in 0.18s"
    r = _parse_pytest_output(output, make_shell(0))
    assert r.passed == 4
    assert r.skipped == 1
    assert r.success is True


def test_parse_with_warnings() -> None:
    output = "2 passed, 3 warnings in 0.12s"
    r = _parse_pytest_output(output, make_shell(0))
    assert r.warnings == 3
    assert r.success is True


def test_parse_no_match_returns_zeros() -> None:
    output = "some other output with no summary"
    r = _parse_pytest_output(output, make_shell(0))
    assert r.passed == 0
    assert r.failed == 0


def test_parse_exit_nonzero_overrides_success() -> None:
    output = "5 passed in 0.5s"
    r = _parse_pytest_output(output, make_shell(1))
    assert r.success is False


# ── _parse_git_log ────────────────────────────────────────────────────────────

def test_parse_git_log_basic() -> None:
    output = "abc12345|Fady|2026-03-14T10:00:00+02:00|feat: add classifier\n"
    commits = _parse_git_log(output)
    assert len(commits) == 1
    assert commits[0].hash == "abc12345"
    assert commits[0].author == "Fady"
    assert commits[0].message == "feat: add classifier"


def test_parse_git_log_multiple() -> None:
    output = (
        "aaa11111|Alice|2026-03-14T10:00:00Z|first commit\n"
        "bbb22222|Bob|2026-03-13T09:00:00Z|second commit\n"
    )
    commits = _parse_git_log(output)
    assert len(commits) == 2
    assert commits[0].hash == "aaa11111"
    assert commits[1].hash == "bbb22222"


def test_parse_git_log_empty_output() -> None:
    assert _parse_git_log("") == []


def test_parse_git_log_skips_malformed_lines() -> None:
    output = "not|enough\nabc|author|date|message\n"
    commits = _parse_git_log(output)
    assert len(commits) == 1
    assert commits[0].hash == "abc"


def test_parse_git_log_message_with_pipe() -> None:
    """Message containing | should not be split."""
    output = "abc12345|Fady|2026-03-14T10:00:00Z|fix: handle pipe | in message\n"
    commits = _parse_git_log(output)
    assert len(commits) == 1
    assert commits[0].message == "fix: handle pipe | in message"


# ── read_file ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_file_basic(tmp_path: Path) -> None:
    f = tmp_path / "main.py"
    f.write_text("line one\nline two\nline three\n")
    result = await read_file(f)
    assert result.success is True
    assert result.line_count == 3
    assert "line one" in result.content


@pytest.mark.asyncio
async def test_read_file_numbered_lines(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("alpha\nbeta\ngamma\n")
    result = await read_file(f)
    assert "1 | alpha" in result.numbered
    assert "2 | beta" in result.numbered
    assert "3 | gamma" in result.numbered


@pytest.mark.asyncio
async def test_read_file_line_range(tmp_path: Path) -> None:
    f = tmp_path / "long.py"
    f.write_text("\n".join(f"line {i}" for i in range(1, 11)))
    result = await read_file(f, start_line=3, end_line=5)
    assert result.success is True
    assert result.line_count == 3
    assert "line 3" in result.content
    assert "line 5" in result.content
    assert "line 1" not in result.content
    assert "line 6" not in result.content


@pytest.mark.asyncio
async def test_read_file_start_line_only(tmp_path: Path) -> None:
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n")
    result = await read_file(f, start_line=3)
    assert result.line_count == 3
    assert "c" in result.content


@pytest.mark.asyncio
async def test_read_file_end_line_only(tmp_path: Path) -> None:
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n")
    result = await read_file(f, end_line=2)
    assert result.line_count == 2
    assert "a" in result.content
    assert "c" not in result.content


@pytest.mark.asyncio
async def test_read_file_not_found() -> None:
    result = await read_file(Path("/nonexistent/file.py"))
    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_read_file_returns_size_bytes(tmp_path: Path) -> None:
    f = tmp_path / "sized.py"
    content = "hello world"
    f.write_text(content)
    result = await read_file(f)
    assert result.size_bytes == len(content.encode())


@pytest.mark.asyncio
async def test_read_file_path_as_string(tmp_path: Path) -> None:
    f = tmp_path / "str_path.py"
    f.write_text("content")
    result = await read_file(str(f))
    assert result.success is True


# ── write_file ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_file_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "output.py"
    result = await write_file(target, "print('hello')")
    assert result.success is True
    assert target.exists()
    assert target.read_text() == "print('hello')"


@pytest.mark.asyncio
async def test_write_file_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "file.py"
    result = await write_file(target, "content")
    assert result.success is True
    assert target.exists()


@pytest.mark.asyncio
async def test_write_file_returns_bytes_written(tmp_path: Path) -> None:
    target = tmp_path / "f.py"
    content = "hello world"
    result = await write_file(target, content)
    assert result.bytes_written == len(content.encode())


@pytest.mark.asyncio
async def test_write_file_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "existing.py"
    target.write_text("old content")
    result = await write_file(target, "new content")
    assert result.success is True
    assert target.read_text() == "new content"


@pytest.mark.asyncio
async def test_write_file_atomic_no_tmp_left(tmp_path: Path) -> None:
    target = tmp_path / "atomic.py"
    await write_file(target, "content")
    tmp_files = list(tmp_path.glob(".write_tmp_*"))
    assert tmp_files == []


@pytest.mark.asyncio
async def test_write_file_path_as_string(tmp_path: Path) -> None:
    target = tmp_path / "str.py"
    result = await write_file(str(target), "data")
    assert result.success is True


@pytest.mark.asyncio
async def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "roundtrip.py"
    original = "def hello():\n    return 'world'\n"
    await write_file(target, original)
    result = await read_file(target)
    assert result.content == original.rstrip("\n")


# ── run_tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_tests_passing_suite(tmp_path: Path) -> None:
    """Write a simple test file and run it."""
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_always_passes():\n    assert 1 + 1 == 2\n")
    result = await run_tests(cwd=tmp_path, args=["-q", "--tb=short"])
    assert result.success is True
    assert result.passed >= 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_run_tests_failing_suite(tmp_path: Path) -> None:
    test_file = tmp_path / "test_fail.py"
    test_file.write_text("def test_always_fails():\n    assert False\n")
    result = await run_tests(cwd=tmp_path, args=["-q", "--tb=no"])
    assert result.success is False
    assert result.failed >= 1


@pytest.mark.asyncio
async def test_run_tests_captures_output(tmp_path: Path) -> None:
    test_file = tmp_path / "test_out.py"
    test_file.write_text("def test_ok():\n    assert True\n")
    result = await run_tests(cwd=tmp_path, args=["-q"])
    assert len(result.output) > 0


@pytest.mark.asyncio
async def test_run_tests_timeout(tmp_path: Path) -> None:
    test_file = tmp_path / "test_slow.py"
    test_file.write_text(
        "import time\ndef test_slow():\n    time.sleep(10)\n    assert True\n"
    )
    result = await run_tests(cwd=tmp_path, args=["-q"], timeout=0.5)
    assert result.success is False
    assert result.timed_out if hasattr(result, "timed_out") else result.error # type: ignore


@pytest.mark.asyncio
async def test_run_tests_returns_shell_result(tmp_path: Path) -> None:
    test_file = tmp_path / "test_s.py"
    test_file.write_text("def test_ok(): assert True\n")
    result = await run_tests(cwd=tmp_path)
    assert result.shell is not None


# ── git_log ───────────────────────────────────────────────────────────────────

@skip_no_git
@pytest.mark.asyncio
async def test_git_log_success(tmp_path: Path) -> None:
    """Init a real git repo and make a commit."""
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path, capture_output=True)

    result = await git_log(cwd=tmp_path)
    assert result.success is True
    assert result.count == 1
    assert result.commits[0].message == "initial commit"


@skip_no_git
@pytest.mark.asyncio
async def test_git_log_returns_branch(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

    result = await git_log(cwd=tmp_path)
    assert result.branch in ("main", "master")


@skip_no_git
@pytest.mark.asyncio
async def test_git_log_respects_limit(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text(str(i))
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"commit {i}"], cwd=tmp_path, capture_output=True)

    result = await git_log(cwd=tmp_path, limit=3)
    assert result.success is True
    assert result.count == 3


@skip_no_git
@pytest.mark.asyncio
async def test_git_log_failure_outside_repo(tmp_path: Path) -> None:
    result = await git_log(cwd=tmp_path)  # not a git repo
    assert result.success is False
    assert result.error != ""