"""
tests/test_shell.py

Tests for ShellRunner and ShellResult.

Uses real subprocesses with simple cross-platform commands (python3 -c, echo).
Timeout tests use a short sleep command to keep the suite fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent.tools.shell import ShellResult, ShellRunner, _normalise_cmd

# ── Helpers ───────────────────────────────────────────────────────────────────

PYTHON = sys.executable  # ensures we use the same Python as the test suite


def make_runner(timeout: float = 10.0) -> ShellRunner:
    return ShellRunner(default_timeout=timeout)


# ── ShellResult ───────────────────────────────────────────────────────────────

def test_shell_result_success_true_on_exit_0() -> None:
    r = ShellResult(cmd=["echo", "hi"], stdout="hi", stderr="", exit_code=0)
    assert r.success is True


def test_shell_result_success_false_on_nonzero() -> None:
    r = ShellResult(cmd=["false"], stdout="", stderr="", exit_code=1)
    assert r.success is False


def test_shell_result_success_false_on_timeout() -> None:
    r = ShellResult(cmd=["sleep"], stdout="", stderr="", exit_code=-1, timed_out=True)
    assert r.success is False


def test_shell_result_output_combines_stdout_stderr() -> None:
    r = ShellResult(cmd=["x"], stdout="out\n", stderr="err\n", exit_code=0)
    assert "out" in r.output
    assert "err" in r.output


def test_shell_result_output_only_stdout_when_no_stderr() -> None:
    r = ShellResult(cmd=["x"], stdout="hello", stderr="", exit_code=0)
    assert r.output == "hello"


def test_shell_result_output_empty_when_both_empty() -> None:
    r = ShellResult(cmd=["x"], stdout="", stderr="", exit_code=0)
    assert r.output == ""


def test_shell_result_short_cmd() -> None:
    r = ShellResult(cmd=["git", "status"], stdout="", stderr="", exit_code=0)
    assert r.short_cmd == "git status"


def test_shell_result_str_ok() -> None:
    r = ShellResult(cmd=["echo", "hi"], stdout="hi", stderr="", exit_code=0)
    assert "ok" in str(r)


def test_shell_result_str_timeout() -> None:
    r = ShellResult(cmd=["sleep"], stdout="", stderr="", exit_code=-1, timed_out=True)
    assert "timeout" in str(r)


def test_shell_result_str_nonzero() -> None:
    r = ShellResult(cmd=["false"], stdout="", stderr="", exit_code=1)
    assert "exit 1" in str(r)


# ── _normalise_cmd ────────────────────────────────────────────────────────────

def test_normalise_cmd_list_passthrough() -> None:
    assert _normalise_cmd(["git", "status"]) == ["git", "status"]


def test_normalise_cmd_string_split() -> None:
    assert _normalise_cmd("git status --short") == ["git", "status", "--short"]


def test_normalise_cmd_string_with_quotes() -> None:
    result = _normalise_cmd('echo "hello world"')
    assert result == ["echo", "hello world"]


# ── ShellRunner construction ──────────────────────────────────────────────────

def test_runner_default_timeout() -> None:
    runner = ShellRunner()
    assert runner.default_timeout == 30.0


def test_runner_custom_timeout() -> None:
    runner = ShellRunner(default_timeout=5.0)
    assert runner.default_timeout == 5.0


def test_runner_zero_timeout_raises() -> None:
    with pytest.raises(ValueError, match="timeout"):
        ShellRunner(default_timeout=0)


def test_runner_negative_timeout_raises() -> None:
    with pytest.raises(ValueError, match="timeout"):
        ShellRunner(default_timeout=-1.0)


# ── run() — success cases ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_echo_succeeds() -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "print('hello')"])
    assert result.success is True
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_run_captures_stdout() -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "print('stdout content')"])
    assert "stdout content" in result.stdout


@pytest.mark.asyncio
async def test_run_captures_stderr() -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "import sys; sys.stderr.write('err content\\n')"])
    assert "err content" in result.stderr


@pytest.mark.asyncio
async def test_run_exit_code_zero_on_success() -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "pass"])
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_run_returns_shell_result() -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "pass"])
    assert isinstance(result, ShellResult)


@pytest.mark.asyncio
async def test_run_accepts_string_cmd() -> None:
    runner = make_runner()
    result = await runner.run(f"{PYTHON} -c \"print('from string')\"")
    assert "from string" in result.stdout


@pytest.mark.asyncio
async def test_run_with_cwd(tmp_path: Path) -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "import os; print(os.getcwd())"], cwd=tmp_path)
    assert result.success is True
    assert str(tmp_path) in result.stdout.strip()


@pytest.mark.asyncio
async def test_run_records_cwd(tmp_path: Path) -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "pass"], cwd=tmp_path)
    assert str(tmp_path) in result.cwd


# ── run() — failure cases ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_nonzero_exit_code() -> None:
    runner = make_runner()
    result = await runner.run([PYTHON, "-c", "import sys; sys.exit(42)"])
    assert result.success is False
    assert result.exit_code == 42


@pytest.mark.asyncio
async def test_run_nonzero_does_not_raise() -> None:
    runner = make_runner()
    # Should return a result, not raise
    result = await runner.run([PYTHON, "-c", "raise SystemExit(1)"])
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_run_command_not_found() -> None:
    runner = make_runner()
    result = await runner.run(["this_command_definitely_does_not_exist_xyz"])
    assert result.success is False
    assert result.exit_code == 127
    assert "not found" in result.stderr.lower() or "Command not found" in result.stderr


@pytest.mark.asyncio
async def test_run_never_raises() -> None:
    """run() must never propagate exceptions — always return ShellResult."""
    runner = make_runner()
    # Even a completely broken input should return, not raise
    result = await runner.run(["nonexistent_binary_xyz_abc"])
    assert isinstance(result, ShellResult)


# ── Timeout ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_timeout_sets_timed_out_flag() -> None:
    runner = make_runner()
    result = await runner.run(
        [PYTHON, "-c", "import time; time.sleep(10)"],
        timeout=0.2,
    )
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_run_timeout_exit_code_minus_one() -> None:
    runner = make_runner()
    result = await runner.run(
        [PYTHON, "-c", "import time; time.sleep(10)"],
        timeout=0.2,
    )
    assert result.exit_code == -1


@pytest.mark.asyncio
async def test_run_timeout_does_not_raise() -> None:
    runner = make_runner()
    result = await runner.run(
        [PYTHON, "-c", "import time; time.sleep(10)"],
        timeout=0.2,
    )
    assert isinstance(result, ShellResult)


@pytest.mark.asyncio
async def test_run_per_call_timeout_overrides_default() -> None:
    runner = ShellRunner(default_timeout=60.0)
    # per-call timeout of 0.2s should fire even though default is 60s
    result = await runner.run(
        [PYTHON, "-c", "import time; time.sleep(10)"],
        timeout=0.2,
    )
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_run_completes_before_timeout() -> None:
    runner = make_runner()
    result = await runner.run(
        [PYTHON, "-c", "print('fast')"],
        timeout=5.0,
    )
    assert result.success is True
    assert not result.timed_out


# ── run_shell_string ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_shell_string_basic() -> None:
    runner = make_runner()
    result = await runner.run_shell_string(f"{PYTHON} -c \"print('shell string')\"")
    assert result.success is True
    assert "shell string" in result.stdout


@pytest.mark.asyncio
async def test_run_shell_string_timeout() -> None:
    runner = make_runner()
    result = await runner.run_shell_string(
        f"{PYTHON} -c \"import time; time.sleep(10)\"",
        timeout=0.2,
    )
    assert result.timed_out is True


@pytest.mark.asyncio
async def test_run_shell_string_captures_stderr() -> None:
    runner = make_runner()
    result = await runner.run_shell_string(
        f"{PYTHON} -c \"import sys; sys.stderr.write('shell err\\n')\""
    )
    assert "shell err" in result.stderr


@pytest.mark.asyncio
async def test_run_shell_string_with_cwd(tmp_path: Path) -> None:
    runner = make_runner()
    result = await runner.run_shell_string(
        f"{PYTHON} -c \"import os; print(os.getcwd())\"",
        cwd=tmp_path,
    )
    assert result.success is True