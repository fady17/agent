"""
agent/tools/shell.py

Async shell runner — executes system commands safely and returns
structured results the agent can reason about.

Design:
    - Uses asyncio.create_subprocess_exec (not shell=True) by default.
      This avoids shell injection — arguments are passed as a list,
      never interpolated into a shell string.
    - shell=True is available via run_shell_string() for cases where
      pipes, redirects, or shell builtins are genuinely needed.
      Use it deliberately, never by default.
    - Every run has an enforced timeout. The process is killed on timeout —
      first SIGTERM, then SIGKILL after a grace period.
    - stdout and stderr are always captured. The caller decides what
      to surface to the user or log.
    - Working directory defaults to the current directory but can be
      overridden per-call — important for project-scoped commands.

ShellResult is the only output type. The agent reads:
    result.success    — did it exit 0?
    result.stdout     — captured standard output
    result.stderr     — captured standard error
    result.exit_code  — raw exit code for diagnostic logging
    result.timed_out  — True if the timeout was hit
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from agent.core.config import cfg
from agent.core.logger import get_logger

log = get_logger(__name__)

# Default timeout — most dev commands finish well within 30s.
# Long-running commands (builds, test suites) should pass an explicit timeout.
DEFAULT_TIMEOUT_S  = 30.0
SIGTERM_GRACE_S    = 3.0    # wait this long after SIGTERM before SIGKILL
MAX_OUTPUT_BYTES   = 1_000_000  # 1 MB — truncate runaway output


# ── ShellResult ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShellResult:
    """
    Structured result of a shell command execution.

    stdout and stderr are decoded UTF-8 with replacement for non-UTF bytes.
    If the process timed out, timed_out=True and exit_code=-1.
    """

    cmd:       list[str]     # the command as executed (list form)
    stdout:    str
    stderr:    str
    exit_code: int
    timed_out: bool = False
    cwd:       str  = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined stdout + stderr, stdout first."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout)
        if self.stderr.strip():
            parts.append(self.stderr)
        return "\n".join(parts)

    @property
    def short_cmd(self) -> str:
        """Command as a single string for logging."""
        return " ".join(self.cmd)

    def __str__(self) -> str:
        status = "ok" if self.success else ("timeout" if self.timed_out else f"exit {self.exit_code}")
        return f"ShellResult({status}: {self.short_cmd!r})"


# ── ShellRunner ───────────────────────────────────────────────────────────────

class ShellRunner:
    """
    Async shell command executor.

    Stateless — safe to share across the session.
    Instantiate once and reuse: runner = ShellRunner()
    """

    def __init__(self, default_timeout: float = DEFAULT_TIMEOUT_S) -> None:
        if default_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {default_timeout}")
        self.default_timeout = default_timeout

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        cmd: list[str] | str,
        *,
        cwd: Path | str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        """
        Execute a command and return a ShellResult.

        Args:
            cmd:     Command as a list of strings (preferred) or a single
                     string that will be split via shlex.split().
                     Use run_shell_string() if you need shell features.
            cwd:     Working directory. Defaults to current directory.
            timeout: Per-call timeout in seconds. Defaults to self.default_timeout.
            env:     Optional environment variables (merged with current env
                     if partial, or used exclusively if a full env dict).

        Returns:
            ShellResult — always returns, never raises.
            On timeout: ShellResult.timed_out=True, exit_code=-1.
            On OS error: ShellResult.exit_code=-2, stderr contains the error.
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        cmd_list = _normalise_cmd(cmd)
        cwd_str  = str(cwd) if cwd else None

        log.debug(
            "shell.run",
            cmd=cmd_list,
            cwd=cwd_str,
            timeout=effective_timeout,
        )

        try:
            return await self._execute(
                cmd_list,
                cwd=cwd_str,
                timeout=effective_timeout,
                env=env,
            )
        except Exception as exc:
            log.error("shell.unexpected_error", cmd=cmd_list, error=str(exc))
            return ShellResult(
                cmd=cmd_list,
                stdout="",
                stderr=f"Unexpected error: {exc}",
                exit_code=-2,
                cwd=cwd_str or "",
            )

    async def run_shell_string(
        self,
        command: str,
        *,
        cwd: Path | str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        """
        Execute a raw shell string with shell=True.

        Use this only when you need:
            - Pipes:       "rg pattern | head -20"
            - Redirects:   "command > output.txt"
            - Shell builtins: "cd dir && command"

        The command is NOT sanitised — never interpolate untrusted input.

        Returns ShellResult same as run().
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        cwd_str = str(cwd) if cwd else None

        log.debug(
            "shell.run_string",
            cmd=command,
            cwd=cwd_str,
            timeout=effective_timeout,
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd_str,
                env=env,
            )
            return await self._wait_with_timeout(
                proc,
                cmd=[command],
                timeout=effective_timeout,
                cwd=cwd_str or "",
            )
        except Exception as exc:
            log.error("shell.run_string_error", cmd=command, error=str(exc))
            return ShellResult(
                cmd=[command],
                stdout="",
                stderr=f"Unexpected error: {exc}",
                exit_code=-2,
                cwd=cwd_str or "",
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _execute(
        self,
        cmd: list[str],
        *,
        cwd: str | None,
        timeout: float,
        env: dict[str, str] | None,
    ) -> ShellResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        except FileNotFoundError:
            return ShellResult(
                cmd=cmd,
                stdout="",
                stderr=f"Command not found: {cmd[0]!r}",
                exit_code=127,    # POSIX convention for command not found
                cwd=cwd or "",
            )
        except PermissionError as exc:
            return ShellResult(
                cmd=cmd,
                stdout="",
                stderr=f"Permission denied: {exc}",
                exit_code=126,    # POSIX convention for permission denied
                cwd=cwd or "",
            )

        return await self._wait_with_timeout(
            proc, cmd=cmd, timeout=timeout, cwd=cwd or ""
        )

    async def _wait_with_timeout(
        self,
        proc: asyncio.subprocess.Process,
        *,
        cmd: list[str],
        timeout: float,
        cwd: str,
    ) -> ShellResult:
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            exit_code = proc.returncode if proc.returncode is not None else -1

            stdout = _decode(stdout_bytes)
            stderr = _decode(stderr_bytes)

            result = ShellResult(
                cmd=cmd,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=False,
                cwd=cwd,
            )

            log.debug(
                "shell.done",
                cmd=cmd,
                exit_code=exit_code,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
            )

            if not result.success:
                log.warning(
                    "shell.nonzero_exit",
                    cmd=cmd,
                    exit_code=exit_code,
                    stderr_preview=stderr[:200],
                )

            return result

        except asyncio.TimeoutError:
            log.warning("shell.timeout", cmd=cmd, timeout=timeout)
            await _kill_process(proc)
            # Collect whatever output was produced before timeout
            try:
                stdout_bytes = await proc.stdout.read() if proc.stdout else b""
                stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            except Exception:
                stdout_bytes = b""
                stderr_bytes = b""

            return ShellResult(
                cmd=cmd,
                stdout=_decode(stdout_bytes),
                stderr=f"Process timed out after {timeout}s",
                exit_code=-1,
                timed_out=True,
                cwd=cwd,
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_cmd(cmd: list[str] | str) -> list[str]:
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return list(cmd)


def _decode(data: bytes) -> str:
    text = data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if len(data) > MAX_OUTPUT_BYTES:
        text += f"\n[output truncated at {MAX_OUTPUT_BYTES // 1024}KB]"
    return text


async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    """Send SIGTERM then SIGKILL after grace period."""
    try:
        proc.terminate()
        await asyncio.sleep(SIGTERM_GRACE_S)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
    except ProcessLookupError:
        pass  # Already exited
    except Exception as exc:
        log.warning("shell.kill_failed", error=str(exc))