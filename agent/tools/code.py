"""
agent/tools/code.py

Code agent tools — the four core operations the code agent performs.

Every tool returns a typed result dataclass. The orchestrator and
sub-agents never parse raw strings — they read structured fields.

Tools:
    read_file   — read a file with optional line range, returns numbered lines
    write_file  — atomic write via temp + rename, never partial-writes
    run_tests   — run pytest, parse summary into structured TestResult
    git_log     — return recent commits as structured GitCommit list

Design:
    - All tools are async functions, not methods on a class.
      They are stateless utilities — no shared state needed.
    - All filesystem operations are atomic where possible.
    - All tools return a result object with a .success property.
      Callers check result.success before using other fields.
    - Errors are captured in result.error — never raised to the caller.
      The agent should always get a result it can reason about.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from agent.core.logger import get_logger
from agent.tools.shell import ShellResult, ShellRunner

log = get_logger(__name__)

_runner = ShellRunner(default_timeout=60.0)


# ── ReadResult ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReadResult:
    path:     Path
    content:  str          # raw file content
    lines:    list[str]    # content split into lines (no trailing newline per line)
    numbered: str          # content with "  N | " prefixed to each line
    encoding: str
    size_bytes: int
    success:  bool
    error:    str = ""

    @property
    def line_count(self) -> int:
        return len(self.lines)


# ── WriteResult ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WriteResult:
    path:       Path
    bytes_written: int
    success:    bool
    error:      str = ""


# ── TestResult ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TestResult:
    passed:   int
    failed:   int
    errors:   int
    skipped:  int
    warnings: int
    duration_s: float
    success:  bool          # True only when failed==0 and errors==0
    output:   str           # full pytest output for display
    shell:    ShellResult | None = None
    error:    str = ""      # set if pytest could not be run at all

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def summary(self) -> str:
        parts = []
        if self.passed:  parts.append(f"{self.passed} passed")
        if self.failed:  parts.append(f"{self.failed} failed")
        if self.errors:  parts.append(f"{self.errors} error")
        if self.skipped: parts.append(f"{self.skipped} skipped")
        if not parts:    return "no tests collected"
        return ", ".join(parts) + f" in {self.duration_s:.2f}s"


# ── GitCommit ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GitCommit:
    hash:    str       # short 8-char hash
    author:  str
    date:    str       # ISO-8601 string
    message: str       # subject line only


@dataclass(frozen=True)
class GitLogResult:
    commits:  list[GitCommit]
    branch:   str
    success:  bool
    error:    str = ""

    @property
    def count(self) -> int:
        return len(self.commits)


# ── read_file ─────────────────────────────────────────────────────────────────

async def read_file(
    path: Path | str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    encoding: str = "utf-8",
) -> ReadResult:
    """
    Read a file and return structured content with numbered lines.

    Args:
        path:       File to read.
        start_line: 1-indexed first line to include (inclusive).
        end_line:   1-indexed last line to include (inclusive).
                    If both are None, the whole file is returned.
        encoding:   File encoding (default utf-8).

    Returns:
        ReadResult — always returns, never raises.
        On error: ReadResult.success=False, ReadResult.error is set.
    """
    target = Path(path).resolve()
    log.debug("code.read_file", path=str(target))

    try:
        raw = target.read_text(encoding=encoding, errors="replace")
    except FileNotFoundError:
        return _read_error(target, f"File not found: {target}", encoding)
    except PermissionError:
        return _read_error(target, f"Permission denied: {target}", encoding)
    except Exception as exc:
        return _read_error(target, str(exc), encoding)

    all_lines = raw.splitlines()

    # Apply line range
    if start_line is not None or end_line is not None:
        s = (start_line - 1) if start_line else 0
        e = end_line if end_line else len(all_lines)
        s = max(0, s)
        e = min(len(all_lines), e)
        selected = all_lines[s:e]
        line_offset = s
    else:
        selected = all_lines
        line_offset = 0

    # Build numbered display
    width = len(str(line_offset + len(selected)))
    numbered_lines = [
        f"{line_offset + i + 1:>{width}} | {line}"
        for i, line in enumerate(selected)
    ]
    numbered = "\n".join(numbered_lines)
    content  = "\n".join(selected)

    return ReadResult(
        path=target,
        content=content,
        lines=selected,
        numbered=numbered,
        encoding=encoding,
        size_bytes=target.stat().st_size,
        success=True,
    )


def _read_error(path: Path, error: str, encoding: str) -> ReadResult:
    log.warning("code.read_file_error", path=str(path), error=error)
    return ReadResult(
        path=path, content="", lines=[], numbered="",
        encoding=encoding, size_bytes=0, success=False, error=error,
    )


# ── write_file ────────────────────────────────────────────────────────────────

async def write_file(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    create_parents: bool = True,
) -> WriteResult:
    """
    Atomically write content to a file.

    Uses temp file + os.replace() so a crash mid-write never produces
    a partially-written file. The previous content is preserved until
    the rename succeeds.

    Args:
        path:            Target file path.
        content:         String content to write.
        encoding:        File encoding (default utf-8).
        create_parents:  Create parent directories if they don't exist.

    Returns:
        WriteResult — always returns, never raises.
    """
    target = Path(path).resolve()
    log.debug("code.write_file", path=str(target))

    try:
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)

        encoded = content.encode(encoding)

        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=target.parent,
            prefix=".write_tmp_",
            suffix=target.suffix or ".tmp",
        )
        tmp_path = Path(tmp_path_str)

        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(encoded)
            os.replace(tmp_path, target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        log.debug("code.write_file_ok", path=str(target), bytes=len(encoded))
        return WriteResult(path=target, bytes_written=len(encoded), success=True)

    except PermissionError as exc:
        return _write_error(target, f"Permission denied: {exc}")
    except Exception as exc:
        return _write_error(target, str(exc))


def _write_error(path: Path, error: str) -> WriteResult:
    log.warning("code.write_file_error", path=str(path), error=error)
    return WriteResult(path=path, bytes_written=0, success=False, error=error)


# ── run_tests ─────────────────────────────────────────────────────────────────

async def run_tests(
    *,
    cwd: Path | str | None = None,
    args: list[str] | None = None,
    timeout: float = 120.0,
) -> TestResult:
    """
    Run pytest and return structured results.

    Args:
        cwd:     Working directory (project root). Defaults to current dir.
        args:    Extra pytest arguments (e.g. ["-x", "--tb=short", "tests/"]).
                 Defaults to ["--tb=short", "-q"].
        timeout: Max seconds to wait for pytest. Defaults to 120s.

    Returns:
        TestResult with pass/fail counts parsed from pytest summary line.
    """
    cmd = ["python", "-m", "pytest"] + (args or ["--tb=short", "-q"])
    log.info("code.run_tests", cmd=cmd, cwd=str(cwd) if cwd else None)

    shell = await _runner.run(cmd, cwd=cwd, timeout=timeout)

    if shell.timed_out:
        return TestResult(
            passed=0, failed=0, errors=0, skipped=0, warnings=0,
            duration_s=timeout,
            success=False,
            output=shell.output,
            shell=shell,
            error=f"pytest timed out after {timeout}s",
        )

    output = shell.output
    return _parse_pytest_output(output, shell)


def _parse_pytest_output(output: str, shell: ShellResult) -> TestResult:
    """
    Parse pytest's summary line into structured counts.

    Handles formats like:
        5 passed in 0.23s
        3 passed, 2 failed in 1.05s
        1 failed, 1 error in 0.44s
        4 passed, 1 skipped in 0.18s
        1 passed, 2 warnings in 0.12s
        no tests ran in 0.01s
    """
    passed = failed = errors = skipped = warnings = 0
    duration_s = 0.0

    # Look for the summary line — it ends with "in X.XXs"
    summary_pattern = re.compile(
        r"([\d]+ \w+(?:,\s*[\d]+ \w+)*)\s+in\s+([\d.]+)s",
        re.IGNORECASE,
    )
    match = summary_pattern.search(output)

    if match:
        summary_text = match.group(1)
        duration_s   = float(match.group(2))

        count_pattern = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|warning|warnings)")
        for m in count_pattern.finditer(summary_text):
            n    = int(m.group(1))
            kind = m.group(2).lower()
            if kind == "passed":              passed   = n
            elif kind in ("failed",):         failed   = n
            elif kind in ("error", "errors"): errors   = n
            elif kind == "skipped":           skipped  = n
            elif kind in ("warning","warnings"): warnings = n

    success = (shell.exit_code == 0) and (failed == 0) and (errors == 0)

    return TestResult(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        warnings=warnings,
        duration_s=duration_s,
        success=success,
        output=output,
        shell=shell,
    )


# ── git_log ───────────────────────────────────────────────────────────────────

async def git_log(
    *,
    cwd: Path | str | None = None,
    limit: int = 20,
    branch: str | None = None,
) -> GitLogResult:
    """
    Return recent git commits as structured GitCommit objects.

    Args:
        cwd:    Repository root. Defaults to current directory.
        limit:  Maximum commits to return. Defaults to 20.
        branch: Specific branch to log. Defaults to current branch.

    Returns:
        GitLogResult — always returns, never raises.
    """
    # Use a machine-parseable format: hash|author|date|message
    fmt  = "%h|%an|%aI|%s"
    cmd  = ["git", "log", f"--pretty=format:{fmt}", f"-{limit}"]
    if branch:
        cmd.append(branch)

    log.debug("code.git_log", cwd=str(cwd) if cwd else None, limit=limit)
    shell = await _runner.run(cmd, cwd=cwd, timeout=15.0)

    if not shell.success:
        return GitLogResult(
            commits=[], branch="",
            success=False,
            error=shell.stderr or f"git log failed (exit {shell.exit_code})",
        )

    commits = _parse_git_log(shell.stdout)
    current_branch = await _get_current_branch(cwd)

    return GitLogResult(commits=commits, branch=current_branch, success=True)


def _parse_git_log(output: str) -> list[GitCommit]:
    commits: list[GitCommit] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", maxsplit=3)
        if len(parts) != 4:
            continue
        commits.append(GitCommit(
            hash=parts[0].strip(),
            author=parts[1].strip(),
            date=parts[2].strip(),
            message=parts[3].strip(),
        ))
    return commits


async def _get_current_branch(cwd: Path | str | None) -> str:
    result = await _runner.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
        timeout=5.0,
    )
    return result.stdout.strip() if result.success else "unknown"