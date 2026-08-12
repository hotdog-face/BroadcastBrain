"""Subprocess execution with timeouts, retries and honest error reporting.

Every external tool invocation in BroadcastBrain goes through :func:`run`.
That gives one place for timeout handling, one place for `nice`, and one place
that turns an exit code into a typed exception the queue knows how to classify.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

from ..constants import SUBPROCESS_KILL_GRACE_S
from ..errors import SubprocessError, SubprocessTimeoutError, ToolNotFoundError

log = logging.getLogger(__name__)

CommandArg = Union[str, Path]


@dataclass(frozen=True)
class CompletedCommand:
    """Result of a finished external command."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def command_line(self) -> str:
        return " ".join(_quote(a) for a in self.args)


def _quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


def which(binary: str) -> Optional[str]:
    """Resolve a binary name or absolute path, returning None if unavailable."""
    candidate = Path(binary)
    if candidate.is_absolute():
        return str(candidate) if os.access(candidate, os.X_OK) else None
    return shutil.which(binary)


def require_tool(binary: str, *, hint: str = "") -> str:
    """Resolve a binary or raise :class:`ToolNotFoundError` with install advice."""
    resolved = which(binary)
    if resolved is None:
        suffix = f" ({hint})" if hint else ""
        raise ToolNotFoundError(f"required external tool not found: {binary}{suffix}")
    return resolved


def run(
    args: Sequence[CommandArg],
    *,
    timeout_s: Optional[float] = None,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    nice: int = 0,
    check: bool = True,
    input_text: Optional[str] = None,
    log_command: bool = True,
    stderr_tail_lines: int = 0,
    on_line: Optional[Callable[[str], None]] = None,
) -> CompletedCommand:
    """Run a command to completion.

    Args:
        args: Program and arguments. ``Path`` values are stringified.
        timeout_s: Wall-clock budget. Exceeding it raises
            :class:`~broadcastbrain.errors.SubprocessTimeoutError` (retryable).
        nice: POSIX niceness for the child, so overnight runs stay polite.
        check: Raise :class:`~broadcastbrain.errors.SubprocessError` on a
            non-zero exit. Set False when a non-zero exit is meaningful data.
        stderr_tail_lines: Keep only the last N stderr lines. ffmpeg detector
            passes emit megabytes of stderr; this bounds memory.
        on_line: Called for each stderr line as it arrives. Used by the signal
            collector to parse detector output while ffmpeg is still running.

    Raises:
        SubprocessTimeoutError: the child exceeded ``timeout_s``.
        SubprocessError: non-zero exit and ``check`` is True.
    """
    argv = [str(a) for a in args]
    if not argv:
        raise ValueError("run() requires at least one argument")

    resolved = which(argv[0])
    if resolved is None:
        raise ToolNotFoundError(f"external tool not found on PATH: {argv[0]}")
    argv[0] = resolved

    if log_command:
        log.debug("exec: %s", " ".join(_quote(a) for a in argv))

    preexec = _make_preexec(nice)
    child_env = dict(os.environ)
    if env:
        child_env.update(env)

    started = time.monotonic()
    stderr_lines: list[str] = []
    stdout_chunks: list[str] = []

    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env=child_env,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
            preexec_fn=preexec,
            start_new_session=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - race with `which`
        raise ToolNotFoundError(f"external tool disappeared: {argv[0]}") from exc

    try:
        if on_line is None and stderr_tail_lines <= 0:
            stdout, stderr = process.communicate(input=input_text, timeout=timeout_s)
            stdout_chunks.append(stdout or "")
            stderr_lines.extend((stderr or "").splitlines())
        else:
            streamed_stdout, streamed_stderr = _communicate_streaming(
                process,
                input_text=input_text,
                timeout_s=timeout_s,
                on_line=on_line,
                tail_lines=stderr_tail_lines,
            )
            stdout_chunks.append(streamed_stdout)
            stderr_lines.extend(streamed_stderr)
    except subprocess.TimeoutExpired as exc:
        _terminate(process)
        raise SubprocessTimeoutError(argv[0], timeout_s or 0.0) from exc
    except BaseException:
        _terminate(process)
        raise

    duration = time.monotonic() - started
    result = CompletedCommand(
        args=tuple(argv),
        returncode=process.returncode,
        stdout="".join(stdout_chunks),
        stderr="\n".join(stderr_lines),
        duration_s=duration,
    )
    if check and result.returncode != 0:
        raise SubprocessError(argv[0], result.returncode, result.stderr)
    return result


def _make_preexec(nice: int) -> Optional[Callable[[], None]]:
    if nice == 0 or os.name != "posix":
        return None

    def _apply() -> None:  # pragma: no cover - runs in the forked child
        with contextlib.suppress(OSError):
            os.nice(nice)

    return _apply


def _communicate_streaming(
    process: subprocess.Popen[str],
    *,
    input_text: Optional[str],
    timeout_s: Optional[float],
    on_line: Optional[Callable[[str], None]],
    tail_lines: int,
) -> tuple[str, list[str]]:
    """Read stderr line-by-line while the child runs, bounding memory."""
    if input_text is not None and process.stdin is not None:
        process.stdin.write(input_text)
        process.stdin.close()

    deadline = time.monotonic() + timeout_s if timeout_s else None
    tail: list[str] = []
    assert process.stderr is not None

    for raw in process.stderr:
        line = raw.rstrip("\n")
        if on_line is not None:
            on_line(line)
        if tail_lines > 0:
            tail.append(line)
            if len(tail) > tail_lines:
                del tail[0 : len(tail) - tail_lines]
        if deadline is not None and time.monotonic() > deadline:
            raise subprocess.TimeoutExpired(_command_of(process), timeout_s or 0.0)

    stdout = process.stdout.read() if process.stdout else ""
    remaining = None
    if deadline is not None:
        remaining = max(0.1, deadline - time.monotonic())
    process.wait(timeout=remaining)
    return stdout, tail


def _command_of(process: subprocess.Popen[str]) -> str:
    """The child's command line, for a timeout message."""
    args = process.args
    if isinstance(args, (str, bytes)):
        return args.decode(errors="replace") if isinstance(args, bytes) else args
    if isinstance(args, (list, tuple)):
        return " ".join(str(a) for a in args)
    return str(args)


def _terminate(process: subprocess.Popen[str]) -> None:
    """Kill the child's whole process group; ffmpeg spawns helpers."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        process.wait(timeout=SUBPROCESS_KILL_GRACE_S)
    except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
