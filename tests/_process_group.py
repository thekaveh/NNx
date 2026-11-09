"""Run a command whose descendants must not outlive its timeout (torchrun launches).

``subprocess.run(..., timeout=...)`` kills only the process it started. A
hung ``torchrun`` dies, but its rank processes are reparented to init and
keep running (each holding its share of a CPU) long after the test failed.
torch starts every rank in a session of its own, so the ranks are not in
torchrun's process group: a SIGTERM to torchrun reaches them only if torchrun
forwards it before it exits. :func:`run_group` therefore starts the command
in a new session and, when it gives up on it, lists the command's
descendants and ends their process groups as well as the command's own.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from typing import Any


def run_group(
    command: Sequence[str], *, timeout: float, grace: float = 5.0, **kwargs: Any
) -> subprocess.CompletedProcess:
    """``subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)``,
    except that on a timeout the process groups of the command and of every
    descendant get SIGTERM, then SIGKILL after ``grace`` seconds, before
    ``TimeoutExpired`` (carrying the output written so far, as text) is
    raised. Any other exception (e.g. Ctrl-C, which no longer reaches the
    command in its own session) ends them the same way before it propagates.
    Outside POSIX there are no process groups: plain ``subprocess.run``."""
    if os.name != "posix":
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, **kwargs
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _end_tree(process, grace)
            try:  # bounded: a survivor that inherited the pipes holds them open
                stdout, stderr = process.communicate(timeout=max(grace, 1.0))
            except subprocess.TimeoutExpired as partial:
                stdout, stderr = _decoded(partial.output), _decoded(partial.stderr)
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
        except BaseException:
            _end_tree(process, grace)
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _end_tree(process: subprocess.Popen, grace: float) -> None:
    """SIGTERM the process groups of ``process`` and of its descendants, then
    SIGKILL whatever is still in them after ``grace`` seconds; return once
    they are empty (or after another ``grace`` seconds, if a member somehow
    survives SIGKILL). The descendants are listed before each signal, and the
    first list is kept: once their parent is gone they are reparented to init
    and a fresh walk from ``process`` no longer finds them. The walk runs only
    while ``process`` is unreaped (alive or a zombie still holding its pid),
    so it never follows a pid that has since gone to another process."""
    groups: set[int] = set()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.returncode is None:
            groups |= {process.pid} | _descendant_groups(process.pid)
        for pgid in groups:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
        deadline = time.monotonic() + grace
        while _any_alive(process, groups) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not _any_alive(process, groups):
            return


def _descendant_groups(root: int) -> set[int]:
    """The process groups of ``root``'s live descendants, from ``ps`` (none
    without it: then only the command's own group is ended)."""
    try:
        listing = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pgid="], capture_output=True, text=True).stdout
    except OSError:
        listing = ""
    children: dict[int, list[tuple[int, int]]] = {}
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(field.isdigit() for field in fields):
            pid, ppid, pgid = map(int, fields)
            children.setdefault(ppid, []).append((pid, pgid))
    groups: set[int] = set()
    pending = [root]
    while pending:
        for pid, pgid in children.get(pending.pop(), []):
            groups.add(pgid)
            pending.append(pid)
    groups.discard(os.getpgrp())  # never this test run's own group
    return groups


def _any_alive(process: subprocess.Popen, groups: set[int]) -> bool:
    """Whether any process is left in ``groups`` (reaping ``process`` first,
    so it does not linger as a zombie member). A group found empty is dropped
    from ``groups``: its id is free for reuse, so it is never signalled again."""
    process.poll()
    for pgid in list(groups):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            groups.discard(pgid)
        except PermissionError:  # macOS says so for a group of unreaped zombies: poll again
            pass
    return bool(groups)


def _decoded(output: bytes | str | None) -> str:
    """Partial output from ``communicate()``, which is bytes even in text mode."""
    if isinstance(output, bytes):
        return output.decode("utf-8", "replace")
    return output or ""


def alive_with(text: str) -> list[str]:
    """The ``ps`` lines (pid and command) of live processes whose command line
    holds ``text``. ``-ww``: procps cuts each line to 80 columns when it does
    not write to a terminal, which would hide a marker near the end."""
    listing = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,command="], check=True, capture_output=True, text=True)
    return [line.strip() for line in listing.stdout.splitlines() if text in line]
