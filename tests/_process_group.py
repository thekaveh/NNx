"""Run a command whose descendants must not outlive its timeout (torchrun launches).

``subprocess.run(..., timeout=...)`` kills only the process it started. A
hung ``torchrun`` dies, but its rank processes are reparented to init and
keep running (each holding its share of a CPU) long after the test failed.
:func:`run_group` starts the command in a new session, so on a timeout it
ends the whole process group: torchrun and every rank it spawned.
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
    except that on a timeout the command's whole process group gets SIGTERM,
    then SIGKILL after ``grace`` seconds, before ``TimeoutExpired`` (carrying
    the output written so far, as text) is raised. Outside POSIX there are
    no process groups: plain ``subprocess.run``."""
    if os.name != "posix":
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, **kwargs
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _end_group(process, grace)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
        except BaseException:  # e.g. Ctrl-C: nothing outlives the test run either
            _end_group(process, 0)
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _end_group(process: subprocess.Popen, grace: float) -> None:
    """SIGTERM the group ``process`` leads, then SIGKILL whatever is still in
    it after ``grace`` seconds; return once the group is empty (or after
    another ``grace`` seconds, if a member somehow survives SIGKILL)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        deadline = time.monotonic() + grace
        while _group_alive(process) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not _group_alive(process):
            return


def _group_alive(process: subprocess.Popen) -> bool:
    """Whether any process is left in the group ``process`` leads (reaping
    the leader first, so it does not linger as a zombie member)."""
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def alive_with(text: str) -> list[str]:
    """The ``ps`` lines (pid and command) of live processes whose command line holds ``text``."""
    listing = subprocess.run(["ps", "-A", "-o", "pid=,command="], check=True, capture_output=True, text=True)
    return [line.strip() for line in listing.stdout.splitlines() if text in line]
