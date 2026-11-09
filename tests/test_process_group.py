"""tests/_process_group.py: what the torchrun launches in the DDP and example
tests run through. No torchrun here, so these stay off the serialized
``torchrun`` xdist worker; a python stand-in plays the launcher."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests import _process_group
from tests._process_group import alive_with, run_group

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")

# A rank as torch starts one: in a session of its own, so outside the launcher's
# process group, and here ignoring SIGTERM. The marker on its command line finds it.
_RANK = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"


def _launcher(marker: str) -> str:
    """``python -c`` code that starts a rank like torchrun does, then hangs."""
    return (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {_RANK!r}, {marker!r}], start_new_session=True); "
        "print('started', flush=True); time.sleep(120)"
    )


def test_a_timed_out_launch_ends_every_process_it_started(tmp_path):
    """run_group(), which launch() and the examples' torchrun use: a timeout
    ends the command's descendants too, even one in a session of its own that
    ignores SIGTERM (killed after the grace period), and still returns the
    output written so far."""
    marker = str(tmp_path / "rank")
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_group([sys.executable, "-c", _launcher(marker)], timeout=5, grace=1)
    assert alive_with(marker) == []
    assert "started" in caught.value.stdout


def test_a_timed_out_launch_never_signals_an_id_it_may_have_lost(tmp_path, monkeypatch):
    """Once the command is reaped its pid, like the id of a group found empty,
    may go to another process: the SIGKILL pass (here for a rank that ignores
    SIGTERM) neither walks from the command's pid again nor signals such a group."""
    commands: list[subprocess.Popen] = []
    walks: list[int | None] = []  # the command's returncode at each walk
    gone: set[int] = set()
    late: list[tuple[int, int]] = []
    any_alive, descendant_groups, killpg = _process_group._any_alive, _process_group._descendant_groups, os.killpg

    def recording_any_alive(process: subprocess.Popen, groups: set[int]) -> bool:
        commands.append(process)
        return any_alive(process, groups)

    def recording_descendant_groups(root: int) -> set[int]:
        walks.append(commands[-1].returncode if commands else None)
        return descendant_groups(root)

    def recording_killpg(pgid: int, sig: int) -> None:
        if sig and pgid in gone:
            late.append((pgid, sig))
        try:
            killpg(pgid, sig)
        except ProcessLookupError:
            gone.add(pgid)
            raise

    monkeypatch.setattr(_process_group, "_any_alive", recording_any_alive)
    monkeypatch.setattr(_process_group, "_descendant_groups", recording_descendant_groups)
    monkeypatch.setattr(os, "killpg", recording_killpg)
    marker = str(tmp_path / "rank")
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_group([sys.executable, "-c", _launcher(marker)], timeout=5, grace=1)
    assert "started" in caught.value.stdout and alive_with(marker) == []
    assert walks and all(returncode is None for returncode in walks), walks
    assert late == []


def test_a_timed_out_launch_always_raises_timeout_expired():
    """A command that dies at once on SIGTERM leaves a zombie that macOS
    reports as EPERM to killpg(pgid, 0) until it is reaped; run_group() polls
    again rather than raising PermissionError in place of TimeoutExpired. The
    race is macOS's, so other platforms take a few runs, not 50."""
    for _ in range(50 if sys.platform == "darwin" else 5):
        with pytest.raises(subprocess.TimeoutExpired):
            run_group(["sleep", "60"], timeout=0.1, grace=2)


def test_an_interrupted_launch_ends_every_process_it_started(tmp_path):
    """Ctrl-C reaches the terminal's foreground group, which no longer holds the
    command run_group() started in its own session: on the KeyboardInterrupt
    run_group() ends the command's descendants itself before re-raising."""
    marker = str(tmp_path / "rank")
    runner = tmp_path / "runner.py"  # a file, so the marker is on the command and rank lines only
    runner.write_text(
        # SIGINT raises KeyboardInterrupt even if this test run inherited it ignored (a background job)
        "import signal, sys\nsignal.signal(signal.SIGINT, signal.default_int_handler)\n"
        f"sys.path.insert(0, {str(ROOT)!r})\nfrom tests._process_group import run_group\n"
        f"run_group([sys.executable, '-c', {_launcher(marker)!r}], timeout=120, grace=1)\n",
        encoding="utf-8",
    )
    # The runner stands for the test run in its terminal: its own group gets the Ctrl-C.
    with subprocess.Popen(
        [sys.executable, str(runner)], stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as test_run:
        try:
            deadline = time.monotonic() + 60
            while (started := len(alive_with(marker))) < 2 and time.monotonic() < deadline:  # command and rank
                time.sleep(0.1)
            os.killpg(test_run.pid, signal.SIGINT)
            _, stderr = test_run.communicate(timeout=60)
        finally:  # a failure must not hang on a live runner (Popen.__exit__ waits for it)
            if test_run.poll() is None:  # not reaped, so its pid is still ours to signal
                os.killpg(test_run.pid, signal.SIGKILL)
                for line in alive_with(marker):  # nor leave behind what it started
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(int(line.split()[0]), signal.SIGKILL)
    assert started == 2 and "KeyboardInterrupt" in stderr
    assert alive_with(marker) == []
