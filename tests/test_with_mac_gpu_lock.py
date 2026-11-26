"""The Mac GPU lock wrapper runs its command under the lock and fails clearly when it is busy."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

from tests._repo_files import repo_file

SCRIPT = repo_file("scripts", "with_mac_gpu_lock.py")


def _run(tmp_path: Path, *command: str, env_extra: dict[str, str] | None = None):
    env = {**os.environ, "NNX_MAC_GPU_LOCK": str(tmp_path / "mac-gpu.lock"), **(env_extra or {})}
    return subprocess.run([sys.executable, str(SCRIPT), "--", *command], env=env, capture_output=True, text=True)


def test_the_command_runs_and_its_status_is_returned(tmp_path):
    done = _run(tmp_path, sys.executable, "-c", "import sys; sys.exit(3)")
    assert done.returncode == 3
    assert "holding" in done.stdout


def test_the_lock_is_released_after_the_command(tmp_path):
    _run(tmp_path, sys.executable, "-c", "pass")
    with (tmp_path / "mac-gpu.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # would raise if still held


def test_a_busy_lock_fails_without_running_the_command(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("with_mac_gpu_lock", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("NNX_MAC_GPU_LOCK", str(tmp_path / "mac-gpu.lock"))
    monkeypatch.setattr(module, "WAIT_SECONDS", 0)
    marker = tmp_path / "ran"
    with (tmp_path / "mac-gpu.lock").open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        status = module.main(["--", sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
    assert status == 1
    assert not marker.exists()


def test_no_command_is_a_usage_error(tmp_path):
    done = _run(tmp_path)
    assert done.returncode == 2
    assert "usage" in done.stderr
