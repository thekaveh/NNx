"""Run one command while holding the shared Mac GPU lock.

The MPS workflow runs on a self-hosted Mac that other sessions also use for
GPU work. Each GPU job takes the same exclusive ``fcntl.flock`` lock on
``~/.cache/mac-gpu.lock`` (or ``NNX_MAC_GPU_LOCK``), so two GPU jobs do not
overlap. The lock is advisory: it only helps when every GPU job takes it.

Usage: ``python3 scripts/with_mac_gpu_lock.py -- COMMAND [ARGS...]``. The
exit status is the command's. If the lock stays busy for longer than
``WAIT_SECONDS``, the script exits 1 without running the command.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

WAIT_SECONDS = 15 * 60
POLL_SECONDS = 5


def lock_path() -> Path:
    override = os.environ.get("NNX_MAC_GPU_LOCK")
    return Path(override) if override else Path.home() / ".cache" / "mac-gpu.lock"


def main(argv: list[str]) -> int:
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print("usage: with_mac_gpu_lock.py -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started > WAIT_SECONDS:
                    print(
                        f"gave up after {WAIT_SECONDS // 60} min waiting for {path}; another GPU job holds it",
                        file=sys.stderr,
                    )
                    return 1
                time.sleep(POLL_SECONDS)
        print(f"holding {path} after {time.monotonic() - started:.0f}s", flush=True)
        # The lock is released when the handle closes, after the command ends.
        return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
