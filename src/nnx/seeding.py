"""Reproducibility helpers for nnx.

`set_seed(seed)` pins every common RNG (Python `random`, NumPy, PyTorch CPU
+ CUDA) to the given seed and toggles cuDNN to deterministic mode. Use
`dataloader_worker_init_fn` as `DataLoader(worker_init_fn=...)` to also
fix the seed inside each worker process — without this, DataLoader workers
inherit a non-deterministic numpy/python seed.

Determinism caveats:
- `torch.use_deterministic_algorithms(True)` can degrade performance and
  some ops have no deterministic CUDA kernel. We don't enable it by
  default; pass `strict=True` to set_seed() to opt in.
- cuDNN deterministic + benchmark=False trades throughput for repeatable
  convolutions.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, strict: bool = False) -> None:
    """Pin every RNG that affects training and toggle cuDNN deterministic.

    Args:
        seed: integer seed shared across Python `random`, NumPy, and PyTorch
            (CPU + CUDA).
        strict: when True also calls torch.use_deterministic_algorithms(True)
            and sets CUBLAS_WORKSPACE_CONFIG. Slower and may raise on ops
            that lack a deterministic CUDA implementation; opt in only when
            full bit-for-bit reproducibility matters.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN: turn off benchmarking (which picks the fastest kernel based on
    # input shapes and can introduce non-determinism) and turn on the
    # deterministic flag.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if strict:
        # Required for deterministic CUDA matmul / convolutions; see
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def dataloader_worker_init_fn(worker_id: int) -> None:
    """DataLoader `worker_init_fn` that pins each worker's numpy/python seed
    deterministically from the worker_id + the parent torch seed.

    Pass as: `DataLoader(..., worker_init_fn=dataloader_worker_init_fn)`.
    """
    # torch.initial_seed() returns the base seed propagated to this worker.
    base_seed = torch.initial_seed() % (2**32)
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_ENV_SNAPSHOT_CACHE: Optional[dict] = None


def env_snapshot(force_refresh: bool = False) -> dict:
    """Capture a snapshot of the runtime environment for reproducibility.

    Returned dict is JSON-serializable. Includes Python / torch / numpy
    versions, GPU info if any, OS, and the git commit hash if running
    inside a git repo. Safe to call from anywhere — failures degrade to
    `None` per field rather than raising.

    Result is memoized within the process (env doesn't change between
    calls). Pass ``force_refresh=True`` to re-compute — useful in tests
    that mutate the environment.
    """
    global _ENV_SNAPSHOT_CACHE
    if _ENV_SNAPSHOT_CACHE is not None and not force_refresh:
        return dict(_ENV_SNAPSHOT_CACHE)
    import platform
    import subprocess

    def _git_commit() -> Optional[str]:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode()
                .strip()
            )
        except Exception:
            return None

    def _git_dirty() -> Optional[bool]:
        try:
            out = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode()
                .strip()
            )
            return bool(out)
        except Exception:
            return None

    def _nnx_version() -> Optional[str]:
        # PyPI distribution is `thekaveh-nnx` (PR #49) — the bare `nnx` name
        # is squatted by an abandoned JAX library, so a stale `version("nnx")`
        # here silently returned None on every clean install of the renamed
        # package, defeating metadata.yaml's reproducibility job. Mirror the
        # same lookup `nnx.__version__` uses (`src/nnx/__init__.py`).
        try:
            from importlib.metadata import version

            return version("thekaveh-nnx")
        except Exception:
            return None

    snap = {
        "nnx": _nnx_version(),
        "python": platform.python_version(),
        # torch.__version__ is a TorchVersion subclass that yaml.dump
        # can't serialize as a plain scalar — coerce to str so the
        # resulting metadata.yaml stays yaml.safe_load-compatible.
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }
    _ENV_SNAPSHOT_CACHE = dict(snap)
    return snap
