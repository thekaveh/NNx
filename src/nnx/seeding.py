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

import contextlib
import os
import random
import warnings
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Optional, cast

import numpy as np
import torch


def set_seed(seed: int, strict: bool = False) -> None:
    """Pin every RNG that affects training and toggle cuDNN deterministic.

    Args:
        seed: integer seed shared across Python `random`, NumPy, and PyTorch
            (CPU + CUDA). Also written to `os.environ["PYTHONHASHSEED"]`
            so DataLoader workers started via the `spawn` method (default
            on Windows + macOS/Py3.8+) inherit a deterministic hash seed.
            Note: the current Python interpreter's hash state was fixed at
            startup — this assignment only affects spawned subprocesses.
            For full hash determinism in the current process, set
            `PYTHONHASHSEED=<N>` in the shell BEFORE invoking Python.
        strict: when True also calls torch.use_deterministic_algorithms(True)
            and sets CUBLAS_WORKSPACE_CONFIG. Slower and may raise on ops
            that lack a deterministic CUDA implementation; opt in only when
            full bit-for-bit reproducibility matters.
    """
    _set_seed(seed, strict)


def _set_seed(seed: int, strict: bool = False, *, scoped: bool = False) -> None:
    """:func:`set_seed`; ``scoped=True`` seeds only the streams
    :func:`_seed_scope` restores (see :func:`_seed_streams`)."""
    # PYTHONHASHSEED governs hash randomization in spawned subprocesses.
    # The current interpreter's hash state was fixed at startup and is
    # NOT affected by this assignment — but DataLoader workers using the
    # `spawn` start method (default on Windows + macOS/Py3.8+) and any
    # `subprocess.Popen` started downstream inherit this env var. Without
    # it, those children re-randomize their dict/set hash order, so any
    # code path that iterates a dict/set populated in the worker can
    # differ between runs even when every other RNG is seeded. Setting
    # it here is the cheap defensive move; the explicit caveat is in
    # set_seed's docstring.
    os.environ["PYTHONHASHSEED"] = str(seed)

    _seed_streams(seed, scoped=scoped)

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


# RNG stream snapshots: the training loop saves and restores them with every
# stateful checkpoint (loader / sampler generators included), and `_seed_scope`
# restores them after a probe.


def _loader_generators(train_loader: Any) -> tuple[tuple[str, Any], ...]:
    """Where a loader keeps its ``torch.Generator`` objects, by state key —
    one list for capture and restore, so the two cannot disagree."""
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    return (
        ("train_loader_generator", getattr(train_loader, "generator", None)),
        ("train_sampler_generator", getattr(getattr(train_loader, "sampler", None), "generator", None)),
        ("train_batch_sampler_generator", getattr(batch_sampler, "generator", None)),
        ("train_batch_sampler_sampler_generator", getattr(getattr(batch_sampler, "sampler", None), "generator", None)),
    )


def _capture_rng_state(train_loader: Optional[Iterable[Any]] = None, *, cuda: bool = True) -> dict[str, Any]:
    """The Python / NumPy / torch (CPU, CUDA, MPS) RNG streams and the
    loader's generators. ``cuda=False`` leaves an uninitialized CUDA context
    alone (reading its streams would create it)."""
    numpy_state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda and torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }
    if train_loader is not None:
        generators = _loader_generators(train_loader)
        generator_states: list[dict[str, Any]] = []
        seen: set[int] = set()
        for key, generator in generators:
            if isinstance(generator, torch.Generator):
                state[key] = generator.get_state()
                if id(generator) not in seen:
                    generator_states.append({"initial_seed": generator.initial_seed(), "state": generator.get_state()})
                    seen.add(id(generator))
        state["train_generators"] = generator_states
    return state


def _restore_rng_state(state: dict[str, Any], train_loader: Optional[Iterable[Any]] = None) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if train_loader is not None:
        generators = _loader_generators(train_loader)
        unique_generators: list[torch.Generator] = []
        seen: set[int] = set()
        for _, generator in generators:
            if isinstance(generator, torch.Generator) and id(generator) not in seen:
                unique_generators.append(generator)
                seen.add(id(generator))
        saved_generators = state.get("train_generators")
        if saved_generators is not None:
            if len(saved_generators) != len(unique_generators):
                raise ValueError("resume loader exposes a different number of torch.Generator instances")
            if saved_generators and isinstance(saved_generators[0], dict):
                if len(saved_generators) == 1:
                    unique_generators[0].set_state(saved_generators[0]["state"])
                    return
                saved_by_seed = {entry["initial_seed"]: entry["state"] for entry in saved_generators}
                current_seeds = [generator.initial_seed() for generator in unique_generators]
                if len(saved_by_seed) != len(saved_generators) or len(set(current_seeds)) != len(current_seeds):
                    raise ValueError("resume loader has ambiguous torch.Generator seeds")
                if set(saved_by_seed) != set(current_seeds):
                    raise ValueError("resume loader exposes different torch.Generator identities")
                for generator in unique_generators:
                    generator.set_state(saved_by_seed[generator.initial_seed()])
            else:
                # Version 2 sidecars recorded generators positionally.
                for generator, generator_state in zip(unique_generators, saved_generators, strict=True):
                    generator.set_state(generator_state)
        else:
            for key, generator in generators:
                if isinstance(generator, torch.Generator) and state.get(key) is not None:
                    generator.set_state(state[key])


# The process-wide settings `set_seed` changes besides the RNG streams; keep
# them in step with `set_seed` (`_seed_scope` restores both).
_SEED_ENV = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")


def _global_rng_state() -> dict[str, Any]:
    """The global random streams, read without creating a CUDA context: a
    CPU-only process never touched the CUDA streams, so there is nothing
    of theirs to keep."""
    return _capture_rng_state(None, cuda=torch.cuda.is_initialized())


@contextlib.contextmanager
def _global_rng_kept() -> Iterator[None]:
    """Run a build whose initial values are discarded without moving the
    global random streams."""
    state = _global_rng_state()
    try:
        yield
    finally:
        _restore_rng_state(state, None)


def _capture_seed_settings() -> dict:
    cudnn = torch.backends.cudnn
    return {
        "cudnn": (cudnn.deterministic, cudnn.benchmark),
        "deterministic": (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ),
        "env": {name: os.environ.get(name) for name in _SEED_ENV},
    }


def _restore_seed_settings(state: dict) -> None:
    """Restore each setting, even if another fails (the first failure is
    raised afterwards)."""

    def cudnn_flags() -> None:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state["cudnn"]

    def algorithms() -> None:
        enabled, warn_only = state["deterministic"]
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def environment() -> None:
        for name, value in state["env"].items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    _run_each((cudnn_flags, algorithms, environment))


def _run_each(steps: Iterable[Callable[[], None]]) -> None:
    """Run every step, then raise the first failure, if any."""
    failure: Optional[Exception] = None
    for step in steps:
        try:
            step()
        except Exception as exc:
            failure = failure or exc
    if failure is not None:
        raise failure


def _cuda_in_use() -> bool:
    """Whether a CUDA context exists, so its streams can be read without
    creating one."""
    return torch.cuda.is_available() and torch.cuda.is_initialized()


@contextlib.contextmanager
def _seed_scope() -> Iterator[Callable[[], None]]:
    """Undo every effect of ``set_seed`` inside the block on exit — the
    Python / NumPy / torch RNG streams and the cuDNN, deterministic-algorithm
    and environment settings — whether or not the block raises (used by
    ``ExperimentPlan.probe``). The RNG streams (restored together, as the
    training loop restores them) and each setting are restored even if
    another restore fails, and a failing restore never hides the block's
    own error (it is added to that error as a note, or warned about on
    Python 3.10).

    An idle CUDA context is not created just to read its streams (a CPU
    probe on a GPU host). When the block itself starts CUDA, it calls the
    yielded function once the context exists; the CUDA streams as they are
    then are restored on exit too."""
    streams = _capture_rng_state(None, cuda=_cuda_in_use())
    settings = _capture_seed_settings()

    def cuda_started() -> None:
        if streams["cuda"] is None and _cuda_in_use():
            streams["cuda"] = torch.cuda.get_rng_state_all()

    def restore() -> None:
        _run_each((lambda: _restore_rng_state(streams, None), lambda: _restore_seed_settings(settings)))

    try:
        yield cuda_started
    except BaseException as error:
        try:
            restore()
        except Exception as restore_error:  # the block's error is the one to report; the failure is noted on it
            note = f"restoring the RNG streams and seed settings also failed: {restore_error!r}"
            add_note = getattr(error, "add_note", None)  # Python >= 3.11
            if callable(add_note):
                add_note(note)
            else:
                with contextlib.suppress(Exception):  # a warning turned into an error must not hide the block's
                    warnings.warn(note, RuntimeWarning, stacklevel=3)
        raise
    restore()


def _seed_streams(seed: int, *, scoped: bool) -> None:
    """Seed Python, NumPy and torch. ``torch.manual_seed`` seeds every
    device's generator, and queues CUDA's seed for its first use when CUDA
    is idle. ``scoped=True`` seeds only what :func:`_seed_scope` restores —
    the CPU generator, CUDA when it is in use and MPS — for a probe, which
    must leave nothing behind (``nnx.plans``)."""
    random.seed(seed)
    np.random.seed(seed)
    if not scoped:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return
    torch.default_generator.manual_seed(seed)
    if _cuda_in_use():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


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

    Result is memoized within the process (versions/hardware don't
    change between calls). Caveat: the ``git_commit`` / ``git_dirty``
    fields are frozen at first call too, so a long session that commits
    mid-run records the session-start git state in later runs'
    metadata.yaml. Pass ``force_refresh=True`` to re-compute — useful
    in tests that mutate the environment, or to re-stamp git state.
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
        # Broad catch is deliberate: env_snapshot() is opportunistic —
        # the caller isn't in a git repo (CI tarball install, fresh
        # `pip install` from PyPI, `tempfile.TemporaryDirectory` runs),
        # `git` isn't on PATH, the 2-second timeout fired, or the
        # subprocess crashed for any other reason. metadata.yaml just
        # omits the field; we never want this lookup to surface as an
        # exception to the user.
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
        # Same opportunistic-fallback rationale as `_git_commit` above —
        # the dirty flag is metadata.yaml decoration, not a precondition.
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
