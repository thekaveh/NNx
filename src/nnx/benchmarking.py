"""Bounded forward benchmarks and profiling (FEAT-029).

Measure, don't assume: :mod:`nnx.compilation` guarantees no speedup, and
these helpers show what compilation costs and buys on *your* hardware.

- :func:`benchmark_forward` times a module's forward on fixed inputs:
  the **first call** (which includes any compilation) separately from
  ``repeats`` **warmed** calls after ``warmup`` untimed ones. The device is
  synchronized around every timed call, CUDA peak-memory counters are reset
  before the timed region, and the :class:`BenchmarkReport` carries shapes,
  dtypes, backend, device, repeats and the dispersion of the latencies.
- :func:`compare_compile` benchmarks an eager and a compiled copy made from
  **identical weights** (both deep copies of the module; the reports share a
  ``weights_digest``) and reports the largest output difference.
- :func:`profile_forward` runs ``torch.profiler`` with a finite
  ``wait / warmup / active / repeat`` schedule into an explicit
  ``output_dir``. The profiler is closed on success, error and
  cancellation, and never runs inside a benchmark's timed region.

Nothing here trains, and the module passed in is never modified (each
helper works on a deep copy, in eval mode, under ``torch.no_grad``).
"""

from __future__ import annotations

import copy
import hashlib
import numbers
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from .compilation import CompileSpec, _known_backends

__all__ = [
    "BenchmarkReport",
    "CompileComparison",
    "ProfileReport",
    "benchmark_forward",
    "compare_compile",
    "profile_forward",
]

Inputs = Union[torch.Tensor, Sequence[torch.Tensor]]


def _count(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{what} must be an int >= {minimum}, got {value!r}")
    return int(value)


def _inputs(inputs: Inputs, device: torch.device) -> tuple[torch.Tensor, ...]:
    batch = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if not batch or not all(isinstance(x, torch.Tensor) for x in batch):
        raise TypeError("inputs must be a tensor or a non-empty sequence of tensors")
    return tuple(x.to(device) for x in batch)


def _device(net: torch.nn.Module, device: Optional[Union[str, torch.device]]) -> torch.device:
    if not isinstance(net, torch.nn.Module):
        raise TypeError(f"net must be a torch.nn.Module, got {type(net).__name__}")
    if device is not None:
        return torch.device(device)
    first = next(iter(net.parameters()), None)
    return first.device if first is not None else torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _weights_digest(net: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _prepared(net: torch.nn.Module, device: torch.device, spec: Optional[CompileSpec]) -> tuple[torch.nn.Module, Any]:
    if spec is not None and not isinstance(spec, CompileSpec):
        raise TypeError(f"compile must be an nnx.compilation.CompileSpec or None, got {type(spec).__name__}")
    if spec is not None and spec.backend not in _known_backends():
        raise ValueError(f"CompileSpec.backend {spec.backend!r} is not a torch.compile backend here")
    module = copy.deepcopy(net).to(device).eval()
    call = module if spec is None else torch.compile(module, backend=spec.backend, **spec.options())
    return module, call


@dataclass(frozen=True, kw_only=True, slots=True)
class BenchmarkReport:
    """One benchmarked configuration. Times are seconds; ``latencies`` are
    the warmed per-call times (``repeats`` of them)."""

    label: str
    backend: Optional[str]
    device: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    batch_size: int
    warmup: int
    repeats: int
    first_call_seconds: float
    latencies: tuple[float, ...]
    peak_memory_bytes: Optional[int]
    weights_digest: str
    torch_version: str

    @property
    def mean_seconds(self) -> float:
        return statistics.fmean(self.latencies)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.latencies)

    @property
    def stdev_seconds(self) -> float:
        """Sample standard deviation (``0.0`` for a single repeat)."""
        return statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0.0

    @property
    def min_seconds(self) -> float:
        return min(self.latencies)

    @property
    def max_seconds(self) -> float:
        return max(self.latencies)

    @property
    def throughput(self) -> float:
        """Warmed samples per second (``batch_size / mean_seconds``), where
        ``batch_size`` is the first input's leading dimension (batch-first
        inputs)."""
        return self.batch_size / self.mean_seconds if self.mean_seconds > 0 else float("inf")

    def state(self) -> dict[str, Any]:
        """A plain, JSON-ready summary."""
        return {
            "label": self.label,
            "backend": self.backend,
            "device": self.device,
            "shapes": [list(shape) for shape in self.shapes],
            "dtypes": list(self.dtypes),
            "batch_size": self.batch_size,
            "warmup": self.warmup,
            "repeats": self.repeats,
            "first_call_seconds": self.first_call_seconds,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "stdev_seconds": self.stdev_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "throughput": self.throughput,
            "peak_memory_bytes": self.peak_memory_bytes,
            "weights_digest": self.weights_digest,
            "torch_version": self.torch_version,
        }


def _timed(call: Any, batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[float, Any]:
    _synchronize(device)
    start = time.perf_counter()
    output = call(*batch)
    _synchronize(device)
    return time.perf_counter() - start, output


def _run_benchmark(
    net: torch.nn.Module,
    inputs: Inputs,
    *,
    compile: Optional[CompileSpec],
    warmup: int,
    repeats: int,
    device: Optional[Union[str, torch.device]],
    label: Optional[str],
) -> tuple[BenchmarkReport, Any]:
    warmup = _count(warmup, "warmup", minimum=0)
    repeats = _count(repeats, "repeats", minimum=1)
    target = _device(net, device)
    module, call = _prepared(net, target, compile)
    batch = _inputs(inputs, target)
    with torch.no_grad():
        first, output = _timed(call, batch, target)  # includes compilation, if any
        for _ in range(warmup):
            call(*batch)
        _synchronize(target)
        if target.type == "cuda":  # the warmed region only: not compilation or autotuning
            torch.cuda.reset_peak_memory_stats(target)
        latencies = tuple(_timed(call, batch, target)[0] for _ in range(repeats))
        peak = int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else None
    report = BenchmarkReport(
        label=label or ("eager" if compile is None else "compiled"),
        backend=None if compile is None else compile.backend,
        device=str(target),
        shapes=tuple(tuple(x.shape) for x in batch),
        dtypes=tuple(str(x.dtype).removeprefix("torch.") for x in batch),
        batch_size=int(batch[0].shape[0]) if batch[0].ndim else 1,
        warmup=warmup,
        repeats=repeats,
        first_call_seconds=first,
        latencies=latencies,
        peak_memory_bytes=peak,
        weights_digest=_weights_digest(module),
        torch_version=str(torch.__version__),
    )
    return report, output


def benchmark_forward(
    net: torch.nn.Module,
    inputs: Inputs,
    *,
    compile: Optional[CompileSpec] = None,
    warmup: int = 3,
    repeats: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    label: Optional[str] = None,
) -> BenchmarkReport:
    """Time ``net``'s forward on ``inputs`` (eager, or compiled with
    ``compile``): the first call, then ``repeats`` calls after ``warmup``
    untimed ones, on a deep copy in eval mode under ``torch.no_grad``.
    ``device`` defaults to the module's."""
    return _run_benchmark(net, inputs, compile=compile, warmup=warmup, repeats=repeats, device=device, label=label)[0]


@dataclass(frozen=True, kw_only=True, slots=True)
class CompileComparison:
    """An eager and a compiled benchmark of identical weights, and the
    largest absolute difference between their outputs."""

    eager: BenchmarkReport
    compiled: BenchmarkReport
    max_abs_diff: float

    @property
    def speedup(self) -> float:
        """Warmed eager mean over compiled mean (> 1 means faster compiled;
        nothing is guaranteed)."""
        return self.eager.mean_seconds / self.compiled.mean_seconds

    def state(self) -> dict[str, Any]:
        return {
            "eager": self.eager.state(),
            "compiled": self.compiled.state(),
            "max_abs_diff": self.max_abs_diff,
            "speedup": self.speedup,
        }


def compare_compile(
    net: torch.nn.Module,
    inputs: Inputs,
    *,
    compile: Optional[CompileSpec] = None,
    warmup: int = 3,
    repeats: int = 10,
    device: Optional[Union[str, torch.device]] = None,
) -> CompileComparison:
    """Benchmark ``net`` eager and compiled (``compile``, default
    ``CompileSpec()``) from identical weights."""
    spec = CompileSpec() if compile is None else compile
    eager, eager_out = _run_benchmark(
        net, inputs, compile=None, warmup=warmup, repeats=repeats, device=device, label="eager"
    )
    compiled, compiled_out = _run_benchmark(
        net, inputs, compile=spec, warmup=warmup, repeats=repeats, device=device, label="compiled"
    )
    if eager.weights_digest != compiled.weights_digest:  # pragma: no cover - deep copies of one module
        raise RuntimeError("eager and compiled benchmarks did not start from identical weights")
    from torch.utils._pytree import tree_flatten

    eager_leaves, eager_spec = tree_flatten(eager_out)
    compiled_leaves, compiled_spec = tree_flatten(compiled_out)
    if eager_spec != compiled_spec:
        raise RuntimeError("eager and compiled forwards returned differently structured outputs")
    diff = max(
        (
            float((a.detach().float() - b.detach().float()).abs().max()) if a.numel() else 0.0
            for a, b in zip(eager_leaves, compiled_leaves, strict=True)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        ),
        default=0.0,
    )
    return CompileComparison(eager=eager, compiled=compiled, max_abs_diff=diff)


@dataclass(frozen=True, kw_only=True, slots=True)
class ProfileReport:
    """A finished profile: where its traces were written, the schedule and
    step count, and the top operators by self time."""

    output_dir: str
    trace_files: tuple[str, ...]
    schedule: Mapping[str, int]
    steps: int
    table: str


def profile_forward(
    net: torch.nn.Module,
    inputs: Inputs,
    *,
    output_dir: Union[str, os.PathLike[str]],
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
    compile: Optional[CompileSpec] = None,
    device: Optional[Union[str, torch.device]] = None,
    row_limit: int = 10,
) -> ProfileReport:
    """Profile ``net``'s forward with ``torch.profiler`` for exactly
    ``(wait + warmup + active) * repeat`` steps, writing Chrome traces to
    ``output_dir`` (created if needed). Every count is finite (``active``
    and ``repeat`` at least 1). The profiler is stopped and its traces
    flushed whether the steps finish, raise or are cancelled. With
    ``compile`` and ``wait=warmup=0`` the first active step includes the
    compilation itself."""
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError(f"output_dir must be a path, got {type(output_dir).__name__}")
    counts = {
        "wait": _count(wait, "wait", minimum=0),
        "warmup": _count(warmup, "warmup", minimum=0),
        "active": _count(active, "active", minimum=1),
        "repeat": _count(repeat, "repeat", minimum=1),
    }
    row_limit = _count(row_limit, "row_limit", minimum=1)
    directory = os.fspath(output_dir)
    os.makedirs(directory, exist_ok=True)
    before = set(os.listdir(directory))
    target = _device(net, device)
    _, call = _prepared(net, target, compile)
    batch = _inputs(inputs, target)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if target.type == "cuda" else [])
    steps = (counts["wait"] + counts["warmup"] + counts["active"]) * counts["repeat"]
    with (
        torch.no_grad(),
        profile(
            activities=activities,
            schedule=schedule(**counts),
            on_trace_ready=tensorboard_trace_handler(directory),
        ) as profiler,
    ):
        for _ in range(steps):
            call(*batch)
            _synchronize(target)
            profiler.step()
    table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=row_limit)
    traces = tuple(sorted(os.path.join(directory, name) for name in set(os.listdir(directory)) - before))
    return ProfileReport(output_dir=directory, trace_files=traces, schedule=counts, steps=steps, table=table)
