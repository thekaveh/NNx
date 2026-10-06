"""FEAT-029: bounded forward benchmarks and profiling."""

from __future__ import annotations

import asyncio
import copy
import os

import pytest
import torch

import nnx.benchmarking as benchmarking
from nnx import CompileSpec
from nnx.benchmarking import BenchmarkReport, benchmark_forward, compare_compile, profile_forward

FAST = CompileSpec(backend="aot_eager")


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _net() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.ReLU(), torch.nn.Linear(12, 3))


def _inputs() -> torch.Tensor:
    return torch.randn(5, 6, generator=torch.Generator().manual_seed(0))


def test_first_call_is_separate_from_warmed_latency():
    net = _net().train()
    before = copy.deepcopy(net.state_dict())
    report = benchmark_forward(net, _inputs(), warmup=2, repeats=4)
    # the caller's module is untouched: a deep copy runs, in eval mode
    assert net.training and all(torch.equal(net.state_dict()[k], v) for k, v in before.items())
    assert report.first_call_seconds > 0 and len(report.latencies) == 4
    assert (report.warmup, report.repeats, report.label, report.backend) == (2, 4, "eager", None)
    assert (report.shapes, report.dtypes, report.device, report.batch_size) == (((5, 6),), ("float32",), "cpu", 5)
    assert report.min_seconds <= report.median_seconds <= report.max_seconds
    assert report.stdev_seconds >= 0 and report.throughput == pytest.approx(5 / report.mean_seconds)
    assert report.peak_memory_bytes is None and report.torch_version == torch.__version__
    state = report.state()
    assert state["first_call_seconds"] == report.first_call_seconds and state["mean_seconds"] == report.mean_seconds


def test_compiled_first_call_carries_the_compile_cost():
    report = benchmark_forward(_net(), _inputs(), compile=FAST, warmup=1, repeats=3)
    assert (report.label, report.backend) == ("compiled", "aot_eager")
    assert report.first_call_seconds > report.median_seconds


def test_the_device_is_synchronized_around_every_timed_call(monkeypatch):
    synced = []
    monkeypatch.setattr(benchmarking, "_synchronize", lambda device: synced.append(str(device)))
    benchmark_forward(_net(), _inputs(), warmup=2, repeats=3)
    # before and after the first call and each repeat, plus once after warmup
    assert len(synced) == 2 * (1 + 3) + 1


def test_cuda_peak_memory_counters_are_reset_before_timing(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: events.append("read") or 1234)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    cpu = torch.device("cpu")

    class FakeCuda:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    monkeypatch.setattr(benchmarking, "_device", lambda net, device: FakeCuda())
    monkeypatch.setattr(benchmarking, "_inputs", lambda inputs, device: (_inputs(),))
    monkeypatch.setattr(benchmarking, "_prepared", lambda net, device, spec: (net.to(cpu).eval(), net))
    report = benchmark_forward(_net(), _inputs(), warmup=0, repeats=2)
    assert events == ["reset", "read"] and report.peak_memory_bytes == 1234 and report.device == "cuda:0"


def test_compare_compile_starts_both_sides_from_identical_weights():
    net = _net()
    before = copy.deepcopy(net.state_dict())
    net.train()
    comparison = compare_compile(net, _inputs(), compile=FAST, warmup=1, repeats=3)
    assert comparison.eager.weights_digest == comparison.compiled.weights_digest
    assert comparison.max_abs_diff <= 1e-6 and comparison.speedup > 0
    assert net.training and all(torch.equal(net.state_dict()[k], v) for k, v in before.items())
    assert set(comparison.state()) == {"eager", "compiled", "max_abs_diff", "speedup"}


@pytest.mark.parametrize(
    "kwargs",
    [{"warmup": -1}, {"repeats": 0}, {"repeats": True}, {"repeats": 2.0}],
)
def test_benchmark_counts_are_finite_and_valid(kwargs):
    with pytest.raises(ValueError):
        benchmark_forward(_net(), _inputs(), **kwargs)


def test_benchmark_rejects_bad_inputs():
    with pytest.raises(TypeError):
        benchmark_forward(_net(), [])
    with pytest.raises(TypeError):
        benchmark_forward("not a module", _inputs())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend"):
        benchmark_forward(_net(), _inputs(), compile=CompileSpec(backend="no-such-backend"))


def test_timed_regions_never_run_under_the_profiler():
    seen = []
    net = _net()
    net.register_forward_hook(lambda module, args, output: seen.append(torch.autograd._profiler_enabled()))
    benchmark_forward(net, _inputs(), warmup=1, repeats=2)  # the hook rides along on the deep copy
    assert seen and not any(seen)


# ---------------- profiling ----------------


def test_profile_writes_traces_to_an_explicit_directory(tmp_path):
    report = profile_forward(_net(), _inputs(), output_dir=tmp_path / "profile", wait=1, warmup=1, active=2, repeat=1)
    assert report.steps == 4 and dict(report.schedule) == {"wait": 1, "warmup": 1, "active": 2, "repeat": 1}
    assert report.output_dir == os.fspath(tmp_path / "profile") and report.trace_files
    assert all(os.path.isfile(path) and path.endswith(".json") for path in report.trace_files)
    assert "Name" in report.table and not torch.autograd._profiler_enabled()


@pytest.mark.parametrize(
    "kwargs",
    [{"active": 0}, {"repeat": 0}, {"wait": -1}, {"warmup": True}, {"row_limit": 0}],
)
def test_profile_counts_are_finite(tmp_path, kwargs):
    with pytest.raises(ValueError):
        profile_forward(_net(), _inputs(), output_dir=tmp_path, **kwargs)


def test_profile_needs_an_explicit_output_directory():
    with pytest.raises(TypeError):
        profile_forward(_net(), _inputs())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="output_dir"):
        profile_forward(_net(), _inputs(), output_dir=None)  # type: ignore[arg-type]


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt(), asyncio.CancelledError()])
def test_the_profiler_closes_on_error_and_cancellation(tmp_path, error):
    net = _net()
    calls = []

    def fail_on_third(module, args, output):
        calls.append(1)
        if len(calls) == 3:
            raise error

    net.register_forward_hook(fail_on_third)
    with pytest.raises(type(error)):
        profile_forward(net, _inputs(), output_dir=tmp_path, wait=1, warmup=1, active=3)
    assert not torch.autograd._profiler_enabled()
    # and a later profile starts cleanly
    assert profile_forward(_net(), _inputs(), output_dir=tmp_path / "again", active=1).steps == 3


def test_report_is_frozen():
    report = benchmark_forward(_net(), _inputs(), warmup=0, repeats=1)
    assert isinstance(report, BenchmarkReport) and report.stdev_seconds == 0.0
    with pytest.raises(AttributeError):
        report.repeats = 5  # type: ignore[misc]
