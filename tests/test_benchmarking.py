"""FEAT-029: bounded forward benchmarks and profiling."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import os

import pytest
import torch

import nnx.benchmarking as benchmarking
from nnx import (
    Activations,
    CompileFailed,
    CompileSpec,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
)
from nnx.benchmarking import BenchmarkReport, ProfileReport, benchmark_forward, compare_compile, profile_forward

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
    monkeypatch.setattr(benchmarking, "_prepared", lambda net, device, spec: (net.to(cpu).eval(), None))
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


def test_a_bfloat16_module_benchmarks_and_digests_its_weights():
    """NumPy has no bfloat16: the weights digest reads raw bytes, so a bf16
    module is benchmarked like any other (and its digest is stable)."""
    net = torch.nn.Linear(4, 2).to(torch.bfloat16)
    first = benchmarking._weights_digest(net)
    assert first == benchmarking._weights_digest(net)
    fp32 = torch.nn.Linear(4, 2)
    expected = benchmarking.hashlib.sha256()
    for name, tensor in sorted(fp32.state_dict().items()):
        expected.update(name.encode("utf-8"))
        expected.update(repr((str(tensor.dtype), tuple(tensor.shape))).encode("utf-8"))
        expected.update(tensor.detach().numpy().tobytes())
    assert benchmarking._weights_digest(fp32) == expected.hexdigest()  # fp32 bytes hash as NumPy's
    report = benchmark_forward(net, torch.randn(3, 4, dtype=torch.bfloat16), warmup=1, repeats=2)
    assert isinstance(report, BenchmarkReport)


def test_a_wrapper_subclass_without_its_own_dequantize_is_hashed_by_its_parts():
    import torch
    from torch.testing._internal.two_tensor import TwoTensor

    from nnx.benchmarking import benchmark_forward

    def digest(fill: float) -> str:
        torch.manual_seed(0)  # the same Linear both times
        module = torch.nn.Linear(2, 2)
        module.register_buffer("pair", TwoTensor(torch.ones(2), torch.full((2,), fill)))
        return benchmark_forward(module, torch.randn(3, 2), repeats=1).weights_digest

    assert digest(1.0) == digest(1.0) != digest(2.0)  # finishes, and reads the inner tensors


# ---------------- compilation through nnx.compilation ----------------

_CALLS = {"n": 0}


def _backend(name: str, build) -> CompileSpec:
    """Register a dynamo backend once per process (``build(gm)`` returns its
    compiled callable) and return a spec for it."""
    from torch._dynamo import register_backend

    if name not in torch._dynamo.list_backends(exclude_tags=()):
        register_backend(compiler_fn=lambda gm, example_inputs: build(gm), name=name)
    return CompileSpec(backend=name)


def _counting() -> CompileSpec:
    """Runs dynamo's graph unchanged, counting each call of a compiled graph."""

    def build(gm):
        def run(*args):
            _CALLS["n"] += 1
            return gm.forward(*args)

        return run

    return _backend("nnx_test_counting", build)


def _broken(**changes) -> CompileSpec:
    def build(gm):
        raise RuntimeError("backend failed to compile")

    return dataclasses.replace(_backend("nnx_test_broken", build), **changes)


def _perturbing(name: str, change) -> CompileSpec:
    """Runs dynamo's graph, then alters output ``i`` with ``change(i, t)``:
    a stand-in for a miscompile."""

    def build(gm):
        def run(*args):
            out = gm.forward(*args)
            many = isinstance(out, (list, tuple))
            seq = [change(i, t) for i, t in enumerate(out if many else [out])]
            return tuple(seq) if many else seq[0]

        return run

    return _backend(name, build)


def _feed_fwd_net(depth: int) -> torch.nn.Module:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8] * depth, dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    ).net


def _x() -> torch.Tensor:
    return torch.randn(16, 4, generator=torch.Generator().manual_seed(0))


def test_a_compare_compile_sweep_compiles_every_depth_without_a_reset():
    """Each benchmark gets dynamo's recompile limit afresh, as each fit does:
    past the limit's 8 entries, depths 9 and 10 still compile."""
    spec = _counting()
    for depth in range(1, 11):
        before = _CALLS["n"]
        comparison = compare_compile(_feed_fwd_net(depth), _x(), compile=spec, warmup=2, repeats=3)
        assert _CALLS["n"] - before == 6, depth  # 1 first + 2 warmup + 3 timed calls, all compiled
        compiled = comparison.compiled
        assert (compiled.label, compiled.backend, compiled.compile.effective) == ("compiled", spec.backend, "compiled")
        assert comparison.max_abs_diff == 0.0


def test_compare_compile_after_nine_compiled_fits_of_the_class_compiles(tmp_path, monkeypatch):
    from torch._dynamo.utils import counters

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    for depth in range(1, 10):
        assert _fit(depth, FAST).compile.effective == "compiled", depth
    graphs = counters["stats"]["unique_graphs"]
    comparison = compare_compile(_feed_fwd_net(9), _x(), compile=FAST, warmup=1, repeats=2)
    assert counters["stats"]["unique_graphs"] > graphs  # a no-grad graph compiled for it
    assert (comparison.compiled.label, comparison.compiled.compile.effective) == ("compiled", "compiled")


def _run(helper: str, net: torch.nn.Module, spec: CompileSpec, tmp_path):
    if helper == "benchmark_forward":
        return benchmark_forward(net, _x(), compile=spec, warmup=1, repeats=2)
    if helper == "compare_compile":
        return compare_compile(net, _x(), compile=spec, warmup=1, repeats=2).compiled
    return profile_forward(net, _x(), compile=spec, output_dir=tmp_path / "profile", wait=0, warmup=0, active=2)


HELPERS = ["benchmark_forward", "compare_compile", "profile_forward"]


@pytest.mark.parametrize("helper", HELPERS)
def test_a_failing_backend_raises_compile_failed_under_the_error_policy(helper, tmp_path):
    from torch._dynamo.exc import TorchDynamoException

    with pytest.raises(CompileFailed, match="backend failed to compile|nnx_test_broken") as caught:
        _run(helper, _feed_fwd_net(1), _broken(), tmp_path)
    assert isinstance(caught.value.__cause__, TorchDynamoException)  # torch's own error
    assert caught.value.record.effective == "failed"


@pytest.mark.parametrize("helper", HELPERS)
def test_a_failing_backend_restarts_eagerly_under_the_eager_policy(helper, tmp_path):
    report = _run(helper, _feed_fwd_net(1), _broken(on_failure="eager"), tmp_path)
    assert report.compile.effective == "eager" and report.compile.restart["error"] == "BackendCompilerFailed"
    if isinstance(report, ProfileReport):
        assert report.trace_files
    else:
        assert (report.label, report.backend) == ("eager", None)
        assert report.state()["compile"]["effective"] == "eager"


def test_reports_expose_the_effective_compile_state(tmp_path):
    report = benchmark_forward(_net(), _inputs(), compile=FAST, warmup=0, repeats=1)
    assert report.compile.effective == "compiled" and report.state()["compile"] == report.compile.record()
    eager = benchmark_forward(_net(), _inputs(), warmup=0, repeats=1)
    assert eager.compile is None and eager.state()["compile"] is None
    profile = profile_forward(_net(), _inputs(), compile=FAST, output_dir=tmp_path, wait=0, warmup=0, active=1)
    assert profile.compile.effective == "compiled"
    assert profile_forward(_net(), _inputs(), output_dir=tmp_path, active=1).compile is None


class _Skipped(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    @torch.compiler.disable
    def forward(self, x):
        return self.lin(x)


@pytest.mark.parametrize("helper", HELPERS)
def test_a_skipped_forward_is_never_reported_compiled(helper, tmp_path):
    # A graph cached for torch's own modules: a skipped forward is traced
    # through the same wrapper code, so a held graph proves nothing.
    assert benchmark_forward(_net(), _inputs(), compile=FAST, warmup=0, repeats=1).label == "compiled"
    with pytest.raises(CompileFailed, match="compiled nothing"):
        _run(helper, _Skipped(), FAST, tmp_path)
    for attempt in range(2):  # the second meets the skipped frame dynamo cached
        report = _run(helper, _Skipped(), dataclasses.replace(FAST, on_failure="eager"), tmp_path / str(attempt))
        assert report.compile.effective == "eager" and "compiled nothing" in report.compile.restart["message"]


@pytest.mark.parametrize("policy", ["error", "eager"])
@pytest.mark.parametrize("helper", HELPERS)
def test_a_forward_disabled_on_one_instance_is_never_reported_compiled(helper, policy, tmp_path):
    """The class holds cached graphs from an earlier benchmark, but this
    instance's forward is disabled: none of them runs, so nothing compiled."""
    spec = dataclasses.replace(_counting(), on_failure=policy)
    assert _run(helper, _BreaksFirst(), spec, tmp_path / "first").compile.effective == "compiled"
    net = _BreaksFirst()
    net.forward = torch.compiler.disable(net.forward)
    before = _CALLS["n"]
    try:
        report = _run(helper, net, spec, tmp_path / "disabled")
    except CompileFailed as caught:
        assert policy == "error" and "compiled nothing" in str(caught)
    else:
        assert policy == "eager", "compiled nothing, yet no CompileFailed"
        assert report.compile.effective == "eager" and "compiled nothing" in report.compile.restart["message"]
    assert _CALLS["n"] == before


class _NoOps(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("policy", ["error", "eager"])
@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize("make", [lambda: torch.nn.LSTM(4, 8), torch.nn.Identity, _NoOps], ids=["lstm", "id", "noops"])
def test_a_forward_dynamo_skips_is_never_reported_compiled(helper, policy, make, tmp_path):
    """Torch's own modules share the wrapper code dynamo traces, so the
    graph cached first for an nn.Sequential is held for an nn.LSTM too: it
    proves nothing ran compiled. Dynamo skips an LSTM (``allow_rnn`` is
    off) and a forward with no torch op; an nn.Identity too, except on the
    2.4 floor, which compiles a graph for it."""
    from torch._dynamo.utils import counters

    assert benchmark_forward(_net(), _inputs(), compile=FAST, warmup=0, repeats=1).label == "compiled"
    spec = dataclasses.replace(FAST, on_failure=policy)
    for attempt in range(2):  # the second meets the skipped frame dynamo cached
        graphs = counters["stats"]["unique_graphs"]
        try:
            report = _run(helper, make(), spec, tmp_path / str(attempt))
        except CompileFailed as caught:
            assert policy == "error" and "compiled nothing" in str(caught)
            continue
        if counters["stats"]["unique_graphs"] > graphs:
            assert make is torch.nn.Identity and report.compile.effective == "compiled"
            continue
        assert policy == "eager", "compiled nothing, yet no CompileFailed"
        assert report.compile.effective == "eager" and "compiled nothing" in report.compile.restart["message"]
        if not isinstance(report, ProfileReport):
            assert (report.label, report.backend) == ("eager", None)


def _entries() -> int:
    """Dynamo's cache entries on every code object it traced (the forward's,
    or the module call wrapper's on the 2.4 floor)."""
    from torch._dynamo.convert_frame import input_codes
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    codes = [code for code in (ref() for ref in list(input_codes.seen)) if code is not None]
    return sum(len(_debug_get_cache_entry_list(code)) for code in codes)


@pytest.mark.parametrize("helper", HELPERS)
def test_a_model_benchmarked_again_reuses_its_cached_graphs(helper, tmp_path):
    """Benchmarks with one spec share its counting backend, so a model
    benchmarked again runs the graphs cached for it and adds no cache
    entry, yet each run counts as compiled. (The 2.4 floor keys its graphs
    on the deep copy: it recompiles each time and frees the entry with the
    copy.)"""
    spec, net = _counting(), _feed_fwd_net(1)
    held = []
    for attempt in range(5):
        before = _CALLS["n"]
        report = _run(helper, net, spec, tmp_path / str(attempt))
        assert report.compile.effective == "compiled" and _CALLS["n"] > before, attempt
        if not isinstance(report, ProfileReport):
            assert (report.label, report.backend) == ("compiled", spec.backend)
        held.append(_entries())
    assert held == [held[0]] * 5 and (held[0] >= 1 or torch.__version__ < "2.5")


def test_another_spec_never_reuses_the_cached_graphs():
    """Another backend, or the same backend with other options, has a
    counting backend of its own: it compiles and runs graphs of its own."""
    from torch._dynamo.utils import counters

    net = _feed_fwd_net(1)
    assert benchmark_forward(net, _x(), compile=FAST, warmup=0, repeats=1).compile.effective == "compiled"
    for spec in (_counting(), dataclasses.replace(FAST, dynamic=False)):
        graphs, before = counters["stats"]["unique_graphs"], _CALLS["n"]
        report = benchmark_forward(net, _x(), compile=spec, warmup=0, repeats=1)
        assert report.compile.effective == "compiled" and counters["stats"]["unique_graphs"] > graphs, spec
        assert (_CALLS["n"] > before) == (spec.backend == "nnx_test_counting")


class _BreaksFirst(torch.nn.Module):
    """A forward that graph-breaks before any torch op (a print or log
    call): its graph is in a resume function, not in the forward's entry."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        torch._dynamo.graph_break()
        return self.lin(x)


class _BreaksInside(torch.nn.Module):
    """The leading graph break in a nested module's forward."""

    def __init__(self) -> None:
        super().__init__()
        self.block, self.head = _BreaksFirst(), torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.head(self.block(x))


class _BreaksMidway(torch.nn.Module):
    """A graph break between two torch ops: the second graph is in a
    resume function."""

    def __init__(self) -> None:
        super().__init__()
        self.a, self.b = torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)

    def forward(self, x):
        h = self.a(x)
        torch._dynamo.graph_break()
        return self.b(h)


def _sequential_breaking_first() -> torch.nn.Module:
    return torch.nn.Sequential(_BreaksFirst(), torch.nn.Linear(2, 2))


def _sequential_breaking_midway() -> torch.nn.Module:
    return torch.nn.Sequential(_BreaksMidway(), torch.nn.Linear(2, 2))


def _apply(layer, x):
    """A plain helper function that graph-breaks before its torch op."""
    torch._dynamo.graph_break()
    return layer(x)


class _BreaksInHelper(torch.nn.Module):
    """The graph break in a plain helper function the forward calls first:
    its graphs are cached on the helper's code, not the forward's."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        return _apply(self.lin, x)


class _BreaksInMethod(torch.nn.Module):
    """The graph break in a method other than ``forward``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def project(self, x):
        torch._dynamo.graph_break()
        return self.lin(x)

    def forward(self, x):
        return self.project(x)


@pytest.mark.parametrize("policy", ["error", "eager"])
@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize(
    "make",
    [
        _BreaksFirst,
        _BreaksMidway,
        _BreaksInside,
        _sequential_breaking_first,
        _sequential_breaking_midway,
        _BreaksInHelper,
        _BreaksInMethod,
    ],
    ids=["leading", "midway", "nested", "sequential-leading", "sequential-midway", "helper", "method"],
)
def test_a_graph_breaking_forward_benchmarked_again_is_reported_compiled(helper, policy, make, tmp_path):
    """Benchmarked again in one process (the same model, or a new one of the
    class), wherever its graph break sits: in the forward, a child's, a
    helper function or another method, and through the wrapper torch's own
    modules (an ``nn.Sequential``) share. Each benchmark runs graphs
    compiled for its spec: its own, or those cached for an earlier one."""
    spec = dataclasses.replace(_counting(), on_failure=policy)
    same = make()
    for attempt, net in enumerate([same, same, make(), make()]):
        before = _CALLS["n"]
        report = _run(helper, net, spec, tmp_path / str(attempt))
        assert _CALLS["n"] > before, attempt
        assert report.compile.effective == "compiled" and report.compile.restart is None, attempt
        if not isinstance(report, ProfileReport):
            assert (report.label, report.backend) == ("compiled", spec.backend)


class _WideBreaksInHelper(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, width)

    def forward(self, x):
        return _apply(self.lin, x)


@pytest.mark.parametrize("helper", HELPERS)
def test_a_graph_breaking_forward_keeps_compiling_past_the_recompile_limit(helper, tmp_path):
    """A benchmark adds cache entries to the helper function and the resume
    functions its graph break leaves, not only to the forward: ten widths
    with static shapes give the helper ten entries, past the recompile
    limit's 8, and all compile."""
    spec = dataclasses.replace(_counting(), dynamic=False)
    for width in range(1, 11):
        before = _CALLS["n"]
        report = _run(helper, _WideBreaksInHelper(width), spec, tmp_path / str(width))
        assert report.compile.effective == "compiled" and _CALLS["n"] > before, width


_ACCUMULATED = next(
    n for n in ("accumulated_recompile_limit", "accumulated_cache_size_limit") if hasattr(torch._dynamo.config, n)
)


@pytest.mark.parametrize("helper", HELPERS)
def test_one_model_benchmarked_past_the_accumulated_limit_keeps_compiling(helper, tmp_path):
    """A model benchmarked again and again adds no cache entry, so it never
    nears dynamo's accumulated limit (256 entries of a code object, here
    lowered to 3)."""
    spec, net = _counting(), _feed_fwd_net(1)
    with torch._dynamo.config.patch(**{_ACCUMULATED: 3}):
        for attempt in range(6):
            before = _CALLS["n"]
            report = _run(helper, net, spec, tmp_path / str(attempt))
            assert report.compile.effective == "compiled" and _CALLS["n"] > before, attempt


def _fit(depth: int, spec: CompileSpec):
    generator = torch.Generator().manual_seed(0)
    batches = [(torch.randn(8, 4, generator=generator), torch.randint(0, 2, (8,), generator=generator))]
    params = NNTrainParams(
        n_epochs=1,
        train_loader=batches,
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        overwrite_existing=True,
    )
    torch.manual_seed(0)
    model = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8] * depth, dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    return model.train(params=params, compile=spec)


def test_a_compiled_fit_still_compiles_after_many_benchmarks_of_the_class(tmp_path, monkeypatch):
    """Benchmarks of one model leave no new cache entries behind, so a
    compiled fit of the class afterwards stays under dynamo's accumulated
    limit (here lowered to 4, against 30 benchmarks) and compiles."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    net = _feed_fwd_net(1)
    with torch._dynamo.config.patch(**{_ACCUMULATED: 4}):
        for attempt in range(30):
            report = benchmark_forward(net, _x(), compile=FAST, warmup=0, repeats=1)
            assert report.compile.effective == "compiled", attempt
        assert _fit(1, FAST).compile.effective == "compiled"


def test_neither_fits_nor_benchmarks_extend_the_accumulated_limit(tmp_path, monkeypatch):
    """The accumulated limit (here lowered to 3) stays as configured for a
    fit and a benchmark alike. Where torch cannot make the recompile limit
    an error (the 2.4 floor), neither changes the recompile limit either."""
    config = torch._dynamo.config
    limit = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    seen: list[tuple[int, int]] = []

    def build(gm):
        seen.append((int(getattr(config, _ACCUMULATED)), int(getattr(config, limit))))
        return gm.forward

    spec = _backend("nnx_test_reads_the_limits", build)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    configured = int(getattr(config, limit))
    with config.patch(**{_ACCUMULATED: 3}):
        for depth in (1, 2):  # the second fit's allowance is the first fit's entries
            assert _fit(depth, spec).compile.effective == "compiled", depth
        fits = len(seen)
        report = benchmark_forward(_feed_fwd_net(3), _x(), compile=spec, warmup=0, repeats=1)
        assert report.compile.effective == "compiled"
    assert fits and len(seen) > fits
    assert {accumulated for accumulated, _ in seen} == {3}
    if not hasattr(config, "fail_on_recompile_limit_hit"):
        assert {recompile for _, recompile in seen} == {configured}


@pytest.mark.parametrize("policy", ["error", "eager"])
@pytest.mark.parametrize("make", [_BreaksMidway, _BreaksInHelper], ids=["midway", "helper"])
def test_a_graph_cached_for_another_spec_never_counts_as_compiled(policy, make, monkeypatch):
    """Without the allowance and with a recompile limit of 1, a benchmark
    with another spec (here other options) cannot compile its graphs (the
    2.4 floor still compiles the one before the break, keyed on the new
    copy). Dynamo may then run the graphs cached for the first spec (torch
    2.13 does, past the limit, without failing): they were not compiled for
    this spec, so unless a graph of its own compiled and ran, it compiled
    nothing. The first spec again runs its cached graphs: compiled."""
    from torch._dynamo.utils import counters

    import nnx.compilation as compilation

    config = torch._dynamo.config
    spec = dataclasses.replace(FAST, on_failure=policy)
    monkeypatch.setattr(compilation._InferenceSession, "strict", lambda self: contextlib.nullcontext())
    first = make()
    with config.patch(**{"recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit": 1}):
        assert benchmark_forward(first, _x(), compile=spec, warmup=0, repeats=1).compile.effective == "compiled"
        assert benchmark_forward(first, _x(), compile=spec, warmup=0, repeats=1).compile.effective == "compiled"
        graphs, other = counters["stats"]["unique_graphs"], dataclasses.replace(spec, dynamic=False)
        try:
            report = benchmark_forward(first, _x(), compile=other, warmup=0, repeats=1)
        except CompileFailed as caught:
            assert policy == "error" and "compiled nothing" in str(caught)
            return
    if report.compile.effective == "compiled":
        assert counters["stats"]["unique_graphs"] > graphs and torch.__version__ < "2.5"
        return
    assert policy == "eager", "compiled nothing, yet no CompileFailed"
    assert (report.label, report.backend, report.compile.effective) == ("eager", None, "eager")


@pytest.mark.parametrize("helper", HELPERS)
def test_a_spent_cache_is_never_reported_compiled(helper, tmp_path, monkeypatch):
    """With no allowance for the entries already held, a recompile limit of
    1 is spent by the first depth: a later depth fails, or compiles afresh
    where torch keeps a cache per module instance (the 2.4 floor), but never
    runs eagerly labelled compiled."""
    import nnx.compilation as compilation

    config = torch._dynamo.config
    monkeypatch.setattr(compilation._InferenceSession, "strict", lambda self: compilation._strict_recompiles(0))
    spec, refused = _counting(), []
    # dynamo's own patch, not monkeypatch: pytest >= 9.2 undoes a setattr on
    # its config module with a delattr, which hides the option for the worker.
    with config.patch(**{"recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit": 1}):
        for depth in (1, 2, 3):
            before = _CALLS["n"]
            try:
                report = _run(helper, _feed_fwd_net(depth), spec, tmp_path / str(depth))
            except CompileFailed:
                refused.append(depth)
                continue
            assert report.compile.effective == "compiled" and _CALLS["n"] > before, depth
        if hasattr(config, "fail_on_recompile_limit_hit"):
            assert refused == [2, 3]
            report = _run(helper, _feed_fwd_net(4), dataclasses.replace(spec, on_failure="eager"), tmp_path / "4")
            assert report.compile.effective == "eager"


@pytest.mark.parametrize("policy", ["error", "eager"])
def test_a_failing_backend_after_a_sweep_is_never_reported_compiled(policy):
    for depth in range(1, 9):  # fills dynamo's default recompile limit for the class
        assert benchmark_forward(_feed_fwd_net(depth), _x(), compile=FAST, warmup=0, repeats=1).label == "compiled"
    spec = _broken(on_failure=policy)
    if policy == "error":
        with pytest.raises(CompileFailed):
            benchmark_forward(_feed_fwd_net(9), _x(), compile=spec, warmup=0, repeats=1)
    else:
        report = benchmark_forward(_feed_fwd_net(9), _x(), compile=spec, warmup=0, repeats=1)
        assert (report.label, report.backend, report.compile.effective) == ("eager", None, "eager")


# ---------------- max_abs_diff ----------------


class _TwoHeads(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a, self.b = torch.nn.Linear(4, 2), torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.a(x), self.b(x)


@dataclasses.dataclass
class _Output:
    logits: torch.Tensor


class _DataclassOut(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        return _Output(self.lin(x))


class _MaskedLogits(torch.nn.Module):
    """One head with a class masked to -inf, identical eager and compiled."""

    def __init__(self, masked_first: bool) -> None:
        super().__init__()
        self.lin, self.masked_first = torch.nn.Linear(4, 3), masked_first

    def forward(self, x):
        logits = self.lin(x)
        masked = logits.masked_fill(torch.tensor([False, False, True]), float("-inf"))
        return (masked, logits) if self.masked_first else (logits, masked)


class _Offset(torch.nn.Module):
    def forward(self, x):
        return x + 1e8


class _Complex(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.complex64) * 2


def _diff(net: torch.nn.Module, spec: CompileSpec, x=None) -> float:
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(0)) if x is None else x
    return compare_compile(net, x, compile=spec, warmup=0, repeats=1).max_abs_diff


def test_max_abs_diff_never_hides_a_one_sided_nan():
    nan_head_1 = _perturbing("nnx_test_nan_head_1", lambda i, t: torch.full_like(t, float("nan")) if i == 1 else t)
    nan_head_0 = _perturbing("nnx_test_nan_head_0", lambda i, t: torch.full_like(t, float("nan")) if i == 0 else t)
    assert _diff(_TwoHeads(), nan_head_1) == float("inf")
    assert _diff(_TwoHeads(), nan_head_0) == float("inf")


@pytest.mark.parametrize("infinity", [float("inf"), float("-inf")], ids=["pos", "neg"])
def test_max_abs_diff_counts_a_one_sided_infinity_as_inf(infinity):
    """|finite - inf| is inf, not the largest finite float."""
    to_inf = _perturbing(f"nnx_test_to_{infinity}", lambda i, t: torch.full_like(t, infinity))
    assert _diff(torch.nn.Linear(4, 2), to_inf) == float("inf")


def test_max_abs_diff_reads_every_head_and_dataclass_fields():
    plus_100 = _perturbing("nnx_test_plus_100", lambda i, t: t + 100)
    assert _diff(_TwoHeads(), plus_100) == pytest.approx(100.0, abs=1e-4)
    assert _diff(_DataclassOut(), plus_100) == pytest.approx(100.0, abs=1e-4)


@dataclasses.dataclass
class _WithUnset:
    logits: torch.Tensor
    cache: object = dataclasses.field(init=False)


class _UnsetFieldOut(_DataclassOut):
    def forward(self, x):
        return _WithUnset(self.lin(x))


def test_max_abs_diff_skips_a_dataclass_field_never_assigned():
    plus_100 = _perturbing("nnx_test_plus_100", lambda i, t: t + 100)
    assert _diff(_UnsetFieldOut(), plus_100) == pytest.approx(100.0, abs=1e-4)
    unset, assigned = _WithUnset(torch.zeros(1)), _WithUnset(torch.zeros(1))
    assigned.cache = None
    assert benchmarking._flatten(unset)[1] != benchmarking._flatten(assigned)[1]  # refused as differently structured


@pytest.mark.parametrize("masked_first", [True, False])
def test_identical_infinities_differ_by_zero_in_either_order(masked_first):
    assert _diff(_MaskedLogits(masked_first), CompileSpec(backend="eager")) == 0.0


def test_max_abs_diff_compares_float64_and_complex_outputs_at_full_width():
    plus_one = _perturbing("nnx_test_plus_one", lambda i, t: t + 1)
    assert _diff(_Offset(), plus_one, torch.ones(3, 1, dtype=torch.float64)) == 1.0
    plus_1j = _perturbing("nnx_test_plus_1j", lambda i, t: t + 1j)
    assert _diff(_Complex(), plus_1j) == pytest.approx(1.0)


# ---------------- nothing on disk before validation ----------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"net": "not a module"},
        {"inputs": []},
        {"compile": "inductor"},
        {"compile": CompileSpec(backend="no-such-backend")},
        {"device": "no-such-device"},
        {"active": 0},
    ],
    ids=["net", "inputs", "compile-type", "backend", "device", "count"],
)
def test_an_invalid_profile_creates_no_directory(tmp_path, kwargs):
    arguments = {"net": _net(), "inputs": _inputs(), **kwargs}
    out = tmp_path / "parent" / "traces"
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        profile_forward(output_dir=out, **arguments)
    assert not (tmp_path / "parent").exists()
