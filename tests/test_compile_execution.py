"""FEAT-029: opt-in torch.compile of the built-in FP32 training forward."""

from __future__ import annotations

import copy
import json
import os

import pytest
import torch
import yaml

from nnx import (
    Activations,
    Checkpoints,
    CompileFailed,
    CompileRecord,
    CompileSpec,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    PrecisionPolicy,
)
from nnx.compilation import _CompileSession
from nnx.nn.callbacks import Callback
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_run import NNRun
from nnx.nn.params.nn_transformer_params import NNTransformerParams
from nnx.provenance import ExperimentManifest

FAST = CompileSpec(backend="aot_eager")  # dynamo + AOTAutograd, no C++ toolchain
RTOL, ATOL = 1e-5, 1e-6


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _feed_fwd(**model_kwargs) -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, **model_kwargs),
    )


def _transformer() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            ffn_mult=2,
            max_seq_len=8,
        ),
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _ff_batches():
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(8, 4, generator=generator)
    y = torch.randint(0, 3, (8,), generator=generator)
    return [(X, y)]


def _lm_batches():
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 16, (4, 6), generator=generator)
    return [(tokens, torch.roll(tokens, -1, dims=1))]


def _params(batches, **kwargs) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=kwargs.pop("n_epochs", 1),
        train_loader=batches,
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        overwrite_existing=True,
        **kwargs,
    )


def _break_graph(model: NNModel) -> None:
    """A deliberate graph break in every forward of model.net."""
    model.net.register_forward_pre_hook(lambda module, args: torch._dynamo.graph_break())


@pytest.fixture
def attempts(monkeypatch):
    """Every attempt record written (a failed first epoch leaves no run directory behind)."""
    import nnx.provenance as provenance

    written: list = []
    original = provenance._write_json

    def spy(path, value):
        if os.path.basename(path) == provenance.ATTEMPT_FILE:
            written.append(json.loads(json.dumps(value)))
        return original(path, value)

    monkeypatch.setattr(provenance, "_write_json", spy)
    return written


def _only_run_dir() -> str:
    (name,) = [name for name in os.listdir("runs") if not name.startswith(".")]
    return os.path.join("runs", name)


def _assert_same_weights(a: NNModel, b: NNModel) -> None:
    for name, tensor in a.net.state_dict().items():
        torch.testing.assert_close(b.net.state_dict()[name], tensor, rtol=RTOL, atol=ATOL)


class SessionSpy(Callback):
    """Records, per epoch, the model's compile session and the optimizer's parameters."""

    def __init__(self) -> None:
        self.sessions: list = []
        self.optimizer_params: list = []

    def on_epoch_end(self, ctx) -> None:
        self.sessions.append(ctx.model._compile_session)
        self.optimizer_params.append([id(p) for group in ctx.optimizer.param_groups for p in group["params"]])


# ---------------- disabled / enabled ----------------


def test_disabled_stays_eager_and_records_nothing():
    model, spy = _feed_fwd(), SessionSpy()
    run = model.train(params=_params(_ff_batches()), callbacks=[spy])
    assert spy.sessions == [None] and run.compile is None
    with open(os.path.join("runs", run.id, "metadata.yaml"), encoding="utf-8") as handle:
        assert "compile" not in yaml.safe_load(handle)
    state = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    assert state is not None and state["compile"] is None


def test_enabled_records_backend_options_version_and_policy():
    model, spy = _feed_fwd(), SessionSpy()
    run = model.train(params=_params(_ff_batches()), callbacks=[spy], compile=FAST)
    assert isinstance(spy.sessions[0], _CompileSession) and model._compile_session is None
    record = run.compile
    assert isinstance(record, CompileRecord)
    assert (record.backend, record.on_failure, record.effective) == ("aot_eager", "error", "compiled")
    assert record.options == {"mode": None, "fullgraph": False, "dynamic": None}
    assert record.torch_version == torch.__version__ and record.capture == "full" and record.restart is None
    # the optimizer owns model.net's own parameters, not a wrapper's
    assert spy.optimizer_params[0] == [id(p) for p in model.net.parameters()]


def test_compile_is_not_part_of_the_run_id():
    eager = _feed_fwd().train(params=_params(_ff_batches()))
    compiled = _feed_fwd().train(params=_params(_ff_batches()), compile=FAST)
    assert eager.id == compiled.id


# ---------------- parity (the advertised inductor backend) ----------------


@pytest.mark.parametrize(
    ("make", "batches"), [(_feed_fwd, _ff_batches), (_transformer, _lm_batches)], ids=["feed_fwd", "transformer"]
)
def test_inductor_matches_eager_logits_and_one_sgd_update(make, batches):
    eager, compiled = make(), make()
    tokens = batches()[0][0]
    session = _CompileSession(compiled.net, CompileSpec())
    with torch.no_grad():
        torch.testing.assert_close(session.forward(tokens), eager.net(tokens), rtol=RTOL, atol=ATOL)
    assert session.record.effective == "compiled" and session.record.backend == "inductor"
    torch._dynamo.reset()

    eager.train(params=_params(batches()))
    run = compiled.train(params=_params(batches()), compile=CompileSpec())
    assert run.compile.effective == "compiled" and run.compile.capture == "full"
    _assert_same_weights(eager, compiled)


def test_a_graph_break_is_partial_capture_but_fullgraph_rejects_it():
    eager, partial = _feed_fwd(), _feed_fwd()
    _break_graph(partial)
    run = partial.train(params=_params(_ff_batches()), compile=FAST)
    assert run.compile.effective == "compiled" and run.compile.capture == "partial" and run.compile.graph_breaks >= 1
    eager.train(params=_params(_ff_batches()))
    _assert_same_weights(eager, partial)

    torch._dynamo.reset()
    strict = _feed_fwd()
    _break_graph(strict)
    with pytest.raises(CompileFailed) as caught:
        strict.train(params=_params(_ff_batches()), compile=CompileSpec(backend="aot_eager", fullgraph=True))
    assert caught.value.record.effective == "failed"
    assert caught.value.record.restart["phase"] == "train" and caught.value.record.restart["batch"] == 0
    assert isinstance(caught.value.__cause__, torch._dynamo.exc.Unsupported)
    # the failed forward ran before any update
    _assert_same_weights(_feed_fwd(), strict)


def test_restart_eager_reruns_only_the_forward_and_records_it():
    eager, restarted = _feed_fwd(), _feed_fwd()
    _break_graph(restarted)
    run = restarted.train(
        params=_params(_ff_batches(), n_epochs=2),
        compile=CompileSpec(backend="aot_eager", fullgraph=True, on_failure="eager"),
    )
    assert run.compile.effective == "eager" and run.compile.graph_breaks is None
    assert run.compile.restart["error"] == "Unsupported"
    assert (run.compile.restart["phase"], run.compile.restart["epoch"], run.compile.restart["batch"]) == ("train", 0, 0)
    eager.train(params=_params(_ff_batches(), n_epochs=2))
    _assert_same_weights(eager, restarted)  # each update applied exactly once


def test_a_failure_after_the_optimizer_mutated_is_never_replayed(monkeypatch, attempts):
    model = _feed_fwd()
    before = copy.deepcopy(model.net.state_dict())
    original_step = torch.optim.SGD.step
    calls = []

    def step_then_fail(self, *args, **kwargs):
        original_step(self, *args, **kwargs)
        calls.append(1)
        raise torch._dynamo.exc.TorchRuntimeError("injected after the update")

    monkeypatch.setattr(torch.optim.SGD, "step", step_then_fail)
    manifest = ExperimentManifest.for_model(model, train=_params(_ff_batches()))
    with pytest.raises(torch._dynamo.exc.TorchRuntimeError, match="injected"):
        model.train(
            params=_params(_ff_batches()),
            compile=CompileSpec(backend="aot_eager", on_failure="eager"),
            provenance=manifest,
        )
    assert calls == [1]  # one update, not replayed eagerly
    reference = _feed_fwd()
    reference.net.load_state_dict(before)
    one_step = torch.optim.SGD(reference.net.parameters(), lr=0.1)
    X, y = _ff_batches()[0]
    torch.nn.functional.cross_entropy(reference.net(X), y).backward()
    original_step(one_step)
    _assert_same_weights(reference, model)
    attempt = attempts[-1]
    assert attempt["status"] == "failed" and attempt["compile"]["effective"] == "compiled"
    assert attempt["compile"]["restart"] is None


def test_a_failed_compile_attempt_never_reports_success(attempts):
    model = _feed_fwd()
    _break_graph(model)
    manifest = ExperimentManifest.for_model(model, train=_params(_ff_batches()))
    with pytest.raises(CompileFailed):
        model.train(
            params=_params(_ff_batches()),
            compile=CompileSpec(backend="aot_eager", fullgraph=True),
            provenance=manifest,
        )
    assert attempts[0]["status"] == "running" and attempts[0]["compile"]["effective"] == "pending"
    attempt = attempts[-1]
    assert attempt["status"] == "failed" and attempt["compile"]["effective"] == "failed"
    assert attempt["error"]["type"] == "CompileFailed"
    assert all(record["compile"]["effective"] != "compiled" for record in attempts)
    # nothing was committed: the empty reservation is released
    assert not os.path.exists("runs") or not [name for name in os.listdir("runs") if not name.startswith(".")]


# ---------------- records agree everywhere ----------------


@pytest.mark.parametrize("restart", [False, True], ids=["compiled", "restart-eager"])
def test_requested_and_effective_state_agree_across_run_checkpoint_and_provenance(restart):
    model = _feed_fwd()
    spec = CompileSpec(backend="aot_eager")
    if restart:
        _break_graph(model)
        spec = CompileSpec(backend="aot_eager", fullgraph=True, on_failure="eager")
    params = _params(_ff_batches(), n_epochs=2)
    run = model.train(params=params, compile=spec, provenance=ExperimentManifest.for_model(model, train=params))
    expected = run.compile.record()
    assert expected["effective"] == ("eager" if restart else "compiled")
    assert (expected["restart"] is not None) is restart
    assert NNRun.load(run.id).compile.record() == expected
    state = NNCheckpoint.load_training_state(run.id, Checkpoints.LAST)
    assert state is not None and state["compile"] == expected
    with open(os.path.join("runs", run.id, "attempt.json"), encoding="utf-8") as handle:
        attempt = json.load(handle)
    assert attempt["status"] == "completed" and attempt["compile"] == expected
    assert run.provenance is not None and run.provenance.attempt.compile == expected


# ---------------- the canonical module survives ----------------


def test_checkpoints_hold_eager_keys_and_reload_eagerly():
    model = _feed_fwd()
    run = model.train(params=_params(_ff_batches()), compile=FAST)
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None
    assert not [key for key in checkpoint.net_state if "_orig_mod" in key]
    assert set(checkpoint.net_state) == set(model.net.state_dict())
    reloaded = NNModel.from_checkpoint(checkpoint)
    assert type(reloaded.net) is type(model.net) and reloaded._compile_session is None
    X = _ff_batches()[0][0]
    with torch.no_grad():
        torch.testing.assert_close(reloaded.net(X), model.net(X))
    # run views never name the wrapper
    views = [str(run), run._repr_html_(), json.dumps(run.state(), default=str)]
    with open(os.path.join("runs", run.id, "metadata.yaml"), encoding="utf-8") as handle:
        views.append(handle.read())
    for view in views:
        assert "_orig_mod" not in view and "OptimizedModule" not in view


def test_resume_rebuilds_the_wrapper():
    model = _feed_fwd()
    first, second = SessionSpy(), SessionSpy()
    run = model.train(params=_params(_ff_batches()), callbacks=[first], compile=FAST)
    resumed = _feed_fwd()
    resumed_run = resumed.train(
        params=_params(_ff_batches(), n_epochs=1, resume_from_run_id=run.id), callbacks=[second], compile=FAST
    )
    assert second.sessions[0] is not first.sessions[0]
    assert second.sessions[0]._net is resumed.net and resumed_run.compile.effective == "compiled"
    state = NNCheckpoint.load_training_state(resumed_run.id, Checkpoints.LAST)
    assert state is not None and state["completed_epoch"] == 1 and state["compile"]["effective"] == "compiled"


def test_export_summary_and_tying_survive_a_compiled_fit(tmp_path):
    pytest.importorskip("onnx")
    from nnx.viz import summary

    model = _transformer()
    run = model.train(params=_params(_lm_batches()), compile=FAST)
    assert model.net.lm_head.weight is model.net.tok_embed.weight
    assert "OptimizedModule" not in str(summary(model, input_data=_lm_batches()[0][0]))
    model.save_pretrained(tmp_path / "hub")
    from_hub = NNModel.from_pretrained(tmp_path / "hub")
    _assert_same_weights(model, from_hub)
    reloaded = NNModel.from_checkpoint(NNCheckpoint.load(run.id, Checkpoints.LAST))
    assert reloaded.net.lm_head.weight is reloaded.net.tok_embed.weight
    _assert_same_weights(model, reloaded)
    ff = _feed_fwd()
    ff.train(params=_params(_ff_batches()), compile=FAST)
    ff.to_onnx(str(tmp_path / "ff.onnx"), _ff_batches()[0][0])
    assert (tmp_path / "ff.onnx").stat().st_size > 0


def test_inference_after_a_compiled_fit_is_eager():
    from nnx.tasks import TaskSpec

    model = _feed_fwd(task=TaskSpec.categorical(3))
    model.train(params=_params(_ff_batches()), compile=FAST)
    model.net.train()
    X, y = _ff_batches()[0]
    model.predict_proba(X)
    model.evaluate([(X, y)])
    assert model.net.training and model._compile_session is None


# ---------------- scope, refused before any run is reserved ----------------


def _refused(model: NNModel, error, match, **train_kwargs) -> None:
    with pytest.raises(error, match=match):
        model.train(params=_params(_ff_batches()), **train_kwargs)
    assert not os.path.exists("runs") or not [name for name in os.listdir("runs") if not name.startswith(".")]


def test_a_topology_changing_callback_is_refused_before_the_first_batch():
    pytest.importorskip("torchao")
    from nnx.quantize import QATLifecycleCallback

    model = NNModel(
        net_params=NNParams(
            input_dim=64, output_dim=3, hidden_dims=[64], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    spy = SessionSpy()
    with pytest.raises(ValueError, match="topology"):
        model.train(
            params=_params([(torch.randn(4, 64), torch.randint(0, 3, (4,)))]),
            callbacks=[QATLifecycleCallback(), spy],
            compile=FAST,
        )
    assert spy.sessions == []
    assert not os.path.exists("runs") or not [name for name in os.listdir("runs") if not name.startswith(".")]


def test_out_of_scope_requests_are_refused_up_front():
    from nnx.nn.nn_model import default_train_step

    _refused(_feed_fwd(), TypeError, "CompileSpec", compile={"backend": "inductor"})
    _refused(
        _feed_fwd(), ValueError, "built-in train step", compile=FAST, train_step_fn=lambda ctx: default_train_step(ctx)
    )
    _refused(_feed_fwd(), ValueError, "not a torch.compile backend", compile=CompileSpec(backend="no-such-backend"))
    _refused(_feed_fwd(precision=PrecisionPolicy("bf16")), ValueError, "fp32", compile=FAST)
    custom = NNModel(
        module=torch.nn.Linear(4, 3),
        params=NNModelParams(device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    _refused(custom, ValueError, "built-in non-graph nets", compile=FAST)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"backend": ""}, ValueError),
        ({"mode": "fastest"}, ValueError),
        ({"fullgraph": 1}, TypeError),
        ({"dynamic": "yes"}, TypeError),
        ({"on_failure": "retry"}, ValueError),
    ],
)
def test_compile_spec_validates_its_fields(kwargs, error):
    with pytest.raises(error):
        CompileSpec(**kwargs)


def test_compile_spec_state_omits_defaults_and_round_trips():
    assert CompileSpec().state() == {}
    assert "backend" not in CompileSpec().state() and "on_failure" not in CompileSpec().state()
    spec = CompileSpec(mode="max-autotune", fullgraph=True, dynamic=False, on_failure="eager")
    assert CompileSpec.from_state(spec.state()) == spec
    with pytest.raises(ValueError, match="unknown"):
        CompileSpec.from_state({"backend": "inductor", "speed": "max"})


# ---------------- honest records under dynamo's caches and limits ----------------


def test_a_cached_compile_reports_unknown_capture_not_full():
    """No dynamo reset between fits: the second fit is served from the cache."""
    first, second = _feed_fwd(), _feed_fwd()
    _break_graph(first)
    _break_graph(second)
    assert first.train(params=_params(_ff_batches()), compile=FAST).compile.capture == "partial"
    record = second.train(params=_params(_ff_batches()), compile=FAST).compile
    assert record.effective == "compiled" and record.capture in {None, "partial"} and record.capture != "full"


def _register_failing_backward_backend() -> str:
    from torch._dynamo import register_backend
    from torch._dynamo.backends.common import aot_autograd

    name = "nnx_test_failing_backward"
    if name not in torch._dynamo.list_backends(exclude_tags=()):

        def forward_compiler(graph, example_inputs):
            return graph.forward

        def backward_compiler(graph, example_inputs):
            raise RuntimeError("injected backward compile failure")

        register_backend(
            name=name, compiler_fn=aot_autograd(fw_compiler=forward_compiler, bw_compiler=backward_compiler)
        )
    return name


@pytest.mark.parametrize("policy", ["error", "eager"])
def test_a_backward_compile_failure_is_recorded_failed_and_never_retried(policy, attempts):
    backend = _register_failing_backward_backend()
    model = _feed_fwd()
    params = _params(_ff_batches())
    with pytest.raises(Exception, match="injected backward compile failure"):
        model.train(
            params=params,
            compile=CompileSpec(backend=backend, on_failure=policy),
            provenance=ExperimentManifest.for_model(model, train=params),
        )
    attempt = attempts[-1]
    assert attempt["status"] == "failed" and attempt["compile"]["effective"] == "failed"
    assert attempt["compile"]["restart"]["stage"] == "backward"
    _assert_same_weights(_feed_fwd(), model)  # the update never happened


@pytest.mark.parametrize("policy", ["error", "eager"])
def test_the_recompile_limit_is_a_failure_not_a_silent_eager_fallback(policy, monkeypatch):
    if not hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
        # Older torch (the 2.4 floor) cannot report the limit: the documented
        # limitation is that NNx then adds no strictness of its own.
        import contextlib

        from nnx.compilation import _strict_recompiles

        assert isinstance(_strict_recompiles(3), contextlib.nullcontext)
        return
    limit = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    monkeypatch.setattr(torch._dynamo.config, limit, 2)
    generator = torch.Generator().manual_seed(0)
    batches = [
        (torch.randn(n, 4, generator=generator), torch.randint(0, 3, (n,), generator=generator))
        for n in (2, 3, 4, 5, 6)
    ]
    spec = CompileSpec(backend="aot_eager", dynamic=False, on_failure=policy)
    if policy == "error":
        with pytest.raises(CompileFailed):
            _feed_fwd().train(params=_params(batches), compile=spec)
    else:
        run = _feed_fwd().train(params=_params(batches), compile=spec)
        assert run.compile.effective == "eager" and run.compile.restart["stage"] == "forward"


def test_a_nested_fit_restores_the_outer_session():
    class NestedFit(Callback):
        def __init__(self) -> None:
            self.inner_session = "unset"

        def on_epoch_end(self, ctx) -> None:
            if ctx.epoch == 0:
                inner = _feed_fwd()
                inner.train(params=_params(_ff_batches()))
                ctx.model.train(params=_params(_ff_batches(), n_epochs=1))  # same model, eager
                self.inner_session = ctx.model._compile_session

    model, nested, spy = _feed_fwd(), NestedFit(), SessionSpy()
    run = model.train(params=_params(_ff_batches(), n_epochs=2), callbacks=[nested, spy], compile=FAST)
    assert isinstance(nested.inner_session, _CompileSession) and spy.sessions[1] is spy.sessions[0]
    assert run.compile is not None and NNRun.load(run.id).compile.effective == "compiled"


def test_an_eager_restart_replays_no_forward_side_effect():
    """The retry starts from the RNG state and buffers the compiled call began with."""
    counted, reference = _feed_fwd(), _feed_fwd()
    for model in (counted, reference):
        model.net.register_buffer("calls", torch.zeros(()))

        def bump(module, args):
            module.calls += 1
            torch.rand(1)  # an RNG draw

        model.net.register_forward_pre_hook(bump)
    _break_graph(counted)
    counted.train(
        params=_params(_ff_batches()),
        compile=CompileSpec(backend="aot_eager", fullgraph=True, on_failure="eager"),
    )
    reference.train(params=_params(_ff_batches()))
    assert counted.net.calls.item() == reference.net.calls.item() == 1


def test_a_bug_in_the_forward_is_the_users_error_not_a_compile_failure():
    model = _feed_fwd()
    wrong = [(torch.randn(4, 5), torch.randint(0, 3, (4,)))]  # 5 features into a 4-input net
    for policy in ("error", "eager"):
        torch._dynamo.reset()
        with pytest.raises(RuntimeError) as caught:
            model.train(params=_params(wrong), compile=CompileSpec(backend="aot_eager", on_failure=policy))
        assert not isinstance(caught.value, CompileFailed)
        assert "shapes cannot be multiplied" in str(caught.value) or "mat1" in str(caught.value)


def test_mode_is_for_inductor_only():
    with pytest.raises(ValueError, match="inductor"):
        CompileSpec(backend="aot_eager", mode="max-autotune")
    assert CompileSpec(mode="max-autotune").mode == "max-autotune"


def test_compile_record_hashes():
    record = _feed_fwd().train(params=_params(_ff_batches()), compile=FAST).compile
    assert hash(record) == hash(CompileRecord.from_record(record.record()))


def test_many_compiled_fits_of_one_class_in_one_process_keep_working():
    """The recompile limit is process-wide per code object: each fit gets it afresh."""
    val = _ff_batches()
    for depth in range(1, 12):
        torch.manual_seed(0)
        model = NNModel(
            net_params=NNParams(
                input_dim=4, output_dim=3, hidden_dims=[8] * depth, dropout_prob=0.0, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )
        run = model.train(params=_params(_ff_batches(), val_loader=val), compile=FAST)
        assert run.compile.effective == "compiled", depth


def _register_second_backward_failure_backend() -> str:
    from torch._dynamo import register_backend
    from torch._dynamo.backends.common import aot_autograd

    name = "nnx_test_second_backward_fails"
    if name not in torch._dynamo.list_backends(exclude_tags=()):
        compiled_backwards = []

        def forward_compiler(graph, example_inputs):
            return graph.forward

        def backward_compiler(graph, example_inputs):
            compiled_backwards.append(1)
            if len(compiled_backwards) == 2:
                raise RuntimeError("second backward compile exploded")
            return graph.forward

        register_backend(
            name=name, compiler_fn=aot_autograd(fw_compiler=forward_compiler, bw_compiler=backward_compiler)
        )
    return name


def test_a_backward_compile_failure_after_a_recompile_is_recorded(attempts):
    backend = _register_second_backward_failure_backend()
    generator = torch.Generator().manual_seed(0)
    batches = [(torch.randn(n, 4, generator=generator), torch.randint(0, 3, (n,), generator=generator)) for n in (8, 5)]
    model = _feed_fwd()
    params = _params(batches)
    with pytest.raises(RuntimeError, match="second backward compile exploded"):
        model.train(
            params=params,
            compile=CompileSpec(backend=backend, dynamic=False, on_failure="eager"),
            provenance=ExperimentManifest.for_model(model, train=params),
        )
    record = attempts[-1]["compile"]
    assert attempts[-1]["status"] == "failed" and record["effective"] == "failed"
    assert (record["restart"]["stage"], record["restart"]["batch"]) == ("backward", 1)


def test_a_compile_failure_under_the_error_policy_leaves_buffers_as_found():
    model = _feed_fwd()
    model.net.register_buffer("calls", torch.zeros(()))

    def bump(module, args):
        module.calls += 1

    model.net.register_forward_pre_hook(bump)
    _break_graph(model)
    with pytest.raises(CompileFailed):
        model.train(params=_params(_ff_batches()), compile=CompileSpec(backend="aot_eager", fullgraph=True))
    assert model.net.calls.item() == 0
