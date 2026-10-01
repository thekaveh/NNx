"""FEAT-028: executing a precision policy.

BF16 autocasts the forward without a scaler, FP16 keeps the scaler, the
backward runs outside autocast, parameters stay float32 (no
``model.half()``), and the update keeps its unscale → normalize →
finite-check → clip → step order. Seeded classification and KD fixtures
with accumulation stay within the predeclared FP32-reference tolerances on
the cells this host can run; absent cells report ``"unverified"``.
"""

from __future__ import annotations

import contextlib
import copy
import os

import numpy as np
import pytest
import torch

import nnx.nn.nn_model as nn_model_module
from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainerParams,
    NNTrainParams,
    Optims,
    PrecisionPolicy,
    PrecisionUnsupportedError,
    ResolvedPrecision,
    Trainer,
    kd_objective,
    precision_support,
    supervised_objective,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.nn_model import GradientAccumulationState, TrainStepContext, default_train_step
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.precision import REFERENCE_TOLERANCES
from nnx.prediction import ProbabilitySpec

_NET = NNParams(input_dim=4, output_dim=3, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU)
_CATEGORICAL = ProbabilitySpec("categorical")
_PLATEAU = NNSchedulerParams(min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _model(precision=None, seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, precision=precision)
    return NNModel(net_params=_NET, params=params)


def _batches(n: int = 6, size: int = 8, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(size, 4, generator=generator), torch.randint(0, 3, (size,), generator=generator)) for _ in range(n)
    ]


def _train_params(**fields) -> NNTrainParams:
    fields.setdefault("overwrite_existing", True)
    return NNTrainParams(
        n_epochs=fields.pop("n_epochs", 2),
        train_loader=fields.pop("train_loader", _batches()),
        optim=NNOptimParams(
            name=Optims.SGD,
            max_lr=fields.pop("lr", 0.05),
            momentum=0.0,
            weight_decay=0.0,
            accumulate_grad_batches=fields.pop("accumulate", 1),
            grad_clip_norm=fields.pop("grad_clip_norm", None),
        ),
        scheduler=_PLATEAU,
        save_phase_checkpoints=False,
        **fields,
    )


class _ForwardDtypes:
    """Records each forward's output dtype and whether autocast was on."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.seen: list[tuple[torch.dtype, bool]] = []
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        self.seen.append((output.dtype, torch.is_autocast_enabled("cpu")))


# --- BF16 / FP16 execution ---------------------------------------------------------------------------------


def test_bf16_autocasts_the_forward_scaler_free_and_backward_outside_autocast():
    model = _model(PrecisionPolicy("bf16"))
    dtypes = _ForwardDtypes(model.net)
    backward_autocast: list[bool] = []
    for parameter in model.net.parameters():
        parameter.register_hook(lambda grad: backward_autocast.append(torch.is_autocast_enabled("cpu")) or grad)
    contexts: list[TrainStepContext] = []

    def recording_step(ctx):
        contexts.append(ctx)
        return default_train_step(ctx)

    model.train(_train_params(n_epochs=1), train_step_fn=recording_step)
    assert dtypes.seen and all(dtype is torch.bfloat16 and on for dtype, on in dtypes.seen[: len(contexts)])
    assert backward_autocast and not any(backward_autocast)  # the backward runs outside autocast
    assert all(ctx.scaler is None and ctx.precision is not None for ctx in contexts)
    assert contexts[0].precision is not None and contexts[0].precision.effective == "bf16"
    assert all(parameter.dtype is torch.float32 for parameter in model.net.parameters())  # never model.half()


class _RecordingScaler:
    """A CPU stand-in for GradScaler recording the AMP protocol."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def scale(self, tensor):
        self.calls.append("scale")
        return tensor * 4.0

    def unscale_(self, optimizer):
        self.calls.append("unscale")
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad /= 4.0

    def step(self, optimizer):
        self.calls.append("step")
        optimizer.step()

    def update(self):
        self.calls.append("update")


def _record_update_order(monkeypatch, calls: list[str]) -> None:
    real_scale, real_clip = nn_model_module._scale_gradients, torch.nn.utils.clip_grad_norm_
    real_finite = nn_model_module._check_finite_gradients

    def scale_gradients(module, factor):
        calls.append("normalize")
        return real_scale(module, factor)

    def finite(module):
        calls.append("finite")
        return real_finite(module)

    def clip(parameters, norm, *args, **kwargs):
        calls.append("clip")
        return real_clip(parameters, norm, *args, **kwargs)

    monkeypatch.setattr(nn_model_module, "_scale_gradients", scale_gradients)
    monkeypatch.setattr(nn_model_module, "_check_finite_gradients", finite)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)


def _step_window(model, precision, scaler, *, accumulate=2, n=2):
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.05)
    state = GradientAccumulationState()
    for idx, batch in enumerate(_batches(n)):
        default_train_step(
            TrainStepContext(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=1.0,
                extra_metrics=None,
                accumulate_grad_batches=accumulate,
                batch_idx=idx,
                epoch_idx=0,
                is_last_batch=idx == n - 1,
                accumulation_state=state,
                precision=precision,
            )
        )


def test_fp16_keeps_the_scaler_and_the_update_order(monkeypatch):
    # CPU simulation of the CUDA fp16 cell: a fp16 resolution with a recording scaler, and a recording
    # autocast (the context, not the kernels, is what the default step owns).
    calls: list[str] = []
    entered: list[tuple[str, torch.dtype]] = []

    @contextlib.contextmanager
    def autocast(device_type, dtype):
        entered.append((device_type, dtype))
        yield

    monkeypatch.setattr(torch, "autocast", autocast)
    _record_update_order(monkeypatch, calls)
    fp16 = ResolvedPrecision(requested="fp16", effective="fp16", device_type="cpu", source="policy")
    _step_window(_model(), fp16, _RecordingScaler(calls))
    assert entered == [("cpu", torch.float16)] * 2  # every forward, under the policy's autocast
    update = [call for call in calls if call != "scale"]
    assert update == ["unscale", "normalize", "clip", "step", "update"]  # no finite check: the scaler skips


def test_bf16_checks_finite_gradients_between_normalize_and_clip(monkeypatch):
    calls: list[str] = []
    _record_update_order(monkeypatch, calls)
    bf16 = PrecisionPolicy("bf16").resolve("cpu")
    _step_window(_model(), bf16, None)
    assert calls == ["normalize", "finite", "clip"]


def test_bf16_never_applies_a_non_finite_gradient():
    model = _model(PrecisionPolicy("bf16"))
    before = copy.deepcopy(model.net.state_dict())
    next(model.net.parameters()).register_hook(lambda grad: grad * float("inf"))
    with pytest.raises(FloatingPointError, match="non-finite gradient .* nothing was stepped"):
        _step_window(model, model.resolved_precision, None, accumulate=1, n=1)
    after = model.net.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)


# --- reference fixtures within the predeclared tolerances --------------------------------------------------


def _classification_fixture(precision, device=Devices.CPU):
    model = _model(precision)
    if device is not Devices.CPU:
        model = NNModel(
            net_params=_NET,
            params=NNModelParams(net=Nets.FEED_FWD, device=device, loss=Losses.CROSS_ENTROPY, precision=precision),
        )
    run = model.train(_train_params(n_epochs=2, accumulate=2, lr=0.02, data_id=f"cls-{precision}"))
    return [idp.train_edp.loss for idp in run.idps if idp.train_edp is not None], model


def _kd_fixture(precision):
    teacher = _model(seed=7)
    student = _model(precision, seed=3)
    run = student.train(
        _train_params(n_epochs=2, accumulate=2, lr=0.02, data_id=f"kd-{precision}"),
        objective=kd_objective(teacher, alpha=0.5, temperature=2.0),
    )
    return [idp.train_edp.loss for idp in run.idps if idp.train_edp is not None], student


def _within(reference, candidate, tolerance) -> None:
    (ref_losses, ref_model), (losses, model) = reference, candidate
    assert np.all(np.isfinite(losses))
    assert np.max(np.abs(np.asarray(ref_losses) - np.asarray(losses))) <= tolerance.loss
    weight_gap = max(
        float((a - b).abs().max()) for a, b in zip(ref_model.net.parameters(), model.net.parameters(), strict=True)
    )
    assert weight_gap <= tolerance.weights


@pytest.mark.parametrize("fixture", [_classification_fixture, _kd_fixture])
def test_bf16_stays_within_its_fp32_reference_tolerance_on_cpu(fixture):
    assert precision_support("cpu")["cpu"]["bf16"] == "verified"
    _within(fixture(None), fixture(PrecisionPolicy("bf16")), REFERENCE_TOLERANCES["bf16"])


@pytest.mark.parametrize("mode", ["fp16", "bf16"])
def test_cuda_cells_are_verified_only_on_cuda_hardware(mode):
    if not torch.cuda.is_available():
        # No hardware here: the cell is reported unverified — never claimed as support.
        assert precision_support("cuda")["cuda"][mode] == "unverified"
        return
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        assert precision_support("cuda")["cuda"][mode] == "unsupported"
        return
    _within(
        _classification_fixture(None, Devices.CUDA),
        _classification_fixture(PrecisionPolicy(mode), Devices.CUDA),
        REFERENCE_TOLERANCES[mode],
    )


# --- one policy, reused in evaluation and prediction -------------------------------------------------------


def test_evaluation_and_prediction_reuse_the_policy_and_keep_their_schema():
    model = _model(PrecisionPolicy("bf16"))
    dtypes = _ForwardDtypes(model.net)
    params = _train_params(n_epochs=1, val_loader=_batches(2, seed=5))
    run = model.train(params)
    assert run.idps is not None and run.idps[-1].val_edp is not None  # validation ran
    assert all(dtype is torch.bfloat16 for dtype, _ in dtypes.seen)  # training and validation forwards
    X = torch.randn(5, 4)
    probabilities = model.predict_proba(X, _CATEGORICAL).probabilities
    result = model.predict(X)
    assert dtypes.seen[-1] == (torch.bfloat16, True)
    assert probabilities is not None and result.logits.dtype == np.float32  # a safe CPU dtype
    full = _model().predict_proba(X, _CATEGORICAL).probabilities
    assert full is not None and full.dtype == probabilities.dtype  # the same schema as full precision


def test_the_legacy_flag_never_changes_evaluation_or_prediction():
    legacy = NNModel(
        net_params=_NET,
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, mixed_precision=True),
    )
    dtypes = _ForwardDtypes(legacy.net)
    legacy.predict(torch.randn(3, 4))
    assert dtypes.seen == [(torch.float32, False)]


def test_a_trainer_objective_runs_in_the_policy():
    model = _model(PrecisionPolicy("bf16"))
    dtypes = _ForwardDtypes(model.net)
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_batches(4),
        val_loader=_batches(2, seed=5),
        optims={"main": NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0)},
        save_phase_checkpoints=False,
    )
    run = Trainer(model).train(params, objective=supervised_objective())
    assert run.precision is not None and run.precision.effective == "bf16"
    assert dtypes.seen and all(dtype is torch.bfloat16 for dtype, _ in dtypes.seen)


def test_a_trainer_step_function_refuses_a_reduced_policy_before_any_run(tmp_path):
    def step(ctx):
        return NNEvaluationDataPoint(loss=0.0)

    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_batches(2),
        optims={"main": NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0)},
        save_phase_checkpoints=False,
    )
    with pytest.raises(PrecisionUnsupportedError, match="Trainer step functions run in full precision"):
        Trainer(_model(PrecisionPolicy("bf16"))).train(params, trainer_step_fn=step)
    assert not os.path.exists(tmp_path / "runs")


# --- a reloaded model re-resolves on its destination device ------------------------------------------------


def test_from_checkpoint_re_resolves_on_the_destination_device():
    model = _model(PrecisionPolicy("bf16"))
    run = model.train(_train_params(n_epochs=1))
    checkpoint = NNCheckpoint.load(run.id, Checkpoints.LAST)
    assert checkpoint is not None
    reloaded = NNModel.from_checkpoint(checkpoint, device=Devices.CPU)
    assert reloaded.resolved_precision.effective == "bf16"
    assert reloaded.predict(torch.randn(2, 4)).logits.dtype == np.float32  # bf16 converts to a safe CPU dtype
    # The saved metadata says CUDA fp16; the destination is CPU, where fp16 cannot run.
    cuda_fp16 = copy.copy(checkpoint)
    object.__setattr__(
        cuda_fp16,
        "model_params",
        NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CUDA, loss=Losses.CROSS_ENTROPY, precision=PrecisionPolicy("fp16")
        ),
    )
    with pytest.raises(PrecisionUnsupportedError, match="CUDA only"):
        NNModel.from_checkpoint(cuda_fp16, device=Devices.CPU)
    overridden = NNModel.from_checkpoint(cuda_fp16, device=Devices.CPU, precision=PrecisionPolicy("bf16"))
    assert overridden.resolved_precision.effective == "bf16" and overridden.params.mixed_precision is False


# --- review round 3 ----------------------------------------------------------------------------------------


class _PerOptimizerScaler:
    """Like GradScaler: one optimizer's gradients overflow at unscale_, step() skips only that optimizer, and
    update() backs the scale off."""

    def __init__(self, poisoned: torch.optim.Optimizer) -> None:
        self.poisoned, self.value, self.seen_inf = poisoned, 4.0, False

    def scale(self, tensor):
        return tensor * self.value

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad /= self.value
                    if optimizer is self.poisoned:
                        parameter.grad.fill_(float("inf"))
                        self.seen_inf = True

    def step(self, optimizer):
        if optimizer is not self.poisoned:
            optimizer.step()

    def update(self):
        if self.seen_inf:
            self.value /= 2.0

    def get_scale(self):
        return self.value


def test_a_multi_optimizer_fp16_window_is_all_or_nothing():
    from nnx._update_engine import UpdateEngine
    from nnx.nn.nn_model import _objective_microbatch

    model = _model()
    first, second = model.net.layers[0], model.net.layers[-1]
    optimizers = {
        "encoder": torch.optim.SGD(first.parameters(), lr=0.1),
        "head": torch.optim.SGD([p for n, p in model.net.named_parameters() if not n.startswith("layers.0")], lr=0.1),
    }
    before = {key: value.clone() for key, value in model.net.state_dict().items()}
    events: list = []
    engine = UpdateEngine(
        optimizers=optimizers,
        scaler=_PerOptimizerScaler(optimizers["encoder"]),
        clip_norms={},
        listeners=[events.append],
    )
    _objective_microbatch(
        engine,
        supervised_objective(),
        model=model,
        batch=_batches(1)[0],
        epoch_idx=0,
        batch_idx=0,
        extra_metrics=None,
        close_window=True,
    )
    after = model.net.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)  # the head did not step alone
    assert events == [] and engine.skipped == 1 and engine.commits == 0
    assert second.weight.grad is None  # the window was released


def _two_optimizer_engine(model, optimizers, scaler):
    from nnx._update_engine import UpdateEngine

    events: list = []
    engine = UpdateEngine(optimizers=optimizers, scaler=scaler, clip_norms={}, listeners=[events.append])
    return engine, events


class _CheckingScaler(_PerOptimizerScaler):
    """GradScaler's protocol rules: step() needs a prior unscale_() of that optimizer."""

    def __init__(self) -> None:
        super().__init__(poisoned=None)  # type: ignore[arg-type]
        self.unscaled: set[int] = set()

    def unscale_(self, optimizer):
        # Like GradScaler: inf checks are recorded only for gradients that exist.
        if any(p.grad is not None for group in optimizer.param_groups for p in group["params"]):
            self.unscaled.add(id(optimizer))
        super().unscale_(optimizer)

    def step(self, optimizer):
        assert id(optimizer) in self.unscaled, "No inf checks were recorded for this optimizer."
        optimizer.step()


def test_an_fp16_window_skips_the_scaler_for_an_optimizer_without_gradients():
    from nnx.nn.nn_model import _objective_microbatch

    model = _model()
    first = model.net.layers[0]
    for parameter in first.parameters():
        parameter.requires_grad_(False)  # frozen during warm-up: no gradient this window
    optimizers = {
        "backbone": torch.optim.SGD(first.parameters(), lr=0.1),
        "head": torch.optim.SGD([p for n, p in model.net.named_parameters() if not n.startswith("layers.0")], lr=0.1),
    }
    engine, events = _two_optimizer_engine(model, optimizers, _CheckingScaler())
    _objective_microbatch(
        engine,
        supervised_objective(),
        model=model,
        batch=_batches(1)[0],
        epoch_idx=0,
        batch_idx=0,
        extra_metrics=None,
        close_window=True,
    )
    assert {event.optimizer for event in events} == {"backbone", "head"} and engine.commits == 1


def test_fp16_refuses_optimizers_sharing_a_parameter():
    from nnx._update_engine import UpdateEngine

    model = _model()
    shared = list(model.net.parameters())
    with pytest.raises(ValueError, match="share a parameter.*unscale twice"):
        UpdateEngine(
            optimizers={"a": torch.optim.SGD(shared, lr=0.1), "b": torch.optim.SGD(shared[:1], lr=0.1)},
            scaler=_CheckingScaler(),
        )


# --- review round 5 ------------------------------------------------------------------------------------------


def _multilabel_model(precision) -> NNModel:
    from nnx import TaskSpec

    torch.manual_seed(0)
    params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.BINARY_CROSS_ENTROPY,
        task=TaskSpec.multilabel(3),
        precision=precision,
    )
    return NNModel(net_params=_NET, params=params)


def _multilabel_batches(n: int = 4, size: int = 8, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(size, 4, generator=generator), torch.randint(0, 2, (size, 3), generator=generator))
        for _ in range(n)
    ]


@pytest.mark.parametrize("use_objective", [False, True], ids=["default-step", "objective"])
def test_bf16_task_steps_read_targets_in_full_precision(use_objective):
    # A multilabel task casts its integer targets to the output's dtype: the
    # output must reach it in float32 (NumPy has no bfloat16).
    model = _multilabel_model(PrecisionPolicy("bf16"))
    dtypes = _ForwardDtypes(model.net)
    run = model.train(
        _train_params(n_epochs=1, train_loader=_multilabel_batches()),
        objective=supervised_objective() if use_objective else None,
    )
    assert dtypes.seen and dtypes.seen[0] == (torch.bfloat16, True)
    assert all(idp.train_edp.kind == "multilabel" and np.isfinite(idp.train_edp.loss) for idp in run.idps)


def test_the_back_compat_train_step_applies_an_explicit_policy():
    model = _model(PrecisionPolicy("bf16"))
    dtypes = _ForwardDtypes(model.net)
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.05)
    model._train_step(_batches(1)[0], optimizer, None)
    assert dtypes.seen == [(torch.bfloat16, True)]
    legacy = _model()
    dtypes = _ForwardDtypes(legacy.net)
    legacy._train_step(_batches(1)[0], torch.optim.SGD(legacy.net.parameters(), lr=0.05), None)
    assert dtypes.seen == [(torch.float32, False)]


def test_a_refused_trainer_step_function_leaves_the_global_rng_untouched():
    def step(ctx):
        return NNEvaluationDataPoint(loss=0.0)

    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_batches(2),
        optims={"main": NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0)},
        save_phase_checkpoints=False,
        seed=7,
    )
    trainer = Trainer(_model(PrecisionPolicy("bf16")))
    before = torch.get_rng_state()
    with pytest.raises(PrecisionUnsupportedError):
        trainer.train(params, trainer_step_fn=step)
    assert torch.equal(torch.get_rng_state(), before)  # refused before set_seed


# --- review round 6 ------------------------------------------------------------------------------------------


def test_an_fp16_step_without_a_scaler_is_refused_before_the_forward():
    model = _model()
    dtypes = _ForwardDtypes(model.net)
    before = copy.deepcopy(model.net.state_dict())
    fp16 = ResolvedPrecision(requested="fp16", effective="fp16", device_type="cpu", source="policy")
    ctx = TrainStepContext(
        model=model,
        batch=_batches(1)[0],
        optimizer=torch.optim.SGD(model.net.parameters(), lr=0.1),
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
        precision=fp16,
    )
    with pytest.raises(ValueError, match="fp16 trains through a GradScaler"):
        default_train_step(ctx)
    assert dtypes.seen == []
    assert all(torch.equal(value, model.net.state_dict()[key]) for key, value in before.items())


def test_the_record_claims_training_only_where_nnx_applies_the_policy():
    from nnx.precision import EVALUATE, PREDICT, TRAIN

    def custom_step(ctx):
        return default_train_step(ctx)  # it may apply ctx.precision, but NNx cannot vouch for it

    custom = _model(PrecisionPolicy("bf16")).train(_train_params(n_epochs=1), train_step_fn=custom_step)
    assert custom.precision is not None and custom.precision.covers == (EVALUATE, PREDICT)
    default = _model(PrecisionPolicy("bf16")).train(_train_params(n_epochs=1))
    assert default.precision is not None and default.precision.covers == (TRAIN, EVALUATE, PREDICT)


@pytest.mark.parametrize(("student_mode", "teacher_mode"), [("bf16", None), (None, "bf16")])
def test_a_kd_teacher_runs_in_its_own_precision(student_mode, teacher_mode):
    student = _model(PrecisionPolicy(student_mode) if student_mode else None)
    teacher = _model(PrecisionPolicy(teacher_mode) if teacher_mode else None, seed=3)
    teacher_dtypes = _ForwardDtypes(teacher.net)
    student.train(_train_params(n_epochs=1, train_loader=_batches(2)), objective=kd_objective(teacher))
    expected = (torch.bfloat16, True) if teacher_mode == "bf16" else (torch.float32, False)
    assert teacher_dtypes.seen and set(teacher_dtypes.seen) == {expected}


def test_a_cuda_ordinal_this_host_lacks_is_unsupported(monkeypatch):
    from nnx import precision as precision_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    resolved = PrecisionPolicy("bf16", fallback="fp32").resolve("cuda:3")
    assert resolved.effective == "fp32" and "cuda:3 does not exist" in str(resolved.fallback_reason)
    with pytest.raises(PrecisionUnsupportedError, match="cuda:3 does not exist"):
        PrecisionPolicy("fp16").resolve("cuda:3")
    assert set(precision_module.precision_support("cuda:3")["cuda"].values()) == {"unsupported"}


# --- review round 7 ------------------------------------------------------------------------------------------


def test_a_teacher_keeps_its_own_policy_beside_a_legacy_student_on_cpu():
    legacy_student = NNModel(
        net_params=_NET,
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, mixed_precision=True),
    )
    teacher = _model(PrecisionPolicy("bf16"), seed=3)
    teacher_dtypes = _ForwardDtypes(teacher.net)
    legacy_student.train(_train_params(n_epochs=1, train_loader=_batches(2)), objective=kd_objective(teacher))
    assert teacher_dtypes.seen and set(teacher_dtypes.seen) == {(torch.bfloat16, True)}


def test_an_fp16_objective_engine_needs_a_scaler():
    fp16 = ResolvedPrecision(requested="fp16", effective="fp16", device_type="cpu", source="policy")
    model = _model()
    with pytest.raises(ValueError, match="fp16 trains through a GradScaler"):
        nn_model_module._objective_engine(
            supervised_objective(),
            optimizers={"default": torch.optim.SGD(model.net.parameters(), lr=0.1)},
            clip_norms={},
            scaler=None,
            precision=fp16,
        )


def test_a_scaler_hook_disagreeing_with_the_precision_is_refused(monkeypatch):
    from nnx import precision as precision_module

    fp32 = ResolvedPrecision(requested="fp32", effective="fp32", device_type="cuda")
    with pytest.raises(ValueError, match="resolves to fp32: AMP is decided by NNModelParams.precision"):
        nn_model_module._check_scaler_hook(fp32, _RecordingScaler([]), "cuda")
    nn_model_module._check_scaler_hook(fp32, _RecordingScaler([]), "cpu")  # never switched anything on off CUDA

    real = precision_module._unsupported
    monkeypatch.setattr(
        precision_module, "_unsupported", lambda mode, *device: None if mode == "fp16" else real(mode, *device)
    )
    monkeypatch.setattr(NNModel, "_build_grad_scaler", lambda _self: None)
    model = _model(PrecisionPolicy("fp16"))
    before = copy.deepcopy(model.net.state_dict())
    with pytest.raises(ValueError, match="_build_grad_scaler returned none"):
        model.train(_train_params(n_epochs=1))
    assert all(torch.equal(model.net.state_dict()[k], v) for k, v in before.items())


def test_a_trainer_builds_its_fp16_scaler_through_the_model_hook(monkeypatch):
    from nnx import precision as precision_module

    real = precision_module._unsupported
    monkeypatch.setattr(
        precision_module, "_unsupported", lambda mode, *device: None if mode == "fp16" else real(mode, *device)
    )

    class Built(Exception):
        pass

    def hook(_self):
        raise Built

    monkeypatch.setattr(NNModel, "_build_grad_scaler", hook)
    params = NNTrainerParams(
        n_epochs=1,
        train_loader=_batches(2),
        optims={"main": NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0)},
        save_phase_checkpoints=False,
    )
    with pytest.raises(Built):
        Trainer(_model(PrecisionPolicy("fp16"))).train(params, objective=supervised_objective())


def test_the_legacy_flag_beside_an_fp16_policy_keeps_the_policys_run_id():
    def params(**fields):
        return NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, **fields)

    both = params(mixed_precision=True, precision=PrecisionPolicy("fp16"))
    assert not both.mixed_precision
    assert both.state() == params(precision=PrecisionPolicy("fp16")).state()
