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
