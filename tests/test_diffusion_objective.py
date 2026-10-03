"""FEAT-040: DDPM noise prediction as an objective.

The adapter returns a differentiable squared-error sum over a valid-element
denominator and detached metrics, and never zeroes, back-propagates, scales
or steps; uneven microbatches with fixed timesteps and noise take the
full-batch update; the engine's clipping and non-finite policy apply; the
loss reaches records, events, monitors and BEST without classification
fields; and a ``sample`` preview never perturbs the objective's generator.
"""

from __future__ import annotations

import copy
import os

import pytest
import torch

from nnx import (
    Activations,
    Callback,
    Checkpoints,
    Devices,
    DiffusionMLP,
    DiffusionObjective,
    Losses,
    MonitorSpec,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainerParams,
    NNTrainParams,
    NoiseSchedulers,
    ObjectiveContext,
    Optims,
    Trainer,
    diffusion_objective,
    diffusion_train_step_factory,
    sample,
    set_seed,
)

SCHEDULE = NoiseSchedulers.LINEAR(T=50)
_PLATEAU = NNSchedulerParams(min_lr=0.0, factor=0.5, patience=10, cooldown=0, threshold=0.0)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _model(seed: int = 0) -> NNModel:
    """The 2-D DiffusionMLP fixture of tests/test_diffusion_training.py."""
    set_seed(seed)
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net = DiffusionMLP(input_dim=2, hidden_dims=[32, 32], time_embed_dim=16).to(model.device)
    return model


def _points(n: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))


def _params(loader, *, n_epochs: int = 1, accumulate: int = 1, **fields) -> NNTrainParams:
    fields.setdefault("overwrite_existing", True)
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=loader,
        optim=NNOptimParams(
            name=Optims.SGD,
            max_lr=fields.pop("lr", 0.05),
            momentum=0.0,
            weight_decay=0.0,
            accumulate_grad_batches=accumulate,
            grad_clip_norm=fields.pop("grad_clip_norm", None),
        ),
        scheduler=_PLATEAU,
        save_phase_checkpoints=False,
        **fields,
    )


def _fixed_noise(n: int, seed: int = 7):
    """A noise_fn handing out fixed per-row timesteps and noise in loader
    order, however the rows are split into microbatches."""
    generator = torch.Generator().manual_seed(seed)
    t_all = torch.randint(0, SCHEDULE.T, (n,), generator=generator)
    eps_all = torch.randn(n, 2, generator=generator)
    state = {"offset": 0}

    def noise_fn(x_0, generator):
        start = state["offset"]
        state["offset"] += x_0.shape[0]
        return t_all[start : state["offset"]], eps_all[start : state["offset"]]

    return noise_fn, t_all, eps_all


# --- AC1: what the adapter returns, and what it never does ----------------------------------------------------


def test_the_adapter_returns_a_squared_error_sum_and_never_owns_the_update(monkeypatch):
    model = _model()
    x_0 = _points(3)
    noise_fn, t_all, eps_all = _fixed_noise(3)
    objective = diffusion_objective(SCHEDULE, noise_fn=noise_fn)

    def forbidden(*args, **kwargs):
        raise AssertionError("an objective never zeroes, back-propagates, scales or steps")

    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "zero_grad", forbidden)
    monkeypatch.setattr(torch.nn.Module, "zero_grad", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    torch.optim.SGD(model.net.parameters(), lr=0.1)  # an optimizer exists; nothing may touch it
    result = objective(ObjectiveContext(model=model, batch=(x_0, torch.zeros(3)), epoch_idx=0, batch_idx=0))

    (term,) = result.terms
    assert (term.name, term.reduction, term.denominator, term.weight) == ("noise_mse", "mean", 6.0, 1.0)
    assert term.numerator.requires_grad
    sqrt_a = SCHEDULE.sqrt_alphas_cumprod[t_all].unsqueeze(-1)
    sqrt_1ma = SCHEDULE.sqrt_one_minus_alphas_cumprod[t_all].unsqueeze(-1)
    with torch.no_grad():
        expected = (model.net(sqrt_a * x_0 + sqrt_1ma * eps_all, t_all) - eps_all).pow(2).sum()
    torch.testing.assert_close(term.numerator.detach(), expected)
    record = result.record
    assert record is not None and record.loss == pytest.approx(float(expected) / 6)
    assert dict(record.metrics) == {"noise_mse": record.loss}
    assert (record.f1, record.recall, record.accuracy, record.precision, record.error) == (None,) * 5
    assert all(p.grad is None for p in model.net.parameters())


def test_an_empty_batch_is_rejected_before_the_forward():
    model = _model()
    calls: list[int] = []
    model.net.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError, match="empty batch"):
        diffusion_objective(SCHEDULE)(
            ObjectiveContext(model=model, batch=(torch.zeros(0, 2), torch.zeros(0)), epoch_idx=0, batch_idx=0)
        )
    assert calls == []


def test_arguments_are_validated():
    with pytest.raises(TypeError, match="NoiseSchedule"):
        DiffusionObjective(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        diffusion_objective(SCHEDULE, seed=-1)
    with pytest.raises(ValueError, match="nonfinite"):
        diffusion_objective(SCHEDULE, nonfinite="ignore")
    bad = diffusion_objective(SCHEDULE, noise_fn=lambda x_0, g: (torch.zeros(1, dtype=torch.long), x_0))
    with pytest.raises(ValueError, match="noise_fn must return"):
        bad(ObjectiveContext(model=_model(), batch=(_points(3), torch.zeros(3)), epoch_idx=0, batch_idx=0))


# --- AC2: uneven microbatches match the full batch --------------------------------------------------------------


def test_uneven_microbatches_match_the_full_batch_reference():
    x = _points(3)
    full_model, split_model = _model(), _model()
    split_model.net.load_state_dict(full_model.net.state_dict())

    full_noise, _, _ = _fixed_noise(3)
    full_events: list = []
    full_model.train(
        _params([(x, torch.zeros(3))]),
        objective=diffusion_objective(SCHEDULE, noise_fn=full_noise),
        callbacks=[_EventRecorder(full_events)],
    )
    split_noise, _, _ = _fixed_noise(3)
    split_events: list = []
    split_model.train(
        _params([(x[:2], torch.zeros(2)), (x[2:], torch.zeros(1))], accumulate=4),  # a short [2, 1] window
        objective=diffusion_objective(SCHEDULE, noise_fn=split_noise),
        callbacks=[_EventRecorder(split_events)],
    )
    assert len(full_events) == len(split_events) == 1 and split_events[0].microbatches == 2
    assert split_events[0].losses["noise_mse"] == pytest.approx(full_events[0].losses["noise_mse"], rel=1e-6)
    for key, value in full_model.net.state_dict().items():
        torch.testing.assert_close(split_model.net.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)


class _EventRecorder(Callback):
    def __init__(self, events: list) -> None:
        self.events = events
        self.records: list = []

    def on_optimizer_update(self, ctx, event) -> None:
        self.events.append(event)

    def on_epoch_end(self, ctx) -> None:
        self.records.append(ctx.idp)


# --- AC3: the engine's clipping and non-finite policy -----------------------------------------------------------


def test_clipping_and_the_nonfinite_policy_are_the_engines():
    x = _points(4)
    clipped, unclipped = _model(), _model()
    unclipped.net.load_state_dict(clipped.net.state_dict())
    before = copy.deepcopy(clipped.net.state_dict())
    clipped.train(
        _params([(x, torch.zeros(4))], grad_clip_norm=1e-3, lr=1.0), objective=diffusion_objective(SCHEDULE, seed=0)
    )
    unclipped.train(_params([(x, torch.zeros(4))], lr=1.0), objective=diffusion_objective(SCHEDULE, seed=0))
    step = torch.sqrt(sum((clipped.net.state_dict()[k] - v).pow(2).sum() for k, v in before.items()))
    assert float(step) == pytest.approx(1e-3, rel=1e-3)  # lr 1 × a gradient clipped to norm 1e-3
    assert any(not torch.equal(unclipped.net.state_dict()[k], v) for k, v in clipped.net.state_dict().items())

    poisoned = (torch.full((2, 2), float("nan")), torch.zeros(2))
    model = _model()
    before = copy.deepcopy(model.net.state_dict())
    with pytest.raises(FloatingPointError, match="non-finite loss term 'noise_mse'"):
        model.train(_params([poisoned]), objective=diffusion_objective(SCHEDULE, seed=0))
    assert all(torch.equal(model.net.state_dict()[k], v) for k, v in before.items())
    run = model.train(_params([poisoned]), objective=diffusion_objective(SCHEDULE, seed=0, nonfinite="skip"))
    assert run.idps[-1].update_count == 0
    assert all(torch.equal(model.net.state_dict()[k], v) for k, v in before.items())


def test_the_imperative_step_and_the_objective_keep_their_own_slots():
    model = _model()
    loader = [(_points(2), torch.zeros(2))]
    with pytest.raises(ValueError, match="imperative diffusion step"):
        model.train(_params(loader), objective=diffusion_train_step_factory(SCHEDULE))
    with pytest.raises(ValueError, match="DiffusionObjective, an objective"):
        model.train(_params(loader), train_step_fn=diffusion_objective(SCHEDULE))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="imperative diffusion step"):
        Trainer(model).train(_trainer_params(loader), objective=diffusion_train_step_factory(SCHEDULE))
    assert not os.path.exists("runs")


def _trainer_params(loader, n_epochs: int = 1) -> NNTrainerParams:
    return NNTrainerParams(
        n_epochs=n_epochs,
        train_loader=loader,
        optims={"main": NNOptimParams(name=Optims.SGD, max_lr=0.05, momentum=0.0, weight_decay=0.0)},
        save_phase_checkpoints=False,
        overwrite_existing=True,
    )


def test_a_trainer_runs_the_objective_on_its_engine():
    model = _model()
    loader = [(_points(4, seed=s), torch.zeros(4)) for s in range(3)]
    run = Trainer(model).train(_trainer_params(loader), objective=diffusion_objective(SCHEDULE, seed=0))
    assert run.idps[-1].update_count == 3
    assert all(idp.train_edp.loss is not None and idp.train_edp.accuracy is None for idp in run.idps)


# --- AC6: the loss reaches records, events, monitors and BEST ---------------------------------------------------


def test_the_named_loss_reaches_records_events_monitors_and_best():
    model = _model()
    loader = [(_points(n, seed=n), torch.zeros(n)) for n in (4, 3, 1)]
    events: list = []
    recorder = _EventRecorder(events)
    noise_fn, _, _ = _fixed_noise(8)
    run = model.train(
        _params(loader, n_epochs=1, monitor=MonitorSpec(metric="loss", split="train")),
        objective=diffusion_objective(SCHEDULE, noise_fn=noise_fn),
        callbacks=[recorder],
    )
    assert [event.losses.keys() for event in events] == [{"noise_mse"}] * 3
    records = [idp.train_edp for idp in run.idps]
    assert all(
        set(record.metrics) == {"noise_mse"} and record.f1 is None and record.error is None for record in records
    )
    # The epoch's loss is Σ numerators / Σ elements, not a mean of batch means.
    expected = sum(record.loss * n * 2 for record, n in zip(records, (4, 3, 1), strict=True)) / 16
    summary = run.idps[-1].train_summary
    assert summary is not None and summary.loss == pytest.approx(expected, rel=1e-6)
    assert summary.accuracy is None and run.idps[-1].selection is not None
    assert run.idps[-1].selection.value == pytest.approx(expected, rel=1e-6)
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 0


# --- AC7: a sampling preview never perturbs the objective ------------------------------------------------------


class _Preview(Callback):
    """Samples with its own generator at the end of the first epoch."""

    def __init__(self, objective: DiffusionObjective) -> None:
        self.objective = objective
        self.before = self.after = None
        self.modes: list[bool] = []

    def on_epoch_end(self, ctx) -> None:
        if ctx.epoch != 0:
            return
        self.before = (self.objective.generator.get_state(), torch.get_rng_state())
        self.modes.append(ctx.model.net.training)
        sample(ctx.model, SCHEDULE, (8, 2), generator=torch.Generator().manual_seed(5))
        self.modes.append(ctx.model.net.training)
        self.after = (self.objective.generator.get_state(), torch.get_rng_state())


def _loader():
    return [(_points(4, seed=s), torch.zeros(4)) for s in range(2)]


def test_a_sample_preview_leaves_the_objective_rng_and_the_resumed_update_unchanged():
    reference = _model()
    reference.train(_params(_loader(), n_epochs=2, seed=3), objective=diffusion_objective(SCHEDULE))

    previewed = _model()
    objective = diffusion_objective(SCHEDULE)
    preview = _Preview(objective)
    first = previewed.train(_params(_loader(), n_epochs=1, seed=3), objective=objective, callbacks=[preview])
    assert preview.before is not None and preview.after is not None
    assert torch.equal(preview.before[0], preview.after[0]) and torch.equal(preview.before[1], preview.after[1])
    assert preview.modes == [True, True]  # sample() put the net back in train mode

    resumed = _model(seed=99)
    second = resumed.train(
        _params(_loader(), n_epochs=1, seed=3, resume_from_run_id=first.id), objective=diffusion_objective(SCHEDULE)
    )
    assert second.resume_status is not None and "diffusion.objective" in second.resume_status.restored_components
    for key, value in reference.net.state_dict().items():
        torch.testing.assert_close(resumed.net.state_dict()[key], value, msg=key)


def test_plans_apply_the_same_owner_rule():
    from nnx.plans import ExperimentPlan

    base = (
        ExperimentPlan()
        .with_net(NNParams(input_dim=2, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU))
        .with_model(NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY))
        .with_train(NNTrainParams(n_epochs=1))
        .with_data([(_points(2), torch.zeros(2))])
    )
    report = base.with_objective(diffusion_train_step_factory(SCHEDULE)).validate()
    assert report.paths == ("objective",) and "imperative diffusion step" in report.diagnostics[0].message
    report = base.with_step_fns(train_step_fn=diffusion_objective(SCHEDULE)).validate()
    assert report.paths == ("train_step_fn",) and "an objective" in report.diagnostics[0].message
    assert base.with_objective(diffusion_objective(SCHEDULE)).validate().ok


# --- review round 1 ---------------------------------------------------------------------------------------


def test_a_reused_objective_restarts_its_stream_for_each_run():
    objective = diffusion_objective(SCHEDULE)  # seeded from the (just seeded) global RNG at first use
    loader = [(_points(4, seed=s), torch.zeros(4)) for s in range(2)]
    first, second = _model(), _model()
    run_a = first.train(_params(loader, seed=0), objective=objective)
    run_b = second.train(_params(loader, seed=0), objective=objective)
    assert [idp.train_edp.loss for idp in run_a.idps] == [idp.train_edp.loss for idp in run_b.idps]
    for key, value in first.net.state_dict().items():
        torch.testing.assert_close(second.net.state_dict()[key], value, msg=key)


def test_seeds_and_custom_timesteps_are_validated_before_use():
    with pytest.raises(ValueError, match=r"\[0, 2\*\*64\)"):
        diffusion_objective(SCHEDULE, seed=2**70)
    model = _model()
    calls: list[int] = []
    model.net.register_forward_hook(lambda *args: calls.append(1))
    ctx = ObjectiveContext(model=model, batch=(_points(3), torch.zeros(3)), epoch_idx=0, batch_idx=0)
    floats = diffusion_objective(SCHEDULE, noise_fn=lambda x_0, g: (torch.zeros(3), torch.zeros_like(x_0)))
    with pytest.raises(ValueError, match="int64"):
        floats(ctx)
    late = diffusion_objective(
        SCHEDULE, noise_fn=lambda x_0, g: (torch.full((3,), SCHEDULE.T, dtype=torch.long), torch.zeros_like(x_0))
    )
    with pytest.raises(ValueError, match=r"outside \[0, 50\)"):
        late(ctx)
    assert calls == []


def test_only_objectives_with_per_commit_work_get_a_commit_hook():
    from nnx import supervised_objective
    from nnx.nn.nn_model import _objective_engine
    from nnx.precision import resolve_precision

    model = _model()

    def engine(objective):
        return _objective_engine(
            objective,
            optimizers={"default": torch.optim.SGD(model.net.parameters(), lr=0.1)},
            clip_norms={},
            scaler=None,
            precision=resolve_precision(model.params, torch.device("cpu")),
        )

    assert engine(supervised_objective()).commit_hooks == []
    assert engine(diffusion_objective(SCHEDULE)).commit_hooks == []


# --- review round 2 ---------------------------------------------------------------------------------------


def test_narrow_integer_timesteps_are_refused_and_numpy_seeds_accepted():
    import numpy as np

    model = _model()
    ctx = ObjectiveContext(model=model, batch=(_points(3), torch.zeros(3)), epoch_idx=0, batch_idx=0)
    for dtype in (torch.uint8, torch.int8, torch.int16):
        narrow = diffusion_objective(
            SCHEDULE, noise_fn=lambda x_0, g, dtype=dtype: (torch.full((3,), 5, dtype=dtype), torch.zeros_like(x_0))
        )
        with pytest.raises(ValueError, match="int64"):
            narrow(ctx)
    int32 = diffusion_objective(
        SCHEDULE, noise_fn=lambda x_0, g: (torch.full((3,), 5, dtype=torch.int32), torch.zeros_like(x_0))
    )
    t, _ = int32.draw(_points(3))
    assert t.dtype == torch.int64 and t.tolist() == [5, 5, 5]
    assert diffusion_objective(SCHEDULE, seed=np.int64(7)).seed == 7


# --- review round 3 ---------------------------------------------------------------------------------------


def test_the_objective_splits_batches_through_the_models_adapter():
    from nnx import NNModelParams as Params
    from nnx.models import KeywordInputs, PositionalInputs

    set_seed(0)
    keyword = NNModel(
        module=DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8),
        params=Params(loss=Losses.MEAN_SQUARED_ERROR),
        batch_adapter=KeywordInputs(["x"], target=None),
    )
    run = keyword.train(
        _params([{"x": _points(4)}, {"x": _points(3, seed=2)}]), objective=diffusion_objective(SCHEDULE)
    )
    assert run.idps[-1].update_count == 2 and all(idp.train_edp.loss is not None for idp in run.idps)

    two = NNModel(
        module=DiffusionMLP(input_dim=2, hidden_dims=[16], time_embed_dim=8),
        params=Params(loss=Losses.MEAN_SQUARED_ERROR),
        batch_adapter=PositionalInputs(2),
    )
    with pytest.raises(ValueError, match="exactly one input"):
        two.train(_params([(_points(2), _points(2), torch.zeros(2))]), objective=diffusion_objective(SCHEDULE))


# --- review round 4 ---------------------------------------------------------------------------------------


def test_noise_of_another_dtype_is_named():
    model = _model()
    ctx = ObjectiveContext(model=model, batch=(_points(3), torch.zeros(3)), epoch_idx=0, batch_idx=0)
    double = diffusion_objective(
        SCHEDULE,
        noise_fn=lambda x_0, g: (torch.zeros(3, dtype=torch.long), torch.zeros(x_0.shape, dtype=torch.float64)),
    )
    with pytest.raises(ValueError, match="typed like x_0"):
        double(ctx)
