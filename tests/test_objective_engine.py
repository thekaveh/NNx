"""FEAT-004: objectives over one shared update engine.

Loss terms carry explicit denominators; the engine combines them per update
window so uneven microbatches, short windows and masks give the full-batch
update, runs autocast → unscale → clip → step, applies a declared
non-finite policy and fires one committed-update event per successful
optimizer update — for NNModel.train and Trainer.train alike.
"""

from __future__ import annotations

import contextlib
import math
import os

import pytest
import torch
import torch.nn.functional as F

from nnx import (
    Activations,
    Callback,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNRun,
    NNTrainerParams,
    NNTrainParams,
    Optims,
    Trainer,
)
from nnx._step_helpers import softened_kl
from nnx._update_engine import UpdateEngine
from nnx.nn.nn_model import _objective_microbatch
from nnx.objectives import LossTerm, ObjectiveResult, kd_objective, supervised_objective
from tests._surgery import SURGERY_FORMS, unrecorded_surgery

RTOL, ATOL = 1e-6, 1e-7


# --- terms -------------------------------------------------------------------


def _engine(param: torch.nn.Parameter, **kwargs) -> UpdateEngine:
    return UpdateEngine(optimizers={"default": torch.optim.SGD([param], lr=0.0)}, **kwargs)


def test_summed_and_normalized_terms_keep_explicit_denominators():
    param = torch.nn.Parameter(torch.tensor(1.0))
    engine = _engine(param)
    engine.accumulate([LossTerm("m", param * 8, 2), LossTerm("s", param * 8, reduction="sum")])
    engine.accumulate([LossTerm("m", param * 1, 1), LossTerm("s", param * 1, reduction="sum")])
    values, total = engine.window_values()
    assert values == {"m": pytest.approx(3.0), "s": pytest.approx(9.0)}  # (8 + 1) / (2 + 1) and 8 + 1
    assert total == pytest.approx(12.0)
    (event,) = engine.commit(epoch_idx=0, batch_idx=1)
    assert event.losses == {"m": pytest.approx(3.0), "s": pytest.approx(9.0)} and event.microbatches == 2
    # d/dparam of the window loss: 9/3 + 9 = 12
    assert engine.commits == 1


def test_mixing_summed_and_normalized_terms_fails():
    param = torch.nn.Parameter(torch.tensor(1.0))
    engine = _engine(param)
    engine.accumulate([LossTerm("a", param * 8, 2)])
    with pytest.raises(ValueError, match="mixes 'mean' and 'sum' reductions"):
        engine.accumulate([LossTerm("a", param * 1, reduction="sum")])
    with pytest.raises(ValueError, match="takes no denominator"):
        LossTerm("a", param * 1, 2, reduction="sum")
    with pytest.raises(ValueError, match="needs an explicit finite denominator"):
        LossTerm("a", param * 1)
    with pytest.raises(ValueError, match="scalar tensor"):
        LossTerm("a", torch.ones(2), 2)
    with pytest.raises(ValueError, match="duplicate loss terms"):
        _engine(param).accumulate([LossTerm("a", param, 1), LossTerm("a", param, 1)])


# --- parity with a full-batch reference ----------------------------------------


def _linear(output_dim: int = 2, weight=((0.25,), (-0.25,)), **model_kwargs) -> NNModel:
    model = NNModel(
        net_params=NNParams(
            input_dim=1, output_dim=output_dim, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, **model_kwargs),
    )
    with torch.no_grad():
        model.net.layers[0].weight.copy_(torch.tensor(weight))
        model.net.layers[0].bias.zero_()
    return model


X = torch.tensor([[1.0], [2.0], [4.0]])
Y = torch.tensor([0, 1, 1])


def _sgd(accumulate: int = 2, clip=None) -> NNOptimParams:
    return NNOptimParams(
        name=Optims.SGD,
        max_lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        accumulate_grad_batches=accumulate,
        grad_clip_norm=clip,
    )


def _fit(model: NNModel, batches, objective, *, epochs: int = 1, optim=None, callbacks=None, data_id=None) -> NNRun:
    return model.train(
        params=NNTrainParams(
            n_epochs=epochs,
            train_loader=batches,
            optim=optim or _sgd(),
            save_phase_checkpoints=False,
            data_id=data_id,
        ),
        objective=objective,
        callbacks=callbacks,
    )


def _reference_step(model: NNModel, loss_fn, clip=None) -> None:
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), clip)
    optimizer.step()


def _assert_same_weights(a: NNModel, b: NNModel) -> None:
    for key, value in a.net.state_dict().items():
        torch.testing.assert_close(b.net.state_dict()[key], value, rtol=RTOL, atol=ATOL, msg=key)


@pytest.mark.parametrize(
    ("variant", "targets", "clip"),
    [
        ("evidence", Y, None),
        ("masked", torch.tensor([0, -100, 1]), None),
        ("all-ignored-microbatch", torch.tensor([0, 1, -100]), None),
        ("clipped", Y, 0.05),
    ],
)
def test_supervised_objective_matches_the_full_batch_update(tmp_path, monkeypatch, variant, targets, clip):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    reference = _linear()
    _reference_step(reference, lambda m: F.cross_entropy(m.net(X), targets), clip)
    model = _linear()
    _fit(model, [(X[:2], targets[:2]), (X[2:], targets[2:])], supervised_objective(), optim=_sgd(clip=clip))
    _assert_same_weights(reference, model)


def test_sum_reduction_losses_are_summed_not_normalized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    reference = _linear()
    _reference_step(reference, lambda m: F.cross_entropy(m.net(X), Y, reduction="sum"))
    model = _linear()
    model.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    _fit(model, [(X[:2], Y[:2]), (X[2:], Y[2:])], supervised_objective())
    _assert_same_weights(reference, model)


def test_short_final_windows_commit_at_the_epoch_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X5 = torch.tensor([[1.0], [2.0], [4.0], [-1.0], [0.5]])
    Y5 = torch.tensor([0, 1, 1, 0, 1])
    reference = _linear()
    _reference_step(reference, lambda m: F.cross_entropy(m.net(X5[:4]), Y5[:4]))  # window of 4 microbatches
    _reference_step(reference, lambda m: F.cross_entropy(m.net(X5[4:]), Y5[4:]))  # short window of 1
    model = _linear()
    run = _fit(model, [(X5[i : i + 1], Y5[i : i + 1]) for i in range(5)], supervised_objective(), optim=_sgd(4))
    _assert_same_weights(reference, model)
    assert [idp.update_count for idp in run.idps] == [0, 0, 0, 1, 2]


def _teacher() -> NNModel:
    return _linear(weight=((0.6,), (-0.3,)))


@pytest.mark.parametrize("targets", [Y, torch.tensor([0, -100, 1])], ids=["labels", "masked"])
def test_kd_objective_matches_the_full_batch_update(tmp_path, monkeypatch, targets):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    alpha, temperature = 0.3, 2.0
    teacher = _teacher()
    teacher_state = {k: v.clone() for k, v in teacher.net.state_dict().items()}
    reference = _linear()
    with torch.no_grad():
        teacher_logits = teacher.net(X)
    _reference_step(
        reference,
        lambda m: (
            alpha * softened_kl(m.net(X), teacher_logits, temperature)
            + (1 - alpha) * F.cross_entropy(m.net(X), targets)
        ),
    )
    model = _linear()
    _fit(
        model,
        [(X[:2], targets[:2]), (X[2:], targets[2:])],
        kd_objective(teacher, alpha=alpha, temperature=temperature),
    )
    _assert_same_weights(reference, model)
    for key, value in teacher.net.state_dict().items():
        assert torch.equal(value, teacher_state[key]), key  # the teacher never moves


# --- ordering and non-finite policy ---------------------------------------------


class _FakeScaler:
    """Records the AMP protocol (CPU stand-in for torch.amp.GradScaler)."""

    def __init__(self, calls: list, params, *, skip: bool = False):
        self.calls, self.params, self.scale_value, self.skip = calls, params, 4.0, skip

    def scale(self, tensor):
        self.calls.append("scale")
        return tensor * self.scale_value

    def unscale_(self, optimizer):
        self.calls.append("unscale")
        for param in self.params:
            if param.grad is not None:
                param.grad /= self.scale_value

    def step(self, optimizer):
        self.calls.append("step")
        if not self.skip:
            optimizer.step()

    def update(self):
        self.calls.append("update")
        if self.skip:
            self.scale_value /= 2  # what GradScaler does after an inf/NaN step

    def get_scale(self):
        return self.scale_value


def _run_window(engine: UpdateEngine, objective, model: NNModel) -> None:
    _objective_microbatch(
        engine, objective, model=model, batch=(X, Y), epoch_idx=0, batch_idx=0, extra_metrics=None, close_window=True
    )


def test_autocast_unscale_clip_and_step_run_in_that_order(monkeypatch):
    calls: list[str] = []
    model = _linear()
    params = list(model.net.parameters())

    @contextlib.contextmanager
    def autocast():
        calls.append("autocast-enter")
        yield
        calls.append("autocast-exit")

    def objective(ctx):
        calls.append("forward")
        return supervised_objective()(ctx)

    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(parameters, max_norm, *args, **kwargs):
        calls.append("clip")
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    events = []
    engine = UpdateEngine(
        optimizers={"default": torch.optim.SGD(params, lr=0.1)},
        scaler=_FakeScaler(calls, params),
        clip_norms={"default": 1.0},
        autocast=autocast,
        listeners=[events.append],
    )
    _run_window(engine, objective, model)
    assert calls == ["autocast-enter", "forward", "autocast-exit", "scale", "unscale", "clip", "step", "update"]
    assert len(events) == 1

    # the reference update: the scaler's scaling cancels out exactly
    reference = _linear()
    _reference_step(reference, lambda m: F.cross_entropy(m.net(X), Y), clip=1.0)
    _assert_same_weights(reference, model)


def test_a_scaler_skipped_step_fires_no_success_hook():
    calls: list[str] = []
    model = _linear()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    params = list(model.net.parameters())
    events = []
    engine = UpdateEngine(
        optimizers={"default": torch.optim.SGD(params, lr=0.1)},
        scaler=_FakeScaler(calls, params, skip=True),
        listeners=[events.append],
    )
    _run_window(engine, supervised_objective(), model)
    assert events == [] and engine.skipped == 1 and engine.commits == 0
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_every_named_optimizer_counts_a_committed_window():
    """A committed window advances every optimizer's update count — one
    whose parameters got no gradient (frozen) is not stepped, but its
    update clock stays aligned with the others."""
    model = _linear()
    weight, bias = model.net.layers[0].weight, model.net.layers[0].bias
    bias.requires_grad_(False)  # frozen for this window
    before = bias.detach().clone()
    events = []
    bias_optimizer = torch.optim.SGD([bias], lr=0.1)
    bias_steps = []
    bias_optimizer.register_step_post_hook(lambda optimizer, args, kwargs: bias_steps.append(1))
    engine = UpdateEngine(
        optimizers={"weight": torch.optim.SGD([weight], lr=0.1), "bias": bias_optimizer},
        listeners=[events.append],
    )
    _run_window(engine, supervised_objective(), model)
    assert sorted((event.optimizer, event.update_idx) for event in events) == [("bias", 1), ("weight", 1)]
    assert engine.commits == 1
    assert torch.equal(bias.detach(), before)  # counted, not changed
    assert bias_steps == []  # not stepped: its step hooks never fire


def _nan_objective(ctx):
    term = supervised_objective()(ctx).terms[0]
    return ObjectiveResult((LossTerm("loss", term.numerator * math.nan, term.denominator),))


def _inf_gradient_objective(ctx):
    weight = ctx.model.net.layers[0].weight
    # finite value (0), infinite gradient (d sqrt(x) / dx at 0)
    numerator = torch.sqrt((weight - weight.detach()).sum())
    return ObjectiveResult((LossTerm("loss", numerator, 1),))


@pytest.mark.parametrize("objective", [_nan_objective, _inf_gradient_objective], ids=["loss", "gradients"])
def test_non_finite_windows_fail_or_skip_as_declared_and_never_step(objective):
    for policy in ("fail", "skip"):
        model = _linear()
        before = {k: v.clone() for k, v in model.net.state_dict().items()}
        events = []
        engine = UpdateEngine(
            optimizers={"default": torch.optim.SGD(model.net.parameters(), lr=0.1)},
            nonfinite=policy,
            listeners=[events.append],
        )
        if policy == "fail":
            with pytest.raises(FloatingPointError, match="nothing was stepped"):
                _run_window(engine, objective, model)
        else:
            _run_window(engine, objective, model)
            assert engine.skipped == 1
        assert events == [] and engine.commits == 0
        for key, value in model.net.state_dict().items():
            assert torch.equal(value, before[key]), (policy, key)
        assert all(p.grad is None for p in model.net.parameters())


# --- committed-update events and counters ---------------------------------------


class _Counter(Callback):
    def __init__(self):
        self.events = []
        self.counts = []

    def on_optimizer_update(self, ctx, event):
        self.events.append(event)
        self.counts.append(ctx.update_count)


def test_one_committed_update_event_per_successful_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X10 = torch.linspace(-1, 1, 10).reshape(10, 1)
    Y10 = (X10[:, 0] > 0).long()
    batches = [(X10[i : i + 2], Y10[i : i + 2]) for i in range(0, 10, 2)]  # five microbatches per epoch
    counter = _Counter()
    run = _fit(_linear(), batches, supervised_objective(), epochs=2, optim=_sgd(2), callbacks=[counter])
    # windows [0, 1], [2, 3], [4] per epoch → three updates per epoch
    assert [e.update_idx for e in counter.events] == [1, 2, 3, 4, 5, 6] == counter.counts
    assert [e.microbatches for e in counter.events] == [2, 2, 1, 2, 2, 1]
    assert [(e.epoch_idx, e.batch_idx) for e in counter.events] == [(0, 1), (0, 3), (0, 4), (1, 1), (1, 3), (1, 4)]
    assert all(isinstance(v, float) for e in counter.events for v in (e.loss, *e.losses.values()))
    # counters stay distinct in the records and survive idps.csv
    assert [(i.iter_idx, i.epoch_idx, i.update_count) for i in run.idps] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 1),
        (3, 0, 2),
        (4, 0, 3),
        (5, 1, 3),
        (6, 1, 4),
        (7, 1, 4),
        (8, 1, 5),
        (9, 1, 6),
    ]
    assert [i.update_count for i in NNRun.load(run.id).idps] == [i.update_count for i in run.idps]

    # a skipped window takes no update and fires no event
    skip_at = iter(range(100))

    class SkipThird:
        nonfinite = "skip"

        def __call__(self, ctx):
            result = supervised_objective()(ctx)
            if next(skip_at) == 2:
                return _nan_objective(ctx)
            return result

    counter = _Counter()
    _fit(_linear(), batches, SkipThird(), epochs=2, optim=_sgd(2), callbacks=[counter], data_id="skip")
    assert [e.update_idx for e in counter.events] == [1, 2, 3, 4, 5]


def test_step_functions_and_objectives_are_never_both_owners(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Exploding:
        def __iter__(self):
            raise AssertionError("the loader must not be iterated")

    class Spy(Callback):
        began = False

        def on_train_begin(self, ctx):
            Spy.began = True

    model = _linear()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(ValueError, match="train_step_fn or objective, not both"):
        model.train(
            params=NNTrainParams(n_epochs=1, train_loader=Exploding()),
            callbacks=[Spy()],
            train_step_fn=lambda ctx: None,
            objective=supervised_objective(),
        )

    trainer_params = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(Exploding())
        .optimizer("main", NNOptimParams.builder().sgd(max_lr=0.1).build())
        .build()
    )
    with pytest.raises(ValueError, match="trainer_step_fn or objective, not both"):
        Trainer(model).train(
            params=trainer_params,
            trainer_step_fn=lambda ctx: None,
            objective=supervised_objective(),
            callbacks=[Spy()],
        )
    with pytest.raises(ValueError, match="trainer_step_fn is required \\(or pass an objective\\)"):
        Trainer(model).train(params=trainer_params, callbacks=[Spy()])
    mismatched = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(Exploding())
        .optimizer(
            "a",
            NNOptimParams.builder()
            .sgd(max_lr=0.1)
            .accumulate_grad(2)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.weight")])
            .build(),
        )
        .optimizer(
            "b",
            NNOptimParams.builder()
            .sgd(max_lr=0.1)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.bias")])
            .build(),
        )
        .build()
    )
    with pytest.raises(ValueError, match="accumulate_grad_batches must agree"):
        Trainer(model).train(params=mismatched, objective=supervised_objective(), callbacks=[Spy()])
    assert Spy.began is False and not os.path.exists("runs")
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_trainer_runs_objectives_through_the_same_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    reference = _linear()
    _reference_step(reference, lambda m: F.cross_entropy(m.net(X), Y))
    model = _linear()
    params = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader([(X[:2], Y[:2]), (X[2:], Y[2:])])
        .optimizer(
            "weight",
            NNOptimParams.builder()
            .sgd(max_lr=0.1, momentum=0.0)
            .accumulate_grad(2)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.weight")])
            .build(),
        )
        .optimizer(
            "bias",
            NNOptimParams.builder()
            .sgd(max_lr=0.1, momentum=0.0)
            .accumulate_grad(2)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.bias")])
            .build(),
        )
        .save_phase_checkpoints(False)
        .build()
    )
    counter = _Counter()
    run = Trainer(model).train(params=params, objective=supervised_objective(), callbacks=[counter])
    _assert_same_weights(reference, model)  # both named optimizers stepped once, on the combined window
    assert sorted((e.optimizer, e.update_idx) for e in counter.events) == [("bias", 1), ("weight", 1)]
    assert [i.update_count for i in run.idps] == [0, 1]


# --- review hardening -------------------------------------------------------------


class _FreezeWeightInEpoch(Callback):
    """Gradual (un)freezing: the weight is frozen for epoch 0 only."""

    def __init__(self):
        self.weights = []

    def on_epoch_begin(self, ctx):
        ctx.model.net.layers[0].weight.requires_grad = ctx.epoch != 0

    def on_epoch_end(self, ctx):
        self.weights.append(ctx.model.net.layers[0].weight.detach().clone())


@pytest.mark.parametrize("accumulate", [1, 2], ids=["single-microbatch-window", "accumulated-window"])
def test_parameters_frozen_between_epochs_train_like_the_default_step(tmp_path, monkeypatch, accumulate):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    batches = [(X[:2], Y[:2]), (X[2:], Y[2:])]
    initial = _linear().net.layers[0].weight.detach().clone()
    reference, ref_freeze = _linear(), _FreezeWeightInEpoch()
    reference.train(
        params=NNTrainParams(
            n_epochs=2, train_loader=batches, optim=_sgd(accumulate), save_phase_checkpoints=False, data_id="step"
        ),
        callbacks=[ref_freeze],
    )
    model, freeze = _linear(), _FreezeWeightInEpoch()
    _fit(model, batches, supervised_objective(), epochs=2, optim=_sgd(accumulate), callbacks=[freeze])
    assert torch.equal(freeze.weights[0], initial)  # frozen in epoch 0: never updated, no crash
    assert not torch.equal(freeze.weights[1], initial)  # unfrozen again: trained
    _assert_same_weights(reference, model)


def test_objective_runs_record_the_exact_whole_epoch_training_loss(tmp_path, monkeypatch):
    from nnx.monitors import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    targets = torch.tensor([0, -100, 1])  # the first microbatch has one valid target of two
    with torch.no_grad():
        expected = float(F.cross_entropy(_linear().net(X), targets))  # one window: weights fixed in the epoch
    monitor = MonitorSpec(metric="loss", split="train", mode="min")

    def params(batches, data_id):
        return NNTrainParams(
            n_epochs=1,
            train_loader=batches,
            optim=_sgd(2),
            save_phase_checkpoints=False,
            monitor=monitor,
            data_id=data_id,
        )

    run = _linear().train(
        params=params([(X[:2], targets[:2]), (X[2:], targets[2:])], "masked"), objective=supervised_objective()
    )
    summary = run.idps[-1].train_summary
    assert summary is not None and summary.loss == pytest.approx(expected, rel=1e-6)
    assert summary.metrics in (None, {})  # like step functions: declared metrics are validation-only

    summed = _linear()
    summed.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        total = float(F.cross_entropy(summed.net(X), Y, reduction="sum"))
    run = summed.train(params=params([(X[:2], Y[:2]), (X[2:], Y[2:])], "summed"), objective=supervised_objective())
    summary = run.idps[-1].train_summary
    assert summary is not None and summary.loss == pytest.approx(total, rel=1e-6)  # a total, not a mean


class _FetchLog:
    def __init__(self, log):
        self.log = log

    def __iter__(self):
        for i in range(3):
            self.log.append(("fetch", i))
            yield X[i : i + 1], Y[i : i + 1]


def test_trainer_step_functions_keep_the_plain_fetch_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    log: list[tuple[str, int]] = []

    def step(ctx):
        log.append(("step", ctx.batch_idx))
        from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

        return NNEvaluationDataPoint(loss=0.0)

    params = (
        NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(_FetchLog(log))
        .optimizer("default", NNOptimParams.builder().sgd(max_lr=0.1, momentum=0.0).build())
        .save_phase_checkpoints(False)
        .build()
    )
    Trainer(_linear()).train(params=params, trainer_step_fn=step)
    # each batch is fetched only when its step runs — no lookahead
    assert log == [("fetch", 0), ("step", 0), ("fetch", 1), ("step", 1), ("fetch", 2), ("step", 2)]


def test_single_microbatch_windows_take_one_backward_pass_and_zero_weights_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    calls = []
    grad = torch.autograd.grad

    def counting_grad(*args, **kwargs):
        calls.append(1)
        return grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counting_grad)
    teacher = _teacher()
    with torch.no_grad():
        teacher_logits = teacher.net(X)
    alpha, temperature = 0.3, 2.0
    reference = _linear()
    _reference_step(
        reference,
        lambda m: (
            alpha * softened_kl(m.net(X), teacher_logits, temperature) + (1 - alpha) * F.cross_entropy(m.net(X), Y)
        ),
    )
    model = _linear()
    _fit(model, [(X, Y)], kd_objective(teacher, alpha=alpha, temperature=temperature), optim=_sgd(1))
    assert len(calls) == 1  # two terms, one window of one microbatch: one backward pass
    _assert_same_weights(reference, model)

    calls.clear()
    _fit(_linear(), [(X[:2], Y[:2]), (X[2:], Y[2:])], kd_objective(teacher, alpha=1.0), optim=_sgd(2), data_id="a1")
    assert len(calls) == 2  # the zero-weight supervised term is never back-propagated


def test_each_term_is_read_from_the_device_once(monkeypatch):
    param = torch.nn.Parameter(torch.tensor(1.0))
    engine = _engine(param)
    result = ObjectiveResult((LossTerm("a", param * 2, 1), LossTerm("b", param * 3, reduction="sum")))
    reads = []
    to_float = torch.Tensor.__float__

    def counting_float(self):
        reads.append(1)
        return to_float(self)

    monkeypatch.setattr(torch.Tensor, "__float__", counting_float)
    result.loss()
    [term.value for term in result.terms]
    engine.accumulate(result.terms)
    result.loss()
    assert len(reads) == 2


def _resume_params(epochs: int, **resume) -> NNTrainParams:
    return NNTrainParams(n_epochs=epochs, train_loader=[(X[:2], Y[:2]), (X[2:], Y[2:])], optim=_sgd(1), **resume)


def test_committed_update_counters_continue_across_a_stateful_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    first = _linear().train(params=_resume_params(2), objective=supervised_objective())
    assert [i.update_count for i in first.idps] == [1, 2, 3, 4]
    counter = _Counter()
    resumed = _linear().train(
        params=_resume_params(1, resume_from_run_id=first.id), objective=supervised_objective(), callbacks=[counter]
    )
    assert resumed.resume_status is not None and "nnx.update_engine" in resumed.resume_status.restored_components
    assert [i.update_count for i in resumed.idps] == [5, 6]
    assert [e.update_idx for e in counter.events] == [5, 6] == counter.counts
    fresh = _linear().train(
        params=_resume_params(1, resume_from_run_id=first.id, resume_mode="weights_only", data_id="fresh"),
        objective=supervised_objective(),
    )
    assert [i.update_count for i in fresh.idps] == [1, 2]  # a weights-only resume starts fresh


class _UpdateLog(Callback):
    def __init__(self):
        self.events = []

    def on_optimizer_update(self, ctx, event):
        self.events.append((event.optimizer, event.update_idx, ctx.update_count))


class _FreezeWeightInEpoch0(Callback):
    """The "weight" optimizer gets no gradient in epoch 0."""

    def on_epoch_begin(self, ctx):
        ctx.model.net.layers[0].weight.requires_grad = ctx.epoch != 0


def _two_optimizer_params(epochs: int, resume=None) -> NNTrainerParams:
    X4, Y4 = torch.tensor([[1.0], [2.0], [4.0], [3.0]]), torch.tensor([0, 1, 1, 0])
    nan = (torch.tensor([[math.nan]]), torch.tensor([1]))  # a window dropped by nonfinite="skip"
    loader = [(X4[:2], Y4[:2]), (X4[2:], Y4[2:]), nan, nan, (X4[:1], Y4[:1])]
    builder = NNTrainerParams.builder().n_epochs(epochs).train_loader(loader)
    for name in ("weight", "bias"):
        optim = (
            NNOptimParams.builder()
            .sgd(max_lr=0.1, momentum=0.0)
            .accumulate_grad(2)
            .param_groups([NNParamGroupSpec(name_pattern=f"layers.0.{name}")])
            .build()
        )
        builder = builder.optimizer(name, optim)
    return (builder.resume_from(resume) if resume else builder).build()


def test_named_optimizers_share_one_update_clock_across_a_resume(tmp_path, monkeypatch):
    """Every named optimizer — one frozen in epoch 0 included — emits the
    engine's committed-update count, a dropped non-finite window counts for
    none, the saved state keeps its per-name ``update_counts`` mapping (the
    format released NNx loads) and a stateful resume continues the clock."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    objective = supervised_objective(nonfinite="skip")
    log = _UpdateLog()
    first = Trainer(_linear()).train(
        params=_two_optimizer_params(2), objective=objective, callbacks=[_FreezeWeightInEpoch0(), log]
    )
    assert log.events == [(name, i, i) for i in (1, 2, 3, 4) for name in ("weight", "bias")]
    saved = NNCheckpoint.load_training_state(run=first.id, type=Checkpoints.LAST)["components"]["nnx.update_engine"]
    assert saved["state"] == {"commits": 4, "skipped": 2, "update_counts": {"weight": 4, "bias": 4}}

    log = _UpdateLog()
    Trainer(_linear()).train(params=_two_optimizer_params(1, resume=first.id), objective=objective, callbacks=[log])
    assert log.events == [(name, i, i) for i in (5, 6) for name in ("weight", "bias")]


def test_the_engine_keeps_one_commit_counter():
    """``commits`` is the engine's only update counter: no per-name mirror
    and no unread window properties."""
    engine = _engine(torch.nn.Parameter(torch.tensor(1.0)))
    for name in ("update_counts", "pending_microbatches", "window_nonfinite"):
        assert not hasattr(engine, name), name
    assert engine.component_state() == {"commits": 0, "skipped": 0, "update_counts": {"default": 0}}


def test_saved_update_counters_restore_the_commit_clock():
    """A saved state (per-name counts equal to ``commits``, as every NNx
    release writes) resumes its clock; one whose per-name counts disagree
    with ``commits`` is refused rather than silently re-counted."""
    model = _linear()
    weight, bias = model.net.layers[0].weight, model.net.layers[0].bias
    events = []
    engine = UpdateEngine(
        optimizers={"weight": torch.optim.SGD([weight], lr=0.1), "bias": torch.optim.SGD([bias], lr=0.1)},
        listeners=[events.append],
    )
    saved = {"commits": 4, "skipped": 2, "update_counts": {"weight": 4, "bias": 4}}
    assert engine.check_component_state(saved, version=1) == []
    engine.load_component_state(saved, version=1)
    _run_window(engine, supervised_objective(), model)
    assert sorted((event.optimizer, event.update_idx) for event in events) == [("bias", 5), ("weight", 5)]
    assert engine.commits == 5 and engine.skipped == 2
    assert engine.component_state()["update_counts"] == {"weight": 5, "bias": 5}

    disagreeing = {"commits": 4, "skipped": 0, "update_counts": {"weight": 4, "bias": 3}}
    (problem,) = engine.check_component_state(disagreeing, version=1)
    assert "disagree" in problem
    (problem,) = engine.check_component_state({"commits": 4, "skipped": 0, "update_counts": [4]}, version=1)
    assert "malformed" in problem


def _feed_fwd() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def test_objective_runs_keep_the_topology_reconstruction_guard(tmp_path, monkeypatch):
    from nnx import low_rank_factorize

    monkeypatch.chdir(tmp_path)
    model = _feed_fwd()
    model.net.layers[1] = low_rank_factorize(model.net.layers[1], rank=2)
    with pytest.raises(ValueError, match="no reconstruction recipe"):
        _fit(model, [(torch.randn(2, 4), torch.tensor([0, 1]))], supervised_objective())
    assert not os.path.exists("runs")  # rejected before any run is reserved


@pytest.mark.parametrize("form", SURGERY_FORMS)
def test_trainer_objective_runs_keep_the_topology_reconstruction_guard(form, tmp_path, monkeypatch):
    """#508: Trainer.train(objective=...) refuses unrecorded surgery with
    NNModel.train's own message, before any run is reserved."""
    monkeypatch.chdir(tmp_path)
    batches = [(torch.randn(2, 4), torch.tensor([0, 1]))]
    model = _feed_fwd()
    unrecorded_surgery(model.net, model.net.layers, form)
    with pytest.raises(ValueError) as from_nnmodel:
        _fit(model, batches, supervised_objective())
    params = NNTrainerParams.builder().n_epochs(1).train_loader(batches).optimizer("main", _sgd(1)).build()
    with pytest.raises(ValueError) as from_trainer:
        Trainer(model).train(params=params, objective=supervised_objective())
    assert str(from_trainer.value) == str(from_nnmodel.value)
    assert not os.path.exists("runs")  # rejected before any run is reserved


def test_objective_runs_accept_a_substituted_composite_net(tmp_path, monkeypatch):
    """The guard checks only reparameterized or replaced versions of the
    descriptor's own keys: a composite substituted into model.net, even
    with a spectral-normed submodule, still trains in both loops."""

    class Composite(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(4, 8)
            self.head = torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(8, 2))

        def forward(self, x):
            return self.head(torch.relu(self.encoder(x)))

        def unpack_batch(self, batch):
            X, Y = batch
            return (X,), Y

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    batches = [(torch.randn(2, 4), torch.tensor([0, 1]))]
    model = _feed_fwd()
    model.net = Composite()
    _fit(model, batches, supervised_objective(), optim=_sgd(1))
    params = NNTrainerParams.builder().n_epochs(1).train_loader(batches).optimizer("main", _sgd(1)).build()
    run = Trainer(model).train(params=params, objective=supervised_objective(), salt="trainer")
    assert run.idps[-1].update_count == 1


def test_trainer_objectives_say_they_run_in_full_precision():
    from types import SimpleNamespace

    from nnx.trainer.trainer import _warn_full_precision_objective

    amp_cuda = SimpleNamespace(params=SimpleNamespace(mixed_precision=True), device=torch.device("cuda"))
    with pytest.warns(RuntimeWarning, match="full precision"):
        _warn_full_precision_objective(amp_cuda)
    with warnings_as_errors():
        _warn_full_precision_objective(SimpleNamespace(params=amp_cuda.params, device=torch.device("cpu")))
        _warn_full_precision_objective(
            SimpleNamespace(params=SimpleNamespace(mixed_precision=False), device=torch.device("cuda"))
        )


@contextlib.contextmanager
def warnings_as_errors():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def test_a_frozen_optimizers_schedule_advances_without_a_torch_warning():
    """An optimizer frozen in its first window is not stepped, but its
    schedule advances with the committed window: torch's "scheduler before
    optimizer" warning would be misleading, so it is not shown."""
    import warnings

    from nnx._scheduler_clock import step_scheduler

    model = _linear()
    weight, bias = model.net.layers[0].weight, model.net.layers[0].bias
    bias.requires_grad_(False)
    bias_optimizer = torch.optim.SGD([bias], lr=0.1)
    schedule = torch.optim.lr_scheduler.StepLR(bias_optimizer, step_size=1, gamma=0.5)
    engine = UpdateEngine(
        optimizers={"weight": torch.optim.SGD([weight], lr=0.1), "bias": bias_optimizer},
        listeners=[lambda event: step_scheduler(schedule) if event.optimizer == "bias" else None],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run_window(engine, supervised_objective(), model)
    assert bias_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_a_skipped_nonfinite_window_never_reaches_the_epoch_training_loss(tmp_path, monkeypatch):
    """nonfinite="skip" drops a window from the update; the whole-epoch
    training loss drops it too (it was NaN for the whole epoch)."""
    from nnx.monitors import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    calls = iter(range(100))

    class SkipSecond:
        nonfinite = "skip"

        def __call__(self, ctx):
            result = supervised_objective()(ctx)
            return _nan_objective(ctx) if next(calls) == 1 else result

    batches = [(X[:1], Y[:1]), (X[1:2], Y[1:2]), (X[2:], Y[2:])]
    run = _linear().train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=batches,
            optim=_sgd(1),
            save_phase_checkpoints=False,
            monitor=MonitorSpec(metric="loss", split="train", mode="min"),
            data_id="skip-summary",
        ),
        objective=SkipSecond(),
    )
    summary = run.idps[-1].train_summary
    assert summary is not None and summary.loss is not None and math.isfinite(summary.loss)


def test_an_epoch_whose_every_window_is_dropped_has_a_nonfinite_loss(tmp_path, monkeypatch):
    """Under "skip", an epoch that drops every window has a non-finite
    whole-epoch loss (the monitor sees "nonfinite" and EarlyStopping counts
    it), never a missing one that a diverged run trains through."""
    from nnx.monitors import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    class AllNaN:
        nonfinite = "skip"

        def __call__(self, ctx):
            return _nan_objective(ctx)

    run = _linear().train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=[(X, Y)],
            optim=_sgd(1),
            save_phase_checkpoints=False,
            monitor=MonitorSpec(metric="loss", split="train", mode="min"),
            data_id="all-dropped",
        ),
        objective=AllNaN(),
    )
    summary = run.idps[-1].train_summary
    assert summary is not None and summary.loss is not None and math.isnan(summary.loss)
    assert run.idps[-1].selection.status == "nonfinite"


def test_a_finite_microbatch_in_a_dropped_window_is_left_out_of_the_epoch_loss(tmp_path, monkeypatch):
    """The rule is per window: a finite microbatch sharing a window with a
    non-finite one is dropped with it (it never reaches an update)."""
    import torch.nn.functional as F

    from nnx.monitors import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    xs, ys = torch.tensor([[1.0], [2.0], [4.0], [1.0]]), torch.tensor([0, 1, 1, 0])
    calls = iter(range(100))

    class NaNSecond:
        nonfinite = "skip"

        def __call__(self, ctx):
            result = supervised_objective()(ctx)
            return _nan_objective(ctx) if next(calls) == 1 else result

    model = _linear()
    with torch.no_grad():  # window 1 is dropped, so window 2 sees the initial weights
        expected = F.cross_entropy(model.net(xs[2:]), ys[2:]).item()
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=[(xs[i : i + 1], ys[i : i + 1]) for i in range(4)],
            optim=_sgd(2),
            save_phase_checkpoints=False,
            monitor=MonitorSpec(metric="loss", split="train", mode="min"),
            data_id="window-rule",
        ),
        objective=NaNSecond(),
    )
    assert run.idps[-1].train_summary.loss == pytest.approx(expected)


def test_an_objective_whose_every_term_is_detached_is_refused(tmp_path, monkeypatch):
    """A detached numerator trained nothing, yet the window counted as a
    committed update (clocks, schedulers, listeners): refuse it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    def detached(ctx):
        term = supervised_objective()(ctx).terms[0]
        return ObjectiveResult((LossTerm("loss", term.numerator.detach(), term.denominator),))

    with pytest.raises(ValueError, match="carry no gradient"):
        _linear().train(
            params=NNTrainParams(n_epochs=1, train_loader=[(X, Y)], optim=_sgd(1), save_phase_checkpoints=False),
            objective=detached,
        )


def test_a_window_dropped_for_non_finite_gradients_is_left_out_of_the_epoch_loss(tmp_path, monkeypatch):
    """Finite terms, non-finite gradients: the engine drops the window, and
    so does the whole-epoch loss (the pre-commit term flag missed it)."""
    import torch.nn.functional as F

    from nnx.monitors import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    xs, ys = torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1])
    calls = iter(range(100))

    class InfGradFirst:
        nonfinite = "skip"

        def __call__(self, ctx):
            return _inf_gradient_objective(ctx) if next(calls) == 0 else supervised_objective()(ctx)

    model = _linear()
    with torch.no_grad():  # window 1 is dropped, so window 2 sees the initial weights
        expected = F.cross_entropy(model.net(xs[1:]), ys[1:]).item()
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=[(xs[i : i + 1], ys[i : i + 1]) for i in range(2)],
            optim=_sgd(1),
            save_phase_checkpoints=False,
            monitor=MonitorSpec(metric="loss", split="train", mode="min"),
            data_id="gradient-drop",
        ),
        objective=InfGradFirst(),
    )
    assert run.idps[-1].train_summary.loss == pytest.approx(expected)


def test_trainer_refuses_a_stop_requested_at_an_update(tmp_path, monkeypatch):
    """stop_at_update set in on_optimizer_update was ignored for the whole
    epoch, then refused with advice to use the hook already in use."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    class StopAtUpdate(Callback):
        calls = 0

        def on_optimizer_update(self, ctx, event):
            StopAtUpdate.calls += 1
            ctx.stop_at_update = True

    params = (
        NNTrainerParams.builder()
        .n_epochs(2)
        .train_loader([(X[:1], Y[:1]), (X[1:2], Y[1:2]), (X[2:], Y[2:])])
        .optimizer(
            "weight",
            NNOptimParams.builder()
            .sgd(max_lr=0.1, momentum=0.0)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.weight")])
            .build(),
        )
        .optimizer(
            "bias",
            NNOptimParams.builder()
            .sgd(max_lr=0.1, momentum=0.0)
            .param_groups([NNParamGroupSpec(name_pattern="layers.0.bias")])
            .build(),
        )
        .save_phase_checkpoints(False)
        .build()
    )
    model = _linear()
    before = model.net.layers[0].weight.detach().clone()
    with pytest.raises(ValueError, match="should_stop in on_epoch_end"):
        Trainer(model).train(params=params, objective=supervised_objective(), callbacks=[StopAtUpdate()])
    # Refused at the first update event, not after the whole epoch.
    assert StopAtUpdate.calls == 1
    assert not torch.equal(model.net.layers[0].weight, before)


# --- #504: windows without usable signal --------------------------------------


def _train_via(via: str, model: NNModel, loader, *, objective, optim=None, epochs=1, **options) -> NNRun:
    """One objective run through ``NNModel.train`` or ``Trainer.train``
    (``scheduler``, ``monitor``, ``callbacks`` and ``data_id`` pass through)."""
    optim = optim or _sgd(1)
    scheduler, monitor = options.get("scheduler"), options.get("monitor")
    callbacks, data_id = options.get("callbacks"), options.get("data_id")
    if via == "trainer":
        builder = (
            NNTrainerParams.builder()
            .n_epochs(epochs)
            .train_loader(loader)
            .optimizer("main", optim)
            .save_phase_checkpoints(False)
        )
        if scheduler is not None:
            builder.scheduler("main", scheduler)
        if monitor is not None:
            builder.monitor(monitor)
        if data_id is not None:
            builder.data_id(data_id)
        return Trainer(model).train(params=builder.build(), objective=objective, callbacks=callbacks)
    kwargs = {"scheduler": scheduler} if scheduler is not None else {}
    params = NNTrainParams(
        n_epochs=epochs,
        train_loader=loader,
        optim=optim,
        save_phase_checkpoints=False,
        monitor=monitor,
        data_id=data_id,
        **kwargs,
    )
    return model.train(params=params, objective=objective, callbacks=callbacks)


def _feed_fwd_4x2() -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


class _WithAuxiliary:
    """A summed cross-entropy plus, optionally, an auxiliary term of a given
    weight (a regulariser annealed from 0, or a logged-only term)."""

    def __init__(self, aux_weight):
        self.aux_weight = aux_weight
        self.after_updates = 0
        self.snapshots: list[torch.Tensor] = []

    def __call__(self, ctx):
        self.snapshots.append(torch.cat([p.detach().flatten().clone() for p in ctx.model.net.parameters()]))
        x, t = ctx.batch
        ctx.model.net.train()
        logits = ctx.model.net(x)
        ce = F.cross_entropy(logits, t, ignore_index=-100, reduction="sum")
        terms = [LossTerm("ce", ce, int((t != -100).sum()))]
        if self.aux_weight is not None:
            terms.append(LossTerm("l2", logits.pow(2).sum(), logits.numel(), weight=self.aux_weight))
        return ObjectiveResult(terms)

    def after_update(self, events):
        self.after_updates += 1


class _UpdateBatches(Callback):
    def __init__(self):
        self.batches: list[int] = []

    def on_optimizer_update(self, ctx, event):
        if event.optimizer in ("default", "main"):
            self.batches.append(event.batch_idx)


@pytest.mark.parametrize("via", ["model", "trainer"])
@pytest.mark.parametrize("aux_weight", [None, 0.0], ids=["no-aux", "aux-weight-0"])
def test_a_weight_zero_term_never_commits_an_all_masked_window(tmp_path, monkeypatch, via, aux_weight):
    """#504: a weight-0 term with samples made an all-masked window commit:
    no parameter moved, yet the update count, the optimizer_update
    scheduler clock, on_optimizer_update and after_update all advanced."""
    from nnx import NNSchedulerParams

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(12, 4, generator=g), torch.randint(0, 2, (12,), generator=g)
    y[4:8] = -100  # batch 1: every target masked
    scheduler = (
        NNSchedulerParams.builder()
        .step(step_size=1, min_lr=0.0, factor=0.5, patience=0, cooldown=0, threshold=0.0)
        .clock("optimizer_update")
        .build()
    )
    objective, events = _WithAuxiliary(aux_weight), _UpdateBatches()
    run = _train_via(
        via,
        _feed_fwd_4x2(),
        [(x[i : i + 4], y[i : i + 4]) for i in (0, 4, 8)],
        objective=objective,
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        scheduler=scheduler,
        callbacks=[events],
        data_id=f"aux-{via}-{aux_weight}",
    )
    assert torch.equal(objective.snapshots[1], objective.snapshots[2])  # the masked batch moved nothing
    assert [i.update_count for i in run.idps] == [1, 1, 2]
    assert [i.lr for i in run.idps] == pytest.approx([0.1, 0.05, 0.05])
    assert events.batches == [0, 2]
    assert objective.after_updates == 2


def test_a_window_whose_every_term_has_weight_zero_commits_nothing():
    """#504: only a term with a nonzero weight and samples trains, so a
    window of weight-0 terms takes no committed update."""
    param = torch.nn.Parameter(torch.tensor(1.0))
    heard = []
    engine = _engine(param, listeners=[heard.append])
    engine.accumulate([LossTerm("a", param * 2, 1, weight=0.0), LossTerm("b", param * 3, reduction="sum", weight=0.0)])
    engine.accumulate([LossTerm("a", param * 2, 1, weight=0.0), LossTerm("b", param * 3, reduction="sum", weight=0.0)])
    assert engine.commit(epoch_idx=0, batch_idx=1) == ()
    assert engine.commits == 0 and heard == []
    engine.accumulate([LossTerm("a", param * 2, 1, weight=0.0), LossTerm("b", param * 3, reduction="sum")])
    (event,) = engine.commit(epoch_idx=0, batch_idx=2)
    assert engine.commits == 1 and event.losses == {"a": pytest.approx(2.0), "b": pytest.approx(3.0)}
    assert event.loss == pytest.approx(3.0)  # the weight-0 term is reported but adds nothing


@pytest.mark.parametrize("via", ["model", "trainer"])
@pytest.mark.parametrize("aux_weight", [None, 0.0], ids=["no-aux", "aux-weight-0"])
def test_an_all_masked_epoch_with_a_weight_zero_term_has_no_loss(tmp_path, monkeypatch, via, aux_weight):
    """#504: a weight-0 term made an all-masked epoch report loss 0.0 —
    a train-loss monitor elected it BEST, and no later epoch of a
    nonnegative loss could beat it."""
    from nnx.monitors import MonitorSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
    y = torch.full((8,), -100)
    run = _train_via(
        via,
        _feed_fwd_4x2(),
        [(x[:4], y[:4]), (x[4:], y[4:])],
        objective=_WithAuxiliary(aux_weight),
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        monitor=MonitorSpec(metric="loss", split="train", mode="min"),
        data_id=f"all-masked-{via}-{aux_weight}",
    )
    last = run.idps[-1]
    assert [i.train_edp.loss for i in run.idps] == [None, None]
    assert last.train_summary is not None and last.train_summary.loss is None
    assert (last.selection.status, last.selection.improved) == ("missing", False)
    assert [i.update_count for i in run.idps] == [0, 0]


def test_an_objective_records_loss_skips_weight_zero_terms():
    """#504: the microbatch loss covers the terms that train — a weight-0
    term beside a fully masked one is no loss at all, never 0.0."""
    param = torch.nn.Parameter(torch.tensor(1.0))
    masked = LossTerm("ce", param * 0, 0)
    assert ObjectiveResult((masked, LossTerm("l2", param * 4, 2, weight=0.0))).loss() is None
    assert ObjectiveResult((LossTerm("ce", param * 6, 3), LossTerm("l2", param * 4, 2, weight=0.0))).loss() == 2.0


_MASKED_TAIL = [(X[:2], Y[:2]), (X[2:], torch.tensor([-100]))]  # 2nd batch: every target ignored


def _task_free_objective(name: str):
    return supervised_objective() if name == "supervised" else kd_objective(_teacher(), alpha=0.0)


@pytest.mark.parametrize("via", ["model", "trainer"])
@pytest.mark.parametrize("name", ["supervised", "kd-alpha-0"])
def test_a_task_free_all_masked_microbatch_records_the_windows_running_loss(tmp_path, monkeypatch, via, name):
    """#504: a task-free objective recorded NaN for an all-masked
    microbatch, so the default ReduceLROnPlateau skipped its step on it.
    It records the window's running loss, as the default step does."""
    import warnings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    default = _linear().train(
        params=NNTrainParams(
            n_epochs=1, train_loader=_MASKED_TAIL, optim=_sgd(2), save_phase_checkpoints=False, data_id="default"
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = _train_via(via, _linear(), _MASKED_TAIL, objective=_task_free_objective(name), optim=_sgd(2))
    losses = [i.train_edp.loss for i in run.idps]
    assert losses == pytest.approx([i.train_edp.loss for i in default.idps])
    assert losses[1] == pytest.approx(losses[0])  # the window's running mean: the valid microbatch's loss
    assert not [w for w in caught if "ReduceLROnPlateau" in str(w.message)]


@pytest.mark.parametrize("via", ["model", "trainer"])
@pytest.mark.parametrize("name", ["supervised", "kd-alpha-0"])
def test_a_task_free_window_masked_throughout_records_no_loss(tmp_path, monkeypatch, via, name):
    """#504: with no valid target anywhere in the window there is no
    running loss either: the records hold None, never NaN."""
    import warnings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    masked = [(X[:2], torch.tensor([-100, -100])), (X[2:], torch.tensor([-100]))]
    model = _linear()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nothing to schedule on: the plateau scheduler says so
        run = _train_via(via, model, masked, objective=_task_free_objective(name), optim=_sgd(2))
    assert [i.train_edp.loss for i in run.idps] == [None, None]
    assert [i.update_count for i in run.idps] == [0, 0]
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize("via", ["model", "trainer"])
def test_early_stopping_on_the_training_loss_trains_through_an_all_masked_tail(tmp_path, monkeypatch, via):
    """#504: the NaN record of a masked tail counted as an epoch without
    improvement, so EarlyStopping(train_edp.loss, patience=0) stopped after
    one epoch where the default step trains all three."""
    from nnx import EarlyStopping

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    default = _linear().train(
        params=NNTrainParams(
            n_epochs=3, train_loader=_MASKED_TAIL, optim=_sgd(2), save_phase_checkpoints=False, data_id="default"
        ),
        callbacks=[EarlyStopping(monitor="train_edp.loss", patience=0)],
    )
    run = _train_via(
        via,
        _linear(),
        _MASKED_TAIL,
        objective=supervised_objective(),
        optim=_sgd(2),
        epochs=3,
        callbacks=[EarlyStopping(monitor="train_edp.loss", patience=0)],
    )
    assert len({i.epoch_idx for i in default.idps}) == 3
    assert len({i.epoch_idx for i in run.idps}) == 3
