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
    Devices,
    Losses,
    Nets,
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


def test_objective_runs_keep_the_topology_reconstruction_guard(tmp_path, monkeypatch):
    from nnx import low_rank_factorize

    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net.layers[1] = low_rank_factorize(model.net.layers[1], rank=2)
    with pytest.raises(ValueError, match="no reconstruction recipe"):
        _fit(model, [(torch.randn(2, 4), torch.tensor([0, 1]))], supervised_objective())
    assert not os.path.exists("runs")  # rejected before any run is reserved


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
