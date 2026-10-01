"""FEAT-014: explicit scheduler clock units.

``NNSchedulerParams(clock="optimizer_update")`` steps a scheduler once per
committed update of its optimizer — never per microbatch, masked window or
skipped step — with horizons counted in updates; ``clock="epoch"`` (the
default) keeps the epoch boundary.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest import mock

import pytest
import torch
from torch.optim import lr_scheduler

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
    NNTrainerParams,
    NNTrainParams,
    Optims,
    Trainer,
)
from nnx.components import ComponentRestoreError
from nnx.nn.callbacks import LRMonitor
from nnx.nn.enum.schedulers import Schedulers
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.objectives import LossTerm, Objective, ObjectiveResult, supervised_objective

_PLATEAU = {"min_lr": 0.0, "factor": 0.5, "patience": 0, "cooldown": 0, "threshold": 0.0}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _model(seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _batches(n: int = 5, size: int = 4, seed: int = 1) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(size, 4, generator=generator), torch.randint(0, 2, (size,), generator=generator)) for _ in range(n)
    ]


class _Unsized:
    """A re-iterable batch source without ``len()`` (a streaming loader)."""

    def __init__(self, batches: list[Any]) -> None:
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def _sgd(accumulate: int = 2, lr: float = 0.1) -> NNOptimParams:
    return NNOptimParams(name=Optims.SGD, max_lr=lr, momentum=0.0, weight_decay=0.0, accumulate_grad_batches=accumulate)


def _sched(kind: Schedulers, clock: str = "optimizer_update", **config: Any) -> NNSchedulerParams:
    return NNSchedulerParams(kind=kind, clock=clock, **{**_PLATEAU, **config})  # type: ignore[arg-type]


def _train(
    model, *, epochs=2, loader=None, accumulate=2, scheduler=None, callbacks=(), objective=None, step=None, **train
):
    train.setdefault("overwrite_existing", True)  # several runs per test share a configuration
    params = NNTrainParams(
        n_epochs=epochs,
        train_loader=loader if loader is not None else _batches(),
        optim=_sgd(accumulate, lr=train.pop("lr", 0.1)),
        scheduler=scheduler if scheduler is not None else _sched(Schedulers.STEP, step_size=1),
        save_phase_checkpoints=False,
        **train,
    )
    return model.train(params, callbacks=list(callbacks), objective=objective, train_step_fn=step)


# --- params ------------------------------------------------------------------------------------------------


def test_the_clock_is_serialized_only_when_it_is_not_the_epoch_default():
    epoch = _sched(Schedulers.ONE_CYCLE, clock="epoch", total_steps=12)
    updates = _sched(Schedulers.ONE_CYCLE, total_steps=12)
    assert "clock" not in epoch.state() and updates.state()["clock"] == "optimizer_update"
    assert NNSchedulerParams.from_state(updates.state()) == updates
    legacy = dict(epoch.state())  # a state written before FEAT-014
    assert NNSchedulerParams.from_state(legacy).clock == "epoch"
    built = (
        NNSchedulerParams.builder()
        .clock("optimizer_update")
        .one_cycle(max_lr=0.5, total_steps=12, **_PLATEAU)  # the variant keeps the clock
        .build()
    )
    assert built == _sched(Schedulers.ONE_CYCLE, max_lr=0.5, total_steps=12)
    with pytest.raises(ValueError, match="clock must be 'epoch' or 'optimizer_update'"):
        _sched(Schedulers.STEP, clock="batch")


def test_a_plateau_scheduler_cannot_take_the_update_clock():
    with pytest.raises(ValueError, match="plateau scheduler reads a monitored metric at the epoch boundary"):
        NNSchedulerParams(clock="optimizer_update", **_PLATEAU)
    with pytest.raises(ValueError, match="plateau scheduler"):
        _sched(Schedulers.REDUCE_LR_ON_PLATEAU)


def test_the_epoch_clock_keeps_its_run_id():
    from nnx.nn.params.nn_run import NNRun

    model = _model()
    legacy = NNSchedulerParams(kind=Schedulers.STEP, step_size=1, **_PLATEAU)
    explicit = _sched(Schedulers.STEP, clock="epoch", step_size=1)

    def run_id(scheduler):
        params = NNTrainParams(n_epochs=1, train_loader=_batches(), optim=_sgd(), scheduler=scheduler)
        return NNRun(train=params, model=model.params, net=model.net_params).id

    assert run_id(legacy) == run_id(explicit)
    assert run_id(_sched(Schedulers.STEP, step_size=1)) != run_id(legacy)


# --- update events -----------------------------------------------------------------------------------------


def _count_steps(scheduler_cls):
    calls = []
    real = scheduler_cls.step

    def spy(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    return calls, mock.patch.object(scheduler_cls, "step", spy)


def test_update_events_not_microbatches_and_epoch_steps_on_the_boundary():
    # 2 epochs x 5 microbatches, accumulation 2: windows of 2, 2 and 1 per epoch.
    calls, spy = _count_steps(lr_scheduler.StepLR)
    with spy:
        _train(_model(), scheduler=_sched(Schedulers.STEP, step_size=1))
    construction = 1  # LRScheduler.__init__ takes one initial step
    assert len(calls) - construction == 6
    calls, spy = _count_steps(lr_scheduler.StepLR)
    with spy:
        _train(_model(), scheduler=_sched(Schedulers.STEP, clock="epoch", step_size=1))
    assert len(calls) - construction == 2


class _SkipWindow(Objective):
    """The supervised objective, with a non-finite loss in one window."""

    def __init__(self, at: tuple[int, int]) -> None:
        super().__init__(nonfinite="skip")
        self.inner = supervised_objective()
        self.at = at

    def __call__(self, ctx):
        result = self.inner(ctx)
        if (ctx.epoch_idx, ctx.batch_idx) == self.at:
            (term,) = result.terms
            poisoned = LossTerm(term.name, term.numerator * float("nan"), term.denominator, term.reduction)
            return ObjectiveResult([poisoned], result.record)
        return result


def test_a_skipped_update_does_not_advance_the_clock():
    monitor = LRMonitor()
    _train(
        _model(), scheduler=_sched(Schedulers.STEP, step_size=1), callbacks=[monitor], objective=supervised_objective()
    )
    assert [k for k, _ in monitor.update_history] == [1, 2, 3, 4, 5, 6]
    skipped = LRMonitor()
    _train(_model(), scheduler=_sched(Schedulers.STEP, step_size=1), callbacks=[skipped], objective=_SkipWindow((1, 2)))
    assert [k for k, _ in skipped.update_history] == [1, 2, 3, 4, 5]
    assert len(skipped.history) == 2  # still one LR snapshot per epoch


class _FakeScaler:
    """A CPU stand-in for GradScaler: the second step finds inf gradients, so the scale backs off. A fused
    optimizer is still called (its kernel skips the update itself); a classic one is not."""

    def __init__(self, fused: bool) -> None:
        self.fused, self.value, self.steps = fused, 1024.0, 0

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        self.steps += 1
        if self.steps != 2 or self.fused:
            optimizer.step()

    def update(self):
        if self.steps == 2:
            self.value /= 2.0


@pytest.mark.parametrize("fused", [False, True])
def test_a_skipped_amp_step_is_not_a_committed_update(fused):
    from nnx._scheduler_clock import SchedulerClock
    from nnx._update_engine import scaler_step

    param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    clock = SchedulerClock("default", lr_scheduler.StepLR(optimizer, step_size=1), horizon=None)
    scaler = _FakeScaler(fused)
    committed = []
    for _ in range(3):  # what default_train_step does under AMP when a clock listens
        param.grad = torch.ones(1)
        committed.append(scaler_step(scaler, (optimizer,)))
        if committed[-1]:
            clock.committed()
    assert committed == [True, False, True]
    assert clock.count == 2


def test_the_default_step_reports_each_committed_update_once():
    from nnx.nn.nn_model import GradientAccumulationState, TrainStepContext, default_train_step

    model = _model()
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
    reported = []
    state = GradientAccumulationState()
    batches = _batches()
    for idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=2,
                batch_idx=idx,
                epoch_idx=0,
                is_last_batch=idx == len(batches) - 1,
                accumulation_state=state,
                report_update=lambda idx=idx: reported.append(idx),
            )
        )
    assert reported == [1, 3, 4]  # windows close at microbatches 2, 4 and the short final one


# --- traces against native references ----------------------------------------------------------------------


def _initial_lr():
    seen: list[float] = []

    class Initial(Callback):
        def on_train_begin(self, ctx):
            seen.append(ctx.optimizer.param_groups[0]["lr"])

    return seen, Initial()


def test_the_warmup_oracle():
    seen, initial = _initial_lr()
    monitor = LRMonitor()
    _train(
        _model(),
        epochs=1,
        loader=_batches(8),
        accumulate=1,
        lr=0.12,
        scheduler=_sched(Schedulers.LINEAR_WARMUP_DECAY, warmup_steps=3, total_steps=8),
        callbacks=[initial, monitor],
    )
    used = [seen[0], *(lr for _, lr in monitor.update_history)][:4]
    assert used == pytest.approx([0.04, 0.08, 0.12, 0.12])


def test_one_cycle_matches_native_stepping_under_accumulation_and_a_short_final_window():
    seen, initial = _initial_lr()
    monitor = LRMonitor()
    # 4 epochs x 5 microbatches, accumulation 2 -> 3 updates per epoch (2, 2, 1): 12 updates.
    _train(
        _model(),
        epochs=4,
        scheduler=_sched(Schedulers.ONE_CYCLE, max_lr=0.5, total_steps=12),
        callbacks=[initial, monitor],
    )
    param = torch.nn.Parameter(torch.zeros(1))
    native_optimizer = torch.optim.SGD([param], lr=0.1)
    native = lr_scheduler.OneCycleLR(native_optimizer, max_lr=0.5, total_steps=12)
    reference = [native_optimizer.param_groups[0]["lr"]]
    for _ in range(12):
        native_optimizer.step()
        native.step()
        reference.append(native_optimizer.param_groups[0]["lr"])
    assert [k for k, _ in monitor.update_history] == list(range(1, 13))
    assert [seen[0], *(lr for _, lr in monitor.update_history)] == pytest.approx(reference)


def test_an_unknown_length_loader_needs_an_explicit_budget_and_overrun_is_refused():
    with pytest.raises(ValueError, match="needs an explicit total_steps"):
        _train(_model(), loader=_Unsized(_batches()), scheduler=_sched(Schedulers.ONE_CYCLE))
    with pytest.raises(ValueError, match="committed update 5, beyond its scheduler's budget of 4"):
        _train(_model(), loader=_Unsized(_batches()), scheduler=_sched(Schedulers.ONE_CYCLE, total_steps=4))
    with pytest.raises(ValueError, match=r"total_steps \(4\) must be >= the run's planned optimizer updates \(6\)"):
        _train(_model(), scheduler=_sched(Schedulers.ONE_CYCLE, total_steps=4))


# --- Trainer: independent clocks -------------------------------------------------------------------------


def _two_optimizer_params(*, auto_step=True, epochs=1, schedulers=None, **trainer):
    def optim(pattern):
        return NNOptimParams(
            name=Optims.SGD,
            max_lr=0.1,
            momentum=0.0,
            weight_decay=0.0,
            param_groups=[NNParamGroupSpec(name_pattern=pattern, lr=0.1)],
        )

    return NNTrainerParams(
        n_epochs=epochs,
        train_loader=_batches(6),
        optims={"a": optim("layers.0.*"), "b": optim("layers.1.*")},
        schedulers=schedulers
        or {
            "a": _sched(Schedulers.STEP, step_size=1),
            "b": _sched(Schedulers.STEP, step_size=1, factor=0.25),
        },
        auto_step_schedulers=auto_step,
        save_phase_checkpoints=False,
        **trainer,
    )


def _two_rate_step(ctx):
    """Optimizer "a" updates every batch, "b" every other batch."""
    x, y = ctx.batch
    loss = ctx.model.loss_fn(ctx.model.net(x), y)
    for optimizer in ctx.optimizers.values():
        optimizer.zero_grad()
    loss.backward()
    ctx.optimizers["a"].step()
    ctx.report_update("a")
    if ctx.batch_idx % 2 == 0:
        ctx.optimizers["b"].step()
        ctx.report_update("b")
    return NNEvaluationDataPoint(loss=float(loss.detach()))


def test_two_optimizers_keep_separate_counters_and_schedules():
    captured = {}

    def step(ctx):
        captured["schedulers"] = ctx.schedulers
        return _two_rate_step(ctx)

    Trainer(_model()).train(_two_optimizer_params(), trainer_step_fn=step)
    a, b = captured["schedulers"]["a"], captured["schedulers"]["b"]
    assert (a.last_epoch, b.last_epoch) == (6, 3)
    assert a.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**6)
    assert b.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.25**3)


def test_auto_step_schedulers_false_detaches_both_subscriptions_even_after_restore():
    calls, spy = _count_steps(lr_scheduler.StepLR)
    with spy:
        first = Trainer(_model()).train(_two_optimizer_params(auto_step=False), trainer_step_fn=_two_rate_step)
        constructions = len(calls)  # one per scheduler, at construction
        assert constructions == 2
        resumed = Trainer(_model(seed=3)).train(
            _two_optimizer_params(auto_step=False, resume_from_run_id=first.id), trainer_step_fn=_two_rate_step
        )
    assert resumed.resume_status is not None and resumed.resume_status.mode == "stateful"
    assert len(calls) == 4  # only the two constructions of the resumed run: no automatic step
    from nnx.nn.enum.checkpoints import Checkpoints
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    state = NNCheckpoint.load_training_state(resumed.id, Checkpoints.LAST)
    assert state is not None
    counts = {name: entry["state"]["count"] for name, entry in state["components"].items() if "clock" in name}
    assert counts == {"nnx.scheduler_clock.a": 0, "nnx.scheduler_clock.b": 0}  # a detached schedule never moved


def test_report_update_names_a_real_optimizer():
    def step(ctx):
        ctx.report_update("c")
        return NNEvaluationDataPoint(loss=0.0)

    with pytest.raises(ValueError, match=r"report_update\('c'\): no optimizer of that name"):
        Trainer(_model()).train(_two_optimizer_params(), trainer_step_fn=step)


def test_a_step_function_that_never_reports_is_warned_about():
    def silent(ctx):
        x, y = ctx.batch
        loss = ctx.model.loss_fn(ctx.model.net(x), y)
        ctx.optimizers["a"].zero_grad()
        loss.backward()
        ctx.optimizers["a"].step()
        return NNEvaluationDataPoint(loss=float(loss.detach()))

    with pytest.warns(UserWarning, match=r"reported no update for \['a', 'b'\]"):
        Trainer(_model()).train(_two_optimizer_params(), trainer_step_fn=silent)


# --- resume ------------------------------------------------------------------------------------------------


def _one_cycle(total_steps=12):
    return _sched(Schedulers.ONE_CYCLE, max_lr=0.5, total_steps=total_steps)


def test_a_split_run_reproduces_the_remaining_lr_sequence():
    whole = LRMonitor()
    _train(_model(), epochs=4, scheduler=_one_cycle(), callbacks=[whole], data_id="whole")
    first_half = LRMonitor()
    parent = _train(_model(), epochs=2, scheduler=_one_cycle(), callbacks=[first_half], data_id="split")
    second_half = LRMonitor()
    _train(
        _model(seed=9),
        epochs=2,
        scheduler=_one_cycle(),
        callbacks=[second_half],
        data_id="split",
        resume_from_run_id=parent.id,
    )
    assert first_half.update_history + second_half.update_history == pytest.approx(whole.update_history)


def test_a_resume_refuses_a_mismatched_clock_or_horizon():
    parent = _train(_model(), epochs=1, scheduler=_one_cycle())
    with pytest.raises(ComponentRestoreError, match="scheduler budget is 12 optimizer updates, this run's is 15"):
        _train(_model(), epochs=1, scheduler=_one_cycle(15), resume_from_run_id=parent.id)
    with pytest.raises(ComponentRestoreError, match="unknown required component 'nnx.scheduler_clock'"):
        _train(
            _model(),
            epochs=1,
            scheduler=_sched(Schedulers.ONE_CYCLE, clock="epoch", max_lr=0.5, total_steps=12),
            resume_from_run_id=parent.id,
        )
    # 3 committed updates + 9 planned fit the budget of 12 exactly; 12 planned do not.
    with pytest.raises(
        ComponentRestoreError, match="3 with 12 planned updates would pass the scheduler's budget of 12"
    ):
        _train(_model(), epochs=4, scheduler=_one_cycle(), resume_from_run_id=parent.id)


def test_an_epoch_run_cannot_be_resumed_on_the_update_clock():
    parent = _train(
        _model(), epochs=1, scheduler=_sched(Schedulers.ONE_CYCLE, clock="epoch", max_lr=0.5, total_steps=12)
    )
    with pytest.raises(ComponentRestoreError, match="missing required component 'nnx.scheduler_clock'"):
        _train(_model(), epochs=1, scheduler=_one_cycle(), resume_from_run_id=parent.id)


# --- review round 1 ----------------------------------------------------------------------------------------


def test_the_default_warmup_follows_total_steps_not_one_runs_length():
    def trace(n_updates):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1.0)
        scheduler = Schedulers.LINEAR_WARMUP_DECAY(
            optimizer, _sched(Schedulers.LINEAR_WARMUP_DECAY, total_steps=100), n_epochs=1, n_updates=n_updates
        )
        lrs = []
        for _ in range(12):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return lrs

    whole = trace(100)
    assert whole[:2] == pytest.approx([0.1, 0.2])  # a tenth of total_steps: 10 warm-up updates
    assert trace(50) == pytest.approx(whole) and trace(None) == pytest.approx(whole)


def test_a_detached_schedule_does_not_move_and_reattaches_from_its_own_position():
    schedulers = {
        "a": _sched(Schedulers.ONE_CYCLE, max_lr=0.5, total_steps=8),
        "b": _sched(Schedulers.STEP, step_size=1),
    }
    captured = {}

    def step(ctx):
        captured["schedulers"] = ctx.schedulers
        return _two_rate_step(ctx)

    detached = Trainer(_model()).train(
        _two_optimizer_params(auto_step=False, schedulers=schedulers), trainer_step_fn=step
    )  # six reports of "a", none of which steps its schedule
    assert captured["schedulers"]["a"].last_epoch == 0
    Trainer(_model(seed=3)).train(
        _two_optimizer_params(schedulers=schedulers, resume_from_run_id=detached.id), trainer_step_fn=step
    )
    assert captured["schedulers"]["a"].last_epoch == 6  # from its own position, within its budget of 8


def test_a_detached_clock_is_not_warned_about():
    import warnings

    def manual(ctx):
        x, y = ctx.batch
        loss = ctx.model.loss_fn(ctx.model.net(x), y)
        ctx.optimizers["a"].zero_grad()
        loss.backward()
        ctx.optimizers["a"].step()
        ctx.schedulers["a"].step()  # the step function owns its schedule: nothing to report
        return NNEvaluationDataPoint(loss=float(loss.detach()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Trainer(_model()).train(_two_optimizer_params(auto_step=False), trainer_step_fn=manual)
    assert not [w for w in caught if "reported no update" in str(w.message)]


def test_a_custom_step_without_a_budget_is_told_why():
    def custom(ctx):
        return NNEvaluationDataPoint(loss=0.0)

    with pytest.raises(ValueError, match="needs an explicit T_max .*only when it owns the update windows"):
        _model().train(
            NNTrainParams(
                n_epochs=1,
                train_loader=_batches(),
                optim=_sgd(),
                scheduler=_sched(Schedulers.COSINE_ANNEALING),
                overwrite_existing=True,
            ),
            train_step_fn=custom,
        )


# --- review round 2 ----------------------------------------------------------------------------------------


def test_an_optimizer_name_that_is_not_a_slug_still_gets_a_clock():
    from nnx._scheduler_clock import component_name

    assert component_name("a b") != component_name("a_b")  # replaced characters never collide
    params = _two_optimizer_params()
    renamed = NNTrainerParams(
        n_epochs=1,
        train_loader=params.train_loader,
        optims={"main opt": params.optims["a"], "b": params.optims["b"]},
        schedulers={"main opt": _sched(Schedulers.STEP, step_size=1)},
        save_phase_checkpoints=False,
    )

    def step(ctx):
        x, y = ctx.batch
        loss = ctx.model.loss_fn(ctx.model.net(x), y)
        for optimizer in ctx.optimizers.values():
            optimizer.zero_grad()
        loss.backward()
        ctx.optimizers["main opt"].step()
        ctx.report_update("main opt")
        return NNEvaluationDataPoint(loss=float(loss.detach()))

    Trainer(_model()).train(renamed, trainer_step_fn=step)


def test_iteration_records_keep_the_learning_rate_each_batch_trained_with():
    seen_by_callbacks = []

    class Seen(Callback):
        def on_optimizer_update(self, ctx, event):
            seen_by_callbacks.append(ctx.optimizer.param_groups[0]["lr"])

    run = _train(_model(), epochs=1, loader=_batches(3), accumulate=1, scheduler=_sched(Schedulers.STEP, step_size=1))
    assert [idp.lr for idp in run.idps] == pytest.approx([0.1, 0.05, 0.025])  # not the next update's rate
    objective_run = _train(
        _model(),
        epochs=1,
        loader=_batches(3),
        accumulate=1,
        scheduler=_sched(Schedulers.STEP, step_size=1),
        objective=supervised_objective(),
        callbacks=[Seen()],
    )
    assert [idp.lr for idp in objective_run.idps] == pytest.approx([0.1, 0.05, 0.025])
    assert seen_by_callbacks == pytest.approx([0.1, 0.05, 0.025])  # callbacks run before the clock steps


class _UnderReported(_Unsized):
    """A loader whose ``len()`` reports fewer batches than it yields."""

    def __len__(self) -> int:
        return len(self.batches) - 2


def test_a_default_budget_is_guarded_too():
    # len() says 3 batches (2 updates per epoch); 5 are yielded (3 updates).
    with pytest.raises(
        ValueError, match=r"committed update 3, beyond its scheduler's default budget of 2 .*len\(train_loader\)"
    ):
        _train(_model(), epochs=1, loader=_UnderReported(_batches()), scheduler=_sched(Schedulers.ONE_CYCLE))


def test_an_empty_loader_reports_itself_on_the_update_clock():
    with pytest.raises(ValueError, match="train_loader yielded no batches"):
        _train(_model(), loader=[], scheduler=_sched(Schedulers.ONE_CYCLE))


# --- review round 3 ----------------------------------------------------------------------------------------


def test_callbacks_see_every_event_of_a_commit_before_any_clock_steps():
    seen = []

    class Seen(Callback):
        def on_optimizer_update(self, ctx, event):
            seen.append((event.optimizer, event.update_idx, ctx.optimizer.param_groups[0]["lr"]))

    Trainer(_model()).train(_two_optimizer_params(), objective=supervised_objective(), callbacks=[Seen()])
    # Both events of the first commit see optimizer "a" (the primary) before its clock steps.
    assert seen[:2] == [("a", 1, pytest.approx(0.1)), ("b", 1, pytest.approx(0.1))]


def test_an_explicit_default_step_owns_its_windows_too():
    from nnx.nn.nn_model import default_train_step

    params = NNTrainParams(
        n_epochs=2,
        train_loader=_batches(),
        optim=_sgd(),
        scheduler=_one_cycle(total_steps=None),
        save_phase_checkpoints=False,
        overwrite_existing=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default step reports its updates: no silent-step warning
        run = _model().train(params, train_step_fn=default_train_step)  # budget: the 6 planned updates
    assert len(run.idps) == 10


def test_a_build_scheduler_override_without_n_updates_still_works():
    built = []

    class Custom(NNModel):
        def _build_scheduler(self, optimizer, params):
            built.append(params.scheduler.clock)
            return super()._build_scheduler(optimizer, params)

    torch.manual_seed(0)
    model = Custom(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    _train(model, scheduler=_sched(Schedulers.STEP, step_size=1))
    _train(model, scheduler=_sched(Schedulers.COSINE_ANNEALING))  # super() still gets the default horizon
    assert built == ["optimizer_update", "optimizer_update"]
    assert "_planned_scheduler_updates" not in vars(model)  # the plan is only lent for the build


def test_a_loaded_update_clock_survives_a_variant_call():
    from nnx.nn.params.nn_scheduler_params_builder import NNSchedulerParamsBuilder

    retuned = NNSchedulerParamsBuilder.from_params(_one_cycle()).one_cycle(max_lr=0.5, total_steps=24, **_PLATEAU)
    assert retuned.build().clock == "optimizer_update"  # a horizon never silently switches to counting epochs
    with pytest.raises(ValueError, match=r"builder: \.clock\('epoch'\)"):
        NNSchedulerParamsBuilder.from_params(_one_cycle()).reduce_on_plateau(**_PLATEAU).build()
    plateau = NNSchedulerParamsBuilder.from_params(_one_cycle()).reduce_on_plateau(**_PLATEAU).clock("epoch").build()
    assert plateau.clock == "epoch"


# --- review round 4 ----------------------------------------------------------------------------------------


def test_a_built_in_paradigm_step_reports_its_updates():
    from nnx import mixup_train_step_factory

    monitor = LRMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "reported no update" warning
        _train(
            _model(),
            accumulate=1,
            scheduler=_sched(Schedulers.STEP, step_size=1),
            callbacks=[monitor],
            step=mixup_train_step_factory(alpha=0.2),
        )
    assert [k for k, _ in monitor.update_history] == list(range(1, 11))  # 2 epochs x 5 batches
    assert monitor.update_history[-1][1] == pytest.approx(0.1 * 0.5**10)


def test_an_nnmodel_step_cannot_report_by_optimizer_name():
    from nnx.nn.nn_model import default_train_step

    def named(ctx):
        result = default_train_step(ctx)
        ctx.report_update("net")
        return result

    with pytest.raises(TypeError, match="takes no optimizer name in NNModel.train"):
        _train(_model(), scheduler=_sched(Schedulers.STEP, step_size=1), step=named)


# --- review round 5 ----------------------------------------------------------------------------------------


def test_a_step_built_on_finalize_step_needs_no_report_of_its_own():
    from nnx._step_helpers import finalize_step

    def custom(ctx):
        x, y = ctx.batch
        ctx.optimizer.zero_grad()
        loss = ctx.model.loss_fn(ctx.model.net(x), y)
        return NNEvaluationDataPoint(loss=finalize_step(loss, ctx, paradigm="custom"))  # steps and reports

    monitor = LRMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # reported: no silent-step warning
        _train(_model(), accumulate=1, scheduler=_one_cycle(total_steps=10), callbacks=[monitor], step=custom)
    assert [k for k, _ in monitor.update_history] == list(range(1, 11))  # 10 updates, each reported once


def test_an_nnmodel_step_reports_without_a_name_on_the_epoch_clock_too():
    from nnx.nn.nn_model import default_train_step

    def named(ctx):
        result = default_train_step(ctx)
        ctx.report_update("default")
        return result

    with pytest.raises(TypeError, match="takes no optimizer name in NNModel.train"):
        _train(_model(), scheduler=_sched(Schedulers.STEP, clock="epoch", step_size=1), step=named)


# --- review round 6 ----------------------------------------------------------------------------------------


class _Lookahead(torch.optim.Optimizer):
    """A wrapper optimizer that skips ``Optimizer.__init__`` (as some third-party wrappers do), so it has no
    step hooks."""

    def __init__(self, inner: torch.optim.Optimizer) -> None:  # noqa: B027 - deliberately no super().__init__
        self.inner = inner
        self.param_groups = inner.param_groups
        self.state = inner.state
        self.defaults = inner.defaults

    def step(self, closure=None):
        return self.inner.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none)

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.inner.load_state_dict(state_dict)


def _factory_optim(factory_id: str, factory) -> Any:
    from nnx.optimizers import NNOptimFactoryParams, OptimizerFactorySpec, register_optimizer_factory

    register_optimizer_factory(factory_id, 1, factory, replace=True)
    return NNOptimFactoryParams(factory=OptimizerFactorySpec(id=factory_id, version=1), max_lr=0.1)


def test_a_wrapper_optimizer_drives_the_clock():
    from nnx.optimizers import unregister_optimizer_factory

    optim = _factory_optim("clock-lookahead", lambda groups, config: _Lookahead(torch.optim.SGD(groups)))
    try:
        monitor = LRMonitor()
        _model().train(
            NNTrainParams(
                n_epochs=2,
                train_loader=_batches(),
                optim=optim,
                scheduler=_sched(Schedulers.STEP, step_size=1),
                save_phase_checkpoints=False,
                overwrite_existing=True,
            ),
            callbacks=[monitor],
        )
    finally:
        unregister_optimizer_factory("clock-lookahead", 1)
    assert [k for k, _ in monitor.update_history] == list(range(1, 11))  # every report counts


def test_iteration_records_snapshot_a_tensor_learning_rate():
    from nnx.optimizers import unregister_optimizer_factory

    def tensor_lr(groups, config):
        for group in groups:
            group["lr"] = torch.tensor(float(group["lr"]))
        return torch.optim.SGD(groups, foreach=False)

    optim = _factory_optim("clock-tensor-lr", tensor_lr)
    try:
        run = _model().train(
            NNTrainParams(
                n_epochs=1,
                train_loader=_batches(3),
                optim=optim,
                scheduler=_sched(Schedulers.STEP, step_size=1),
                save_phase_checkpoints=False,
                overwrite_existing=True,
            )
        )
    finally:
        unregister_optimizer_factory("clock-tensor-lr", 1)
    assert [idp.lr for idp in run.idps] == pytest.approx([0.1, 0.05, 0.025])


# --- review round 7 ----------------------------------------------------------------------------------------


class _TwoStep(torch.optim.Optimizer):
    """A SAM-style optimizer: it calls ``Optimizer.__init__`` but updates the weights through an inner
    optimizer, so its own ``step`` is never called."""

    def __init__(self, groups) -> None:
        self.base = torch.optim.SGD(groups)
        super().__init__(self.base.param_groups, self.base.defaults)

    def first_step(self) -> None:
        pass

    def second_step(self) -> None:
        self.base.step()


@pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)`:UserWarning"
)  # the outer step() never runs
def test_reports_are_authoritative():
    from nnx.optimizers import unregister_optimizer_factory

    def sam_step(ctx):
        x, y = ctx.batch
        ctx.optimizer.zero_grad()
        loss = ctx.model.loss_fn(ctx.model.net(x), y)
        loss.backward()
        ctx.optimizer.first_step()
        ctx.optimizer.second_step()
        ctx.report_update()
        return NNEvaluationDataPoint(loss=float(loss.detach()))

    optim = _factory_optim("clock-two-step", lambda groups, config: _TwoStep(groups))
    try:
        sam = LRMonitor()
        _model().train(
            NNTrainParams(
                n_epochs=2,
                train_loader=_batches(),
                optim=optim,
                scheduler=_sched(Schedulers.STEP, step_size=1),
                save_phase_checkpoints=False,
                overwrite_existing=True,
            ),
            callbacks=[sam],
            train_step_fn=sam_step,
        )
    finally:
        unregister_optimizer_factory("clock-two-step", 1)
    assert [k for k, _ in sam.update_history] == list(range(1, 11))  # its own step() never runs

    def two_updates(ctx):  # two updates on one batch, reported together afterwards
        x, y = ctx.batch
        for _ in range(2):
            ctx.optimizer.zero_grad()
            ctx.model.loss_fn(ctx.model.net(x), y).backward()
            ctx.optimizer.step()
        ctx.report_update()
        ctx.report_update()
        return NNEvaluationDataPoint(loss=0.0)

    twice = LRMonitor()
    _train(_model(), accumulate=1, scheduler=_sched(Schedulers.STEP, step_size=1), callbacks=[twice], step=two_updates)
    assert len(twice.update_history) == 20


# --- review round 9 ----------------------------------------------------------------------------------------


def test_a_cosine_schedule_past_a_default_period_warns_but_a_configured_one_does_not():
    # len() says 6 batches (12 planned updates over 2 epochs); 8 are yielded per epoch.
    with pytest.warns(UserWarning, match=r"passed its T_max of 12 .*set T_max") as caught:
        _train(
            _model(), accumulate=1, loader=_UnderReported(_batches(8)), scheduler=_sched(Schedulers.COSINE_ANNEALING)
        )
    assert len([w for w in caught if "T_max" in str(w.message)]) == 1  # once per run
    monitor = LRMonitor()  # an explicit T_max may be run past on purpose, as torch allows
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _train(_model(), accumulate=1, scheduler=_sched(Schedulers.COSINE_ANNEALING, T_max=4), callbacks=[monitor])
    assert len(monitor.update_history) == 10


# --- review round 10 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_max", [None, 10])
def test_a_default_period_cosine_run_resumes_and_warns_past_the_restored_period(t_max):
    parent = _train(_model(), epochs=1, accumulate=1, scheduler=_sched(Schedulers.COSINE_ANNEALING), data_id="cos")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed = _train(
            _model(seed=9),
            epochs=1,
            accumulate=1,
            scheduler=_sched(Schedulers.COSINE_ANNEALING, T_max=t_max),
            data_id="cos",
            resume_from_run_id=parent.id,
        )
    assert resumed.resume_status is not None and resumed.resume_status.mode == "stateful"
    # The restored period is the parent's 5 planned updates, whatever this run configures (round 11): passing
    # it warns, and the warning says a resume keeps the checkpoint's T_max.
    warned = [w for w in caught if "passed its T_max of 5" in str(w.message)]
    assert len(warned) == 1
    assert "stateful resume keeps the checkpoint's T_max" in str(warned[0].message)


# --- review rounds 12-13 ---------------------------------------------------------------------------------------


def test_a_subclass_cosine_period_is_its_own_choice():
    class Cosine(NNModel):  # a subclass building a short cosine period on purpose
        def _build_scheduler(self, optimizer, params):
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    torch.manual_seed(0)
    model = Cosine(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    monitor = LRMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # run past on purpose: no period warning
        _train(model, accumulate=1, scheduler=_sched(Schedulers.STEP, step_size=1), callbacks=[monitor])
    assert len(monitor.update_history) == 10


# --- review round 14 ---------------------------------------------------------------------------------------


def _subclass_model(build):
    class Custom(NNModel):
        def _build_scheduler(self, optimizer, params):
            return build(optimizer)

    torch.manual_seed(0)
    return Custom(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def test_a_subclass_cosine_period_is_its_own_choice_under_a_cosine_configuration_too():
    model = _subclass_model(lambda optimizer: lr_scheduler.CosineAnnealingLR(optimizer, T_max=3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _train(model, accumulate=1, scheduler=_sched(Schedulers.COSINE_ANNEALING))  # T_max left unset


@pytest.mark.parametrize(
    "build, reason",
    [
        (
            lambda optimizer: lr_scheduler.ChainedScheduler(
                [lr_scheduler.LinearLR(optimizer), lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)]
            ),
            "ChainedScheduler has no last_epoch",
        ),
        (lambda optimizer: lr_scheduler.ReduceLROnPlateau(optimizer), "ReduceLROnPlateau reads a monitored metric"),
    ],
)
def test_a_scheduler_the_update_clock_cannot_drive_is_refused_up_front(build, reason):
    with pytest.raises(ValueError, match=f"{reason}.*use clock='epoch'"):
        _train(_subclass_model(build), scheduler=_sched(Schedulers.STEP, step_size=1))


# --- review round 15 ---------------------------------------------------------------------------------------


def test_a_checkpoint_from_before_component_state_cannot_resume_on_the_update_clock(tmp_path):
    from nnx.nn.enum.checkpoints import Checkpoints
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    parent = _train(_model(), epochs=1, scheduler=_one_cycle(), data_id="legacy")
    checkpoint = NNCheckpoint.load(parent.id, Checkpoints.LAST)
    assert checkpoint is not None
    sidecar = tmp_path / "runs" / parent.id / "checkpoints" / f"last.pt.opt.{checkpoint.training_state_id}.pt"
    state = torch.load(sidecar, weights_only=True)
    for key in ("components", "optimizers", "optimizer_types", "schedulers", "scheduler_types"):
        state.pop(key, None)
    state["nnx_training_state_version"] = 3  # as written before FEAT-005: its scheduler position counts epochs
    torch.save(state, sidecar)
    with pytest.raises(ComponentRestoreError, match="predates scheduler clocks.*clock='epoch'"):
        _train(_model(seed=9), epochs=1, scheduler=_one_cycle(), data_id="legacy", resume_from_run_id=parent.id)
