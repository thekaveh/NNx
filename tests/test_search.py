"""FEAT-033: budgeted experiment search through the optional Optuna adapter."""

from __future__ import annotations

import os
import re
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

optuna = pytest.importorskip("optuna")

from nnx.monitors import MonitorSpec  # noqa: E402
from nnx.nn.callbacks import Callback  # noqa: E402
from nnx.nn.enum.activations import Activations  # noqa: E402
from nnx.nn.enum.checkpoints import Checkpoints  # noqa: E402
from nnx.nn.enum.devices import Devices  # noqa: E402
from nnx.nn.enum.losses import Losses  # noqa: E402
from nnx.nn.enum.nets import Nets  # noqa: E402
from nnx.nn.params.nn_checkpoint import NNCheckpoint  # noqa: E402
from nnx.nn.params.nn_model_params import NNModelParams  # noqa: E402
from nnx.nn.params.nn_optim_params import NNOptimParams  # noqa: E402
from nnx.nn.params.nn_params import NNParams  # noqa: E402
from nnx.nn.params.nn_train_params import NNTrainParams  # noqa: E402
from nnx.plans import ExperimentPlan  # noqa: E402
from nnx.search import (  # noqa: E402
    CategoricalParam,
    FloatParam,
    IntParam,
    SearchBudget,
    SearchSpace,
    _TrialControl,
    search,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOSS = MonitorSpec("loss")
NET = NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)
MODEL = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
DATA = torch.Generator().manual_seed(11)
X, Y = torch.randn(36, 4, generator=DATA), torch.arange(36) % 3
X_VAL, Y_VAL = torch.randn(12, 4, generator=DATA), torch.arange(12) % 3
SPACE = SearchSpace(FloatParam(name="lr", low=1e-3, high=1e-1, log=True))


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


class Calls:
    def __init__(self) -> None:
        self.train = self.val = self.callbacks = 0


def plan(calls: Calls | None = None, *, n_epochs: int = 2, batch_size: int = 6, accumulate: int = 1) -> ExperimentPlan:
    calls = calls or Calls()

    def train():
        calls.train += 1
        return DataLoader(TensorDataset(X, Y), batch_size=batch_size)

    def val():
        calls.val += 1
        return DataLoader(TensorDataset(X_VAL, Y_VAL), batch_size=6)

    def callback():
        calls.callbacks += 1
        return Callback()

    params = NNTrainParams(
        n_epochs=n_epochs,
        optim=NNOptimParams.builder().sgd(max_lr=0.05).accumulate_grad(accumulate).build(),
        monitor=LOSS,
    )
    return (
        ExperimentPlan()
        .with_net(NET)
        .with_model(MODEL)
        .with_train(params)
        .with_data(train, val=val, identity="toy")
        .with_callback_factories(callback)
        .with_seed(3)
    )


def with_lr(base: ExperimentPlan, params) -> ExperimentPlan:
    optim = replace(base.train.optim, max_lr=params["lr"])
    return base.with_optim(optim)


def run(base=None, *, budget=SearchBudget(trials=2), name="s", storage=None, **kwargs):
    return search(
        base or plan(),
        kwargs.pop("space", SPACE),
        apply=kwargs.pop("apply", with_lr),
        monitor=kwargs.pop("monitor", LOSS),
        budget=budget,
        study_name=name,
        storage=storage,
        sampler=kwargs.pop("sampler", optuna.samplers.RandomSampler(seed=0)),
        **kwargs,
    )


def runs_on_disk() -> list[str]:
    return (
        sorted(name for name in os.listdir("runs") if not name.startswith(".") and name != "best")
        if os.path.isdir("runs")
        else []
    )


# ---------------- validated first ----------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: FloatParam(name="lr", low=0.1, high=0.1),
        lambda: FloatParam(name="lr", low=0.0, high=1.0, log=True),
        lambda: FloatParam(name="lr", low=0.0, high=1.0, step=0.0),
        lambda: FloatParam(name="lr", low=0.0, high=float("inf")),
        lambda: IntParam(name="n", low=3, high=2),
        lambda: IntParam(name="n", low=0, high=4, log=True),
        lambda: CategoricalParam(name="c", choices=()),
        lambda: CategoricalParam(name="c", choices=("a", "a")),
        lambda: SearchSpace(),
        lambda: SearchSpace(FloatParam(name="x", low=0, high=1), IntParam(name="x", low=0, high=2)),
        lambda: SearchBudget(trials=0),
        lambda: SearchBudget(trials=2, updates_per_trial=0),
        lambda: SearchBudget(trials=2, deadline_seconds=-1.0),
    ],
)
def test_bounds_and_budgets_validate_first(make):
    with pytest.raises((ValueError, TypeError)):
        make()


def test_numpy_bounds_budgets_and_choices_are_the_plain_values():
    """#544: NumPy numbers from a sweep grid are the plain values they equal
    — the same state(), digest() and stored types (digest()'s json.dumps
    refuses a raw numpy.int64)."""
    import numpy as np

    for numpy_float in (np.float32, np.float64):
        numpy = FloatParam(name="lr", low=numpy_float(0.25), high=numpy_float(1.0), step=numpy_float(0.125))
        plain = FloatParam(name="lr", low=0.25, high=1.0, step=0.125)
        assert numpy.state() == plain.state()
        assert SearchSpace(numpy).digest() == SearchSpace(plain).digest()
        assert all(type(getattr(numpy, field)) is float for field in ("low", "high", "step"))
    integral = FloatParam(name="lr", low=np.int64(0), high=np.int64(2), step=np.int64(1))
    assert SearchSpace(integral).digest() == SearchSpace(FloatParam(name="lr", low=0, high=2, step=1)).digest()
    width = IntParam(name="w", low=np.int64(1), high=np.int32(8), step=np.int64(1))
    assert width.state() == IntParam(name="w", low=1, high=8).state()
    assert all(type(getattr(width, field)) is int for field in ("low", "high", "step"))
    assert SearchSpace(width).digest() == SearchSpace(IntParam(name="w", low=1, high=8)).digest()
    choices = CategoricalParam(name="c", choices=(np.int64(16), np.int64(32)))
    assert choices.state() == CategoricalParam(name="c", choices=(16, 32)).state()
    assert SearchSpace(choices).digest() == SearchSpace(CategoricalParam(name="c", choices=(16, 32))).digest()
    budget = SearchBudget(trials=np.int64(3), updates_per_trial=np.int32(2), deadline_seconds=np.float32(1.5))
    assert (budget.trials, budget.updates_per_trial, budget.deadline_seconds) == (3, 2, 1.5)
    assert (
        type(budget.trials) is int and type(budget.updates_per_trial) is int and type(budget.deadline_seconds) is float
    )


def test_enum_choices_stay_the_members_suggest_returns():
    """#544 review: only NumPy scalars are normalized; an IntEnum member is
    kept as given (state() and digest() as before), so suggest() hands it back."""
    import enum

    class Width(enum.IntEnum):
        NARROW = 16
        WIDE = 32

    choices = CategoricalParam(name="w", choices=(Width.NARROW, Width.WIDE))
    assert all(type(choice) is Width for choice in choices.choices)
    assert choices.state() == CategoricalParam(name="w", choices=(16, 32)).state()


@pytest.mark.parametrize(
    ("make", "field"),
    [
        (lambda: FloatParam(name="lr", low=10**400, high=1.0), "lr.low"),
        (lambda: FloatParam(name="lr", low=1e-4, high=10**400), "lr.high"),
        (lambda: FloatParam(name="lr", low=1e-4, high=1.0, step=10**400), "lr.step"),
        (lambda: FloatParam(name="lr", low=-(10**400), high=1.0), "lr.low"),
        (lambda: SearchBudget(trials=2, deadline_seconds=10**400), "deadline_seconds"),
    ],
)
def test_an_integer_too_large_for_a_float_is_refused_by_name(make, field):
    """#544: 10**400 raised a raw OverflowError."""
    with pytest.raises(ValueError, match=re.escape(field)):
        make()


@pytest.mark.parametrize(
    "make",
    [
        lambda: IntParam(name="w", low=1.0, high=8),
        lambda: IntParam(name="w", low=True, high=8),
        lambda: IntParam(name="w", low="1", high=8),
        lambda: SearchBudget(trials=True),
        lambda: SearchBudget(trials=3.0),
        lambda: SearchBudget(trials=2, updates_per_trial="2"),
        lambda: FloatParam(name="lr", low=True, high=2.0),
    ],
)
def test_bools_floats_and_strings_stay_refused_as_counts(make):
    with pytest.raises((ValueError, TypeError)):
        make()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda p: p, None),
        (lambda p: p.with_data(lambda: DataLoader(TensorDataset(X, Y)), val=None), "val_loader"),
        (lambda p: p.with_data(DataLoader(TensorDataset(X, Y), batch_size=6), val=lambda: None), "factories"),
        (lambda p: p.with_callbacks(Callback()), "fresh callbacks"),
        (lambda p: p.with_metrics(monitor=MonitorSpec("error")), "must equal the search monitor"),
    ],
)
def test_the_plan_and_monitor_are_checked_before_any_trial(change, match):
    if match is None:
        with pytest.raises(ValueError, match="validation monitor"):
            run(monitor=MonitorSpec("loss", split="train"))
    else:
        with pytest.raises(ValueError, match=match):
            run(change(plan()))
    assert runs_on_disk() == []


# ---------------- fresh trials, budget ----------------


def test_each_trial_is_fresh_and_the_base_plan_unchanged():
    calls = Calls()
    base = plan(calls)
    snapshot = (base.train, base.callback_factories, base.seed)
    result = run(base, budget=SearchBudget(trials=3))
    assert (base.train, base.callback_factories, base.seed) == snapshot
    assert (calls.train, calls.val, calls.callbacks) == (3, 3, 3)
    attempts = [outcome.attempt_id for outcome in result.outcomes]
    runs = [outcome.run_id for outcome in result.outcomes]
    assert len(set(attempts)) == 3 and len(set(runs)) == 3 and sorted(runs) == runs_on_disk()
    for outcome in result.outcomes:
        assert outcome.state == "completed" and set(outcome.params) == {"lr"}
        assert outcome.epochs == 2 and outcome.updates == 12 and outcome.seconds >= 0
        assert outcome.monitor_value is not None and outcome.checkpoint and os.path.exists(outcome.checkpoint)


def test_every_trial_state_counts_against_the_budget(tmp_path):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    first = run(storage=storage, name="budget", budget=SearchBudget(trials=2))
    assert first.asked == 2
    again = run(storage=storage, name="budget", budget=SearchBudget(trials=3))
    assert again.asked == 1 and len(again.outcomes) == 3
    assert run(storage=storage, name="budget", budget=SearchBudget(trials=3)).asked == 0

    # failed, pruned and orphaned priors count too
    study = optuna.load_study(study_name="budget", storage=storage)
    orphan = study.ask()  # left RUNNING, as by a crashed process
    assert orphan.number == 3
    outcome = run(storage=storage, name="budget", budget=SearchBudget(trials=5))
    assert outcome.asked == 1
    states = [o.state for o in outcome.outcomes]
    assert (
        states.count("running") == 1 and next(o for o in outcome.outcomes if o.state == "running").reason == "orphaned"
    )


def test_two_priors_and_a_budget_of_three_ask_once(tmp_path):
    storage = f"sqlite:///{tmp_path / 'priors.db'}"
    run(storage=storage, name="priors", budget=SearchBudget(trials=1))
    study = optuna.load_study(study_name="priors", storage=storage)
    failed = study.ask()
    failed.suggest_float("lr", 1e-3, 1e-1, log=True)
    study.tell(failed, state=optuna.trial.TrialState.FAIL)
    assert len(study.trials) == 2
    assert run(storage=storage, name="priors", budget=SearchBudget(trials=3)).asked == 1


def test_a_changed_search_identity_is_refused_before_any_trial(tmp_path):
    storage = f"sqlite:///{tmp_path / 'id.db'}"
    run(storage=storage, name="id", budget=SearchBudget(trials=1))
    other = SearchSpace(FloatParam(name="lr", low=1e-4, high=1e-1, log=True))
    with pytest.raises(ValueError, match="space"):
        run(storage=storage, name="id", budget=SearchBudget(trials=3), space=other)
    with pytest.raises(ValueError, match="plan"):
        run(plan(n_epochs=3), storage=storage, name="id", budget=SearchBudget(trials=3))
    assert len(optuna.load_study(study_name="id", storage=storage).trials) == 1


def test_a_plan_without_a_resume_keeps_its_stored_identity():
    """#505: the resume joins the study identity only when the plan resumes,
    so a study stored before it reopens unchanged (the e69fd74 digest)."""
    from nnx.search import _plan_identity

    assert _plan_identity(plan()) == "7a6811c4d14139c6551356fccfda1a8efdc89cbc8697e43428135c2f558844f3"
    assert _plan_identity(plan().resuming("source", mode="weights_only").without_resume()) == _plan_identity(plan())


class StopAfterEpoch(Callback):
    """Interrupts a fit after epoch 1 (zero-based), leaving the rest of its plan."""

    def on_epoch_end(self, ctx) -> None:
        if ctx.epoch >= 1:
            ctx.should_stop = True


def epochs_search(base: ExperimentPlan, *, name: str, storage=None):
    return run(
        base,
        space=SearchSpace(CategoricalParam(name="epochs", choices=(1, 2))),
        apply=lambda p, q: p.with_epochs(q["epochs"]),
        sampler=optuna.samplers.GridSampler({"epochs": [1, 2]}),
        name=name,
        storage=storage,
    )


def with_resume(base: ExperimentPlan, **changes) -> ExperimentPlan:
    """The resume set on the training parameters, not with resuming()."""
    return base.with_train(replace(base.train, **changes))


def resumed(base: ExperimentPlan, run_id: str, how: str, mode: str | None = None) -> ExperimentPlan:
    """``base`` resuming ``run_id`` through ``resuming()`` or the training
    parameters, in ``mode`` (else the default ``"auto"``)."""
    if how == "resuming":
        return base.resuming(run_id) if mode is None else base.resuming(run_id, mode=mode)
    return with_resume(base, resume_from_run_id=run_id, **({} if mode is None else {"resume_mode": mode}))


@pytest.mark.parametrize("mode", [None, "stateful"], ids=["auto", "stateful"])
@pytest.mark.parametrize("how", ["resuming", "train"])
def test_a_stateful_resume_is_refused_before_any_trial(tmp_path, how, mode):
    """#505: a stateful resume restores the parent's monitor best, so BEST
    (and the trial's value) would come only from an epoch that beats the
    parent's; trials that validated finite epochs failed as
    no_validated_epoch. The search refuses the plan before any trial, naming
    the weights-only resume, which scores each trial by its own epochs."""
    parent = plan(n_epochs=6).fit().run  # its best epoch is 4: no resumed epoch beats it
    calls = Calls()
    before = runs_on_disk()
    storage = f"sqlite:///{tmp_path / 'stateful.db'}"
    with pytest.raises(ValueError, match=rf"resume_mode={mode or 'auto'!r}.*mode='weights_only'"):
        epochs_search(resumed(plan(calls), parent.id, how, mode), name="stateful", storage=storage)
    assert (calls.train, calls.val, calls.callbacks) == (0, 0, 0)
    assert runs_on_disk() == before
    assert optuna.study.get_all_study_names(storage) == []
    # What a trial would have been: validated epochs, but no BEST (no value).
    child = resumed(plan(), parent.id, how, mode).fit().run
    assert child.resume_status.mode == "stateful" and any(idp.val_edp is not None for idp in child.idps)
    assert NNCheckpoint.load(run=child.id, type=Checkpoints.BEST) is None

    weights = epochs_search(resumed(plan(), parent.id, how, "weights_only"), name="weights")
    assert weights.best is not None and len(weights.outcomes) == 2
    for outcome in weights.outcomes:  # each trial is scored by its own validated epochs
        assert outcome.state == "completed" and outcome.observations
        assert outcome.monitor_value == min(value for _, value in outcome.observations)
        best_epoch = min(outcome.observations, key=lambda item: item[1])[0]
        best = NNCheckpoint.load(run=outcome.run_id, type=Checkpoints.BEST)
        assert best is not None and best.idp.epoch_idx == best_epoch


@pytest.mark.parametrize("on_train", [False, True], ids=["resuming", "train"])
def test_a_planned_resume_is_refused_before_any_trial(on_train):
    """#505: ``resume_epochs="planned"`` continues the source's training
    state, monitor best included, so the search refuses it too."""
    source = plan(n_epochs=3).with_callback_factories(StopAfterEpoch).fit().run
    calls = Calls()
    before = runs_on_disk()
    base = plan(calls, n_epochs=3)
    planned = (
        with_resume(base, resume_from_run_id=source.id, resume_epochs="planned")
        if on_train
        else base.resuming(source.id, epochs="planned")
    )
    with pytest.raises(ValueError, match=r"resume_mode='auto'.*mode='weights_only'.*'additional'"):
        run(planned, space=SearchSpace(CategoricalParam(name="noop", choices=(0,))), apply=lambda p, q: p)
    assert (calls.train, calls.val, calls.callbacks) == (0, 0, 0)
    assert runs_on_disk() == before


def test_a_trial_plan_that_resumes_statefully_stops_the_search(tmp_path):
    """#505: apply() returning a stateful resume is a configuration error:
    the trial is recorded failed and counts against the budget, and no fit
    runs."""
    parent = plan().fit().run
    storage = f"sqlite:///{tmp_path / 'resume.db'}"
    before = runs_on_disk()
    with pytest.raises(ValueError, match="resume_mode='auto'"):
        run(
            apply=lambda p, q: with_lr(p, q).resuming(parent.id),
            storage=storage,
            name="resume",
            budget=SearchBudget(trials=3),
        )
    (trial,) = optuna.load_study(study_name="resume", storage=storage).trials  # one trial spent, then stopped
    assert trial.state == optuna.trial.TrialState.FAIL
    assert trial.user_attrs["reason"] == "config_error:ValueError"
    assert runs_on_disk() == before


def test_a_stored_study_over_another_resume_is_refused_before_any_trial(tmp_path):
    """#505: the study identity includes the effective resume (source run,
    checkpoint, mode and epochs), so a study over a fine-tune of run A is
    never continued with a fine-tune of B, of A's BEST or with no resume."""
    run_a = plan().fit().run.id
    run_b = plan(n_epochs=3).fit().run.id
    storage = f"sqlite:///{tmp_path / 'resume.db'}"

    def trials(name: str) -> int:
        return len(optuna.load_study(study_name=name, storage=storage).trials)

    first = run(plan().resuming(run_a, mode="weights_only"), storage=storage, name="finetune")
    assert first.asked == 2 and trials("finetune") == 2
    for other in (
        plan().resuming(run_b, mode="weights_only"),
        plan().resuming(run_a, checkpoint="best", mode="weights_only"),
        plan(),
        with_resume(plan(), resume_from_run_id=run_a, resume_mode="weights_only"),
    ):
        with pytest.raises(ValueError, match=r"belongs to a different search \(plan differ\)"):
            run(other, storage=storage, name="finetune", budget=SearchBudget(trials=5))
        assert trials("finetune") == 2
    # ... and a study without a resume is never continued with one.
    run(plan(), storage=storage, name="fresh", budget=SearchBudget(trials=1))
    with pytest.raises(ValueError, match=r"belongs to a different search \(plan differ\)"):
        run(plan().resuming(run_a, mode="weights_only"), storage=storage, name="fresh", budget=SearchBudget(trials=3))
    assert trials("fresh") == 1
    # The same resume continues its study.
    assert (
        run(
            plan().resuming(run_a, mode="weights_only"), storage=storage, name="finetune", budget=SearchBudget(trials=3)
        ).asked
        == 1
    )


# ---------------- per-trial caps ----------------


class StepCount(Callback):
    """Counts optimizer steps and snapshots the net at every committed epoch."""

    steps: list[int] = []
    committed: list[dict] = []

    def on_train_begin(self, ctx) -> None:
        StepCount.steps.append(0)
        original = ctx.optimizer.step

        def counted(*args, **kwargs):
            StepCount.steps[-1] += 1
            return original(*args, **kwargs)

        ctx.optimizer.step = counted

    def on_epoch_end(self, ctx) -> None:
        StepCount.committed.append({k: v.clone() for k, v in ctx.model.net.state_dict().items()})


def test_an_update_cap_binds_and_a_partial_window_is_no_update():
    StepCount.steps, StepCount.committed = [], []
    base = plan(n_epochs=2, batch_size=4, accumulate=3).with_callback_factories(StepCount)  # 9 batches: 3 windows/epoch
    result = run(base, budget=SearchBudget(trials=1, updates_per_trial=2))
    (outcome,) = result.outcomes
    assert StepCount.steps == [2] and outcome.updates == 2  # no third committed step
    assert outcome.reason == "update_cap" and outcome.epochs == 0 and outcome.state == "failed"
    # stopped inside epoch 0: no epoch committed, so no LAST at all
    assert NNCheckpoint.load(run=outcome.run_id, type=Checkpoints.LAST) is None


def test_a_mid_epoch_stop_leaves_the_last_commit_unrelabelled():
    StepCount.steps, StepCount.committed = [], []
    base = plan(n_epochs=3, batch_size=4, accumulate=3).with_callback_factories(StepCount)  # 3 updates per epoch
    result = run(base, budget=SearchBudget(trials=1, updates_per_trial=4))  # stops in epoch 1, after its 1st update
    (outcome,) = result.outcomes
    assert StepCount.steps == [4] and outcome.updates == 4 and outcome.epochs == 1
    assert outcome.state == "completed" and outcome.reason == "update_cap"
    last = NNCheckpoint.load(run=outcome.run_id, type=Checkpoints.LAST)
    assert last is not None and last.idp.epoch_idx == 0
    for name, tensor in StepCount.committed[0].items():  # epoch 0's tensors, not the live ones
        assert torch.equal(last.net_state[name], tensor)
    trained = result.trial_runs[outcome.number]
    assert {idp.epoch_idx for idp in trained.idps} == {0}


class Clock:
    """A clock the tests move; the search reads it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def advancing(clock: Clock, *, at_update: int | None = None, at_end: bool = False, to: float):
    """A callback factory moving ``clock`` to ``to`` at the given point."""

    class Advance(Callback):
        def on_train_begin(self, ctx) -> None:
            if at_update is not None:
                ctx.update_listeners.append(self.on_update)

        def on_update(self, ctx) -> None:
            if ctx.committed_updates == at_update:
                clock.now = to

        def on_train_end(self, ctx) -> None:
            if at_end:
                clock.now = to

    return Advance


def test_an_expired_deadline_starts_no_trial_and_records_the_overrun():
    clock = Clock()
    base = plan().with_callback_factories(advancing(clock, at_end=True, to=12.0))  # past 10 s after trial 0
    result = run(base, budget=SearchBudget(trials=5, deadline_seconds=10.0), clock=clock)
    assert result.asked == 1 and result.stop_reason == "deadline"
    assert result.deadline_overrun_seconds == pytest.approx(2.0)


def test_a_deadline_stops_a_running_trial_at_the_next_boundary():
    clock = Clock()
    base = plan().with_callback_factories(advancing(clock, at_update=1, to=11.0))
    result = run(base, budget=SearchBudget(trials=3, deadline_seconds=10.0), clock=clock)
    (first,) = result.outcomes
    assert first.reason == "deadline" and first.overrun_seconds == pytest.approx(1.0)
    assert first.updates == 1 and first.epochs == 0  # stopped at the update boundary, mid-epoch
    assert result.asked == 1 and result.stop_reason == "deadline"


# ---------------- pruning, non-finite values, selection ----------------


def test_pruning_sees_validation_observations_at_increasing_steps():
    seen = []
    original = _TrialControl.on_epoch_end

    def spy(self, ctx):
        original(self, ctx)
        seen.append(list(self.observations))

    _TrialControl.on_epoch_end = spy
    try:
        result = run(
            plan(n_epochs=3),
            budget=SearchBudget(trials=2),
            pruner=optuna.pruners.ThresholdPruner(upper=0.0),  # every observation prunes
        )
    finally:
        _TrialControl.on_epoch_end = original
    for outcome in result.outcomes:
        assert outcome.state == "pruned" and outcome.reason == "pruned"
        assert [step for step, _ in outcome.observations] == [0]
    assert result.best is None  # pruned trials are never winners
    assert all([step for step, _ in obs] == sorted({step for step, _ in obs}) for obs in seen)


def test_a_non_finite_monitor_fails_the_trial_and_never_wins():
    def poison(base, params):
        bad = torch.full_like(X_VAL, float("nan"))
        return with_lr(base, params).with_data(val=lambda: DataLoader(TensorDataset(bad, Y_VAL), batch_size=6))

    result = run(apply=poison, budget=SearchBudget(trials=2))
    assert {outcome.state for outcome in result.outcomes} == {"failed"}
    assert {outcome.reason for outcome in result.outcomes} == {"non_finite"}
    assert result.best is None


def test_the_best_completed_trial_is_selected_by_the_monitor(monkeypatch):
    values = {0: 0.5, 1: 0.3, 2: 0.4}
    original = _TrialControl.on_epoch_end

    def scripted(self, ctx):
        original(self, ctx)
        if self.observations:
            self.observations[-1] = (self.observations[-1][0], values[self.trial.number])
            self.best = values[self.trial.number]

    monkeypatch.setattr(_TrialControl, "on_epoch_end", scripted)
    result = run(budget=SearchBudget(trials=3))
    assert [outcome.monitor_value for outcome in result.outcomes] == [0.5, 0.3, 0.4]
    best = result.best
    assert best is not None and best.number == 1 and best.state == "completed"
    assert best.run_id in runs_on_disk() and best.checkpoint and os.path.exists(best.checkpoint)
    assert result.direction == "minimize" and result.monitor == "val.loss"


# ---------------- cancellation ----------------


class Interrupt(Callback):
    def __init__(self) -> None:
        self.cleaned = False

    def on_epoch_end(self, ctx) -> None:
        if ctx.epoch == 1:
            raise KeyboardInterrupt

    def on_train_end(self, ctx) -> None:
        Interrupt.cleanup.append(True)


Interrupt.cleanup = []


def test_cancellation_runs_cleanup_releases_the_lease_and_keeps_the_last_commit(tmp_path):
    storage = f"sqlite:///{tmp_path / 'cancel.db'}"
    Interrupt.cleanup = []
    base = plan(n_epochs=3).with_callback_factories(Interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(base, storage=storage, name="cancel", budget=SearchBudget(trials=2))
    assert Interrupt.cleanup == [True]  # on_train_end ran
    study = optuna.load_study(study_name="cancel", storage=storage)
    (trial,) = study.trials
    assert trial.state == optuna.trial.TrialState.FAIL and trial.user_attrs["reason"] == "error:KeyboardInterrupt"
    (run_id,) = runs_on_disk()
    last = NNCheckpoint.load(run=run_id, type=Checkpoints.LAST)
    assert last is not None and last.idp.epoch_idx == 0  # epoch 1 never committed
    from filelock import FileLock

    with FileLock(os.path.join("runs", ".leases", f"{run_id}.lock"), timeout=0):  # the lease was released
        pass


# ---------------- review regressions ----------------


def test_the_trial_value_is_the_one_its_best_checkpoint_was_chosen_by():
    monitor = MonitorSpec("loss", min_delta=10.0)  # only the first epoch ever "improves"
    base = plan(n_epochs=3).with_metrics(monitor=monitor)
    result = run(base, monitor=monitor, budget=SearchBudget(trials=1))
    (outcome,) = result.outcomes
    best = NNCheckpoint.load(run=outcome.run_id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 0
    assert outcome.monitor_value == pytest.approx(best.idp.val_edp.loss)
    assert outcome.monitor_value == outcome.observations[0][1] != min(value for _, value in outcome.observations)


def test_a_failed_fit_is_recorded_with_its_message_and_the_search_goes_on():
    def broken_fit(base, params):
        return with_lr(base, params).with_data(lambda: DataLoader(TensorDataset(X[:, :3], Y), batch_size=6))

    result = run(apply=broken_fit, budget=SearchBudget(trials=2))
    assert result.asked == 2 and {o.state for o in result.outcomes} == {"failed"}
    assert all(o.reason == "error:RuntimeError" and "mat1 and mat2" in o.error for o in result.outcomes)


@pytest.mark.parametrize(
    ("apply", "error", "match"),
    [
        (lambda base, params: (_ for _ in ()).throw(KeyError("typo")), KeyError, "typo"),
        (lambda base, params: "not a plan", TypeError, "ExperimentPlan"),
        (lambda base, params: with_lr(base, params).with_callbacks(Callback()), ValueError, "fresh callbacks"),
        (lambda base, params: with_lr(base, params).with_metrics(monitor=MonitorSpec("error")), ValueError, "monitor"),
    ],
    ids=["apply-raises", "not-a-plan", "borrowed-callbacks", "other-monitor"],
)
def test_a_configuration_error_stops_the_search(tmp_path, apply, error, match):
    storage = f"sqlite:///{tmp_path / 'config.db'}"
    with pytest.raises(error, match=match):
        run(apply=apply, storage=storage, name="config", budget=SearchBudget(trials=3))
    (trial,) = optuna.load_study(study_name="config", storage=storage).trials  # one trial spent, then stopped
    assert trial.state == optuna.trial.TrialState.FAIL and trial.user_attrs["reason"].startswith("config_error:")


def _schedule(kind_name: str, **fields):
    from nnx.nn.enum.schedulers import Schedulers
    from nnx.nn.params.nn_scheduler_params import NNSchedulerParams

    return NNSchedulerParams(
        kind=Schedulers[kind_name], min_lr=1e-6, factor=0.5, patience=1, cooldown=0, threshold=0.0, **fields
    )


def with_scheduler(base: ExperimentPlan, scheduler) -> ExperimentPlan:
    return base.with_train(replace(base.train, scheduler=scheduler))


def test_a_schedule_horizon_the_build_refuses_stops_the_search_before_any_trial(tmp_path):
    """#506: a total_steps shorter than the run fails every trial alike, so
    the base plan is refused before a study, trial or factory call."""
    from nnx.plans import PlanError

    calls = Calls()
    base = with_scheduler(plan(calls), _schedule("ONE_CYCLE", max_lr=0.05, total_steps=1))
    storage = f"sqlite:///{tmp_path / 'horizon.db'}"
    with pytest.raises(PlanError, match="train.scheduler"):
        run(base, storage=storage, name="horizon", budget=SearchBudget(trials=4))
    assert (calls.train, calls.val, calls.callbacks) == (0, 0, 0) and runs_on_disk() == []
    assert optuna.study.get_all_study_names(storage) == []


def test_an_apply_that_varies_the_epochs_past_the_horizon_stops_the_search(tmp_path):
    """#506: apply() turning a valid horizon into a refused one (n_epochs
    past an explicit total_steps) is a configuration error of that trial."""
    from nnx.plans import PlanError

    base = with_scheduler(plan(n_epochs=2), _schedule("ONE_CYCLE", max_lr=0.05, total_steps=2))
    storage = f"sqlite:///{tmp_path / 'epochs.db'}"
    with pytest.raises(PlanError, match="n_epochs"):
        run(base, apply=lambda p, q: with_lr(p, q).with_epochs(3), storage=storage, name="epochs")
    (trial,) = optuna.load_study(study_name="epochs", storage=storage).trials
    assert trial.state == optuna.trial.TrialState.FAIL and trial.user_attrs["reason"] == "config_error:PlanError"


def test_a_factory_returning_a_dataset_stops_the_search_and_a_fixed_rerun_continues(tmp_path):
    """#506: fit()'s PlanError for what a factory returned is a configuration
    error: one trial, recorded as config_error:PlanError, then the search
    stops — so a rerun with the fixed factory still has budget to ask."""
    from nnx.plans import PlanError

    calls = Calls()
    broken = plan(calls).with_data(lambda: TensorDataset(X, Y))  # forgot the DataLoader
    assert broken.validate().ok  # the factory is called only by fit()
    storage = f"sqlite:///{tmp_path / 'factory.db'}"
    with pytest.raises(PlanError, match="TensorDataset"):
        run(broken, storage=storage, name="factory", budget=SearchBudget(trials=3))
    (trial,) = optuna.load_study(study_name="factory", storage=storage).trials  # one trial spent, then stopped
    assert trial.state == optuna.trial.TrialState.FAIL
    assert trial.user_attrs["reason"] == "config_error:PlanError"
    assert "wrap it in a DataLoader" in trial.user_attrs["error"]
    assert calls.val == 1 and runs_on_disk() == []
    fixed = run(plan(), storage=storage, name="factory", budget=SearchBudget(trials=3))
    assert fixed.asked == 2 and fixed.best is not None  # the same stored study, with budget left


class _Stream(torch.utils.data.IterableDataset):
    """A training stream without a length."""

    def __iter__(self):
        yield from zip(X.split(6), Y.split(6), strict=True)


def _early_stopping_on_accuracy():
    from nnx.nn.callbacks import EarlyStopping

    return EarlyStopping(monitor=MonitorSpec(metric="accuracy", split="val"), patience=1)  # not declared


@pytest.mark.parametrize(
    ("change", "path"),
    [
        (lambda p: p.with_data(lambda: iter(DataLoader(TensorDataset(X, Y), batch_size=6))), "data.train"),
        (lambda p: p.with_callback_factories(lambda: 3), "callback_factories[0]"),
        (lambda p: p.with_callback_factories(_early_stopping_on_accuracy), "callback_factories[0].monitor"),
        (
            lambda p: with_scheduler(p, _schedule("COSINE_ANNEALING", clock="optimizer_update")).with_data(
                lambda: DataLoader(_Stream(), batch_size=None)
            ),
            "train.scheduler",
        ),
        (
            lambda p: with_scheduler(p, _schedule("ONE_CYCLE", max_lr=0.05, total_steps=5, clock="optimizer_update")),
            "train.scheduler",
        ),
    ],
    ids=["data-factory-output", "callback-factory-output", "factory-callback-monitor", "no-budget", "short-budget"],
)
def test_every_fit_time_plan_error_stops_the_search(tmp_path, change, path):
    """#506: each PlanError fit() raises for what the factories made stops
    the search after its trial; a fit's other errors stay failed trials
    (test_a_failed_fit_is_recorded_with_its_message_and_the_search_goes_on)."""
    from nnx.plans import PlanError

    base = change(plan())  # 6 batches of the train factory: 12 planned updates over 2 epochs
    assert base.validate().ok
    storage = f"sqlite:///{tmp_path / 'fit.db'}"
    with pytest.raises(PlanError) as caught:
        run(base, storage=storage, name="fit", budget=SearchBudget(trials=3))
    assert path in [d.path for d in caught.value.diagnostics]
    (trial,) = optuna.load_study(study_name="fit", storage=storage).trials
    assert trial.state == optuna.trial.TrialState.FAIL and trial.user_attrs["reason"] == "config_error:PlanError"
    assert runs_on_disk() == []


def test_an_update_cap_is_refused_with_a_custom_step():
    from nnx.nn.nn_model import default_train_step

    base = plan().with_step_fns(train_step_fn=lambda ctx: default_train_step(ctx))
    with pytest.raises(ValueError, match="updates_per_trial"):
        run(base, budget=SearchBudget(trials=1, updates_per_trial=2))


def test_an_exhausted_grid_stops_the_search_without_spending_budget():
    grid = optuna.samplers.GridSampler({"lr": [0.01, 0.02]}, seed=0)
    space = SearchSpace(CategoricalParam(name="lr", choices=(0.01, 0.02)))
    result = run(space=space, sampler=grid, budget=SearchBudget(trials=5))
    assert result.asked == 2 and result.stop_reason == "exhausted"
    assert sorted(o.params["lr"] for o in result.outcomes) == [0.01, 0.02]


def test_a_search_over_lr_on_a_resuming_plan_never_ranks_trials_that_trained_alike():
    """#502: on a stateful resuming plan every trial would train with the
    source's learning rate (the checkpoint's param groups) while its run.yaml
    recorded the trial's own, so the search would rank identical runs. The
    search refuses a stateful resume before any trial (#505); a weights-only
    resume trains each at its lr."""
    from nnx.nn.params.nn_run import NNRun

    source = plan(n_epochs=1).fit().run
    space = SearchSpace(CategoricalParam(name="lr", choices=(0.5, 1e-7)))

    def grid():
        return optuna.samplers.GridSampler({"lr": [0.5, 1e-7]}, seed=0)

    before = runs_on_disk()
    with pytest.raises(ValueError, match=r"resume_mode='auto'.*mode='weights_only'"):
        run(plan().resuming(source.id), space=space, sampler=grid(), name="stateful")
    assert runs_on_disk() == before
    weights = run(plan().resuming(source.id, mode="weights_only"), space=space, sampler=grid(), name="weights")
    completed = [o for o in weights.outcomes if o.state == "completed"]
    assert len(completed) == 2
    for outcome in completed:
        assert outcome.run_id is not None
        assert NNRun.load(id=outcome.run_id).idps[0].lr == outcome.params["lr"]  # trained at its own lr
    assert completed[0].monitor_value != completed[1].monitor_value


def test_the_missing_extra_is_named(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(ImportError, match=r'pip install "thekaveh-nnx\[optuna\]"'):
        run(budget=SearchBudget(trials=1))


def test_space_identity_is_normalized():
    assert FloatParam(name="x", low=0, high=1).state() == FloatParam(name="x", low=0.0, high=1.0).state()
    assert "log" not in FloatParam(name="x", low=0, high=1).state()
    assert "step" not in IntParam(name="n", low=1, high=4).state()


def test_a_journal_history_keeps_its_committed_window_after_a_mid_epoch_stop():
    from nnx.history import HistoryJournal
    from nnx.nn.nn_model import NNModel

    seen = {}

    class StopInEpochOne(Callback):
        def on_train_begin(self, ctx) -> None:
            ctx.update_listeners.append(self.on_update)

        def on_update(self, ctx) -> None:
            if ctx.committed_updates == 6 + 2:  # 6 updates per epoch: two into epoch 1
                ctx.stop_at_update = True

        def on_train_end(self, ctx) -> None:
            seen["idps"] = [(idp.epoch_idx, idp.batch_idx) for idp in ctx.idps]

    torch.manual_seed(0)
    model = NNModel(net_params=NET, params=MODEL)
    run_ = model.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=DataLoader(TensorDataset(X, Y), batch_size=6),
            optim=NNOptimParams.builder().sgd(max_lr=0.05).build(),
        ),
        callbacks=[StopInEpochOne()],
        history=HistoryJournal(retention=3, chunk_size=1),
    )
    committed = [(0, 3), (0, 4), (0, 5)]  # the window's last three records of epoch 0
    assert [(idp.epoch_idx, idp.batch_idx) for idp in run_.idps] == committed
    assert seen["idps"] == committed
    last = NNCheckpoint.load(run=run_.id, type=Checkpoints.LAST)
    assert last is not None and last.idp.epoch_idx == 0


def _world(monkeypatch, size: int) -> None:
    """Pretend a torchrun process group of ``size`` ranks is initialized."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: size)


def test_search_refuses_inside_a_multi_rank_process_group(tmp_path, monkeypatch):
    """FIX-028: under torchrun every rank would run its own study; the search
    is sequential and single-process, so it refuses before any study,
    factory or run exists."""
    _world(monkeypatch, 2)
    calls = Calls()
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    with pytest.raises(RuntimeError, match="single process.*world size 2"):
        run(plan(calls), storage=storage)
    assert not (tmp_path / "study.db").exists()
    assert (calls.train, calls.val, calls.callbacks) == (0, 0, 0)
    assert runs_on_disk() == []


def test_trial_control_declares_no_rank_behaviour():
    """Trials never run under DDP (the search refuses a multi-rank group), so
    the per-trial callback claims no distributed behaviour."""
    assert "distributed" not in vars(_TrialControl)


@pytest.mark.parametrize("choices", [(True, 1, 1.0), (0, False)])
def test_categorical_choices_that_compare_equal_are_refused(choices):
    """optuna stores a choice by its index: equal choices would collapse, so
    a trial would report a value it did not run with."""
    with pytest.raises(ValueError, match="distinct"):
        CategoricalParam(name="c", choices=choices)


def test_an_update_cap_binds_with_the_default_step_passed_explicitly():
    from nnx.nn.nn_model import default_train_step

    base = plan(n_epochs=2, batch_size=4, accumulate=3).with_step_fns(train_step_fn=default_train_step)
    result = run(base, budget=SearchBudget(trials=1, updates_per_trial=2))
    (outcome,) = result.outcomes
    assert outcome.updates == 2 and outcome.reason == "update_cap"
