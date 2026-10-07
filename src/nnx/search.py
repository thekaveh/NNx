"""Budgeted experiment search through an optional Optuna adapter (FEAT-033).

``search(plan, space, apply=..., monitor=..., budget=..., study_name=...)``
asks an Optuna study for parameter sets one at a time; ``apply(plan,
params)`` turns each into a **fresh** :class:`~nnx.plans.ExperimentPlan`
(the base plan is never changed), which is fitted — a new model, optimizer,
loaders and callbacks per trial, its own attempt id and run directory — and
scored by one declared **validation** monitor. Requires the ``optuna``
extra (``pip install "thekaveh-nnx[optuna]"``); importing ``nnx`` never
imports Optuna or creates a study.

- **Validated first.** The space (bounds, steps, log scales, choices), the
  budget, the monitor (a validation monitor with a direction, equal to the
  plan's own ``NNTrainParams.monitor`` so the BEST checkpoint is chosen the
  same way) and the plan (validation data, data and callbacks as factories)
  are checked before any trial; reopening a stored study whose space,
  monitor or plan identity differs is refused before any trial too.
- **The budget counts every trial.** Completed, failed, pruned and orphaned
  (left running by a crashed process) trials all count against
  ``SearchBudget.trials``: a study holding two trials, searched with a
  budget of 3, asks once.
- **Per-trial caps.** ``updates_per_trial`` stops a trial right after that
  many committed optimizer updates (a partial accumulation window is not an
  update). ``deadline_seconds`` (measured on ``clock``, from the search's
  start) starts no new trial once expired and stops a running one at its
  next update or epoch boundary, recording the overrun. Neither is
  preemptive: a long batch finishes first. A trial stopped mid-epoch keeps
  only its committed epochs — LAST, its tensors and the history are those
  of the last completed epoch (no LAST before a first completed one).
- **Pruning** (Optuna's ``pruner``) sees only the monitor's validation
  value of each completed epoch, reported at step = the epoch index
  (strictly increasing). A missing or non-finite value fails the trial; it
  is never a candidate. A trial's value is the run's own monitor decision:
  the value of the epoch its BEST checkpoint holds (``min_delta`` included).
- **Errors.** A fit that raises fails its trial (``TrialOutcome.error``
  keeps the message) and the search continues; a configuration error —
  ``apply`` raising or returning a plan the search cannot score — stops the
  search. An exhausted sampler ends it (``stop_reason="exhausted"``).
- **Outcomes.** Every trial's :class:`TrialOutcome` keeps its parameters,
  state and terminal reason, resource counters (epochs, committed updates,
  seconds), best monitor value, attempt id, run id and BEST checkpoint path.
  :attr:`SearchResult.best` is the completed trial with the best finite
  monitor value under the monitor's direction.
- **Cancellation** (``KeyboardInterrupt``) runs the fit's callback cleanup,
  releases its run lease, records the trial as failed and propagates.

Trials run sequentially in this process, which is the study's only
writer: inside a ``torch.distributed`` process group of more than one rank
(a ``torchrun`` launch) ``search`` raises ``RuntimeError`` before loading or
creating a study, rather than run one study per rank. NNx artifacts live
under ``runs/`` and Optuna's in ``storage`` (an in-memory study when
``None``). Samplers keep their own limits (a seeded
``TPESampler`` is reproducible only sequentially).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Union

from .nn.callbacks import Callback

__all__ = [
    "CategoricalParam",
    "FloatParam",
    "IntParam",
    "SearchBudget",
    "SearchResult",
    "SearchSpace",
    "TrialOutcome",
    "search",
]

_STUDY_ATTR = "nnx.search"


def _optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            'nnx.search needs Optuna: pip install "thekaveh-nnx[optuna]" (or pip install optuna)'
        ) from error
    return optuna


def _name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"a search parameter needs a non-empty name, got {value!r}")
    return value


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True, kw_only=True, slots=True)
class FloatParam:
    """A float in ``[low, high]``: ``log`` samples on a log scale (``low >
    0``), ``step`` discretizes (not with ``log``)."""

    name: str
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None

    def __post_init__(self) -> None:
        _name(self.name)
        low, high = _finite(self.low, f"{self.name}.low"), _finite(self.high, f"{self.name}.high")
        if low >= high:
            raise ValueError(f"{self.name}: low must be < high, got [{low}, {high}]")
        if not isinstance(self.log, bool):
            raise TypeError(f"{self.name}.log must be a bool")
        if self.log and low <= 0:
            raise ValueError(f"{self.name}: a log scale needs low > 0, got {low}")
        if self.step is not None:
            step = _finite(self.step, f"{self.name}.step")
            if step <= 0 or self.log:
                raise ValueError(f"{self.name}: step must be > 0 and cannot combine with log")

    def state(self) -> dict[str, Any]:
        """Fields that differ from their defaults; bounds as floats (so
        ``low=0`` and ``low=0.0`` are one space)."""
        state: dict[str, Any] = {"kind": "float", "name": self.name, "low": float(self.low), "high": float(self.high)}
        if self.log:
            state["log"] = True
        if self.step is not None:
            state["step"] = float(self.step)
        return state

    def suggest(self, trial: Any) -> float:
        return trial.suggest_float(self.name, self.low, self.high, log=self.log, step=self.step)


@dataclass(frozen=True, kw_only=True, slots=True)
class IntParam:
    """An int in ``[low, high]`` (inclusive): ``log`` (``low >= 1``) or a
    ``step``."""

    name: str
    low: int
    high: int
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        _name(self.name)
        for what, value in (("low", self.low), ("high", self.high), ("step", self.step)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{self.name}.{what} must be an int, got {value!r}")
        if self.low >= self.high:
            raise ValueError(f"{self.name}: low must be < high, got [{self.low}, {self.high}]")
        if self.step < 1:
            raise ValueError(f"{self.name}: step must be >= 1")
        if not isinstance(self.log, bool):
            raise TypeError(f"{self.name}.log must be a bool")
        if self.log and (self.low < 1 or self.step != 1):
            raise ValueError(f"{self.name}: a log scale needs low >= 1 and step 1")

    def state(self) -> dict[str, Any]:
        """Fields that differ from their defaults."""
        state: dict[str, Any] = {"kind": "int", "name": self.name, "low": self.low, "high": self.high}
        if self.log:
            state["log"] = True
        if self.step != 1:
            state["step"] = self.step
        return state

    def suggest(self, trial: Any) -> int:
        return trial.suggest_int(self.name, self.low, self.high, log=self.log, step=self.step)


@dataclass(frozen=True, kw_only=True, slots=True)
class CategoricalParam:
    """One of ``choices`` (distinct ``None`` / bool / int / float / str)."""

    name: str
    choices: tuple[Any, ...]

    def __post_init__(self) -> None:
        _name(self.name)
        choices = tuple(self.choices)
        if len(choices) < 1:
            raise ValueError(f"{self.name}: choices must not be empty")
        if not all(choice is None or isinstance(choice, (bool, int, float, str)) for choice in choices):
            raise TypeError(f"{self.name}: choices must be None, bool, int, float or str")
        if len({repr(choice) for choice in choices}) != len(choices):
            raise ValueError(f"{self.name}: choices must be distinct")
        object.__setattr__(self, "choices", choices)

    def state(self) -> dict[str, Any]:
        return {"kind": "categorical", "name": self.name, "choices": list(self.choices)}

    def suggest(self, trial: Any) -> Any:
        return trial.suggest_categorical(self.name, list(self.choices))


SearchParam = Union[FloatParam, IntParam, CategoricalParam]


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """The parameters a trial draws, by unique name."""

    params: tuple[SearchParam, ...]

    def __init__(self, *params: SearchParam) -> None:
        if not params:
            raise ValueError("a search space needs at least one parameter")
        for param in params:
            if not isinstance(param, (FloatParam, IntParam, CategoricalParam)):
                raise TypeError(f"search parameters are FloatParam / IntParam / CategoricalParam, got {param!r}")
        names = [param.name for param in params]
        if len(set(names)) != len(names):
            raise ValueError(f"search parameter names must be unique, got {names}")
        object.__setattr__(self, "params", tuple(params))

    def state(self) -> list[dict[str, Any]]:
        return [param.state() for param in self.params]

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True, slots=True)
class SearchBudget:
    """``trials`` — the study's total (every state counts);
    ``updates_per_trial`` — committed optimizer updates per trial;
    ``deadline_seconds`` — wall time from the search's start."""

    trials: int
    updates_per_trial: Optional[int] = None
    deadline_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.trials, bool) or not isinstance(self.trials, int) or self.trials < 1:
            raise ValueError(f"SearchBudget.trials must be an int >= 1, got {self.trials!r}")
        if self.updates_per_trial is not None and (
            isinstance(self.updates_per_trial, bool)
            or not isinstance(self.updates_per_trial, int)
            or self.updates_per_trial < 1
        ):
            raise ValueError(f"SearchBudget.updates_per_trial must be an int >= 1, got {self.updates_per_trial!r}")
        if self.deadline_seconds is not None and _finite(self.deadline_seconds, "deadline_seconds") <= 0:
            raise ValueError(f"SearchBudget.deadline_seconds must be > 0, got {self.deadline_seconds!r}")


@dataclass(frozen=True, kw_only=True, slots=True)
class TrialOutcome:
    """One trial: its Optuna ``number``, ``params``, ``state``
    (``"completed"``, ``"pruned"``, ``"failed"`` or ``"running"`` for an
    orphan) and terminal ``reason``; the resources it used (``epochs``,
    committed ``updates``, ``seconds``, ``overrun_seconds`` past the
    deadline); its best finite ``monitor_value``, the validation
    ``observations`` ``(epoch, value)`` pruning saw; and its ``attempt_id``,
    ``run_id`` and BEST ``checkpoint`` path."""

    number: int
    params: Mapping[str, Any]
    state: str
    reason: str
    monitor_value: Optional[float] = None
    epochs: int = 0
    updates: int = 0
    seconds: float = 0.0
    overrun_seconds: Optional[float] = None
    observations: tuple[tuple[int, float], ...] = ()
    attempt_id: Optional[str] = None
    run_id: Optional[str] = None
    checkpoint: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True, kw_only=True, slots=True)
class SearchResult:
    """Every trial of the study (priors included), the ``best`` completed
    one, how many this call asked, and why it stopped (``"budget"``,
    ``"deadline"`` or ``"exhausted"`` — a sampler ran out of points)."""

    study_name: str
    outcomes: tuple[TrialOutcome, ...]
    best: Optional[TrialOutcome]
    asked: int
    stop_reason: str
    direction: str
    monitor: str
    deadline_overrun_seconds: Optional[float] = None
    trial_runs: Mapping[int, Any] = field(default_factory=dict)


class _TrialControl(Callback):
    """The per-trial callback: caps, deadline, monitor observations and
    pruning. Built fresh for every trial."""

    def __init__(
        self, *, trial: Any, monitor: Any, budget: SearchBudget, deadline: Optional[float], clock: Callable[[], float]
    ) -> None:
        self.trial = trial
        self.monitor = monitor
        self.budget = budget
        self.deadline = deadline
        self.clock = clock
        self.observations: list[tuple[int, float]] = []
        self.best: Optional[float] = None  # the value of the epoch BEST holds
        self.best_epoch: Optional[int] = None
        self.non_finite: Optional[tuple[int, float]] = None
        self.missing = False
        self.pruned = False
        self.reason: Optional[str] = None
        self.overrun: Optional[float] = None
        self.updates = 0
        self.epochs = 0

    def on_train_begin(self, ctx: Any) -> None:
        ctx.update_listeners.append(self.on_update)

    def _past_deadline(self) -> bool:
        if self.deadline is None:
            return False
        now = self.clock()
        if now >= self.deadline:
            self.overrun = now - self.deadline
            return True
        return False

    def on_update(self, ctx: Any) -> None:
        self.updates = ctx.committed_updates
        cap = self.budget.updates_per_trial
        if cap is not None and self.updates >= cap:
            ctx.stop_at_update = True
            self.reason = self.reason or "update_cap"
        if self._past_deadline():
            ctx.stop_at_update = True
            self.reason = "deadline"

    def on_epoch_end(self, ctx: Any) -> None:
        self.epochs += 1
        idp = ctx.idp
        value = self.monitor.value(train=idp.train_edp, val=idp.val_edp) if idp is not None else None
        if value is None or not math.isfinite(value):
            self.missing = value is None
            self.non_finite = (ctx.epoch, float("nan") if value is None else float(value))
            ctx.should_stop = True
            return
        self.observations.append((ctx.epoch, float(value)))
        # The trial's value is the one its BEST checkpoint was chosen by: the
        # run's monitor decision (min_delta and tie rules included).
        selection = getattr(idp, "selection", None)
        if selection is not None:
            if selection.improved:
                self.best, self.best_epoch = float(value), ctx.epoch
        elif self.best is None or (value < self.best if self.monitor.mode == "min" else value > self.best):
            self.best, self.best_epoch = float(value), ctx.epoch
        self.trial.report(float(value), step=ctx.epoch)
        if self.trial.should_prune():
            self.pruned = True
            ctx.should_stop = True
        elif self._past_deadline():
            self.reason = "deadline"
            ctx.should_stop = True


def _plan_identity(plan: Any) -> str:
    """The plan configuration a stored study belongs to."""
    parts = {
        "net": None if plan.net is None else plan.net.state(),
        "model": None if plan.model is None else plan.model.state(),
        "train": None if plan.train is None else plan.train.state(),
        "seed": plan.seed,
        "data": None if plan.data_identity is None else str(plan.data_identity),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _check_plan(plan: Any, monitor: Any, budget: Optional[SearchBudget] = None) -> Any:
    from .monitors import MonitorSpec
    from .plans import ExperimentPlan, _is_factory

    if not isinstance(plan, ExperimentPlan):
        raise TypeError(f"search needs an nnx.plans.ExperimentPlan, got {type(plan).__name__}")
    if not isinstance(monitor, MonitorSpec):
        raise TypeError(f"monitor must be an nnx.monitors.MonitorSpec, got {type(monitor).__name__}")
    if monitor.split != "val":
        raise ValueError(
            "search scores trials by a validation monitor (MonitorSpec(split='val')): never a training or last-batch "
            "value"
        )
    plan.validate().raise_for_errors()
    train = plan.train
    assert train is not None  # a valid plan has training parameters
    declared = tuple(train.metrics or ())
    resolved = monitor.resolve(declared, owner="search monitor")
    if train.monitor is None or train.monitor.resolve(declared) != resolved:
        raise ValueError(
            f"the plan's NNTrainParams.monitor ({train.monitor!r}) must equal the search monitor ({resolved!r}), so "
            "each trial's BEST checkpoint is chosen by the same monitor and direction"
        )
    sources = plan._sources(train)
    if sources[1] is None:
        raise ValueError("search needs validation data in the plan (with_data(val=...))")
    if not all(_is_factory(source) for source in sources if source is not None):
        raise ValueError(
            "each trial builds fresh loaders: pass the plan's data as zero-argument factories "
            "(with_data(train=lambda: ..., val=lambda: ...))"
        )
    if plan.callbacks:
        raise ValueError(
            "each trial builds fresh callbacks: pass them as factories (with_callback_factories), not borrowed "
            "instances (with_callbacks)"
        )
    if budget is not None and budget.updates_per_trial is not None and plan.train_step_fn is not None:
        raise ValueError(
            "SearchBudget.updates_per_trial counts the updates NNx commits: a plan with its own train_step_fn may "
            "not report them (ctx.report_update()); use an objective, the default step, or no update cap"
        )
    return resolved


class _TrialFailed(Exception):
    """A trial that ran but has no admissible value (non-finite, or stopped
    before any validated epoch): recorded as failed, never a candidate."""


def _spent(study: Any) -> int:
    """Trials that count against the budget: every state but WAITING
    (enqueued, not yet run) — completed, failed, pruned and orphaned
    (RUNNING) ones alike."""
    return sum(1 for trial in study.get_trials(deepcopy=False) if trial.state.name != "WAITING")


def _state_name(state: Any) -> str:
    return {"COMPLETE": "completed", "PRUNED": "pruned", "FAIL": "failed", "RUNNING": "running", "WAITING": "waiting"}[
        state.name
    ]


def _outcome(frozen: Any) -> TrialOutcome:
    attrs = frozen.user_attrs
    state = _state_name(frozen.state)
    return TrialOutcome(
        number=frozen.number,
        params=MappingProxyType(dict(frozen.params)),
        state=state,
        reason=attrs.get("reason", "orphaned" if state == "running" else state),
        error=attrs.get("error"),
        monitor_value=attrs.get("monitor_value"),
        epochs=int(attrs.get("epochs", 0)),
        updates=int(attrs.get("updates", 0)),
        seconds=float(attrs.get("seconds", 0.0)),
        overrun_seconds=attrs.get("overrun_seconds"),
        observations=tuple(tuple(item) for item in attrs.get("observations", ())),
        attempt_id=attrs.get("attempt_id"),
        run_id=attrs.get("run_id"),
        checkpoint=attrs.get("checkpoint"),
    )


def search(
    plan: Any,
    space: SearchSpace,
    *,
    apply: Callable[[Any, Mapping[str, Any]], Any],
    monitor: Any,
    budget: SearchBudget,
    study_name: str,
    storage: Optional[str] = None,
    sampler: Any = None,
    pruner: Any = None,
    clock: Callable[[], float] = time.monotonic,
) -> SearchResult:
    """Run the budgeted search; see the module documentation."""
    from .distributed import _refuse_multi_rank

    _refuse_multi_rank("nnx.search.search()")  # sequential and single-process: never one study per rank
    if not isinstance(space, SearchSpace):
        raise TypeError(f"space must be an nnx.search.SearchSpace, got {type(space).__name__}")
    if not isinstance(budget, SearchBudget):
        raise TypeError(f"budget must be an nnx.search.SearchBudget, got {type(budget).__name__}")
    if not callable(apply):
        raise TypeError("apply must be a callable (plan, params) -> ExperimentPlan")
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError(f"study_name must be a non-empty string, got {study_name!r}")
    resolved = _check_plan(plan, monitor, budget)
    optuna = _optuna()

    direction = "minimize" if resolved.mode == "min" else "maximize"
    identity = {
        "space": space.digest(),
        "monitor": resolved.state(),
        "direction": direction,
        "plan": _plan_identity(plan),
    }
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )
    stored = study.user_attrs.get(_STUDY_ATTR)
    if stored is None:
        if study.trials:
            raise ValueError(f"study {study_name!r} holds trials NNx did not run: refusing to search it")
        study.set_user_attr(_STUDY_ATTR, identity)
    elif stored != identity:
        changed = sorted(key for key in identity if stored.get(key) != identity[key])
        raise ValueError(f"study {study_name!r} belongs to a different search ({', '.join(changed)} differ)")

    start = clock()
    deadline = None if budget.deadline_seconds is None else start + budget.deadline_seconds
    asked = 0
    stop_reason = "budget"
    deadline_overrun: Optional[float] = None
    trial_runs: dict[int, Any] = {}

    def objective(trial: Any) -> float:
        params = {param.name: param.suggest(trial) for param in space.params}
        return _run_trial(optuna, trial, plan, params, apply, resolved, budget, deadline, clock, trial_runs)

    while _spent(study) < budget.trials:
        now = clock()
        if deadline is not None and now >= deadline:
            stop_reason = "deadline"
            deadline_overrun = now - deadline
            break
        if isinstance(study.sampler, optuna.samplers.GridSampler) and study.sampler.is_exhausted(study):
            stop_reason = "exhausted"  # every grid point has been tried
            break
        asked += 1
        # One trial per call: Optuna records its state (a failure, a prune,
        # an interruption) and the sampler's own bookkeeping.
        # A failed fit is a failed trial and the search goes on; anything else
        # (a broken ``apply``, a plan the search cannot score) stops it.
        study.optimize(objective, n_trials=1, catch=(_TrialFailed,), gc_after_trial=False)
        if getattr(study, "_stop_flag", False):  # a sampler (a GridSampler, exhausted) stopped the study
            stop_reason = "exhausted"
            break
    now = clock()
    if stop_reason == "budget" and deadline is not None and now >= deadline and asked:
        deadline_overrun = now - deadline

    outcomes = tuple(_outcome(frozen) for frozen in study.get_trials(deepcopy=False))
    candidates = [
        outcome
        for outcome in outcomes
        if outcome.state == "completed" and outcome.monitor_value is not None and math.isfinite(outcome.monitor_value)
    ]
    best = None
    if candidates:
        pick = min if direction == "minimize" else max
        best = pick(
            candidates,
            key=lambda outcome: (
                (outcome.monitor_value, outcome.number) if pick is min else (outcome.monitor_value, -outcome.number)
            ),
        )
    return SearchResult(
        study_name=study_name,
        outcomes=outcomes,
        best=best,
        asked=asked,
        stop_reason=stop_reason,
        direction=direction,
        monitor=resolved.key,
        deadline_overrun_seconds=deadline_overrun,
        trial_runs=MappingProxyType(trial_runs),
    )


def _run_trial(
    optuna: Any,
    trial: Any,
    plan: Any,
    params: Mapping[str, Any],
    apply: Callable[[Any, Mapping[str, Any]], Any],
    monitor: Any,
    budget: SearchBudget,
    deadline: Optional[float],
    clock: Callable[[], float],
    trial_runs: dict[int, Any],
) -> float:
    """One trial, as Optuna's objective: its best monitor value, or
    ``TrialPruned`` / an exception (recorded as failed by ``optimize``)."""
    from .nn.enum.checkpoints import Checkpoints
    from .nn.params.nn_checkpoint import _checkpoint_path
    from .plans import ExperimentPlan

    attempt_id = f"search-{trial.number}-{uuid.uuid4().hex}"
    control = _TrialControl(trial=trial, monitor=monitor, budget=budget, deadline=deadline, clock=clock)
    started = clock()

    def record(**attrs: Any) -> None:
        for key, value in {
            "attempt_id": attempt_id,
            "epochs": control.epochs,
            "updates": control.updates,
            "seconds": clock() - started,
            "observations": [list(item) for item in control.observations],
            "overrun_seconds": control.overrun,
            **attrs,
        }.items():
            trial.set_user_attr(key, value)

    try:
        # A configuration problem stops the search (Optuna marks the trial
        # failed, then the error propagates): the plan apply() returns must
        # be one the search can score, with fresh callbacks and loaders.
        trial_plan = apply(plan, MappingProxyType(dict(params)))
        if not isinstance(trial_plan, ExperimentPlan):
            raise TypeError(f"apply must return an ExperimentPlan, got {type(trial_plan).__name__}")
        _check_plan(trial_plan, monitor, budget)
    except Exception as error:
        record(reason=f"config_error:{type(error).__name__}", error=f"{type(error).__name__}: {error}")
        raise
    except BaseException as error:  # an interruption: recorded, then propagated
        record(reason=f"error:{type(error).__name__}", error=f"{type(error).__name__}: {error}")
        raise
    try:
        trial_plan = trial_plan.with_callback_factories(*trial_plan.callback_factories, lambda: control)
        fitted = trial_plan.fit(attempt=attempt_id)
    except Exception as error:
        # A fit that raised is a failed trial, recorded with its message; the
        # search goes on.
        record(reason=f"error:{type(error).__name__}", error=f"{type(error).__name__}: {error}")
        raise _TrialFailed(f"the fit raised {type(error).__name__}: {error}") from error
    except BaseException as error:
        # An interruption (KeyboardInterrupt) is recorded, then propagates.
        record(reason=f"error:{type(error).__name__}", error=f"{type(error).__name__}: {error}")
        raise
    run = fitted.run
    trial_runs[trial.number] = run
    best = control.best
    checkpoint = _checkpoint_path(run.id, Checkpoints.BEST)
    common = {
        "run_id": run.id,
        "checkpoint": checkpoint if os.path.exists(checkpoint) else None,
        "monitor_value": best,
    }
    if control.non_finite is not None:
        record(reason="missing" if control.missing else "non_finite", **common)
        raise _TrialFailed(f"non-finite monitor value at epoch {control.non_finite[0]}")
    if control.pruned:
        record(reason="pruned", **common)
        raise optuna.TrialPruned()
    if best is None:
        record(reason=control.reason or "no_validated_epoch", **common)
        raise _TrialFailed("stopped before any validated epoch: nothing to score")
    record(reason=control.reason or "completed", **common)
    return float(best)
