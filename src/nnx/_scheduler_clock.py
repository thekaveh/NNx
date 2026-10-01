"""Optimizer-update scheduler clocks (FEAT-014). Internal.

A scheduler configured with ``clock="optimizer_update"`` steps once per
*committed* update of the optimizer it belongs to — never per microbatch, an
all-masked window or a skipped AMP step. The update source is explicit:

- the shared update engine (FEAT-004) for an objective run;
- ``default_train_step`` (never for a step the AMP scaler skipped) and
  ``finalize_step`` (so every built-in paradigm step), after each optimizer
  step they take;
- a custom step function that calls ``ctx.report_update()`` (``NNModel``)
  or ``ctx.report_update(name)`` (``Trainer``) after each optimizer step it
  takes itself — NNx never infers updates around an opaque step.

Whoever calls ``optimizer.step()`` reports it, and every report counts: a
step function that delegates its optimizer step to ``default_train_step`` or
``finalize_step`` does not report that step again.

A :class:`SchedulerClock` steps the scheduler after each committed update of
its optimizer while attached (``Trainer``'s ``auto_step_schedulers=False``
detaches it: the step function then owns every scheduler step). Its position
is the scheduler's own step count, restored with the scheduler's state; its
owner, clock and budget are checkpointed component state, so a stateful
resume refuses a mismatched configuration before anything steps.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from collections.abc import Mapping
from typing import Any, Optional

from torch.optim import lr_scheduler

CLOCK = "optimizer_update"
# Scheduler kinds with a hard ``total_steps`` budget (shared with the
# resume-horizon check in ``nn_model``).
HORIZON_KINDS = frozenset({"one_cycle", "linear_warmup_decay"})


def refuse_optimizer_name(names: tuple[Any, ...]) -> None:
    """``NNModel.train`` trains one optimizer: its steps report without a
    name, and a name is refused rather than ignored."""
    if names:
        raise TypeError(
            f"report_update() takes no optimizer name in NNModel.train (it trains one optimizer); got "
            f"{names[0]!r}. Trainer step functions report by name"
        )


class _NoUpdateListener:
    """The ``report_update`` of a step context nothing listens to (built
    outside a training loop, or a run without an update clock): a report
    does nothing. ``refuse_names`` holds an ``NNModel`` step to its
    nameless form whatever the clock (a stable ``repr`` keeps generated
    signatures deterministic)."""

    def __init__(self, *, refuse_names: bool = False) -> None:
        self.refuse_names = refuse_names

    def __call__(self, *names: Any) -> None:
        if self.refuse_names:
            refuse_optimizer_name(names)

    def __repr__(self) -> str:
        return "<no update listener>"


# ``TrainerStepContext`` (reports by name) and ``TrainStepContext`` (no name).
NO_UPDATE_LISTENER = _NoUpdateListener()
NO_UPDATE_REPORTER = _NoUpdateListener(refuse_names=True)


def listens(report_update: Any) -> bool:
    """Whether a context's ``report_update`` reaches a clock (a step can
    then skip work, such as host syncs, only a listener needs)."""
    return not isinstance(report_update, _NoUpdateListener)


def uses_update_clock(scheduler_params: Any) -> bool:
    return getattr(scheduler_params, "clock", "epoch") == CLOCK


def planned_updates(loader: Any, window: int, n_epochs: int) -> Optional[int]:
    """The run's planned committed updates when NNx owns the windows:
    ``ceil(len(loader) / window)`` per epoch (a short final window still
    commits); ``None`` for a loader without a length. At least 1, so an
    empty loader builds a valid schedule and the loop reports it."""
    try:
        batches = len(loader)
    except TypeError:
        return None
    return max(1, n_epochs * math.ceil(batches / max(1, window)))


def update_horizon(scheduler_params: Any, planned: Optional[int] = None) -> Optional[int]:
    """The hard budget an update-clock scheduler may not step past: a
    one-cycle / warmup-decay schedule's ``total_steps``, or the planned
    updates it defaults to (``None`` for the open-ended kinds)."""
    if str(getattr(scheduler_params, "kind", None)) not in HORIZON_KINDS:
        return None
    if scheduler_params.total_steps is not None:
        return scheduler_params.total_steps
    return planned


def chosen_period(scheduler: Any, scheduler_params: Any, planned: Optional[int]) -> Optional[int]:
    """The cosine period a run built on purpose: the built scheduler's
    ``T_max``, unless it is the planned-updates default the configuration
    left it to (``None`` then, and for a non-cosine scheduler). A subclass
    building its own period is its choice, whatever the configuration."""
    if not isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
        return None
    period = int(scheduler.T_max)
    left_to_default = (
        str(getattr(scheduler_params, "kind", None)) == "cosine_annealing" and scheduler_params.T_max is None
    )
    return None if left_to_default and period == planned else period


def component_name(owner: Optional[str] = None) -> str:
    """The checkpointed component name of an optimizer's clock: a
    filename-safe slug of its name, with a short hash of the original when
    characters had to be replaced (so two names never collide)."""
    if owner is None:
        return "nnx.scheduler_clock"
    slug = re.sub(r"[^A-Za-z0-9._:-]", "_", owner)
    if slug != owner:
        slug = f"{slug}-{hashlib.sha256(owner.encode()).hexdigest()[:8]}"
    return f"nnx.scheduler_clock.{slug}"


class SchedulerClock:
    """Steps one optimizer's scheduler once per committed update of that
    optimizer. Its position is the scheduler's own step count."""

    def __init__(
        self,
        owner: str,
        scheduler: Any,
        *,
        horizon: Optional[int],
        planned: Optional[int] = None,
        attached: bool = True,
        component_name: str = "nnx.scheduler_clock",
        default_budget: bool = False,
        configured_period: Optional[int] = None,
    ) -> None:
        # The clock steps its scheduler without a metric and positions it by
        # last_epoch: a plateau scheduler, or one without last_epoch (such as
        # torch's ChainedScheduler), cannot run on it.
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau) or not hasattr(scheduler, "last_epoch"):
            reason = (
                "reads a monitored metric at the epoch boundary"
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau)
                else "has no last_epoch to position it by"
            )
            raise ValueError(
                f"optimizer {owner!r}'s scheduler {type(scheduler).__name__} {reason}, so it cannot run on the "
                "optimizer_update clock; use clock='epoch' for it"
            )
        self.owner = owner
        self.scheduler = scheduler
        self.horizon = horizon
        self.planned = planned
        self.attached = attached
        self.component_name = component_name
        # Whether the horizon is the planned updates (total_steps unset), so
        # an overrun names len(train_loader) as its source.
        self.default_budget = default_budget
        # A cosine schedule warns once when it passes its live T_max (past
        # it the rate climbs back up), unless that is the period this run
        # chose — an explicit T_max, or a subclass's own — which torch lets
        # a schedule run past on purpose. Live, because a stateful resume
        # restores the checkpoint's T_max.
        self.configured_period = configured_period
        self._period_warned = False
        # (scheduler step, learning rate after it) since the epoch began.
        self.trace: list[tuple[int, float]] = []

    @classmethod
    def for_schedule(
        cls, owner: str, scheduler: Any, scheduler_params: Any, *, planned: Optional[int], **options: Any
    ) -> SchedulerClock:
        """A clock for ``scheduler`` configured by ``scheduler_params``, its
        horizon the configured ``total_steps`` or, unset, the ``planned``
        updates."""
        horizon = update_horizon(scheduler_params, planned)
        return cls(
            owner,
            scheduler,
            horizon=horizon,
            planned=planned,
            default_budget=horizon is not None and scheduler_params.total_steps is None,
            configured_period=chosen_period(scheduler, scheduler_params, planned),
            **options,
        )

    @property
    def count(self) -> int:
        """The scheduler steps taken — one per committed update while the
        clock is attached — restored with the scheduler's own state."""
        return int(self.scheduler.last_epoch)

    def committed(self) -> None:
        """One committed update of this clock's optimizer: step the
        scheduler, unless the clock is detached."""
        if not self.attached:
            return
        if self.horizon is not None and self.count >= self.horizon:
            if self.default_budget:
                raise ValueError(
                    f"optimizer {self.owner!r} committed update {self.count + 1}, beyond its scheduler's default "
                    f"budget of {self.horizon} optimizer updates, planned from len(train_loader); the loader "
                    "yielded more batches than its len() reports (an IterableDataset read by several workers "
                    "can), so set total_steps to cover every update of the run"
                )
            raise ValueError(
                f"optimizer {self.owner!r} committed update {self.count + 1}, beyond its scheduler's budget of "
                f"{self.horizon} optimizer updates (total_steps); set total_steps to cover every update of the run"
            )
        self.scheduler.step()
        self.trace.append((self.count, float(self.scheduler.optimizer.param_groups[0]["lr"])))
        if not self._period_warned and isinstance(self.scheduler, lr_scheduler.CosineAnnealingLR):
            period = self.scheduler.T_max  # the live period, a restored one included
            if self.count > period and period != self.configured_period:
                self._period_warned = True
                warnings.warn(
                    f"optimizer {self.owner!r}'s cosine schedule passed its T_max of {period} optimizer updates, "
                    "so its learning rate now rises again; set T_max to cover every update of the run (a "
                    "stateful resume keeps the checkpoint's T_max, so set it in the run that starts the schedule)",
                    UserWarning,
                    stacklevel=2,
                )

    def report_update(self, *names: Any) -> None:
        """``TrainStepContext.report_update()`` in ``NNModel.train``: one
        committed update of the run's one optimizer. A name is refused
        rather than ignored (``Trainer`` steps report by name)."""
        refuse_optimizer_name(names)
        self.committed()

    # ---------- checkpointable component (FEAT-005) ----------

    def component_spec(self) -> Any:
        from .components import ComponentSpec

        return ComponentSpec(self.component_name, version=1, required=True)

    def component_state(self) -> dict[str, Any]:
        return {"clock": CLOCK, "owner": self.owner, "count": self.count, "horizon": self.horizon}

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        problems: list[str] = []
        if state.get("clock") != CLOCK or state.get("owner") != self.owner:
            problems.append(
                f"the checkpoint's scheduler clock is {state.get('clock')!r} owned by {state.get('owner')!r}, this "
                f"run's is {CLOCK!r} owned by {self.owner!r}"
            )
        if state.get("horizon") != self.horizon:
            problems.append(
                f"the checkpoint's scheduler budget is {state.get('horizon')} optimizer updates, this run's is "
                f"{self.horizon}; configure the same total_steps"
            )
        count = state.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            problems.append(f"malformed update count {count!r}")
        elif self.horizon is not None and self.planned is not None and count + self.planned > self.horizon:
            problems.append(
                f"resuming at update {count} with {self.planned} planned updates would pass the scheduler's budget "
                f"of {self.horizon} (total_steps); configure one horizon covering the original and resumed updates"
            )
        return problems

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        # The position is the scheduler's own step count, restored with the
        # scheduler state; this component only guards the configuration.
        return None
