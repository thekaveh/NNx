"""Objective-based training over one shared update engine (FEAT-004).

An imperative ``train_step_fn`` owns everything — forward, backward,
accumulation, mixed precision, clipping and ``optimizer.step`` — so each
paradigm re-implements (or, through ``finalize_step``, refuses) the update
mechanics. An **objective** only describes the loss; NNx's shared update
engine does the rest, for ``NNModel.train`` and ``Trainer.train`` alike
(``Trainer`` has no mixed-precision setting, so it runs objectives in full
precision)::

    from nnx.objectives import kd_objective

    model.train(params=NNTrainParams(...), objective=kd_objective(teacher, alpha=0.5, temperature=4.0))

For each microbatch an objective returns :class:`LossTerm`\\ s: a
differentiable **numerator** (a sum over the term's samples) and either an
explicit **denominator** (``reduction="mean"``: the number — or total
weight — of samples the numerator sums over) or none (``reduction="sum"``).
The engine never infers a reduction from a scalar. At the end of an update
window (``accumulate_grad_batches`` microbatches, fewer at the epoch's end)
every normalized term is ``Σ numerators / Σ denominators`` over the window
and summed terms are ``Σ numerators``, so uneven microbatches, a short last
window and masked (ignored) targets give exactly the full-batch update. The
engine then runs autocast (around the objective's forward) → unscale →
clip → step, applies the objective's non-finite policy (``"fail"`` raises
before anything is stepped; ``"skip"`` drops the window) and fires one
committed-update event per successful optimizer update, delivered to
``Callback.on_optimizer_update``.

Imperative step functions and ``finalize_step`` are unchanged; a run uses
either a step function or an objective, never both.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import torch

from ._update_engine import REDUCTIONS, UpdateEvent, check_nonfinite_policy

if TYPE_CHECKING:
    from .nn.nn_model import NNModel
    from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

__all__ = [
    "KDObjective",
    "LossTerm",
    "Objective",
    "ObjectiveContext",
    "ObjectiveResult",
    "SupervisedObjective",
    "UpdateEvent",
    "kd_objective",
    "supervised_objective",
]


@dataclass(frozen=True)
class LossTerm:
    """One named part of an objective's loss for one microbatch.

    Args:
        name: identifies the term across the microbatches of a window.
        numerator: differentiable scalar — the **sum** of the term over its
            samples (or, for ``reduction="sum"``, the term's total).
        denominator: for ``reduction="mean"``, the number (or total weight)
            of samples ``numerator`` sums over — ``0`` when every sample is
            masked; must be ``None`` for ``reduction="sum"``.
        reduction: ``"mean"`` (divided by the window-total denominator) or
            ``"sum"`` (the window total). A term keeps its reduction for
            the whole window; mixing them fails.
        weight: finite factor applied to the term's window value.
    """

    name: str
    numerator: torch.Tensor
    denominator: Optional[Union[int, float]] = None
    reduction: str = "mean"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"LossTerm name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.numerator, torch.Tensor) or self.numerator.numel() != 1:
            raise ValueError(f"LossTerm {self.name!r} numerator must be a scalar tensor")
        if self.reduction not in REDUCTIONS:
            raise ValueError(f"LossTerm {self.name!r} reduction must be 'mean' or 'sum', got {self.reduction!r}")
        if self.reduction == "sum":
            if self.denominator is not None:
                raise ValueError(
                    f"LossTerm {self.name!r} is summed (reduction='sum') and takes no denominator; pass "
                    "reduction='mean' to normalize it"
                )
        else:
            denominator = self.denominator
            if (
                denominator is None
                or isinstance(denominator, bool)
                or not isinstance(denominator, numbers.Real)
                or not math.isfinite(denominator)
                or denominator < 0
            ):
                raise ValueError(
                    f"LossTerm {self.name!r} is normalized (reduction='mean') and needs an explicit finite "
                    f"denominator >= 0 (the number of samples its numerator sums over), got {denominator!r}"
                )
            object.__setattr__(self, "denominator", float(denominator))
        weight = self.weight
        if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or not math.isfinite(weight):
            raise ValueError(f"LossTerm {self.name!r} weight must be a finite number, got {weight!r}")
        object.__setattr__(self, "weight", float(weight))

    @cached_property
    def _numerator_value(self) -> float:
        # Read from the device once per term: the engine, ``value`` and
        # ``ObjectiveResult.loss`` all reuse it.
        return float(self.numerator.detach())

    @property
    def value(self) -> Optional[float]:
        """This microbatch's own value (``None`` when fully masked)."""
        numerator = self._numerator_value
        if self.reduction == "sum":
            return numerator
        return numerator / self.denominator if self.denominator else None


@dataclass(frozen=True)
class ObjectiveContext:
    """What an objective sees for one microbatch."""

    model: NNModel
    batch: Any
    epoch_idx: int
    batch_idx: int
    extra_metrics: Optional[Mapping[str, Callable]] = None


@dataclass(frozen=True)
class ObjectiveResult:
    """An objective's output: the loss terms and, optionally, the
    microbatch's metric record (its ``loss`` is filled from the terms when
    absent)."""

    terms: Sequence[LossTerm]
    record: Optional[NNEvaluationDataPoint] = None

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not terms or not all(isinstance(term, LossTerm) for term in terms):
            raise ValueError("ObjectiveResult.terms must be a non-empty sequence of LossTerm")
        object.__setattr__(self, "terms", terms)

    def loss(self) -> Optional[float]:
        """The microbatch's weighted loss over its unmasked terms."""
        values = [(term.weight, term.value) for term in self.terms]
        present = [(w, v) for w, v in values if v is not None]
        return sum(w * v for w, v in present) if present else None


class Objective:
    """Base class for objectives: override :meth:`__call__`.

    ``nonfinite`` (``"fail"`` default, or ``"skip"``) is the engine's policy
    for a non-finite loss term or gradient. Any callable
    ``(ObjectiveContext) -> ObjectiveResult`` works as an objective; a plain
    function uses ``"fail"``.
    """

    nonfinite: str = "fail"

    def __init__(self, *, nonfinite: str = "fail") -> None:
        self.nonfinite = check_nonfinite_policy(nonfinite)

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:  # pragma: no cover - abstract
        raise NotImplementedError


ObjectiveFn = Callable[[ObjectiveContext], ObjectiveResult]


def _supervised_terms(model: Any, logits: torch.Tensor, target: torch.Tensor, name: str, weight: float) -> LossTerm:
    from .nn.nn_model import _loss_terms

    _, numerator, denominator = _loss_terms(model.loss_fn, logits, target)
    if denominator is None:
        return LossTerm(name, numerator, None, "sum", weight)
    return LossTerm(name, numerator, denominator, "mean", weight)


class SupervisedObjective(Objective):
    """The standard supervised loss as an objective: the model's
    ``loss_fn`` on its outputs, normalized by the loss's own denominator
    (non-ignored targets, class weights) — or summed for a
    ``reduction="sum"`` loss — and, with a ``TaskSpec``, the task's masked
    loss over its valid targets."""

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        from .nn.nn_model import _classification_edp_for_loss

        model = ctx.model
        model.net.train()
        adapter = getattr(model, "task_adapter", None)
        if adapter is not None:
            _, target, logits = model._fwd_outputs(ctx.batch)
            output, target, valid = adapter.prepare(logits, target)
            _, numerator, weight = adapter.loss_terms(model.loss_fn, output, target, valid)
            term = LossTerm("loss", numerator, None, "sum") if weight is None else LossTerm("loss", numerator, weight)
            accumulator = adapter.accumulator(keep_arrays=bool(ctx.extra_metrics))
            with torch.no_grad():
                accumulator.update(output.detach(), target, valid)
            record = accumulator.result(loss=term.value, extra_metrics=ctx.extra_metrics)
            return ObjectiveResult((term,), record)
        _, target, logits, prediction = model._fwd_pass(ctx.batch)
        term = _supervised_terms(model, logits, target, "loss", 1.0)
        record = _classification_edp_for_loss(
            loss_fn=model.loss_fn,
            target=target,
            prediction=prediction,
            loss=term.value if term.value is not None else math.nan,
            extra_metrics=ctx.extra_metrics,
        )
        return ObjectiveResult((term,), record)


def supervised_objective(*, nonfinite: str = "fail") -> SupervisedObjective:
    """The standard supervised loss as an objective (see
    :class:`SupervisedObjective`)."""
    return SupervisedObjective(nonfinite=nonfinite)


class KDObjective(Objective):
    """Hinton knowledge distillation as an objective.

    Two terms: ``"distillation"`` — the temperature-softened
    ``KL(teacher ‖ student)`` summed over the rows and normalized by their
    count, scaled by ``T²`` (weight ``alpha``) — and ``"supervised"`` —
    the student's loss on the labels with its own denominator (weight
    ``1 − alpha``). Over a window this is exactly
    ``alpha · KL_batchmean + (1 − alpha) · loss`` of the full batch, however
    the rows are split into microbatches or targets are ignored. The teacher
    is frozen and put in eval mode once, as ``kd_train_step_factory`` does,
    and never updated.
    """

    def __init__(self, teacher: NNModel, *, alpha: float = 0.5, temperature: float = 4.0, nonfinite: str = "fail"):
        from .paradigms.distillation import _check_kd_weights, _freeze_teacher

        super().__init__(nonfinite=nonfinite)
        _check_kd_weights(alpha, temperature)
        self.teacher = teacher
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        _freeze_teacher(teacher)

    def __call__(self, ctx: ObjectiveContext) -> ObjectiveResult:
        from ._step_helpers import softened_kl
        from .nn.nn_model import _classification_edp_for_loss

        model = ctx.model
        model.net.train()
        (X,), Y = cast(Any, model.net).unpack_batch(ctx.batch)
        X, Y = X.to(model.device), Y.to(model.device)
        student = model.net(X)
        with torch.no_grad():
            teacher_logits = self.teacher.net(X.to(self.teacher.device)).to(model.device)
        rows = int(student.shape[0])
        soft = LossTerm(
            "distillation",
            softened_kl(student, teacher_logits, self.temperature) * rows,  # batchmean × rows = row sum
            rows,
            "mean",
            self.alpha,
        )
        hard = _supervised_terms(model, student, Y, "supervised", 1.0 - self.alpha)
        result = ObjectiveResult((soft, hard))
        loss = result.loss()
        record = _classification_edp_for_loss(
            loss_fn=model.loss_fn,
            target=Y,
            prediction=student.detach().argmax(dim=-1),
            loss=loss if loss is not None else math.nan,
            extra_metrics=ctx.extra_metrics,
        )
        return ObjectiveResult((soft, hard), record)


def kd_objective(
    teacher: NNModel, *, alpha: float = 0.5, temperature: float = 4.0, nonfinite: str = "fail"
) -> KDObjective:
    """Knowledge distillation as an objective (see :class:`KDObjective`) —
    the objective counterpart of ``kd_train_step_factory``, which stays
    available (and unchanged) as the imperative step."""
    return KDObjective(teacher, alpha=alpha, temperature=temperature, nonfinite=nonfinite)
