"""Shared utilities for custom train_step_fn factories.

The paradigm step factories (e.g. KD, SimCLR, Mixup, CutMix, diffusion) all
do the same boilerplate after their per-paradigm loss computation:
NaN-guard, gradient clipping, optimizer step. Without a shared helper,
each factory either silently dropped the relevant NNOptimParams knobs
(grad_clip_norm) or never noticed when training diverged.

This module is internal — :func:`finalize_step` is the canonical
post-loss tail. Public step factories call it instead of writing the
sequence by hand. Direct callers of NNModel.train(train_step_fn=...)
in user code can use it too; it's a regular function with no nnx
coupling beyond the TrainStepContext type.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Optional, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F

from ._validation import require_finite_real
from .nn.nn_model import TrainStepContext
from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from .tasks import STEP_TASKS, TaskAdapter, _StepTasks

# Marks an imperative paradigm step (it steps the optimizer itself) with its
# paradigm's name, so passing it as ``objective=`` is refused before any run
# is reserved (FEAT-040) instead of being stepped twice.
IMPERATIVE_STEP = "__nnx_imperative_step__"

_StepT = TypeVar("_StepT")


def first_input(model: Any, batch: Any, *, who: str) -> torch.Tensor:
    """A self-supervised paradigm's one input from a batch, on the model's
    device (labels, if any, are ignored): split by the model's batch adapter
    when it has one (FEAT-006) — which must yield exactly one input,
    positional or keyword — else the net's own ``unpack_batch``, else the
    first element of a tuple / list, else the batch itself (a net swapped in
    for a built-in one)."""
    adapter = getattr(model, "_batch_adapter", None)
    if adapter is not None:
        args, kwargs, _ = adapter.split(batch)
        inputs = [*args, *kwargs.values()]
        if len(inputs) != 1:
            raise ValueError(
                f"{who} needs batches of exactly one input; the model's batch adapter gave {len(args)} positional "
                f"and {len(kwargs)} keyword input(s)"
            )
        x = inputs[0]
    elif hasattr(model.net, "unpack_batch"):
        inputs, _ = cast(Any, model.net).unpack_batch(batch)
        if len(inputs) != 1:
            raise ValueError(f"{who} needs batches of exactly one input; the net's unpack_batch gave {len(inputs)}")
        (x,) = inputs
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch
    return x.to(model.device)


def full_precision(tensor: torch.Tensor) -> torch.Tensor:
    """A mixed-precision (float16 / bfloat16) output as float32, so a loss
    sum accumulates in full precision; other dtypes unchanged."""
    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor


def imperative_step(step: _StepT, *, paradigm: str) -> _StepT:
    """Mark ``step`` as an imperative ``paradigm`` step (see
    :data:`IMPERATIVE_STEP`) and return it."""
    setattr(step, IMPERATIVE_STEP, paradigm)
    return step


def task_step(step: _StepT, tasks: _StepTasks) -> _StepT:
    """Mark ``step`` with the task kinds it trains (#513), so a declared task
    of any other kind is refused before a run is reserved, and return it."""
    setattr(step, STEP_TASKS, tasks)
    return step


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskTerm:
    """A paradigm step's hard (label) term through the model's task adapter
    (#513): ``loss`` is the loss over the valid targets — a graph-connected
    zero when every target is masked (``trains`` is then false) — and
    ``output`` / ``target`` / ``valid`` the prepared batch for the record."""

    loss: torch.Tensor
    trains: bool
    output: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor


def task_term(adapter: TaskAdapter, loss_fn: torch.nn.Module, logits: torch.Tensor, target: torch.Tensor) -> TaskTerm:
    """Validate one batch through ``adapter`` (before any backward) and
    score ``loss_fn`` over its valid targets only, as the default step does:
    ignored categorical rows and NaN multilabel labels are left out."""
    output, target, valid = adapter.prepare(logits, target)
    loss, numerator, weight = adapter.loss_terms(loss_fn, output, target, valid)
    # A sum-reduction loss (weight None) is its numerator; an all-masked batch (weight 0) has a NaN display loss.
    return TaskTerm(loss=loss if weight else numerator, trains=weight != 0, output=output, target=target, valid=valid)


def task_record(
    adapter: TaskAdapter,
    terms: Sequence[tuple[float, TaskTerm]],
    *,
    loss: Optional[float],
    extra_metrics: Optional[Mapping[str, Callable]],
) -> NNEvaluationDataPoint:
    """The task record of one paradigm batch (#513): each ``(weight, term)``
    scored by the adapter's own accumulator and their metrics mixed by the
    weights (Mixup / CutMix score two permutations of one target, so every
    term has the same valid ``count``). ``loss`` is the step's trained loss,
    ``None`` for a batch that took no update; an all-masked batch records
    ``status="empty"``."""
    records = []
    for _, term in terms:
        accumulator = adapter.accumulator(keep_arrays=bool(extra_metrics))
        with torch.no_grad():
            accumulator.update(term.output.detach(), term.target, term.valid)
        records.append(accumulator.result(loss=loss, extra_metrics=extra_metrics))
    record = records[0]
    if len(records) > 1 and record.count:
        weights = [weight for weight, _ in terms]

        def mixed(field: str) -> Optional[float]:
            values = [getattr(r, field) for r in records]
            return None if values[0] is None else float(sum(w * v for w, v in zip(weights, values, strict=True)))

        def mixed_map(field: str) -> dict[str, float]:
            maps = [getattr(r, field) for r in records]
            return {key: float(sum(w * m[key] for w, m in zip(weights, maps, strict=True))) for key in maps[0]}

        record = replace(
            record,
            **{name: mixed(name) for name in ("f1", "recall", "accuracy", "precision", "error")},
            metrics=mixed_map("metrics"),
            extra=mixed_map("extra"),
        )
    # An all-masked batch whose other terms trained (a soft or auxiliary term) keeps the loss it stepped on.
    return record if record.loss is not None or loss is None else record.with_loss(loss)


def finalize_step(
    loss: torch.Tensor,
    ctx: TrainStepContext,
    *,
    paradigm: str,
) -> float:
    """Standard post-loss tail for custom :class:`TrainStepFn` factories.

    Honors ``ctx.grad_clip_norm`` (global L2 grad-clip), runs the
    optimizer step and reports it with ``ctx.report_update()``, so an
    ``optimizer_update``-clock scheduler steps once per call (FEAT-014).
    Raises a clear ``FloatingPointError`` if the loss
    is non-finite — silent divergence leaves checkpoints full of
    garbage weights, same failure mode :func:`default_train_step`
    guards against.

    **Not supported** in paradigm step factories (would silently drop
    if we accepted them): AMP (``ctx.scaler``, or a reduced
    ``ctx.precision`` — FEAT-028) and gradient accumulation
    (``ctx.accumulate_grad_batches != 1``). Both raise loudly rather
    than letting the caller think their NNOptimParams knobs are in
    effect. Honoring them would require per-paradigm care (scaling
    each loss component, cycling zero_grad/step boundaries), which is
    out of scope for the v1 paradigm factories.

    **CPU caveat** (the legacy ``mixed_precision`` flag; an explicit
    reduced ``precision=`` is refused on every device): the legacy flag's
    AMP rejection only fires when ``ctx.scaler`` is non-None, which on CPU
    it never is — :meth:`NNModel._build_grad_scaler`
    returns None whenever ``device.type != "cuda"`` regardless of
    ``NNModelParams.mixed_precision``. A CPU user setting
    ``mixed_precision=True`` therefore gets the same silent no-op as
    they would for any other CPU AMP request (the behavior
    NNModelParams documents as "silently bypassed on CPU/MPS"). The
    explicit ``ValueError`` is the user-facing safety net for the
    CUDA path, where the silent drop would actually matter.

    Args:
        loss: scalar loss tensor with ``requires_grad`` already wired
            from the forward path. The caller must have invoked
            ``net.zero_grad()`` before computing it.
        ctx: the :class:`TrainStepContext` passed into the step.
        paradigm: short label used in error messages — e.g.,
            ``"diffusion"``, ``"mixup"``. Helps users find the
            failing factory.

    Returns:
        The float value of the loss after detach — useful for
        populating the returned :class:`NNEvaluationDataPoint`.

    Raises:
        ValueError: when AMP or gradient accumulation is requested
            (the paradigm factories don't honor those knobs).
        FloatingPointError: when ``loss`` is non-finite.
    """
    _refuse_unsupported_knobs(ctx, paradigm)

    # Check finiteness BEFORE backward so a diverged loss doesn't
    # propagate NaN/Inf into gradients and (via step()) into the
    # model's weights. `default_train_step` checks AFTER for legacy
    # reasons (its tests pin the existing post-step ordering); the
    # paradigm hooks have no such constraint and we choose the cleaner
    # contract here.
    loss_val = float(loss.detach())
    if not np.isfinite(loss_val):
        raise FloatingPointError(
            f"non-finite {paradigm} loss ({loss_val!r}) — training diverged. "
            "Check learning rate, loss-specific hyperparameters, or input normalization."
        )

    loss.backward()

    if ctx.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(ctx.model.net.parameters(), ctx.grad_clip_norm)

    ctx.optimizer.step()
    # Every call commits one update (accumulation is refused above).
    report_update = getattr(ctx, "report_update", None)
    if report_update is not None:
        report_update()

    return loss_val


def finalize_task_step(loss: torch.Tensor, ctx: TrainStepContext, *, trains: bool, paradigm: str) -> Optional[float]:
    """:func:`finalize_step` for a step with a declared task (#513), or —
    when the batch has nothing to learn from (``trains`` false: every target
    masked and every other term of weight 0) — no backward, no optimizer
    step and no reported update, as the default step's masked window
    (``None``). The knobs :func:`finalize_step` refuses are refused either way."""
    if trains:
        return finalize_step(loss, ctx, paradigm=paradigm)
    _refuse_unsupported_knobs(ctx, paradigm)
    ctx.model.net.zero_grad()
    return None


def _refuse_unsupported_knobs(ctx: TrainStepContext, paradigm: str) -> None:
    precision = getattr(ctx, "precision", None)
    if ctx.scaler is not None or (precision is not None and precision.reduced):
        raise ValueError(
            f"{paradigm} train_step_fn does not support mixed precision "
            "(NNModelParams.mixed_precision=True or a reduced PrecisionPolicy). "
            "Train this NNModel in fp32, or write a custom train_step_fn that "
            "applies ctx.precision and the scaler explicitly."
        )
    if ctx.accumulate_grad_batches != 1:
        raise ValueError(
            f"{paradigm} train_step_fn does not support gradient accumulation "
            f"(NNOptimParams.accumulate_grad_batches={ctx.accumulate_grad_batches}). "
            "Set accumulate_grad_batches=1, or write a custom train_step_fn."
        )


def require_real_or_scalar_tensor(value: object, field: str, *, owner: str, **domain: Any) -> float:
    """``require_finite_real`` for a per-step helper's setting that may also
    arrive as a one-element tensor (a learnable or torch-scheduled value):
    the tensor's value is checked against the same domain and returned as a
    float. A bool, complex or multi-element tensor is refused like a bool."""
    if not torch.is_tensor(value):
        return require_finite_real(value, field, owner=owner, **domain)
    if value.dtype == torch.bool or value.is_complex() or value.numel() != 1:
        raise ValueError(
            f"{owner} requires {field} to be a finite real number or a one-element real tensor, "
            f"got a {value.dtype} tensor of shape {tuple(value.shape)}"
        )
    return require_finite_real(float(value.detach()), field, owner=owner, **domain)


def require_row_logits(logits: torch.Tensor, paradigm: str) -> None:
    """The classification paradigm steps score ``(rows, classes)`` logits;
    a sequence model's ``(batch, time, vocab)`` output needs the causal-LM
    task, whose loss and metrics flatten the time axis."""
    if logits.ndim > 2:
        raise ValueError(
            f"the {paradigm} step scores (rows, classes) logits, got shape {tuple(logits.shape)}: train a "
            "sequence model with nnx.lm_tasks.CausalLMTask"
        )


def predicted_labels(loss_fn: Any, logits: torch.Tensor) -> torch.Tensor:
    """Hard predictions as the default step makes them: a non-negative logit
    for a ``BCEWithLogitsLoss`` head (an argmax over one logit is always 0),
    otherwise the last-axis argmax."""
    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        return (logits >= 0).to(dtype=torch.long)
    return logits.argmax(dim=-1)


def softened_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Hinton-style distillation loss: ``KL(teacher_soft || student_soft) * T²``.

    Both logits are softened by ``temperature``. ``F.kl_div``'s contract
    is ``sum target*(log target - input)`` with ``input`` in log-space
    and ``target`` in probability space — that evaluates to
    ``KL(target || exp(input)) = KL(teacher_soft || student_soft)``.
    The ``T²`` factor keeps the soft-loss gradient magnitude comparable
    across temperatures (Hinton et al., 2015, §2). Shared by the
    distillation and feature-KD factories so the direction of the KL
    and the ``T²`` scaling can't drift between them.
    """
    if student_logits.shape != teacher_logits.shape:  # a broadcast would distill a wrong target silently
        raise ValueError(
            f"distillation needs teacher logits of the student's shape {tuple(student_logits.shape)}, got "
            f"{tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 2:  # a (N,) single-logit head would be softmaxed across the batch's rows
        raise ValueError(
            f"distillation needs (rows, classes) logits, got shape {tuple(student_logits.shape)}; a single-logit "
            "(binary) head has no class distribution to soften"
        )
    if student_logits.shape[-1] < 2:  # a softmax over one logit is always 1: the soft loss is 0
        raise ValueError(
            "distillation needs at least 2 class logits per row; a single-logit (binary) head has no class "
            "distribution to soften"
        )
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
