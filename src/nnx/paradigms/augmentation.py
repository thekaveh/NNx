"""Batch-level training augmentations: Mixup and CutMix.

Both interpolate samples (and labels) within a batch and shift the
loss accordingly. They live as :class:`TrainStepFn` factories — not
``collate_fn``\\s — because mixing labels with arbitrary loss functions
needs label-aware computation that the standard ``(X, Y)`` batch
contract doesn't expose. Building this on the ``train_step_fn`` hook
keeps the contract identical and the augmentation visible at the
training-loop level rather than buried in the dataloader.

**Mixup** (zhang:mixup): ``x' = λ x_a + (1−λ) x_b``;
``L = λ L(f(x'), y_a) + (1−λ) L(f(x'), y_b)``. Works for floating
inputs of any shape — tabular, image, sequence features. Integer inputs
cannot be interpolated and are refused before any mixing: cast integer
features to float; for token ids, train a sequence model with
``nnx.lm_tasks.CausalLMTask``.

**CutMix** (yun:cutmix): copies a random rectangular region from
``x_b`` into ``x_a``, then re-weights the loss by the area ratio.
Requires 4D ``(B, C, H, W)`` image tensors; raises on lower-rank input.

**Declared tasks.** A model with a categorical or multilabel ``TaskSpec``
scores each of the two targets through its task adapter, as the default
step scores its one: targets equal to ``ignore_index`` and NaN labels are
left out of that target's loss, each batch is validated before any backward
pass, and the record carries the task's ``kind`` / ``count`` / ``status``
with its metrics mixed by ``λ``. A batch whose every target is masked takes
no update and records ``status="empty"``. A regression task is refused
before any run is reserved.
"""

from __future__ import annotations

import torch

from .._step_helpers import (
    finalize_step,
    finalize_task_step,
    imperative_step,
    predicted_labels,
    require_row_logits,
    task_record,
    task_step,
    task_term,
)
from .._validation import require_finite_real
from ..nn.nn_model import (
    NNModel,
    TrainStepContext,
    TrainStepFn,
    _check_scorable_target,
    _classification_metric_tensors,
    _loss_input,
    _single_input_batch,
)
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..precision import full_precision_only
from ..tasks import _StepTasks


def _unpack_supervised(ctx: TrainStepContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull ``(X, Y)`` from a supervised batch via the net's
    ``unpack_batch`` adapter and move both to the model's device.
    Both augmentation factories share this preamble.

    The batch is split by the model's batch adapter (FEAT-006) and must
    hold a single-tensor input — multi-input nets (e.g., PyG graph data
    with multiple feature tensors) or keyword-input modules raise
    ``ValueError`` here. Use a custom train_step_fn for those.
    """
    return _single_input_batch(ctx.model, ctx.batch, who="mixup / cutmix")


def _mixed_task_step(
    ctx: TrainStepContext,
    m: NNModel,
    logits: torch.Tensor,
    Y_a: torch.Tensor,
    Y_b: torch.Tensor,
    lam: float,
    *,
    paradigm: str,
) -> NNEvaluationDataPoint:
    """The tail of a Mixup / CutMix step for a model with a declared task
    (#513): one adapter ``prepare`` per target, each loss over its own valid
    targets, no update when every target is masked, and the task record
    mixed by ``lam``."""
    adapter = m.task_adapter
    assert adapter is not None
    term_a = task_term(adapter, m.loss_fn, logits, Y_a)
    term_b = task_term(adapter, m.loss_fn, logits, Y_b)
    loss = lam * term_a.loss + (1.0 - lam) * term_b.loss
    trains = term_a.trains or term_b.trains  # Y_b permutes Y_a: both are masked or neither
    loss_val = finalize_task_step(loss, ctx, trains=trains, paradigm=paradigm)
    return task_record(adapter, [(lam, term_a), (1.0 - lam, term_b)], loss=loss_val, extra_metrics=ctx.extra_metrics)


def _weighted_acc(
    loss_fn: torch.nn.Module, Y_hat: torch.Tensor, Y_a: torch.Tensor, Y_b: torch.Tensor, lam: float
) -> float:
    """Lam-weighted accuracy — Mixup's standard "fractional correctness"
    metric. Useful as an ``error`` signal even though it's not a
    standard top-1 accuracy. Targets are read as the default step reads
    them (probability / soft targets, ignored rows)."""

    def accuracy(target: torch.Tensor) -> float:
        scored_target, scored_prediction = _classification_metric_tensors(loss_fn, target, Y_hat)
        if scored_target.numel() == 0:
            return 0.0
        return float((scored_prediction == scored_target).float().mean().item())

    return lam * accuracy(Y_a) + (1.0 - lam) * accuracy(Y_b)


def mixup_train_step_factory(*, alpha: float = 0.4) -> TrainStepFn:
    """Build a Mixup :class:`TrainStepFn`.

    Args:
        alpha: Beta-distribution shape parameter. ``λ ~ Beta(α, α)``;
            α=1.0 yields a uniform mix, lower values concentrate λ
            near 0 or 1 (closer to no-mixing). 0.4 is the
            classification default; image-task papers often use 0.2-1.0.
            Must be positive.

    Returns:
        A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.
        Reports a Mixup-weighted ``error`` and the mixed loss. The
        loss honors the model's ``loss_fn`` (so this works for any
        classification loss, not just CrossEntropy).

    Raises:
        ValueError: if ``alpha`` is not a finite real ``> 0`` (a ``bool``
            included). At step time, before any mixing or RNG draw, on
            non-floating input such as token ids. ``NNModel.train`` refuses
            a model that declares a regression task (``TaskValidationError``)
            before any run is reserved.
    """
    require_finite_real(alpha, "alpha", owner="mixup_train_step_factory", minimum=0, exclusive_min=True)
    # Lambda (and the CutMix box) come from torch's global RNG at step time,
    # like the batch permutation: NNTrainParams.seed controls them, and a
    # stateful resume restores that stream, so a split run continues the
    # uninterrupted one.
    beta = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha)))

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        X, Y = _unpack_supervised(ctx)
        if not X.is_floating_point():
            # Before the draws: a refused batch leaves the RNG stream as it was.
            raise ValueError(
                f"the mixup step interpolates floating inputs, got {X.dtype} input: cast integer features to "
                "float; token ids cannot be mixed, so train a sequence model with nnx.lm_tasks.CausalLMTask"
            )
        B = X.shape[0]

        lam = float(beta.sample())
        perm = torch.randperm(B, device=m.device)
        X_mixed = lam * X + (1.0 - lam) * X[perm]
        Y_a = Y
        Y_b = Y[perm]

        Y_hat_logits = m._net_forward((X_mixed,), {})
        require_row_logits(Y_hat_logits, "mixup")
        if m.task_adapter is not None:
            return _mixed_task_step(ctx, m, Y_hat_logits, Y_a, Y_b, lam, paradigm="mixup")
        loss_input = _loss_input(m.loss_fn, Y_hat_logits)
        loss = lam * m.loss_fn(loss_input, Y_a) + (1.0 - lam) * m.loss_fn(loss_input, Y_b)
        _check_scorable_target(m.loss_fn, Y, paradigm="mixup")  # before the update
        loss_val = finalize_step(loss, ctx, paradigm="mixup")

        Y_hat = predicted_labels(m.loss_fn, Y_hat_logits)
        acc = _weighted_acc(m.loss_fn, Y_hat, Y_a, Y_b, lam)
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=acc,
            precision=0.0,
            loss=loss_val,
            error=float(1.0 - acc),
        )

    # finalize_step: full precision
    return full_precision_only(
        imperative_step(task_step(step, _StepTasks("mixup_train_step_factory")), paradigm="mixup")
    )


def cutmix_train_step_factory(*, alpha: float = 1.0) -> TrainStepFn:
    """Build a CutMix :class:`TrainStepFn` for 4D image batches.

    Args:
        alpha: Beta-distribution shape parameter for the area ratio.
            ``λ ~ Beta(α, α)``; controls the size of the swapped
            rectangle. 1.0 is the original paper default.
            Must be positive.

    Returns:
        A ``TrainStepFn`` for image classification (4D ``(B, C, H, W)``
        inputs). Raises at step time on lower-rank input — CutMix's
        spatial cut isn't well-defined without H and W.

    Raises:
        ValueError: if ``alpha`` is not a finite real ``> 0`` (a ``bool``
            included). At step time, on input that is not 4D.
            ``NNModel.train`` refuses a model that declares a regression
            task (``TaskValidationError``) before any run is reserved.
    """
    require_finite_real(alpha, "alpha", owner="cutmix_train_step_factory", minimum=0, exclusive_min=True)
    # Lambda (and the CutMix box) come from torch's global RNG at step time,
    # like the batch permutation: NNTrainParams.seed controls them, and a
    # stateful resume restores that stream, so a split run continues the
    # uninterrupted one.
    beta = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha)))

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        X, Y = _unpack_supervised(ctx)
        if X.dim() != 4:
            # Mixup refuses integer input (token ids), so it is no alternative there.
            hint = (
                "Use Mixup for tabular / lower-rank data."
                if X.is_floating_point()
                else "Cast integer features to float before mixing them; for token ids, train a sequence model "
                "with nnx.lm_tasks.CausalLMTask."
            )
            raise ValueError(f"CutMix requires 4D image input (B, C, H, W), got rank {X.dim()}. {hint}")
        B, _, H, W = X.shape

        lam = float(beta.sample())
        # Box dimensions: side lengths scale with √(1−λ) so the area
        # ratio is (1−λ). cx, cy are the box center, sampled uniformly.
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        cx = int(torch.randint(0, W, (1,)).item())
        cy = int(torch.randint(0, H, (1,)).item())
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)

        perm = torch.randperm(B, device=m.device)
        X_cut = X.clone()
        X_cut[:, :, y1:y2, x1:x2] = X[perm, :, y1:y2, x1:x2]
        # Re-derive λ from the actual cut area — clipping at the edges
        # makes the realized box smaller than the nominal Beta draw.
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        Y_a = Y
        Y_b = Y[perm]

        Y_hat_logits = m._net_forward((X_cut,), {})
        require_row_logits(Y_hat_logits, "cutmix")
        if m.task_adapter is not None:
            return _mixed_task_step(ctx, m, Y_hat_logits, Y_a, Y_b, lam, paradigm="cutmix")
        loss_input = _loss_input(m.loss_fn, Y_hat_logits)
        loss = lam * m.loss_fn(loss_input, Y_a) + (1.0 - lam) * m.loss_fn(loss_input, Y_b)
        _check_scorable_target(m.loss_fn, Y, paradigm="cutmix")  # before the update
        loss_val = finalize_step(loss, ctx, paradigm="cutmix")

        Y_hat = predicted_labels(m.loss_fn, Y_hat_logits)
        acc = _weighted_acc(m.loss_fn, Y_hat, Y_a, Y_b, lam)
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=acc,
            precision=0.0,
            loss=loss_val,
            error=float(1.0 - acc),
        )

    # finalize_step: full precision
    return full_precision_only(
        imperative_step(task_step(step, _StepTasks("cutmix_train_step_factory")), paradigm="cutmix")
    )
