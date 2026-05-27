"""Batch-level training augmentations: Mixup and CutMix.

Both interpolate samples (and labels) within a batch and shift the
loss accordingly. They live as :class:`TrainStepFn` factories — not
``collate_fn``\\s — because mixing labels with arbitrary loss functions
needs label-aware computation that the standard ``(X, Y)`` batch
contract doesn't expose. Building this on the ``train_step_fn`` hook
keeps the contract identical and the augmentation visible at the
training-loop level rather than buried in the dataloader.

**Mixup** (zhang:mixup): ``x' = λ x_a + (1−λ) x_b``;
``L = λ L(f(x'), y_a) + (1−λ) L(f(x'), y_b)``. Works for any input
shape — tabular, image, sequence.

**CutMix** (yun:cutmix): copies a random rectangular region from
``x_b`` into ``x_a``, then re-weights the loss by the area ratio.
Requires 4D ``(B, C, H, W)`` image tensors; raises on lower-rank input.
"""

from __future__ import annotations

import numpy as np
import torch

from .._step_helpers import finalize_step
from ..nn.nn_model import TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def _unpack_supervised(ctx: TrainStepContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull ``(X, Y)`` from a supervised batch via the net's
    ``unpack_batch`` adapter and move both to the model's device.
    Both augmentation factories share this preamble.

    The ``(X,), Y = unpack_batch(...)`` destructuring asserts a
    single-tensor input — multi-input nets (e.g., PyG graph data
    with multiple feature tensors) would unpack to more elements
    and trip the implicit tuple-arity check here. Use a custom
    train_step_fn for those.
    """
    m = ctx.model
    (X,), Y = m.net.unpack_batch(ctx.batch)
    return X.to(m.device), Y.to(m.device)


def _weighted_acc(Y_hat: torch.Tensor, Y_a: torch.Tensor, Y_b: torch.Tensor, lam: float) -> float:
    """Lam-weighted accuracy — Mixup's standard "fractional correctness"
    metric. Useful as an ``error`` signal even though it's not a
    standard top-1 accuracy."""
    return float(lam * (Y_hat == Y_a).float().mean().item() + (1.0 - lam) * (Y_hat == Y_b).float().mean().item())


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
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    rng = np.random.default_rng()

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        X, Y = _unpack_supervised(ctx)
        B = X.shape[0]

        lam = float(rng.beta(alpha, alpha))
        perm = torch.randperm(B, device=m.device)
        X_mixed = lam * X + (1.0 - lam) * X[perm]
        Y_a = Y
        Y_b = Y[perm]

        Y_hat_logits = m.net(X_mixed)
        loss = lam * m.loss_fn(Y_hat_logits, Y_a) + (1.0 - lam) * m.loss_fn(Y_hat_logits, Y_b)
        loss_val = finalize_step(loss, ctx, paradigm="mixup")

        Y_hat = Y_hat_logits.argmax(dim=-1)
        acc = _weighted_acc(Y_hat, Y_a, Y_b, lam)
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=acc,
            precision=0.0,
            loss=loss_val,
            error=float(1.0 - acc),
        )

    return step


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
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    rng = np.random.default_rng()

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        X, Y = _unpack_supervised(ctx)
        if X.dim() != 4:
            raise ValueError(
                f"CutMix requires 4D image input (B, C, H, W), got rank {X.dim()}. "
                "Use Mixup for tabular / lower-rank data."
            )
        B, _, H, W = X.shape

        lam = float(rng.beta(alpha, alpha))
        # Box dimensions: side lengths scale with √(1−λ) so the area
        # ratio is (1−λ). cx, cy are the box center, sampled uniformly.
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        cx = int(rng.integers(0, W))
        cy = int(rng.integers(0, H))
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

        Y_hat_logits = m.net(X_cut)
        loss = lam * m.loss_fn(Y_hat_logits, Y_a) + (1.0 - lam) * m.loss_fn(Y_hat_logits, Y_b)
        loss_val = finalize_step(loss, ctx, paradigm="cutmix")

        Y_hat = Y_hat_logits.argmax(dim=-1)
        acc = _weighted_acc(Y_hat, Y_a, Y_b, lam)
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=acc,
            precision=0.0,
            loss=loss_val,
            error=float(1.0 - acc),
        )

    return step
