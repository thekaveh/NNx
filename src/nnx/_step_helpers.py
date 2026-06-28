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

import numpy as np
import torch
import torch.nn.functional as F

from .nn.nn_model import TrainStepContext


def finalize_step(
    loss: torch.Tensor,
    ctx: TrainStepContext,
    *,
    paradigm: str,
) -> float:
    """Standard post-loss tail for custom :class:`TrainStepFn` factories.

    Honors ``ctx.grad_clip_norm`` (global L2 grad-clip) and runs the
    optimizer step. Raises a clear ``FloatingPointError`` if the loss
    is non-finite — silent divergence leaves checkpoints full of
    garbage weights, same failure mode :func:`default_train_step`
    guards against.

    **Not supported** in paradigm step factories (would silently drop
    if we accepted them): AMP (``ctx.scaler``) and gradient accumulation
    (``ctx.accumulate_grad_batches != 1``). Both raise loudly rather
    than letting the caller think their NNOptimParams knobs are in
    effect. Honoring them would require per-paradigm care (scaling
    each loss component, cycling zero_grad/step boundaries), which is
    out of scope for the v1 paradigm factories.

    **CPU caveat**: the AMP rejection only fires when ``ctx.scaler`` is
    non-None, which on CPU it never is — :meth:`NNModel._build_grad_scaler`
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
    if ctx.scaler is not None:
        raise ValueError(
            f"{paradigm} train_step_fn does not support mixed precision "
            "(NNModelParams.mixed_precision=True). Disable AMP on this "
            "NNModel or write a custom train_step_fn that handles the "
            "scaler explicitly."
        )
    if ctx.accumulate_grad_batches != 1:
        raise ValueError(
            f"{paradigm} train_step_fn does not support gradient accumulation "
            f"(NNOptimParams.accumulate_grad_batches={ctx.accumulate_grad_batches}). "
            "Set accumulate_grad_batches=1, or write a custom train_step_fn."
        )

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

    return loss_val


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
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
