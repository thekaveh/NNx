"""Mixture-of-Experts training step — adds the load-balancing aux loss
to the standard supervised loss.

A MoE network only learns to *use* its experts if the router is
nudged toward uniform expert utilization. Without the auxiliary
penalty, the router can collapse onto one or two experts, leaving the
rest dead and wasting the parameter budget the MoE layer was created
to spend.

This factory wraps a standard supervised training step (forward → loss
→ backward → optimizer step) and adds

    L = L_supervised + α · Σ_layer layer.last_aux_loss

where the sum runs over every :class:`MoELinear` in the net and ``α``
is :data:`aux_loss_weight`. The supervised forward populates each MoE
layer's ``.last_aux_loss`` as a side effect; the factory just collects
and weights them.

The step routes through :func:`nnx._step_helpers.finalize_step` for
the standard NaN-guard + grad-clip + optimizer.step tail — same shape
as the KD / SimCLR / Mixup / CutMix paradigm factories.
"""

from __future__ import annotations

import torch

from .._metrics import classification_edp
from .._step_helpers import finalize_step
from ..nn.moe import MoELinear
from ..nn.nn_model import TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def moe_train_step_factory(*, aux_loss_weight: float = 0.01) -> TrainStepFn:
    """Build an MoE-aware supervised :class:`TrainStepFn`.

    The returned step performs the standard supervised forward
    (``loss = m.loss_fn(net(X), Y)``) and then *adds* the
    Switch-style load-balancing penalty summed across every
    :class:`MoELinear` layer in ``model.net``, weighted by
    ``aux_loss_weight``. Backward, grad-clip, and optimizer step go
    through :func:`nnx._step_helpers.finalize_step` for the same
    NaN-guard + grad-clip tail as the other paradigm factories.

    Args:
        aux_loss_weight: weight on the aux loss term (``α`` in the
            Switch formulation). Must be non-negative. ``0.0`` turns
            the factory into a plain supervised step (the aux loss is
            still computed by each MoE forward but contributes 0 to
            backward). Defaults to ``0.01`` — the Switch paper's
            tutorial value; small enough not to dominate the main
            loss, large enough to prevent expert collapse.

    Returns:
        A ``TrainStepFn`` for :meth:`NNModel.train`. Works on any
        single-input supervised net that contains ≥ 0
        :class:`MoELinear` layers; if there are no MoE layers, the
        aux loss is 0 and the step is exactly supervised.

    Raises:
        ValueError: if ``aux_loss_weight < 0``.
    """
    if aux_loss_weight < 0:
        raise ValueError(f"aux_loss_weight must be non-negative, got {aux_loss_weight}")

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        # Single-input destructuring — same contract as the other
        # supervised paradigm factories (Mixup / CutMix / KD).
        # Multi-input nets need a custom step.
        (X,), Y = m.net.unpack_batch(ctx.batch)
        X = X.to(m.device)
        Y = Y.to(m.device)

        # The supervised forward populates each MoELinear's
        # ``.last_aux_loss`` as a side effect.
        Y_hat_logits = m.net(X)
        supervised_loss = m.loss_fn(Y_hat_logits, Y)

        # Sum the per-layer aux losses across every MoE layer in the
        # net. ``.last_aux_loss`` is set by every MoELinear forward;
        # we collect them post-forward. If there are no MoELinear
        # layers, the sum is the scalar 0 — the factory degenerates
        # to a plain supervised step. We construct the zero on the
        # same device as the supervised loss so the add below stays
        # device-coherent regardless of model placement.
        aux_loss: torch.Tensor = torch.zeros((), device=supervised_loss.device)
        for module in m.net.modules():
            if isinstance(module, MoELinear) and module.last_aux_loss is not None:
                aux_loss = aux_loss + module.last_aux_loss

        loss = supervised_loss + aux_loss_weight * aux_loss
        loss_val = finalize_step(loss, ctx, paradigm="moe")

        return classification_edp(
            Y=Y,
            Y_hat=Y_hat_logits.argmax(dim=-1),
            loss=loss_val,
            extra_metrics=ctx.extra_metrics,
        )

    return step
