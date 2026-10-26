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
from ..nn.moe import MoELinear
from ..nn.nn_model import (
    TrainStepContext,
    TrainStepFn,
    _check_scorable_target,
    _classification_edp_for_loss,
    _loss_input,
    _single_input_batch,
)
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..precision import full_precision_only
from ..tasks import _StepTasks


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

    A model that declares a ``TaskSpec`` scores the supervised term
    through its task adapter, as the default step does: categorical
    targets equal to ``ignore_index`` and NaN multilabel labels are left
    out, each batch is validated before any backward pass, and the record
    carries the task's ``kind`` / ``count`` / ``status``. A batch with
    nothing to learn from (every target masked, and no weighted aux loss)
    takes no update and records ``status="empty"``. A regression task is
    refused before any run is reserved.

    Returns:
        A ``TrainStepFn`` for :meth:`NNModel.train`. Works on any
        single-input supervised net that contains ≥ 0
        :class:`MoELinear` layers; if there are no MoE layers, the
        aux loss is 0 and the step is exactly supervised.

    Raises:
        ValueError: if ``aux_loss_weight`` is not a finite real ``>= 0``
            (a ``bool`` included). ``NNModel.train`` refuses a model that
            declares a regression task (``TaskValidationError``) before any
            run is reserved.
    """
    require_finite_real(aux_loss_weight, "aux_loss_weight", owner="moe_train_step_factory", minimum=0)

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        # Single-input batches — same contract as the other supervised
        # paradigm factories (Mixup / CutMix / KD). Multi-input nets need a
        # custom step.
        X, Y = _single_input_batch(m, ctx.batch, who="moe_train_step_factory")

        # Clear stale aux losses BEFORE the forward: a MoELinear that's
        # registered but not exercised by this batch (conditional branch,
        # frozen tower) would otherwise contribute the previous step's
        # tensor — whose graph the previous backward already freed —
        # crashing with "backward through the graph a second time".
        for module in m.net.modules():
            if isinstance(module, MoELinear):
                module.last_aux_loss = None

        # The supervised forward populates each MoELinear's
        # ``.last_aux_loss`` as a side effect.
        Y_hat_logits = m._net_forward((X,), {})
        require_row_logits(Y_hat_logits, "moe")
        adapter = m.task_adapter
        # A declared task scores the labels over its valid targets (#513), validated before any backward.
        hard = task_term(adapter, m.loss_fn, Y_hat_logits, Y) if adapter is not None else None
        supervised_loss = hard.loss if hard is not None else m.loss_fn(_loss_input(m.loss_fn, Y_hat_logits), Y)

        # Sum the per-layer aux losses across every MoE layer in the
        # net. ``.last_aux_loss`` is set by every MoELinear forward;
        # we collect them post-forward. If there are no MoELinear
        # layers, the sum is the scalar 0 — the factory degenerates
        # to a plain supervised step. We construct the zero on the
        # same device as the supervised loss so the add below stays
        # device-coherent regardless of model placement.
        aux_loss: torch.Tensor = torch.zeros((), device=supervised_loss.device)
        routed = False  # an MoE layer ran: its aux loss trains the router
        for module in m.net.modules():
            if isinstance(module, MoELinear) and module.last_aux_loss is not None:
                aux_loss = aux_loss + module.last_aux_loss
                routed = True

        loss = supervised_loss + aux_loss_weight * aux_loss
        if adapter is not None:
            assert hard is not None
            # Every target masked and no weighted aux loss: nothing to learn from.
            trains = hard.trains or (aux_loss_weight > 0 and routed)
            loss_val = finalize_task_step(loss, ctx, trains=trains, paradigm="moe")
            return task_record(adapter, [(1.0, hard)], loss=loss_val, extra_metrics=ctx.extra_metrics)
        _check_scorable_target(m.loss_fn, Y, paradigm="moe")  # before the update: a refused batch changes nothing
        loss_val = finalize_step(loss, ctx, paradigm="moe")

        return _classification_edp_for_loss(
            loss_fn=m.loss_fn,
            target=Y,
            prediction=predicted_labels(m.loss_fn, Y_hat_logits),
            loss=loss_val,
            extra_metrics=ctx.extra_metrics,
        )

    # finalize_step: full precision
    return full_precision_only(imperative_step(task_step(step, _StepTasks("moe_train_step_factory")), paradigm="moe"))
