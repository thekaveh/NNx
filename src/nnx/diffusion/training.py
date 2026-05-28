"""DDPM-style training step factory.

The diffusion training paradigm doesn't fit the standard
``loss_fn(net(X), Y)`` shape — for each batch, we sample a random
timestep, corrupt x_0 to x_t via the forward diffusion process, and
train the network to predict the added noise. That is a custom
:class:`nnx.TrainStepFn` and rides on top of :meth:`NNModel.train`'s
existing ``train_step_fn=`` hook (no Trainer needed; only one
optimizer is involved).

``diffusion_train_step_factory(schedule) -> TrainStepFn`` captures the
schedule in a closure and returns a step fn the user passes straight
to :meth:`NNModel.train`. The closure indexes the schedule tensors on
the model's device per-call (via the ``_extract`` helper, which does a
``values.to(t.device)`` on each step) so callers can build the schedule
on CPU and the factory will pull the right per-timestep coefficients
across devices. The schedule itself is a 1-D length-T tensor so the
per-step transfer cost is negligible relative to the forward / backward
pass; for repeated sampling, see :func:`nnx.diffusion.sample`, which
migrates the schedule once up-front.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .._step_helpers import finalize_step
from ..nn.nn_model import TrainStepContext, TrainStepFn
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from .schedules import NoiseSchedule


def _extract(values: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Pull ``values[t]`` and reshape so it broadcasts against a sample
    tensor of shape ``target_shape``. ``values`` is 1D of length T,
    ``t`` is ``(B,)``, output is ``(B, 1, 1, ...)`` matching the rank of
    target_shape."""
    out = values.to(t.device)[t]
    while out.dim() < len(target_shape):
        out = out.unsqueeze(-1)
    return out


def diffusion_train_step_factory(schedule: NoiseSchedule) -> TrainStepFn:
    """Build a DDPM noise-prediction :class:`TrainStepFn`.

    Each call to the returned step fn:

      1. Samples a random per-sample timestep ``t ~ Uniform[0, T)``.
      2. Samples Gaussian noise ``ε ~ N(0, I)`` matching x_0's shape.
      3. Computes ``x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε`` (forward diffusion).
      4. Calls ``model.net(x_t, t)`` to predict ``ε_pred``.
      5. Backprops the MSE between ``ε_pred`` and ``ε``, steps the optimizer.

    Loss is reported as both ``.loss`` and ``.error`` on the returned
    EDP so BEST checkpoint tracking and the ReduceLROnPlateau scheduler
    have a metric to lock onto. The standard supervised classification
    metrics (accuracy/f1/...) are not meaningful for a generative
    paradigm and stay zero.

    Args:
        schedule: a :class:`NoiseSchedule` from :class:`NoiseSchedulers`.
            Built on any device; the step fn lazily migrates the
            indexed tensors to ``model.device`` per call.

    Returns:
        A function suitable for ``NNModel.train(..., train_step_fn=...)``.
    """

    def step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        # Standard NNx dataloader contract: batch is (X, Y) or a single
        # tensor; Y is unused by diffusion. Use unpack_batch when the net
        # supplies one, fall back to direct unpacking otherwise.
        if hasattr(m.net, "unpack_batch"):
            (x_0,), _ = m.net.unpack_batch(ctx.batch)
        elif isinstance(ctx.batch, (list, tuple)):
            x_0 = ctx.batch[0]
        else:
            x_0 = ctx.batch
        x_0 = x_0.to(m.device)

        B = x_0.shape[0]
        t = torch.randint(0, schedule.T, (B,), device=m.device)
        eps = torch.randn_like(x_0)

        sqrt_a = _extract(schedule.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_1ma = _extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_a * x_0 + sqrt_1ma * eps

        eps_pred = m.net(x_t, t)
        loss = F.mse_loss(eps_pred, eps)
        loss_val = finalize_step(loss, ctx, paradigm="diffusion")

        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=loss_val,
            error=loss_val,
        )

    return step
