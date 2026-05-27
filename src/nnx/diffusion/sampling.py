"""Reverse-diffusion sampling — generate samples from a trained DDPM.

Standard ancestral sampling: start from x_T ~ N(0, I), iterate t = T-1
down to 0, and at each step predict the noise via ``model.net(x_t, t)``
and step ``x_{t-1}`` via the DDPM posterior formula.

The sampler is a single free function; no class state. It runs entirely
under ``torch.no_grad`` and ``model.net.eval()`` mode.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..nn.nn_model import NNModel
from .schedules import NoiseSchedule


@torch.no_grad()
def sample(
    model: NNModel,
    schedule: NoiseSchedule,
    shape: tuple[int, ...],
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Run T reverse-diffusion steps and return samples drawn from the
    distribution the model was trained on.

    Args:
        model: an :class:`NNModel` whose ``.net`` is the trained
            denoising network (e.g., :class:`DiffusionMLP` or any
            ``forward(x, t) -> ε`` module).
        schedule: the same :class:`NoiseSchedule` used during training.
            Indexed tensors are moved to ``device`` lazily.
        shape: full tensor shape to generate, e.g., ``(256, 2)`` for
            256 2D samples.
        device: target device. Defaults to ``model.device``.
        generator: optional torch.Generator for reproducible sampling
            (pass one built with ``torch.Generator(device).manual_seed(...)``).

    Returns:
        A tensor of shape ``shape`` carrying the generated samples.
    """
    if device is None:
        device = model.device

    # Migrate the schedule's indexed tensors once so the per-step loop
    # doesn't re-transfer them.
    sched = schedule.to(device)
    model.net.eval()

    x = torch.randn(*shape, device=device, generator=generator)

    for t_int in reversed(range(sched.T)):
        # Batch-broadcast the timestep so the network's t-conditioning
        # works whether called with B==1 or B==shape[0].
        t = torch.full((shape[0],), t_int, dtype=torch.long, device=device)

        eps_pred = model.net(x, t)

        beta_t = sched.betas[t_int]
        alpha_t = sched.alphas[t_int]
        alpha_bar_t = sched.alphas_cumprod[t_int]

        # Posterior mean (DDPM eq. 11): predict x_0 implicitly through eps,
        # then the posterior of q(x_{t-1} | x_t, x_0).
        coef = beta_t / (1.0 - alpha_bar_t).sqrt()
        mean = (1.0 / alpha_t.sqrt()) * (x - coef * eps_pred)

        if t_int > 0:
            # Standard DDPM: re-inject noise scaled by the posterior std
            # at every step except the final t=0 (the boundary case is a
            # deterministic mean — adding noise would not match the training
            # objective).
            noise = (
                torch.randn_like(x)
                if generator is None
                else torch.randn(
                    x.shape,
                    generator=generator,
                    device=device,
                    dtype=x.dtype,
                )
            )
            x = mean + sched.posterior_variance[t_int].sqrt() * noise
        else:
            x = mean

    return x
