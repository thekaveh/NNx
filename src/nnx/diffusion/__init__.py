"""Diffusion (DDPM-style) training and sampling.

Public surface — re-exported from the top-level ``nnx`` package:

  - :class:`NoiseSchedulers` — enum-as-factory for LINEAR and COSINE
    variance schedules.
  - :class:`NoiseSchedule` — frozen dataclass holding the precomputed
    schedule tensors.
  - :class:`DiffusionMLP` — small conditional MLP suitable for
    tabular / low-dim diffusion (forward(x, t) → ε_pred).
  - :func:`diffusion_train_step_factory` — turns a schedule into a
    :class:`nnx.TrainStepFn` for use with :meth:`NNModel.train`.
  - :func:`sample` — reverse-diffusion sampler.

The diffusion paradigm hits NNx through the ``train_step_fn`` hook
introduced earlier — no Trainer or NNModel changes needed.
"""

from __future__ import annotations

from .nets import DiffusionMLP, sinusoidal_time_embed
from .sampling import sample
from .schedules import NoiseSchedule, NoiseSchedulers
from .training import diffusion_train_step_factory

__all__ = [
    "DiffusionMLP",
    "NoiseSchedule",
    "NoiseSchedulers",
    "diffusion_train_step_factory",
    "sample",
    "sinusoidal_time_embed",
]
