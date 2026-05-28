"""Alternative training paradigms.

Each factory returns a :class:`nnx.TrainStepFn` for the
``train_step_fn=`` hook on :meth:`NNModel.train`. The training loop,
checkpoint cadence, callbacks, and persistence are unchanged — only
the per-batch update is swapped.

Public surface — re-exported from the top-level ``nnx`` package:

  - :func:`kd_train_step_factory` — Hinton-style knowledge distillation.
  - :func:`simclr_train_step_factory` + :func:`nt_xent_loss` — SimCLR
    contrastive learning.
  - :func:`mixup_train_step_factory` — Mixup augmentation.
  - :func:`cutmix_train_step_factory` — CutMix augmentation (image data).
  - :func:`moe_train_step_factory` — Mixture-of-Experts supervised step
    with Switch-style load-balancing aux loss.
"""

from __future__ import annotations

from .augmentation import cutmix_train_step_factory, mixup_train_step_factory
from .contrastive import nt_xent_loss, simclr_train_step_factory
from .distillation import kd_train_step_factory
from .moe import moe_train_step_factory

__all__ = [
    "kd_train_step_factory",
    "simclr_train_step_factory",
    "nt_xent_loss",
    "mixup_train_step_factory",
    "cutmix_train_step_factory",
    "moe_train_step_factory",
]
