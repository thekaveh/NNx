"""Alternative training paradigms.

Each factory returns a :class:`nnx.TrainStepFn` for the
``train_step_fn=`` hook on :meth:`NNModel.train`. The training loop,
checkpoint cadence, callbacks, and persistence are unchanged — only
the per-batch update is swapped.

Public surface — re-exported from the top-level ``nnx`` package:

  - :func:`kd_train_step_factory` — Hinton-style knowledge distillation.
  - :func:`feature_kd_train_step_factory` — FitNets-style feature
    distillation (logit-KD + named intermediate-layer MSE).
  - :func:`born_again_train` — iterated self-distillation across G
    generations, layered on top of :func:`kd_train_step_factory`.
  - :func:`simclr_train_step_factory` + :func:`nt_xent_loss` — SimCLR
    contrastive learning.
  - :func:`mixup_train_step_factory` — Mixup augmentation.
  - :func:`cutmix_train_step_factory` — CutMix augmentation (image data).
"""

from __future__ import annotations

from .augmentation import cutmix_train_step_factory, mixup_train_step_factory
from .born_again import born_again_train
from .contrastive import nt_xent_loss, simclr_train_step_factory
from .distillation import feature_kd_train_step_factory, kd_train_step_factory

__all__ = [
    "kd_train_step_factory",
    "feature_kd_train_step_factory",
    "born_again_train",
    "simclr_train_step_factory",
    "nt_xent_loss",
    "mixup_train_step_factory",
    "cutmix_train_step_factory",
]
