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
  - :func:`moe_train_step_factory` — Mixture-of-Experts supervised step
    with Switch-style load-balancing aux loss.
  - :func:`jepa_train_step_factory` + :func:`build_target_encoder`
    + :func:`update_ema` + :func:`random_block_mask` +
    :class:`JEPAPredictor` — I-JEPA self-supervised learning in
    latent space.
  - :func:`jepa_objective` / :class:`JEPAObjective` — I-JEPA as an
    objective (FEAT-040): the shared update engine owns the update and the
    EMA target advances once per committed update.
  - :func:`dpo_train_step_factory` — Direct Preference Optimization
    (Rafailov et al., 2023): chosen-vs-rejected log-ratio objective
    against a frozen reference policy.

Offline teacher distributions (FEAT-022) — exported from ``nnx.paradigms``
(``nnx.paradigms.offline_distillation``), not the top-level package:
:class:`TeacherRecord` / :class:`TeacherDataset` distill a student from
stored, permissioned teacher probabilities, trained by
:class:`OfflineDistillationObjective` (``KL(teacher ‖ student)`` plus an
optional hard CE, at temperature 1) and scored by
:func:`evaluate_offline`.
"""

from __future__ import annotations

from .augmentation import cutmix_train_step_factory, mixup_train_step_factory
from .born_again import born_again_train
from .contrastive import nt_xent_loss, simclr_train_step_factory
from .distillation import feature_kd_train_step_factory, kd_train_step_factory
from .dpo import dpo_train_step_factory
from .jepa import (
    JEPAPredictor,
    JEPATrainStep,
    build_target_encoder,
    jepa_train_step_factory,
    random_block_mask,
    update_ema,
)
from .jepa_objective import JEPAObjective, jepa_objective
from .moe import moe_train_step_factory
from .offline_distillation import (
    OfflineDistillationObjective,
    TeacherDataset,
    TeacherRecord,
    evaluate_offline,
    read_teacher_records,
    write_teacher_records,
)

__all__ = [
    "kd_train_step_factory",
    "feature_kd_train_step_factory",
    "born_again_train",
    "simclr_train_step_factory",
    "nt_xent_loss",
    "mixup_train_step_factory",
    "cutmix_train_step_factory",
    "moe_train_step_factory",
    "jepa_train_step_factory",
    "build_target_encoder",
    "update_ema",
    "random_block_mask",
    "JEPAPredictor",
    "JEPAObjective",
    "jepa_objective",
    "JEPATrainStep",
    "dpo_train_step_factory",
    "TeacherRecord",
    "TeacherDataset",
    "OfflineDistillationObjective",
    "evaluate_offline",
    "read_teacher_records",
    "write_teacher_records",
]
