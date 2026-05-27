"""Multi-optimizer Trainer — parallel to :meth:`NNModel.train` for
scenarios where the per-batch update isn't a single supervised
forward/backward/step. Designed around GAN-style G/D alternation but
applicable to actor-critic, energy-based models, contrastive
multi-head training, or any other multi-optimizer paradigm.

Public surface — re-exported from the top-level ``nnx`` package:

  - :class:`Trainer` — orchestrator, constructed around one NNModel.
  - :class:`NNTrainerParams` — frozen dataclass with name-keyed
    ``optims`` / ``schedulers`` dicts; round-trips through ``state()``
    like every other params dataclass.
  - :class:`TrainerStepContext` — per-batch bundle passed into the
    user-supplied ``trainer_step_fn``.
  - :class:`TrainerStepFn` — type alias for the step function signature.

The Trainer writes the same on-disk artifacts (NNRun + per-tag
NNCheckpoint) NNModel.train does, with an extra ``trainer`` block on
NNRun preserving the multi-optim config.
"""

from __future__ import annotations

from .params import NNTrainerParams
from .trainer import Trainer, TrainerStepContext, TrainerStepFn

__all__ = [
    "Trainer",
    "TrainerStepContext",
    "TrainerStepFn",
    "NNTrainerParams",
]
