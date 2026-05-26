"""Multi-optimizer Trainer.

Public surface mirrors `nnx.finetune` — a focused subpackage you can
import directly (`from nnx.trainer import Trainer, NNTrainerParams`)
or via the top-level package re-exports.
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
