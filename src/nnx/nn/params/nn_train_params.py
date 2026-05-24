from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from torch.utils.data import DataLoader

from ..enum.optims import Optims
from ..params.nn_optim_params import NNOptimParams
from ..params.nn_scheduler_params import NNSchedulerParams


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    n_epochs        : int
    scheduler       : NNSchedulerParams     = NNSchedulerParams(patience=8, cooldown=2, factor=95e-2, threshold=1e-3, min_lr=1e-7)
    optim           : NNOptimParams         = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5)

    train_loader    : Optional[DataLoader]  = field(repr=False, default=None)
    val_loader      : Optional[DataLoader]  = field(repr=False, default=None)

    def with_train_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, train_loader=value)

    def with_val_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, val_loader=value)

    def __str__(self):
        return f"Train={{n_epochs={self.n_epochs}, Optim={self.optim}, Scheduler={self.scheduler}}}"

    def state(self):
        return dict(
            n_epochs    = self.n_epochs
            , optim     = self.optim.state()
            , scheduler = self.scheduler.state()
        )

    @staticmethod
    def from_state(state: dict) -> NNTrainParams:
        return NNTrainParams(
            n_epochs    = state['n_epochs']
            , optim     = NNOptimParams.from_state(state['optim'])
            , scheduler = NNSchedulerParams.from_state(state['scheduler'])
        )
