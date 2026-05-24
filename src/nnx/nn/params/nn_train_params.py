from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from torch.utils.data import DataLoader

from ..enum.optims import Optims
from ..params.nn_optim_params import NNOptimParams
from ..params.nn_scheduler_params import NNSchedulerParams


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    """Training configuration.

    `seed` pins every RNG that affects training (Python random, NumPy,
    torch CPU+CUDA, cuDNN) when NNModel.train() runs. None disables
    seeding (default).

    To preserve back-compat with previously-saved runs, `seed` is included
    in state() ONLY when set — so existing runs with no seed continue to
    hash to the same `run.id`.
    """

    n_epochs        : int
    scheduler       : NNSchedulerParams     = NNSchedulerParams(patience=8, cooldown=2, factor=95e-2, threshold=1e-3, min_lr=1e-7)
    optim           : NNOptimParams         = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5)

    seed            : Optional[int]         = None

    train_loader    : Optional[DataLoader]  = field(repr=False, default=None)
    val_loader      : Optional[DataLoader]  = field(repr=False, default=None)

    def with_train_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, train_loader=value)

    def with_val_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, val_loader=value)

    def __str__(self):
        return f"Train={{n_epochs={self.n_epochs}, seed={self.seed}, Optim={self.optim}, Scheduler={self.scheduler}}}"

    def state(self):
        d = dict(
            n_epochs    = self.n_epochs
            , optim     = self.optim.state()
            , scheduler = self.scheduler.state()
        )
        # Only emit `seed` into state() when it's set, so a NNTrainParams
        # created without a seed hashes to the same run.id as before this
        # field existed. Existing on-disk runs without a seed are loadable
        # via the default below.
        if self.seed is not None:
            d['seed'] = self.seed
        return d

    @staticmethod
    def from_state(state: dict) -> NNTrainParams:
        return NNTrainParams(
            n_epochs    = state['n_epochs']
            , optim     = NNOptimParams.from_state(state['optim'])
            , scheduler = NNSchedulerParams.from_state(state['scheduler'])
            , seed      = state.get('seed')
        )
