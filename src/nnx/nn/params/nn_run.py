from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from typing import Optional

import pandas as pd
import yaml

from ..enum.checkpoints import Checkpoints
from ..params.nn_checkpoint import NNCheckpoint
from ..params.nn_iteration_data_point import NNIterationDataPoint
from ..params.nn_model_params import NNModelParams
from ..params.nn_params import NNParams
from ..params.nn_train_params import NNTrainParams


def _runs_root(root: Optional[str] = None) -> str:
    """Resolve the on-disk root for `runs/`. Defaults to `<cwd>/runs` so
    existing notebook callers (which pass nothing) keep their layout."""
    return os.path.join(root if root is not None else os.getcwd(), "runs")


def _best_err(checkpoint: Optional[NNCheckpoint]) -> float:
    """Pull the error metric from a checkpoint, preferring val over train.
    Returns +inf for missing checkpoints or fully missing metrics so caller
    comparisons always prefer the *new* run when there's no prior signal."""
    if checkpoint is None:
        return float("inf")
    edp = checkpoint.idp.val_edp if checkpoint.idp.val_edp is not None else checkpoint.idp.train_edp
    if edp is None or edp.error is None:
        return float("inf")
    return edp.error

@dataclass(frozen=True, kw_only=True, slots=True)
class NNRun:
    net     : NNParams
    train   : NNTrainParams
    model   : NNModelParams

    _id     : Optional[str]                         = field(repr=False, default=None)
    _state  : Optional[dict]                        = field(repr=False, default=None)
    idps    : Optional[list[NNIterationDataPoint]]  = field(repr=False, default=None)

    def __str__(self):
        return (
            "{"
            f"loss={self.model.loss}"
            f", device={self.model.device}"
            f", net={self.model.net}"

            f", dims={self.net.dims}"
            f", dropout={self.net.dropout_prob}"
            f", activation={self.net.activation}"
            f", n_heads={self.net.n_heads}"

            f", n_epochs={self.train.n_epochs}"

            f", max_lr={self.train.optim.max_lr}"
            f", momentum={self.train.optim.momentum}"
            f", decay={self.train.optim.weight_decay}"

            f", factor={self.train.scheduler.factor}"
            f", patience={self.train.scheduler.patience}"
            f", cooldown={self.train.scheduler.cooldown}"
            f", threshold={self.train.scheduler.threshold}"
            f", min_lr={self.train.scheduler.min_lr}"
            "}"
        )

    @property
    def id(self) -> str:
        return self._id

    def __post_init__(self):
        state = dict(
            model   = self.model.state()
            , net   = self.net.state()
            , train = self.train.state()
        )

        id = hashlib.md5(str(state).encode('utf-8')).hexdigest()

        object.__setattr__(self, '_id', id)
        object.__setattr__(self, '_state', {"id": id, **state})

    def state(self) -> dict:
        return self._state

    def with_idps(self, value: list[NNIterationDataPoint]) -> NNRun:
        return replace(self, idps=value)

    def checkpoints(self, root: Optional[str] = None) -> list[NNCheckpoint]:
        return [
            NNCheckpoint.load(run=self.id, type=Checkpoints.FIRST, root=root)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.Q1, root=root)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.Q2, root=root)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.Q3, root=root)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.LAST, root=root)
        ]

    def save(self, root: Optional[str] = None) -> NNRun:
        runs_root = _runs_root(root)
        run_path = os.path.join(runs_root, self.id)
        best_run_path = os.path.join(runs_root, "best")

        csv_path = os.path.join(run_path, "idps.csv")
        yaml_path = os.path.join(run_path, "run.yaml")

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        with open(yaml_path, 'w') as f:
            yaml.dump(self.state(), f)

        pd.json_normalize(
            data=[idp.state() for idp in self.idps]
        ).to_csv(csv_path)

        if not os.path.lexists(best_run_path):
            os.symlink(src=run_path, dst=best_run_path)
        elif not os.path.exists(best_run_path):
            # Dangling symlink (e.g. after moving the repo) — replace it.
            os.remove(path=best_run_path)
            os.symlink(src=run_path, dst=best_run_path)
        else:
            best_err = _best_err(NNCheckpoint.load(run="best", type=Checkpoints.BEST, root=root))
            curr_err = _best_err(NNCheckpoint.load(run=self.id, type=Checkpoints.BEST, root=root))

            if curr_err < best_err:
                os.remove(path=best_run_path)
                os.symlink(src=run_path, dst=best_run_path)

        return self

    @staticmethod
    def load(id: str, root: Optional[str] = None) -> NNRun:
        run_path = os.path.join(_runs_root(root), id)
        yaml_path = os.path.join(run_path, "run.yaml")
        csv_path = os.path.join(run_path, "idps.csv")

        with open(yaml_path) as f:
            rep = yaml.load(f, Loader=yaml.FullLoader)

        idps = pd.read_csv(csv_path).to_dict(orient='records')

        return NNRun(
            net     = NNParams.from_state(rep['net'])
            , train = NNTrainParams.from_state(rep['train'])
            , model = NNModelParams.from_state(rep['model'])
            , idps  = [NNIterationDataPoint.from_state(idp) for idp in idps]
        )

    @staticmethod
    def all(root: Optional[str] = None) -> list[NNRun]:
        runs_root = _runs_root(root)
        return [NNRun.load(id=id, root=root) for id in os.listdir(runs_root) if id != "best"]
