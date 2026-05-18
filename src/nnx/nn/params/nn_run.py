from __future__ import annotations

import os
import yaml
import torch
import hashlib

import pandas as pd

from typing import List, Optional
from dataclasses import asdict, dataclass, field, replace

from ..enum.checkpoints import Checkpoints

from ..params.nn_params import NNParams
from ..params.nn_checkpoint import NNCheckpoint
from ..params.nn_train_params import NNTrainParams
from ..params.nn_model_params import NNModelParams
from ..params.nn_iteration_data_point import NNIterationDataPoint

@dataclass(frozen=True, kw_only=True, slots=True)
class NNRun:
    net     : NNParams
    train   : NNTrainParams
    model   : NNModelParams 
    
    _id     : Optional[str]                         = field(repr=False, default=None)
    _state  : Optional[dict]                        = field(repr=False, default=None)
    idps    : Optional[List[NNIterationDataPoint]]  = field(repr=False, default=None)
    
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
    
    def with_idps(self, value: List[NNIterationDataPoint]) -> NNRun:
        return replace(self, idps=value)
    
    def checkpoints(self) -> List[NNCheckpoint]:
        return [
            NNCheckpoint.load(run=self.id, type=Checkpoints.FIRST)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.Q1)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.Q2)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.Q3)
            , NNCheckpoint.load(run=self.id, type=Checkpoints.LAST)
        ]
        
    def save(self) -> NNRun:
        run_path = os.path.join(os.getcwd(), "runs", self.id)
        best_run_path = os.path.join(os.getcwd(), "runs", "best")
        
        csv_path = os.path.join(run_path, "idps.csv")
        yaml_path = os.path.join(run_path, "run.yaml")

        if not os.path.exists(run_path):
            os.makedirs(run_path)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.state(), f)
        
        pd.json_normalize(
            data=[idp.state() for idp in self.idps]
        ).to_csv(csv_path)
            
        if not os.path.exists(best_run_path):
            os.symlink(src=run_path, dst=best_run_path)
        else:
            best_err = NNCheckpoint.load(run="best", type=Checkpoints.BEST).idp.val_edp.error
            curr_err = NNCheckpoint.load(run=self.id, type=Checkpoints.BEST).idp.val_edp.error
            
            if curr_err < best_err:
                os.remove(path=best_run_path)
                os.symlink(src=run_path, dst=best_run_path)
        
        return self
    
    @staticmethod
    def load(id: str) -> NNRun:
        run_path = os.path.join(os.getcwd(), "runs", id)
        yaml_path = os.path.join(run_path, "run.yaml")
        csv_path = os.path.join(run_path, "idps.csv")
        
        with open(yaml_path, 'r') as f:
            rep = yaml.load(f, Loader=yaml.FullLoader)
            
        idps = pd.read_csv(csv_path).to_dict(orient='records')
        
        return NNRun(
            net     = NNParams.from_state(rep['net'])
            , train = NNTrainParams.from_state(rep['train'])
            , model = NNModelParams.from_state(rep['model'])
            , idps  = [NNIterationDataPoint.from_state(idp) for idp in idps]
        )
        
    @staticmethod
    def all() -> List[NNRun]:
        return [NNRun.load(id=id) for id in os.listdir(os.path.join(os.getcwd(), "runs")) if id != "best"]