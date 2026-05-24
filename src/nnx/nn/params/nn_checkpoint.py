from __future__ import annotations

import os
import torch

from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict

from ..enum.checkpoints import Checkpoints

from ..params.nn_params import NNParams
from ..params.nn_model_params import NNModelParams
from ..params.nn_iteration_data_point import NNIterationDataPoint
    
@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpoint:
    net_params  : NNParams
    net_state   : OrderedDict
    model_params: NNModelParams
    idp         : NNIterationDataPoint
    
    def to_file(self, path: str) -> None:
        dir_path = os.path.dirname(path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        torch.save(self, path)
        
    def save(self, run: str, type: Checkpoints) -> None:
        self.to_file(
            path=os.path.join(".", "runs", run, "checkpoints", str(type) + ".pt")
        )
        
    @staticmethod
    def from_file(path: str) -> Optional[NNCheckpoint]:
        if not os.path.exists(path):
            return None

        # NNCheckpoint files are pickled Python objects (not bare state dicts),
        # so the weights_only=True default introduced in torch>=2.6 would raise
        # UnpicklingError. Trust the file (it was produced by NNCheckpoint.save
        # on the local machine) and load with weights_only=False.
        ret = torch.load(path, weights_only=False)

        if not isinstance(ret, NNCheckpoint):
            return None

        return ret
    
    @staticmethod
    def load(run: str, type: Checkpoints) -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_file(
            path=os.path.join(".", "runs", run, "checkpoints", str(type) + ".pt")
        )