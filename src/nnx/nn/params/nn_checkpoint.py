from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

from ..enum.checkpoints import Checkpoints
from ..params.nn_iteration_data_point import NNIterationDataPoint
from ..params.nn_model_params import NNModelParams
from ..params.nn_params import NNParams


def _checkpoint_path(run: str, type: Checkpoints, root: Optional[str] = None) -> str:
    """Resolve the on-disk path for a checkpoint. Defaults to cwd-relative
    so existing notebook code stays untouched."""
    base = root if root is not None else "."
    return os.path.join(base, "runs", run, "checkpoints", str(type) + ".pt")


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

    def save(self, run: str, type: Checkpoints, root: Optional[str] = None) -> None:
        self.to_file(path=_checkpoint_path(run, type, root=root))

    @staticmethod
    def from_file(path: str) -> Optional[NNCheckpoint]:
        """Load a pickled NNCheckpoint.

        SECURITY: This calls torch.load(weights_only=False), which unpickles
        arbitrary Python objects from the file. NEVER call this on a
        checkpoint file from an untrusted source — a malicious .pt file can
        execute arbitrary code at load time. The default ./runs/<id>/checkpoints/
        layout assumes the files were produced locally by NNCheckpoint.save.

        Returns None if the path doesn't exist or the loaded object isn't
        an NNCheckpoint instance.
        """
        if not os.path.exists(path):
            return None

        # NNCheckpoint files are pickled Python objects (not bare state dicts),
        # so the weights_only=True default introduced in torch>=2.6 would raise
        # UnpicklingError. See the security note in the docstring before
        # widening this to externally-sourced files.
        ret = torch.load(path, weights_only=False)

        if not isinstance(ret, NNCheckpoint):
            return None

        return ret

    @staticmethod
    def load(run: str, type: Checkpoints, root: Optional[str] = None) -> Optional[NNCheckpoint]:
        return NNCheckpoint.from_file(path=_checkpoint_path(run, type, root=root))
