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


def _atomic_torch_save(obj, path: str) -> None:
    """torch.save(obj, path) wrapped with tmp + rename so a
    KeyboardInterrupt during the pickle never leaves a half-written
    .pt file at the destination."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@dataclass(frozen=True, kw_only=True, slots=True)
class NNCheckpoint:
    net_params: NNParams
    net_state: OrderedDict
    model_params: NNModelParams
    idp: NNIterationDataPoint

    def to_file(self, path: str) -> None:
        """Atomically write this NNCheckpoint to `path`.

        Writes to ``<path>.tmp`` first and renames into place so a
        KeyboardInterrupt during the underlying torch.save can never
        leave a half-written checkpoint at the destination — matching
        the atomicity guarantee NNRun.save offers for YAML/CSV.
        """
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        _atomic_torch_save(self, path)

    def save(
        self,
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
        optimizer_state: Optional[OrderedDict] = None,
    ) -> None:
        """Save the checkpoint to disk atomically.

        When `optimizer_state` is supplied, a sibling file is written at
        ``<id>/checkpoints/<type>.opt.pt`` holding the optimizer state dict.
        This sidecar is used by NNModel.train(resume_from=...) to warm-resume
        with the prior optimizer momentum / Adam state.
        """
        ckpt_path = _checkpoint_path(run, type, root=root)
        self.to_file(path=ckpt_path)
        if optimizer_state is not None:
            _atomic_torch_save(optimizer_state, ckpt_path + ".opt.pt")

    @staticmethod
    def load_optimizer_state(
        run: str,
        type: Checkpoints,
        root: Optional[str] = None,
    ) -> Optional[OrderedDict]:
        """Load the optimizer state sidecar for a checkpoint. Returns None
        when no sidecar exists (e.g., checkpoints written before resume
        support was added).

        Loaded with ``weights_only=True`` — the optimizer state-dict
        contains only tensors and standard scalar/dict/list types, so the
        strict loader works AND it removes the arbitrary-code-execution
        risk that the main NNCheckpoint.from_file documents."""
        path = _checkpoint_path(run, type, root=root) + ".opt.pt"
        if not os.path.exists(path):
            return None
        return torch.load(path, weights_only=True)

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
