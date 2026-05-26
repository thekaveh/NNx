from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

import pandas as pd
import yaml

from ..enum.checkpoints import Checkpoints
from ..params.nn_checkpoint import NNCheckpoint
from ..params.nn_iteration_data_point import NNIterationDataPoint
from ..params.nn_model_params import NNModelParams
from ..params.nn_params import NNParams
from ..params.nn_train_params import NNTrainParams

if TYPE_CHECKING:
    from ...trainer.params import NNTrainerParams


def _runs_root(root: Optional[str] = None) -> str:
    """Resolve the on-disk root for `runs/`. Defaults to `<cwd>/runs` so
    existing notebook callers (which pass nothing) keep their layout."""
    return os.path.join(root if root is not None else os.getcwd(), "runs")


def _atomic_write_text(path: str, content: str) -> None:
    """Write `content` to `path` atomically — fsync, rename. A
    KeyboardInterrupt during the rename either leaves the prior file
    intact OR the new file fully written; never a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fsync isn't supported on every filesystem (e.g., some
            # network mounts). Atomic rename is still useful even
            # without the fsync guarantee.
            pass
    os.replace(tmp, path)


def _point_best(best_run_path: str, run_path: str) -> None:
    """Make `best_run_path` point at `run_path`. Uses a symlink where the
    platform supports it (POSIX and Windows-with-developer-mode); falls
    back to writing a `best/POINTER.txt` text file with the run path.
    Either way ``_read_best_pointer`` can recover the target run."""
    if os.path.lexists(best_run_path):
        if os.path.islink(best_run_path):
            os.remove(best_run_path)
        else:
            # Existing pointer directory from a prior fallback — clear it.
            import shutil
            shutil.rmtree(best_run_path)
    try:
        os.symlink(src=run_path, dst=best_run_path)
    except (OSError, NotImplementedError):
        # Windows without developer mode: write a pointer file instead.
        # Atomic write so a KeyboardInterrupt during the fallback can't
        # leave a half-written POINTER.txt that confuses _read_best_pointer.
        os.makedirs(best_run_path, exist_ok=True)
        _atomic_write_text(os.path.join(best_run_path, "POINTER.txt"), run_path)


def _read_best_pointer(best_run_path: str) -> Optional[str]:
    """Return the run id currently pointed to by `runs/best`, or None when
    nothing is pointed there yet. Supports both symlink and POINTER.txt
    fallback layouts."""
    if not os.path.lexists(best_run_path):
        return None
    if os.path.islink(best_run_path):
        # Resolve the symlink to a run path → the basename is the run id.
        target = os.readlink(best_run_path)
        return os.path.basename(target.rstrip(os.sep)) or None
    pointer_file = os.path.join(best_run_path, "POINTER.txt")
    if os.path.isfile(pointer_file):
        with open(pointer_file) as f:
            target = f.read().strip()
        return os.path.basename(target.rstrip(os.sep)) or None
    return None


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

    # Optional trainer-mode marker. Populated by Trainer.train(); None for
    # NNModel.train()-produced runs. state() omits it when None so existing
    # run.id hashes for NNModel runs are preserved exactly.
    trainer : Optional[NNTrainerParams]           = field(default=None)

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
        # `trainer` is omitted when None so existing NNModel runs hash to
        # the same run.id as before this field existed. Same omit-when-
        # default pattern as NNTrainParams.seed / save_phase_checkpoints
        # and NNOptimParams.param_groups.
        if self.trainer is not None:
            state['trainer'] = self.trainer.state()

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
        metadata_path = os.path.join(run_path, "metadata.yaml")

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        # All three writes go through the atomic write helper so a
        # KeyboardInterrupt mid-save leaves either the old file or the
        # new file, never a half-written one. This is what makes the
        # post-R3 "incremental save" claim actually safe — without
        # atomicity, a partial write here corrupts the run.
        _atomic_write_text(yaml_path, yaml.dump(self.state()))

        # Env snapshot: written separately so it does NOT contribute to
        # run.id (which is md5(state())). Captures library/torch/python
        # versions + git commit so a run.yaml from six months ago is
        # debuggable even if the library has moved on.
        from ...seeding import env_snapshot
        _atomic_write_text(metadata_path, yaml.safe_dump(env_snapshot()))

        _atomic_write_text(
            csv_path,
            pd.json_normalize(data=[idp.state() for idp in self.idps]).to_csv(),
        )

        if not os.path.lexists(best_run_path) or not os.path.exists(best_run_path):
            # Either no symlink yet, or one dangling after a repo move — repoint.
            _point_best(best_run_path, run_path)
        else:
            # Resolve the current best target via the symlink OR pointer file —
            # `NNCheckpoint.load(run="best", ...)` only works under a symlink
            # layout, so on the Windows pointer-file fallback we go through
            # the run id explicitly to make a fair comparison.
            best_run_id = _read_best_pointer(best_run_path)
            best_ckpt = (
                NNCheckpoint.load(run=best_run_id, type=Checkpoints.BEST, root=root)
                if best_run_id is not None else None
            )
            best_err = _best_err(best_ckpt)
            curr_err = _best_err(NNCheckpoint.load(run=self.id, type=Checkpoints.BEST, root=root))

            if curr_err < best_err:
                _point_best(best_run_path, run_path)

        return self

    @staticmethod
    def load(id: str, root: Optional[str] = None) -> NNRun:
        run_path = os.path.join(_runs_root(root), id)
        yaml_path = os.path.join(run_path, "run.yaml")
        csv_path = os.path.join(run_path, "idps.csv")

        with open(yaml_path) as f:
            rep = yaml.load(f, Loader=yaml.FullLoader)

        idps = pd.read_csv(csv_path).to_dict(orient='records')

        # Lazy import for trainer params — keeps `nnx.nn.params` importable
        # without dragging the trainer subpackage in, and avoids a cycle
        # if anything in `nnx.trainer` ever needs to import NNRun.
        trainer_state = rep.get('trainer')
        if trainer_state is not None:
            from ...trainer.params import NNTrainerParams
            trainer = NNTrainerParams.from_state(trainer_state)
        else:
            trainer = None

        return NNRun(
            net     = NNParams.from_state(rep['net'])
            , train = NNTrainParams.from_state(rep['train'])
            , model = NNModelParams.from_state(rep['model'])
            , trainer = trainer
            , idps  = [NNIterationDataPoint.from_state(idp) for idp in idps]
        )

    @staticmethod
    def all(root: Optional[str] = None) -> list[NNRun]:
        """List every saved NNRun under the runs root, skipping the `best`
        pointer. Returns [] when the runs/ directory doesn't exist yet.
        Non-directory entries (stray files, .DS_Store) are filtered out
        so they don't trigger spurious NNRun.load failures."""
        runs_root = _runs_root(root)
        if not os.path.isdir(runs_root):
            return []
        result: list[NNRun] = []
        for entry in os.listdir(runs_root):
            if entry == "best":
                continue
            if not os.path.isdir(os.path.join(runs_root, entry)):
                continue
            # Defensive: a directory in runs/ that lacks run.yaml isn't a
            # real run (could be a leftover from an aborted experiment).
            if not os.path.isfile(os.path.join(runs_root, entry, "run.yaml")):
                continue
            result.append(NNRun.load(id=entry, root=root))
        return result
