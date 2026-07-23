"""Callback protocol and standard callbacks for NNModel.train().

The protocol gives `train()` discrete lifecycle hooks (on_train_begin,
on_epoch_begin, on_epoch_end, on_train_end) and lets a callback signal
early termination by setting `ctx.should_stop = True`.

The legacy callable signature `Callable[[List[NNIterationDataPoint]], None]`
is preserved via _LegacyCallback (which adapts to on_epoch_end) so existing
notebooks keep working.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from .params.nn_checkpoint import NNCheckpoint, NNCheckpointTransform, _snapshot_state_dict
from .params.nn_iteration_data_point import NNIterationDataPoint

if TYPE_CHECKING:
    from .nn_model import _CallbackContext


class Callback:
    """Base class for training callbacks. Override any subset of the hooks."""

    def on_train_begin(self, ctx: _CallbackContext) -> None:
        pass

    def on_epoch_begin(self, ctx: _CallbackContext) -> None:
        pass

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        pass

    def on_train_end(self, ctx: _CallbackContext) -> None:
        pass

    def checkpoint_transforms(self) -> tuple[NNCheckpointTransform, ...]:
        """Completed topology transforms to persist on the final checkpoint."""
        return ()


class _LegacyCallback(Callback):
    """Adapts a plain Callable[[List[IDP]], None] into a Callback.

    Old notebook code: `model.train(params, callbacks=[lambda idps: plot(idps)])`.
    The original train() called the callable after each epoch with the running
    idps list and a `clear_output(wait=True)` first. This shim preserves both.
    """

    def __init__(self, fn: Callable[[list[NNIterationDataPoint]], None]):
        self._fn = fn
        # Lazy resolution of IPython.display.clear_output, cached on first
        # use. Keeps `import nnx` from pulling in IPython for users who
        # never use a legacy lambda-style callback, AND avoids the
        # per-epoch dict-lookup cost of `from ... import` in the hot path.
        self._clear_output: Optional[Callable] = None

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        if self._clear_output is None:
            try:
                from IPython.display import clear_output
            except ImportError:
                clear_output = lambda **_kwargs: None

            self._clear_output = clear_output

        self._clear_output(wait=True)
        self._fn(ctx.idps)


class EarlyStopping(Callback):
    """Stop training when the monitored metric stops improving.

    Args:
        monitor: which IDP field to track. "val_edp.error" (default), "val_edp.loss",
                 "train_edp.error", or "train_edp.loss".
        patience: epochs with no improvement before stopping.
        min_delta: minimum change to qualify as improvement.
        mode: "min" (default) for loss/error; "max" for accuracy/f1.
    """

    def __init__(
        self,
        monitor: str = "val_edp.error",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        valid_monitors = {
            "val_edp.error",
            "val_edp.loss",
            "train_edp.error",
            "train_edp.loss",
        }
        if monitor not in valid_monitors:
            raise ValueError(f"monitor must be one of {sorted(valid_monitors)}, got {monitor!r}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._wait: int = 0

    def _lookup_monitored(self, idp: NNIterationDataPoint) -> Optional[float]:
        # Named distinctly from nnx._metrics._resolve_metric (the
        # val→train / error→loss fallback resolver) — this one just
        # dereferences the user's `monitor` string, e.g. "val_edp.loss".
        edp_name, _, field = self.monitor.partition(".")
        edp = getattr(idp, edp_name, None)
        if edp is None:
            return None
        return getattr(edp, field, None)

    def on_train_begin(self, ctx: _CallbackContext) -> None:
        # Fresh run, fresh patience: without this reset, reusing one
        # EarlyStopping instance across train() calls compares against
        # the previous run's best and can stop the new run immediately.
        self._best = None
        self._wait = 0

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        if ctx.idp is None:
            return
        current = self._lookup_monitored(ctx.idp)
        if current is None:
            return
        if self._best is None or self._is_improvement(current, self._best):
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                ctx.should_stop = True


class ModelCheckpoint(Callback):
    """Save a custom-tagged checkpoint at user-specified epochs.

    The standard train() loop already saves FIRST / Q1 / Q2 / Q3 / LAST / BEST
    via the Checkpoints enum. This callback adds ad-hoc save points outside
    that cycle — useful for sampling at fixed milestones (e.g., epoch 10,
    20, 50) for downstream inspection.

    Each match writes ``<cwd>/runs/<run.id>/checkpoints/<tag>_e<epoch>.pt``
    — cwd-relative, matching what :meth:`NNRun.save` and :class:`NNCheckpoint`
    use when called from inside :meth:`NNModel.train` (the train() entry
    point doesn't accept a ``root=`` parameter). The epoch suffix
    prevents successive matches from overwriting each other when
    ``epochs`` has multiple entries.

    Args:
        epochs: list of 0-indexed epoch numbers at which to save. Empty /
            None means the callback never fires (and never saves anything).
        tag: prefix in the filename, defaults to ``"custom"``.
    """

    def __init__(self, epochs: Optional[list[int]] = None, tag: str = "custom"):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) is None:
            raise ValueError(
                f"ModelCheckpoint tag must be a non-empty filename-safe slug "
                f"containing only letters, digits, '.', '_', or '-'; got {tag!r}"
            )
        self.epochs = set(epochs or [])
        self.tag = tag

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        if ctx.epoch not in self.epochs or ctx.idp is None:
            return
        # Build the NNCheckpoint inline — same shape as NNModel._save_checkpoints
        # but with a custom path so it doesn't collide with the Checkpoints enum
        # tags. Goes through NNCheckpoint.to_file for the atomic-write guarantee.
        ckpt = NNCheckpoint(
            idp=ctx.idp,
            model_params=ctx.model.params,
            net_params=ctx.model.net_params,
            net_state=_snapshot_state_dict(ctx.model.net.state_dict()),
        )
        # Same cwd-relative `runs/<id>/checkpoints/` layout NNCheckpoint.save
        # uses through _checkpoint_path; we hand-build the path here because
        # the user-supplied tag isn't part of the Checkpoints enum.
        path = os.path.join(
            "runs",
            ctx.run.id,
            "checkpoints",
            f"{self.tag}_e{ctx.epoch}.pt",
        )
        # The training loop flushes this only after history and LAST have
        # committed, so callback artifacts cannot advertise a partial epoch.
        ctx.deferred_checkpoint_writes.append(lambda: ckpt.to_file(path))


class LRMonitor(Callback):
    """Logs the current LR each epoch. History exposed at `.history`."""

    def __init__(self):
        self.history: list[float] = []

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        lr = ctx.optimizer.param_groups[0]["lr"]
        self.history.append(lr)


def _edp_metric_iter(edp):
    """Yield (name, value) pairs for the standard EDP fields plus any
    user-supplied extras. Skips None values."""
    if edp is None:
        return
    for name in ("loss", "error", "accuracy", "f1", "precision", "recall"):
        v = getattr(edp, name, None)
        if v is not None:
            yield name, v
    for name, v in (getattr(edp, "extra", None) or {}).items():
        yield f"extra/{name}", v


class TensorBoardCallback(Callback):
    """Stream train/val metrics + LR to a TensorBoard SummaryWriter.

    Requires `tensorboard` to be installed — imported lazily so users who
    don't use this callback don't pay the dependency cost.

    Args:
        log_dir: directory passed to SummaryWriter. None lets TensorBoard
            pick its default (runs/<datetime>).
        flush_each_epoch: when True (default), calls writer.flush() so
            partial training is visible in TB even if the process crashes.
    """

    def __init__(self, log_dir: Optional[str] = None, flush_each_epoch: bool = True):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "TensorBoardCallback requires `tensorboard`. "
                "Install with `pip install thekaveh-nnx[tensorboard]` or `pip install tensorboard`."
            ) from e
        self._writer = SummaryWriter(log_dir=log_dir)
        self._flush_each_epoch = flush_each_epoch

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        idp = ctx.idp
        if idp is None:
            return
        step = idp.epoch_idx

        for name, v in _edp_metric_iter(idp.train_edp):
            self._writer.add_scalar(f"train/{name}", v, step)
        for name, v in _edp_metric_iter(idp.val_edp):
            self._writer.add_scalar(f"val/{name}", v, step)
        self._writer.add_scalar("lr", ctx.optimizer.param_groups[0]["lr"], step)

        if self._flush_each_epoch:
            self._writer.flush()

    def on_train_end(self, ctx: _CallbackContext) -> None:
        self._writer.close()


class WandbCallback(Callback):
    """Stream train/val metrics + LR to Weights & Biases.

    Requires `wandb` — lazily imported. Pass `project=` to start a new run,
    or `wandb_run=` to attach to an externally-managed run.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        wandb_run=None,
        **init_kwargs,
    ):
        if wandb_run is None:
            try:
                import wandb
            except ImportError as e:
                raise ImportError(
                    "WandbCallback requires `wandb`. Install with `pip install thekaveh-nnx[wandb]` or `pip install wandb`."
                ) from e
            self._run = wandb.init(project=project, **init_kwargs)
            self._owns_run = True
        else:
            self._run = wandb_run
            self._owns_run = False

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        idp = ctx.idp
        if idp is None:
            return

        log: dict = {"epoch": idp.epoch_idx, "lr": ctx.optimizer.param_groups[0]["lr"]}
        for name, v in _edp_metric_iter(idp.train_edp):
            log[f"train/{name}"] = v
        for name, v in _edp_metric_iter(idp.val_edp):
            log[f"val/{name}"] = v
        self._run.log(log, step=idp.epoch_idx)

    def on_train_end(self, ctx: _CallbackContext) -> None:
        if self._owns_run:
            self._run.finish()
