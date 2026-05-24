"""Callback protocol and standard callbacks for NNModel.train().

The protocol gives `train()` discrete lifecycle hooks (on_train_begin,
on_epoch_begin, on_epoch_end, on_train_end) and lets a callback signal
early termination by setting `ctx.should_stop = True`.

The legacy callable signature `Callable[[List[NNIterationDataPoint]], None]`
is preserved via _LegacyCallback (which adapts to on_epoch_end) so existing
notebooks keep working.
"""
from __future__ import annotations

from typing import Callable, List, Optional, TYPE_CHECKING

from IPython.display import clear_output

from .params.nn_iteration_data_point import NNIterationDataPoint

if TYPE_CHECKING:
    from .nn_model import _CallbackContext


class Callback:
    """Base class for training callbacks. Override any subset of the hooks."""

    def on_train_begin(self, ctx: "_CallbackContext") -> None:
        pass

    def on_epoch_begin(self, ctx: "_CallbackContext") -> None:
        pass

    def on_epoch_end(self, ctx: "_CallbackContext") -> None:
        pass

    def on_train_end(self, ctx: "_CallbackContext") -> None:
        pass


class _LegacyCallback(Callback):
    """Adapts a plain Callable[[List[IDP]], None] into a Callback.

    Old notebook code: `model.train(params, callbacks=[lambda idps: plot(idps)])`.
    The original train() called the callable after each epoch with the running
    idps list and a `clear_output(wait=True)` first. This shim preserves both.
    """

    def __init__(self, fn: Callable[[List[NNIterationDataPoint]], None]):
        self._fn = fn

    def on_epoch_end(self, ctx: "_CallbackContext") -> None:
        clear_output(wait=True)
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
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._wait: int = 0

    def _resolve_metric(self, idp: NNIterationDataPoint) -> Optional[float]:
        edp_name, _, field = self.monitor.partition(".")
        edp = getattr(idp, edp_name, None)
        if edp is None:
            return None
        return getattr(edp, field, None)

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def on_epoch_end(self, ctx: "_CallbackContext") -> None:
        if ctx.idp is None:
            return
        current = self._resolve_metric(ctx.idp)
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
    """Manual checkpoint trigger — useful for ad-hoc save points beyond the
    fixed Q1/Q2/Q3/BEST/LAST cycle in train().

    The standard cycle already saves BEST and LAST every epoch, so this exists
    for callers who want a named tag at a custom epoch. No-op unless `epochs`
    matches the current epoch.
    """

    def __init__(self, epochs: Optional[List[int]] = None, tag: str = "custom"):
        self.epochs = set(epochs or [])
        self.tag = tag

    def on_epoch_end(self, ctx: "_CallbackContext") -> None:
        # Intentionally minimal — train()'s _save_checkpoints already handles
        # the standard tags. Custom-tag saving was not in the prior API; this
        # callback is a hook for future expansion.
        if ctx.epoch in self.epochs:
            pass


class LRMonitor(Callback):
    """Logs the current LR each epoch. History exposed at `.history`."""

    def __init__(self):
        self.history: List[float] = []

    def on_epoch_end(self, ctx: "_CallbackContext") -> None:
        lr = ctx.optimizer.param_groups[0]["lr"]
        self.history.append(lr)
