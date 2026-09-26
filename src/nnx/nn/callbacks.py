"""Callback protocol and standard callbacks for NNModel.train().

The protocol gives `train()` discrete lifecycle hooks (on_train_begin,
on_epoch_begin, on_epoch_end, on_train_end) and lets a callback signal
early termination by setting `ctx.should_stop = True`.

The legacy callable signature `Callable[[List[NNIterationDataPoint]], None]`
is preserved via _LegacyCallback (which adapts to on_epoch_end) so existing
notebooks keep working.
"""

from __future__ import annotations

import math
import os
import re
import sys
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Optional

from .._metrics import _resolve_metric_with_provenance
from .._validation import require_count, require_finite_real
from ..components import ComponentSpec
from .params.nn_checkpoint import _MODEL_CHECKPOINT_TAG, NNCheckpoint, NNCheckpointTransform, _snapshot_state_dict
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


def _warn_at_user_frame(message: str) -> None:
    """Emit a ``RuntimeWarning`` attributed to the first caller outside nnx.

    ``warnings.warn(stacklevel=...)`` would point at a fixed line inside the
    training loop, and Python's default filter shows a given message from a
    given location only once per process — so a second ``train()`` call in
    the same notebook would be silent. A fresh registry per call keeps user
    filters (``ignore`` / ``error`` / ``once``) authoritative while callers
    bound the volume themselves (``EarlyStopping`` reports once per run).
    """
    frame = sys._getframe(1)
    while frame.f_back is not None and str(frame.f_globals.get("__name__", "")).split(".")[0] == "nnx":
        frame = frame.f_back
    warnings.warn_explicit(
        message,
        RuntimeWarning,
        frame.f_code.co_filename,
        frame.f_lineno,
        module=frame.f_globals.get("__name__"),
        registry=None,
        module_globals=frame.f_globals,
    )


class EarlyStopping(Callback):
    """Stop training when the monitored metric stops improving.

    Args:
        monitor: which data-point field to track. ``None`` (default) selects
                 automatically from the validation data point, once per
                 ``train()`` call: ``val_edp.error`` when the first validated
                 epoch reports a finite error, otherwise ``val_edp.loss`` when it
                 reports a finite loss (regression and other evaluators that
                 leave ``error`` as ``None``). The choice then stays fixed for
                 that run; ``selected_monitor`` reports it. The default never
                 reads training metrics and only supports ``mode="min"``.
                 An explicit ``"val_edp.error"``, ``"val_edp.loss"``,
                 ``"train_edp.error"`` or ``"train_edp.loss"`` reads exactly that
                 field with no fallback. Unlike BEST selection and
                 ReduceLROnPlateau, there is no validation→training fallback.
                 When the tracked field (or its whole data point) is absent, the
                 epoch is not counted toward patience and one ``RuntimeWarning``
                 per run names the monitor and what is missing, instead of the
                 callback going silently inactive. A NaN/±inf value never
                 becomes the best and counts as an epoch without improvement
                 (also reported once per run).
        patience: epochs with no improvement before stopping — a nonnegative
                  integer count (NumPy integers accepted and normalized; zero
                  stops on the first non-improving epoch). Fractional, boolean
                  or string values raise ``ValueError`` at construction.
        min_delta: minimum change to qualify as improvement.
        mode: improvement direction for the monitored field. ``"min"``
              (default): lower is better — the meaning of every accepted
              monitor (loss / error), and the direction BEST selection and
              ReduceLROnPlateau also assume, so it is almost always right.
              ``"max"`` only reverses this callback's comparison for one of
              the four accepted keys; it does not enable accuracy/F1
              monitors, which are rejected, and it does not change how the
              rest of NNx ranks ``error`` / ``loss``. ``"max"`` requires an
              explicit ``monitor``.

        name: component name under which the patience state is checkpointed
              (FEAT-005). By default ``"early_stopping"``, numbered
              (``early_stopping.2`` …) in callback order when a run has
              several; an explicit name must be unique within the run.

    Checkpointable: the best value, the epochs waited and the selected
    monitor are saved with every checkpoint's training state and restored
    on a stateful warm resume *after* ``on_train_begin`` resets them, so a
    resumed run stops at the same epoch an uninterrupted one would. It is
    an optional component: resuming from a checkpoint written without it
    starts with fresh patience.

    Example::

        EarlyStopping(monitor="val_edp.loss", mode="min", patience=5)
    """

    def __init__(
        self,
        monitor: Optional[str] = None,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        name: Optional[str] = None,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        valid_monitors = {
            "val_edp.error",
            "val_edp.loss",
            "train_edp.error",
            "train_edp.loss",
        }
        if monitor is not None and monitor not in valid_monitors:
            raise ValueError(f"monitor must be None or one of {sorted(valid_monitors)}, got {monitor!r}")
        if monitor is None and mode == "max":
            raise ValueError(
                "mode='max' requires an explicit monitor: the default (monitor=None) selects "
                "val_edp.error or val_edp.loss, which improve by decreasing"
            )
        # `patience` is an epoch count, not a real hyperparameter (FIX-021).
        patience = require_count(
            patience,
            "patience",
            owner="EarlyStopping",
            minimum=0,
            domain_message=f"patience must be >= 0, got {patience}",
        )
        require_finite_real(min_delta, "min_delta", owner="EarlyStopping", minimum=0.0)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._component = (
            ComponentSpec("early_stopping", version=1, required=False, numbered=True)
            if name is None
            else ComponentSpec(name, version=1, required=False)
        )
        self._best: Optional[float] = None
        self._wait: int = 0
        # Field compared during the current run: the explicit monitor, or
        # the automatic choice once a validated epoch offers a finite value.
        self._selected: Optional[str] = monitor
        # Diagnostics already emitted this run (one warning per reason).
        self._reported: set[str] = set()

    @property
    def selected_monitor(self) -> Optional[str]:
        """Field compared in the current run (``None`` until the automatic
        default has seen a validated epoch with a finite error or loss)."""
        return self._selected

    def _report_once(self, key: str, detail: str) -> None:
        if key in self._reported:
            return
        self._reported.add(key)
        label = f"EarlyStopping(monitor={self.monitor!r})"
        if self.monitor is None and self._selected is not None:
            label += f" (automatically selected {self._selected!r})"
        _warn_at_user_frame(f"{label}: {detail}")

    def _observe(self, idp: NNIterationDataPoint, epoch: object) -> Optional[tuple[str, float]]:
        """Return ``(field, value)`` to compare this epoch, or ``None`` when
        the epoch does not count (the reason is reported once per run)."""
        # Named distinctly from nnx._metrics._resolve_metric (the
        # val→train / error→loss fallback resolver): this compares exactly
        # one field — the explicit monitor or the run's automatic choice.
        edp_name = self._selected.partition(".")[0] if self._selected else "val_edp"
        edp = getattr(idp, edp_name, None)
        if edp is None:
            if edp_name == "val_edp":
                hint = "Configure a val_loader"
                if self.monitor is None:
                    hint += ", or pass monitor='train_edp.loss' / 'train_edp.error' to stop on a training metric"
                self._report_once(
                    "val_edp:absent",
                    f"epoch {epoch} has no validation data point, so there is nothing to monitor and "
                    f"this epoch is not counted toward patience. {hint}.",
                )
            else:
                self._report_once(
                    "train_edp:absent",
                    f"epoch {epoch} has no training data point, so this epoch is not counted toward patience.",
                )
            return None
        if self._selected is None:
            # Same val-side preference and finiteness rule as BEST selection
            # and ReduceLROnPlateau: first finite of error → loss.
            value, source, _ = _resolve_metric_with_provenance(edp, None)
            if source is not None and value is not None:
                self._selected = source
                return source, float(value)
            for field in ("error", "loss"):
                present = getattr(edp, field, None)
                if present is not None:
                    # Only non-finite values so far (e.g. a run diverging from
                    # the first epoch): count the epoch, keep the choice open.
                    return f"val_edp.{field}", float(present)
            self._report_once(
                "val_edp:no-field",
                f"epoch {epoch}'s validation data point has neither an 'error' nor a 'loss', so this "
                "epoch is not counted toward patience. Have eval_step_fn return a loss (or an error).",
            )
            return None
        field = self._selected.partition(".")[2]
        value = getattr(edp, field, None)
        if value is None:
            self._report_once(
                f"{self._selected}:missing",
                f"epoch {epoch}'s {edp_name} has no {field!r} value, so this epoch is not counted "
                "toward patience. Monitor a field the evaluator populates (regression evaluators "
                "usually report 'loss' and leave 'error' as None).",
            )
            return None
        return self._selected, float(value)

    # ---------- checkpointable component (FEAT-005) ----------

    def component_spec(self) -> ComponentSpec:
        return self._component

    def component_state(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best": self._best,
            "wait": self._wait,
            "selected": self._selected,
        }

    def check_component_state(self, state: Mapping[str, Any], *, version: int) -> list[str]:
        """Reject, before anything is restored, patience tracked for a
        different monitor or direction."""
        if state.get("monitor") != self.monitor or state.get("mode") != self.mode:
            return [
                f"EarlyStopping tracked monitor={state.get('monitor')!r}, mode={state.get('mode')!r} in the "
                f"checkpoint but is configured with monitor={self.monitor!r}, mode={self.mode!r}"
            ]
        return []

    def load_component_state(self, state: Mapping[str, Any], *, version: int) -> None:
        problems = self.check_component_state(state, version=version)
        if problems:
            raise ValueError(problems[0])
        best = state.get("best")
        self._best = None if best is None else float(best)
        self._wait = int(state["wait"])
        self._selected = state.get("selected")

    def on_train_begin(self, ctx: _CallbackContext) -> None:
        # Fresh run, fresh patience: without this reset, reusing one
        # EarlyStopping instance across train() calls compares against
        # the previous run's best and can stop the new run immediately.
        # The automatic monitor choice and the once-per-run diagnostics
        # reset too, so a regression run followed by a classification run
        # each pick their own field.
        self._best = None
        self._wait = 0
        self._selected = self.monitor
        self._reported = set()

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def on_epoch_end(self, ctx: _CallbackContext) -> None:
        if ctx.idp is None:
            return
        epoch = getattr(ctx, "epoch", None)
        observed = self._observe(ctx.idp, epoch)
        if observed is None:
            return
        field, current = observed
        if math.isfinite(current):
            if self._best is None or self._is_improvement(current, self._best):
                self._best = current
                self._wait = 0
                return
        else:
            # A NaN best would make every later comparison False; a
            # non-finite epoch is simply one without improvement.
            self._report_once(
                "non-finite",
                f"epoch {epoch} reported {field}={current}; a non-finite value never becomes the "
                "best and counts as an epoch without improvement.",
            )
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

    Files are **weights-only** and say so (``training_state_present`` is
    ``False``): pass ``resume_from_checkpoint="<tag>_e<epoch>"`` with
    ``resume_mode="weights_only"`` (or the default ``"auto"``) to
    warm-start from one; ``resume_mode="stateful"`` rejects it before
    anything is restored.

    Args:
        epochs: list of 0-indexed epoch numbers at which to save. Empty /
            None means the callback never fires (and never saves anything).
        tag: prefix in the filename, defaults to ``"custom"``.
    """

    def __init__(self, epochs: Optional[list[int]] = None, tag: str = "custom"):
        if re.fullmatch(_MODEL_CHECKPOINT_TAG, tag) is None:
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
        # Weights-only by declaration (FEAT-005): no training-state sidecar
        # is written, so a stateful resume from this file fails before
        # anything is restored; resume_mode="weights_only" (or "auto")
        # warm-starts from its weights.
        ckpt = NNCheckpoint(
            idp=ctx.idp,
            model_params=ctx.model.params,
            net_params=ctx.model.net_params,
            net_state=_snapshot_state_dict(ctx.model.net.state_dict()),
            training_state_present=False,
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
    """Yield (name, value) pairs for the standard EDP fields, a task
    record's own metrics (``mse`` / ``mae``, ``subset_accuracy`` /
    ``element_accuracy`` — FEAT-002) and any user-supplied extras. Skips
    None values, so a regression record emits no classification fields and
    an all-masked record emits nothing rather than zeros."""
    if edp is None:
        return
    for name in ("loss", "error", "accuracy", "f1", "precision", "recall"):
        v = getattr(edp, name, None)
        if v is not None:
            yield name, v
    for name, v in (getattr(edp, "metrics", None) or {}).items():
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
