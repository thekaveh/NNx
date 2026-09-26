from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, Union

from ..._validation import require_count
from ...monitors import MetricSpec, MonitorSpec, _unique_metrics
from ..enum.optims import Optims
from ..params.nn_optim_params import NNOptimParams
from ..params.nn_scheduler_params import NNSchedulerParams

if TYPE_CHECKING:
    from ...optimizers import NNOptimFactoryParams


RESUME_MODES = ("auto", "stateful", "weights_only")


def _validate_resume_mode(mode: object, owner: str) -> None:
    if mode not in RESUME_MODES:
        raise ValueError(f"{owner}.resume_mode must be one of {', '.join(repr(m) for m in RESUME_MODES)}, got {mode!r}")


def _validate_monitoring(params: Any, owner: str) -> None:
    """Normalize ``metrics`` to a tuple of uniquely named specs and check
    that ``monitor`` names ``loss`` / ``error`` or a declared metric (no
    registry lookup: that happens when training starts)."""
    metrics = params.metrics
    if metrics is None or isinstance(metrics, (str, bytes, MetricSpec)) or not isinstance(metrics, Iterable):
        raise TypeError(f"{owner}.metrics must be a sequence of MetricSpec, got {type(metrics).__name__}")
    object.__setattr__(params, "metrics", _unique_metrics(metrics, owner))
    if params.monitor is not None:
        if not isinstance(params.monitor, MonitorSpec):
            raise TypeError(f"{owner}.monitor must be a MonitorSpec, got {type(params.monitor).__name__}")
        params.monitor.check_names(params.metrics, owner=f"{owner}.monitor")


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    """Training configuration.

    `seed` pins every RNG that affects training (Python random, NumPy,
    torch CPU+CUDA, cuDNN) when NNModel.train() runs. None disables
    seeding (default).

    To preserve back-compat with previously-saved runs, `seed` is included
    in state() ONLY when set — so existing runs with no seed continue to
    hash to the same `run.id`.

    `extra_metrics` maps a name to ``callable(y_true, y_pred) -> float``
    (truth first, decoded class predictions second). The default
    classification training step calls each per batch, and `evaluate()`
    (the default validation pass) calls each once on the aggregate
    predictions; a custom `train_step_fn` / `eval_step_fn` decides whether
    and how to call them. Runtime-only: not part of `state()`.

    `metrics` declares named, registered metrics (:class:`~nnx.MetricSpec`)
    computed over the full sample of each epoch — on the validation set by
    `evaluate()` and on the training epoch by the default step — and
    reported in the records' `metrics` under each spec's name. `monitor`
    (:class:`~nnx.MonitorSpec`) names the split and metric that BEST
    selection and a `ReduceLROnPlateau` scheduler track, with one shared
    improvement rule (FEAT-003). Both are serialized only when set, so a
    configuration without them keeps its `state()` and run id.
    """

    n_epochs: int
    scheduler: NNSchedulerParams = NNSchedulerParams(patience=8, cooldown=2, factor=95e-2, threshold=1e-3, min_lr=1e-7)
    # A built-in NNOptimParams or a registered NNOptimFactoryParams
    # (nnx.optimizers); state() carries a `factory` entry for the latter.
    optim: Union[NNOptimParams, NNOptimFactoryParams] = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5
    )

    seed: Optional[int] = None
    # Stable caller-supplied identity for the dataset or split. DataLoader
    # objects are runtime-only and cannot be serialized safely.
    data_id: Optional[str] = None

    # When True (default, back-compat), train() saves FIRST + Q1 + Q2 + Q3
    # phase checkpoints in addition to LAST + BEST. Set False to skip the
    # FIRST/Q* writes — useful for tiny experiments or huge models where
    # per-epoch checkpoint I/O dominates wall-clock time.
    save_phase_checkpoints: bool = True

    # Any re-iterable of batches is accepted (a DataLoader, a list of
    # (X, Y) tuples, NNGraphDataset's one-element full-batch list, ...).
    # Runtime-only: never serialized. Warm resume restores generator state
    # for real DataLoaders; a consumed one-shot iterator cannot be replayed.
    train_loader: Optional[Iterable[Any]] = field(repr=False, default=None)
    val_loader: Optional[Iterable[Any]] = field(repr=False, default=None)

    # FEAT-003: declared (registered, serializable) metrics and the monitor
    # BEST / plateau scheduling track. Omitted from state() when unset.
    metrics: tuple[MetricSpec, ...] = ()
    monitor: Optional[MonitorSpec] = None

    # Custom metrics: name -> callable(y_true, y_pred) -> float. Runtime-only
    # (functions don't round-trip through YAML), so this lives outside
    # state() / from_state() — like train_loader/val_loader. Each is invoked
    # on every train batch and on every evaluate() aggregate.
    extra_metrics: Optional[Mapping[str, Callable]] = field(repr=False, default=None)

    # Resume control. When `resume_from_run_id` is set, train() loads that
    # run's checkpoint of the named type and warm-restarts training from
    # its model weights and complete stateful training bundle when available.
    # The source run and checkpoint are serialized as parent lineage, so a
    # resumed session receives a distinct run id and cannot silently replace
    # the source run's history.
    resume_from_run_id: Optional[str] = field(repr=False, default=None)
    resume_from_checkpoint: Optional[str] = field(repr=False, default="last")
    parent_run_id: Optional[str] = field(repr=False, default=None)
    overwrite_existing: bool = field(repr=False, default=False)
    # FEAT-005 — how a resume restores state (runtime-only, never serialized):
    # "auto" (default) restores the complete training state when the
    # checkpoint has it and falls back to its weights with a warning;
    # "stateful" requires the training state and fails before restoring
    # anything when the checkpoint is weights-only; "weights_only" restores
    # only the model weights and starts every component fresh.
    resume_mode: str = field(repr=False, default="auto")

    def __post_init__(self):
        # Fail-fast: `n_epochs` drives `range(params.n_epochs)` in the train
        # loop, so a value < 1 silently makes training a no-op (empty idps, a
        # degenerate saved run, no BEST checkpoint) rather than erroring.
        # `n_epochs` is always emitted into state(), so this never shifts a
        # run.id for any valid config.
        # `n_epochs` is an integer count (FIX-021): a fractional value would
        # otherwise reach `range()` only when training starts.
        object.__setattr__(
            self,
            "n_epochs",
            require_count(
                self.n_epochs,
                "n_epochs",
                owner="NNTrainParams",
                minimum=1,
                domain_message=f"NNTrainParams requires n_epochs >= 1, got {self.n_epochs}",
            ),
        )
        if self.data_id is not None and not self.data_id.strip():
            raise ValueError("NNTrainParams.data_id must be non-empty when provided")
        if self.parent_run_id is not None and self.resume_from_run_id is not None:
            raise ValueError("set resume_from_run_id or parent_run_id, not both")
        _validate_resume_mode(self.resume_mode, "NNTrainParams")
        _validate_monitoring(self, "NNTrainParams")

    def with_train_loader(self, value: Iterable[Any]) -> NNTrainParams:
        return replace(self, train_loader=value)

    def with_val_loader(self, value: Iterable[Any]) -> NNTrainParams:
        return replace(self, val_loader=value)

    def __str__(self):
        return f"Train={{n_epochs={self.n_epochs}, seed={self.seed}, Optim={self.optim}, Scheduler={self.scheduler}}}"

    def state(self):
        d: dict[str, object] = dict(
            n_epochs=self.n_epochs,
            optim=self.optim.state(),
            scheduler=self.scheduler.state(),
        )
        # Only emit `seed` / `save_phase_checkpoints` into state() when they
        # diverge from their defaults so a NNTrainParams created without
        # them hashes to the same run.id as before these fields existed.
        # Existing on-disk runs without these keys are loadable via .get()
        # defaults in from_state below.
        if self.seed is not None:
            d["seed"] = self.seed
        if self.data_id is not None:
            d["data_id"] = self.data_id
        lineage = self.parent_run_id or self.resume_from_run_id
        if lineage is not None:
            d["parent_run_id"] = lineage
            d["parent_checkpoint"] = self.resume_from_checkpoint
        if self.save_phase_checkpoints is not True:
            d["save_phase_checkpoints"] = self.save_phase_checkpoints
        if self.metrics:
            d["metrics"] = [spec.state() for spec in self.metrics]
        if self.monitor is not None:
            d["monitor"] = self.monitor.state()
        return d

    @staticmethod
    def from_state(state: dict) -> NNTrainParams:
        # Lazy import: nnx.optimizers imports this package's params.
        from ...optimizers import optim_params_from_state

        return NNTrainParams(
            n_epochs=state["n_epochs"],
            optim=optim_params_from_state(state["optim"]),
            scheduler=NNSchedulerParams.from_state(state["scheduler"]),
            seed=state.get("seed"),
            data_id=state.get("data_id"),
            parent_run_id=state.get("parent_run_id"),
            resume_from_checkpoint=state.get("parent_checkpoint", "last"),
            save_phase_checkpoints=state.get("save_phase_checkpoints", True),
            metrics=tuple(MetricSpec.from_state(m) for m in state.get("metrics") or ()),
            monitor=MonitorSpec.from_state(state["monitor"]) if state.get("monitor") is not None else None,
        )
