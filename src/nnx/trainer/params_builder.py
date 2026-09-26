"""Builder for NNTrainerParams — the composite multi-optim configuration.

Composes Plan 1's NNSchedulerParams.builder() + Plan 2's
NNOptimParams.builder() output via name-keyed dicts. `.optimizer(name,
params)` and `.scheduler(name, params)` accept pre-built params
instances — the simplest composition path. (A lambda-Builder
variant is deferred — composability via the lambda is elegant but
makes the call-site harder to read; we ship the direct form first.)

`.build()` enforces `schedulers.keys() ⊆ optims.keys()` BEFORE
constructing the dataclass, so the user sees the constraint failure
at the Builder boundary rather than from inside the dataclass's
__post_init__.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Union

from torch.utils.data import DataLoader

from .._builders import copy_containers, params_init_values
from ..monitors import MetricSpec, MonitorSpec
from ..nn.params.nn_optim_params import NNOptimParams
from ..nn.params.nn_scheduler_params import NNSchedulerParams
from .params import NNTrainerParams

if TYPE_CHECKING:
    from ..optimizers import NNOptimFactoryParams


class NNTrainerParamsBuilder:
    """Composite builder for `NNTrainerParams`.

    Reach via `NNTrainerParams.builder()`. The required setter is
    `.n_epochs(N)`; at least one `.optimizer(name, params)` call is
    also required (`NNTrainerParams.__post_init__` rejects empty
    optims). Schedulers, seed, loaders, etc. are all chained optionals.

    Ownership (see `NNTrainerParams`): the custom ``trainer_step_fn``
    owns every optimizer update; Trainer steps registered schedulers once
    per epoch unless ``.auto_step_schedulers(False)``.

    `copy()` branches a (possibly partial) builder and `from_params()`
    rebuilds one from an existing `NNTrainerParams`, so a shared base
    (epochs, loaders, a common optimizer) is written once and each branch
    adds its own optimizers, schedulers or `data_id`.
    """

    # Every `NNTrainerParams` init field `from_params` can carry.
    _PARAMS_FIELDS: ClassVar[tuple[str, ...]] = (
        "n_epochs",
        "optims",
        "schedulers",
        "seed",
        "data_id",
        "save_phase_checkpoints",
        "auto_step_schedulers",
        "overwrite_existing",
        "train_loader",
        "val_loader",
        "extra_metrics",
        "resume_from_run_id",
        "resume_from_checkpoint",
        "parent_run_id",
        "resume_mode",
        "metrics",
        "monitor",
    )
    # Configuration containers a branch owns (its items stay shared); the
    # `optims` / `schedulers` maps are the builder's own dicts.
    _CONTAINER_FIELDS: ClassVar[tuple[str, ...]] = ("optims", "schedulers", "extra_metrics")

    def __init__(self) -> None:
        self._fields: dict[str, Any] = {}
        self._optims: dict[str, Union[NNOptimParams, NNOptimFactoryParams]] = {}
        self._schedulers: dict[str, NNSchedulerParams] = {}

    def copy(self) -> NNTrainerParamsBuilder:
        """Return an independent branch of this builder, complete or partial.

        The `optims` / `schedulers` maps and the `extra_metrics` mapping
        are copied, so registering, replacing or dropping an entry on one
        builder never shows up on the other (nor in values already
        built). Their values are shared by identity: optimizer and
        scheduler params are immutable, and `DataLoader` s and metric
        callables are runtime objects that are never copied or iterated.
        Nothing is built or validated here — `build()` validates each
        branch, including the `schedulers ⊆ optims` rule.
        """
        branch = type(self)()
        branch._fields = copy_containers(self._fields, ("extra_metrics",))
        branch._optims = dict(self._optims)
        branch._schedulers = dict(self._schedulers)
        return branch

    @classmethod
    def from_params(cls, params: NNTrainerParams) -> NNTrainerParamsBuilder:
        """Return a builder pre-loaded with every field of `params`.

        `from_params(params).build()` reproduces `params` field for field,
        with the same `state()` (sorted keys, omitted defaults) and the
        same runtime insertion order of `optims`. Built-in and
        registered-factory optimizer params, loaders and metric callables
        keep their identity; the builder's own maps are fresh copies, so
        extending it never alters `params`.

        Raises:
            TypeError: if `params` is not exactly an `NNTrainerParams`
                (the single-optimizer `NNTrainParams` has no builder).
            ValueError: if `params` carries a field this builder cannot
                reproduce.
        """
        values = params_init_values(
            "NNTrainerParamsBuilder", params, NNTrainerParams, cls._PARAMS_FIELDS, containers=cls._CONTAINER_FIELDS
        )
        builder = cls()
        builder._optims = values.pop("optims")
        builder._schedulers = values.pop("schedulers", {})
        builder._fields = values
        return builder

    def n_epochs(self, n: int) -> NNTrainerParamsBuilder:
        """Number of training epochs. Required."""
        self._fields["n_epochs"] = n
        return self

    def optimizer(self, name: str, params: Union[NNOptimParams, NNOptimFactoryParams]) -> NNTrainerParamsBuilder:
        """Register one optimizer under `name`. Each name gets its
        own torch.optim.Optimizer at Trainer.train() time. Use
        `NNOptimParams.builder()` (Plan 2) to construct a built-in
        `params`, or `nnx.NNOptimFactoryParams` for a registered factory."""
        self._optims[name] = params
        return self

    def scheduler(self, name: str, params: NNSchedulerParams) -> NNTrainerParamsBuilder:
        """Register one scheduler under `name`. The name must match a
        previously-registered `.optimizer(name, ...)` call — `.build()`
        enforces the subset invariant."""
        self._schedulers[name] = params
        return self

    def seed(self, value: int) -> NNTrainerParamsBuilder:
        """Seed for reproducibility. None at default (no seeding via
        params; the caller's `set_seed()` is the only path)."""
        self._fields["seed"] = value
        return self

    def data_id(self, value: str) -> NNTrainerParamsBuilder:
        """Name the dataset or split this configuration trains on.

        Part of `state()` and therefore of the run id, so two otherwise
        identical configurations (e.g. sibling builder branches) over
        different data get distinct run directories. None at default.
        """
        self._fields["data_id"] = value
        return self

    def metrics(self, *specs: MetricSpec) -> NNTrainerParamsBuilder:
        """Declare named, registered metrics (FEAT-003) — computed over the
        whole validation set by ``evaluate()`` and reported under each
        spec's name. Replaces any earlier declaration."""
        self._fields["metrics"] = tuple(specs)
        return self

    def monitor(self, spec: MonitorSpec) -> NNTrainerParamsBuilder:
        """What BEST selection and plateau schedulers track (FEAT-003)."""
        self._fields["monitor"] = spec
        return self

    def resume_from(self, run_id: str, checkpoint: str = "last", mode: str = "auto") -> NNTrainerParamsBuilder:
        """Warm-resume from ``run_id``'s ``checkpoint`` (FEAT-005): the model,
        every named optimizer and scheduler, the RNG and the registered
        components (callbacks such as ``EarlyStopping``) continue where
        that run stopped. ``mode`` is ``"auto"`` (stateful when the
        checkpoint has training state, else weights-only), ``"stateful"``
        (fail unless it has) or ``"weights_only"``. Serialized only as
        parent lineage, so the resumed run gets its own id."""
        self._fields["resume_from_run_id"] = run_id
        self._fields["resume_from_checkpoint"] = checkpoint
        self._fields["resume_mode"] = mode
        return self

    def overwrite_existing(self, value: bool) -> NNTrainerParamsBuilder:
        """Allow a run to replace an existing run directory with the same
        id. Default False; not part of `state()` or the run id."""
        self._fields["overwrite_existing"] = value
        return self

    def save_phase_checkpoints(self, value: bool) -> NNTrainerParamsBuilder:
        """Whether to write phase checkpoints (FIRST / Q1 / Q2 / Q3 /
        LAST / BEST). Default True. The fluent contract is "last call
        wins" — a prior `.save_phase_checkpoints(False)` followed by
        `.save_phase_checkpoints(True)` leaves the dataclass at the
        default (which `state()` then omits)."""
        self._fields["save_phase_checkpoints"] = value
        return self

    def auto_step_schedulers(self, value: bool) -> NNTrainerParamsBuilder:
        """Choose whether Trainer steps every scheduler after each epoch.

        Disable this when the custom step function owns scheduler timing.
        """
        self._fields["auto_step_schedulers"] = value
        return self

    def train_loader(self, loader: DataLoader) -> NNTrainerParamsBuilder:
        """Training DataLoader. Optional at Builder time (can be wired
        later via NNTrainerParams.with_train_loader)."""
        self._fields["train_loader"] = loader
        return self

    def val_loader(self, loader: DataLoader) -> NNTrainerParamsBuilder:
        """Validation DataLoader. Optional at Builder time (can be wired
        later via NNTrainerParams.with_val_loader)."""
        self._fields["val_loader"] = loader
        return self

    def extra_metrics(self, metrics: Mapping[str, Callable]) -> NNTrainerParamsBuilder:
        """Extra metrics, name-keyed ``callable(y_true, y_pred) -> float``
        (truth first, decoded predictions second). Trainer's built-in
        validation calls each once on the aggregate predictions
        (``NNModel.evaluate``); the custom ``trainer_step_fn`` decides
        whether and how to call them on training batches."""
        self._fields["extra_metrics"] = metrics
        return self

    def build(self) -> NNTrainerParams:
        """Validate the key-subset invariant, then construct the dataclass.

        `schedulers.keys() ⊆ optims.keys()` is the contract
        `NNTrainerParams.__post_init__` enforces. We check here so the
        user sees the violation at the Builder boundary — e.g., they
        called `.scheduler("d", ...)` without first calling
        `.optimizer("d", ...)` — rather than at the dataclass ctor.

        `n_epochs` has no meaningful default — call `.n_epochs(N)` before
        `.build()`. Caught here too, for the same Builder-boundary reason.

        Raises:
            ValueError: if `.n_epochs(N)` was not called before
                `.build()`, OR if a `.scheduler(name, ...)` was
                attached for a name that has no corresponding
                `.optimizer(name, ...)`. Both messages name the
                Builder methods to call so the user can fix the chain
                without consulting the dataclass schema.
        """
        if "n_epochs" not in self._fields:
            raise ValueError(
                "NNTrainerParamsBuilder.n_epochs() must be called before .build() — "
                "n_epochs has no meaningful default. Example: "
                ".n_epochs(50).optimizer('main', NNOptimParams(...)).build()"
            )
        unknown = set(self._schedulers.keys()) - set(self._optims.keys())
        if unknown:
            raise ValueError(
                "NNTrainerParamsBuilder.scheduler() called with names not present in optims: "
                f"{sorted(unknown)}. Call .optimizer({sorted(unknown)[0]!r}, ...) before scheduling. "
                f"Known optim names so far: {sorted(self._optims.keys())}"
            )
        # Always include optims (required by NNTrainerParams.__post_init__).
        # Only include schedulers if non-empty (preserves omit-when-default in state()).
        fields = dict(self._fields)
        fields["optims"] = self._optims
        if self._schedulers:
            fields["schedulers"] = self._schedulers
        return NNTrainerParams(**fields)
