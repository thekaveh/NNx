"""NNTrainerParams — configuration for the multi-optimizer Trainer.

Parallel to NNTrainParams, with `optim` / `scheduler` (singular) replaced
by `optims` / `schedulers` (dicts keyed by user-chosen names). Each
NNOptimParams in `optims` becomes its own torch.optim.Optimizer at
train() time, with NNOptimParams.param_groups (the fine-tuning hook)
used to scope which sub-net's parameters it operates on.

Round-trips through state() / from_state() like every other params
dataclass, with the same back-compat-omit-when-default pattern for
optional fields. Keys are sorted at serialization so two
configurations that differ only in dict insertion order produce the
same run.id.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union

from torch.utils.data import DataLoader

from .._validation import require_count
from ..monitors import MetricSpec, MonitorSpec
from ..nn.params.nn_optim_params import NNOptimParams
from ..nn.params.nn_scheduler_params import NNSchedulerParams
from ..nn.params.nn_train_params import _validate_monitoring, _validate_resume_mode

if TYPE_CHECKING:
    from ..optimizers import NNOptimFactoryParams
    from .params_builder import NNTrainerParamsBuilder

_V = TypeVar("_V")


class _FrozenMapping(Mapping[str, _V], Generic[_V]):
    """Insertion-ordered, read-only snapshot of a name-keyed mapping.

    `NNTrainerParams.__post_init__` wraps `optims`, `schedulers` and
    `extra_metrics` in this so a caller (or a reused builder) mutating
    the dict it passed in can no longer change an already-constructed
    configuration — its `state()` and the `run.id` derived from it must
    describe the optimizers that actually execute. Only the *container*
    is copied: values are the existing immutable params dataclasses or
    intentionally shared runtime callables, never deep-copied.

    Item assignment / deletion raise `TypeError` (the `Mapping` ABC
    defines neither). Runtime insertion order is preserved for
    deterministic optimizer construction; `state()` still sorts keys.
    Shallow/deep copy and pickle rebuild an equivalent snapshot via
    `__reduce__` (pickling requires pickleable values, as before).
    """

    __slots__ = ("_items",)

    def __init__(self, values: Optional[Mapping[str, _V]] = None) -> None:
        self._items: dict[str, _V] = dict(values or {})

    def __getitem__(self, key: str) -> _V:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __hash__(self) -> int:
        return hash(tuple(self._items.items()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._items!r})"

    def __reduce__(self):
        return (_FrozenMapping, (self._items,))


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainerParams:
    """Configuration for `Trainer.train()` — the multi-optimizer parallel
    to `NNModel.train()` / `NNTrainParams`.

    `optims` is a name-keyed mapping of NNOptimParams; each entry
    produces a distinct torch Optimizer. Use `NNOptimParams.param_groups`
    on each entry (the fine-tuning hook from :mod:`nnx.finetune`) to scope an optimizer
    to a subset of the model's parameters — e.g., one optim for the
    generator sub-net (`name_pattern="G.*"`), one for the discriminator
    (`name_pattern="D.*"`) inside a single combined NNModel.

    `schedulers` is similarly keyed and indexes the same names. Missing
    entries default to ReduceLROnPlateau with the same defaults
    NNTrainParams uses, so callers only have to populate schedulers for
    the optims they want to customize.

    `seed`, `save_phase_checkpoints`, `extra_metrics`, `train_loader`,
    `val_loader` mirror NNTrainParams (`extra_metrics` are
    ``callable(y_true, y_pred)``; Trainer's validation calls them on the
    aggregate). The custom `trainer_step_fn` owns every optimizer update;
    by default Trainer steps every scheduler once after each epoch — set
    `auto_step_schedulers=False` when the custom step function owns
    scheduler timing too.

    `optims`, `schedulers` and `extra_metrics` are captured as read-only,
    insertion-ordered snapshots at construction (also via `from_state`,
    `dataclasses.replace` and the `with_*_loader` helpers): mutating the
    mapping you passed in — or reusing the builder that produced this
    configuration — never changes it, and item assignment / deletion on
    the exposed mappings raises `TypeError`. Values are shared by
    identity (immutable params dataclasses; runtime-only metric callables
    and loaders are never copied).
    """

    n_epochs: int
    # Built-in NNOptimParams and/or registered NNOptimFactoryParams.
    optims: Mapping[str, Union[NNOptimParams, NNOptimFactoryParams]]
    schedulers: Mapping[str, NNSchedulerParams] = field(default_factory=dict)

    seed: Optional[int] = None
    data_id: Optional[str] = None
    save_phase_checkpoints: bool = True
    auto_step_schedulers: bool = True
    overwrite_existing: bool = field(repr=False, default=False)

    train_loader: Optional[DataLoader] = field(repr=False, default=None)
    val_loader: Optional[DataLoader] = field(repr=False, default=None)

    extra_metrics: Optional[Mapping[str, Callable]] = field(repr=False, default=None)

    # Warm-resume controls (FEAT-005), mirroring NNTrainParams: the source run
    # and checkpoint are serialized only as parent lineage (`parent_run_id` /
    # `parent_checkpoint`), so a resumed session gets a distinct run id and the
    # default serialization is unchanged. `resume_mode` is runtime-only.
    resume_from_run_id: Optional[str] = field(repr=False, default=None)
    resume_from_checkpoint: str = field(repr=False, default="last")
    parent_run_id: Optional[str] = field(repr=False, default=None)
    resume_mode: str = field(repr=False, default="auto")

    # FEAT-003 declared metrics and monitor, as on NNTrainParams (serialized
    # only when set). Trainer steps are custom, so training-split named
    # metrics are unavailable; validation metrics come from evaluate().
    metrics: tuple[MetricSpec, ...] = ()
    monitor: Optional[MonitorSpec] = None

    def __post_init__(self):
        # Snapshot the configuration containers FIRST so every later check
        # (and every later reader) sees the captured mapping, not the
        # caller's live dict — a reused builder or a mutated source dict
        # must never change this frozen configuration (FIX-010). Values
        # are shared by reference on purpose: params dataclasses are
        # immutable and metric callables are runtime-only objects.
        for name in ("optims", "schedulers", "extra_metrics"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, _FrozenMapping):
                object.__setattr__(self, name, _FrozenMapping(value))
        # Fail-fast: `n_epochs` drives `range(params.n_epochs)` in Trainer.train,
        # so a value < 1 silently makes training a no-op. Symmetric with
        # NNTrainParams.__post_init__.
        # Integer count, normalized to plain `int` (FIX-021).
        object.__setattr__(
            self,
            "n_epochs",
            require_count(
                self.n_epochs,
                "n_epochs",
                owner="NNTrainerParams",
                minimum=1,
                domain_message=f"NNTrainerParams requires n_epochs >= 1, got {self.n_epochs}",
            ),
        )
        if self.data_id is not None and not self.data_id.strip():
            raise ValueError("NNTrainerParams.data_id must be non-empty when provided")
        if self.parent_run_id is not None and self.resume_from_run_id is not None:
            raise ValueError("set resume_from_run_id or parent_run_id, not both")
        _validate_resume_mode(self.resume_mode, "NNTrainerParams")
        _validate_monitoring(self, "NNTrainerParams")
        if not self.optims:
            raise ValueError(
                "NNTrainerParams.optims must have at least one entry — the Trainer constructs one Optimizer per name."
            )
        unknown = set(self.schedulers.keys()) - set(self.optims.keys())
        if unknown:
            raise ValueError(
                "NNTrainerParams.schedulers has keys not present in optims: "
                f"{sorted(unknown)} (known optim names: {sorted(self.optims.keys())})"
            )

    def with_train_loader(self, value: DataLoader) -> NNTrainerParams:
        return replace(self, train_loader=value)

    def with_val_loader(self, value: DataLoader) -> NNTrainerParams:
        return replace(self, val_loader=value)

    def __str__(self):
        return f"Trainer={{n_epochs={self.n_epochs}, optims={sorted(self.optims.keys())}, seed={self.seed}}}"

    def state(self):
        # Keys are sorted so dict insertion order doesn't affect run.id.
        d: dict[str, object] = dict(
            n_epochs=self.n_epochs,
            optims={k: self.optims[k].state() for k in sorted(self.optims.keys())},
        )
        # Match NNTrainParams: emit `schedulers` / `seed` /
        # `save_phase_checkpoints` only when set to a non-default value, so a
        # trainer run with the defaults hashes stably across versions and
        # follows the project-wide omit-when-default convention.
        if self.schedulers:
            d["schedulers"] = {k: self.schedulers[k].state() for k in sorted(self.schedulers.keys())}
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
        if self.auto_step_schedulers is not True:
            d["auto_step_schedulers"] = self.auto_step_schedulers
        if self.metrics:
            d["metrics"] = [spec.state() for spec in self.metrics]
        if self.monitor is not None:
            d["monitor"] = self.monitor.state()
        return d

    @staticmethod
    def from_state(state: dict) -> NNTrainerParams:
        from ..optimizers import optim_params_from_state

        return NNTrainerParams(
            n_epochs=state["n_epochs"],
            optims={k: optim_params_from_state(v) for k, v in state["optims"].items()},
            schedulers={k: NNSchedulerParams.from_state(v) for k, v in state.get("schedulers", {}).items()},
            seed=state.get("seed"),
            data_id=state.get("data_id"),
            parent_run_id=state.get("parent_run_id"),
            resume_from_checkpoint=state.get("parent_checkpoint", "last"),
            save_phase_checkpoints=state.get("save_phase_checkpoints", True),
            auto_step_schedulers=state.get("auto_step_schedulers", True),
            metrics=tuple(MetricSpec.from_state(m) for m in state.get("metrics") or ()),
            monitor=MonitorSpec.from_state(state["monitor"]) if state.get("monitor") is not None else None,
        )

    @classmethod
    def builder(cls) -> NNTrainerParamsBuilder:
        """Return a composite multi-optim builder. See
        `NNTrainerParamsBuilder`. Composes
        `NNOptimParams.builder()` + `NNSchedulerParams.builder()`.
        """
        from .params_builder import NNTrainerParamsBuilder

        return NNTrainerParamsBuilder()
