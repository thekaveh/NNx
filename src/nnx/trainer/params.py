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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

from torch.utils.data import DataLoader

from ..nn.params.nn_optim_params import NNOptimParams
from ..nn.params.nn_scheduler_params import NNSchedulerParams

if TYPE_CHECKING:
    from .params_builder import NNTrainerParamsBuilder


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
    `val_loader` mirror NNTrainParams exactly — the orchestration around
    a single user-supplied `trainer_step_fn` is otherwise identical.
    """

    n_epochs: int
    optims: Mapping[str, NNOptimParams]
    schedulers: Mapping[str, NNSchedulerParams] = field(default_factory=dict)

    seed: Optional[int] = None
    save_phase_checkpoints: bool = True

    train_loader: Optional[DataLoader] = field(repr=False, default=None)
    val_loader: Optional[DataLoader] = field(repr=False, default=None)

    extra_metrics: Optional[Mapping[str, Callable]] = field(repr=False, default=None)

    def __post_init__(self):
        # Fail-fast: `n_epochs` drives `range(params.n_epochs)` in Trainer.train,
        # so a value < 1 silently makes training a no-op. Symmetric with
        # NNTrainParams.__post_init__.
        if self.n_epochs < 1:
            raise ValueError(f"NNTrainerParams requires n_epochs >= 1, got {self.n_epochs}")
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
        d = dict(
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
        if self.save_phase_checkpoints is not True:
            d["save_phase_checkpoints"] = self.save_phase_checkpoints
        return d

    @staticmethod
    def from_state(state: dict) -> NNTrainerParams:
        return NNTrainerParams(
            n_epochs=state["n_epochs"],
            optims={k: NNOptimParams.from_state(v) for k, v in state["optims"].items()},
            schedulers={k: NNSchedulerParams.from_state(v) for k, v in state.get("schedulers", {}).items()},
            seed=state.get("seed"),
            save_phase_checkpoints=state.get("save_phase_checkpoints", True),
        )

    @classmethod
    def builder(cls) -> NNTrainerParamsBuilder:
        """Return a composite multi-optim builder. See
        `NNTrainerParamsBuilder`. Composes
        `NNOptimParams.builder()` + `NNSchedulerParams.builder()`.
        """
        from .params_builder import NNTrainerParamsBuilder

        return NNTrainerParamsBuilder()
