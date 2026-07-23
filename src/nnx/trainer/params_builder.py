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
from typing import Any

from torch.utils.data import DataLoader

from ..nn.params.nn_optim_params import NNOptimParams
from ..nn.params.nn_scheduler_params import NNSchedulerParams
from .params import NNTrainerParams


class NNTrainerParamsBuilder:
    """Composite builder for `NNTrainerParams`.

    Reach via `NNTrainerParams.builder()`. The required setter is
    `.n_epochs(N)`; at least one `.optimizer(name, params)` call is
    also required (`NNTrainerParams.__post_init__` rejects empty
    optims). Schedulers, seed, loaders, etc. are all chained optionals.
    """

    def __init__(self) -> None:
        self._fields: dict[str, Any] = {}
        self._optims: dict[str, NNOptimParams] = {}
        self._schedulers: dict[str, NNSchedulerParams] = {}

    def n_epochs(self, n: int) -> NNTrainerParamsBuilder:
        """Number of training epochs. Required."""
        self._fields["n_epochs"] = n
        return self

    def optimizer(self, name: str, params: NNOptimParams) -> NNTrainerParamsBuilder:
        """Register one optimizer under `name`. Each name gets its
        own torch.optim.Optimizer at Trainer.train() time. Use
        `NNOptimParams.builder()` (Plan 2) to construct `params`."""
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
        """Extra metrics callables, name-keyed. Each is called with
        (y_pred, y_true) at every validation step."""
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
