from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Optional

from torch.utils.data import DataLoader

from ..enum.optims import Optims
from ..params.nn_optim_params import NNOptimParams
from ..params.nn_scheduler_params import NNSchedulerParams


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTrainParams:
    """Training configuration.

    `seed` pins every RNG that affects training (Python random, NumPy,
    torch CPU+CUDA, cuDNN) when NNModel.train() runs. None disables
    seeding (default).

    To preserve back-compat with previously-saved runs, `seed` is included
    in state() ONLY when set — so existing runs with no seed continue to
    hash to the same `run.id`.
    """

    n_epochs: int
    scheduler: NNSchedulerParams = NNSchedulerParams(patience=8, cooldown=2, factor=95e-2, threshold=1e-3, min_lr=1e-7)
    optim: NNOptimParams = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5)

    seed: Optional[int] = None

    # When True (default, back-compat), train() saves FIRST + Q1 + Q2 + Q3
    # phase checkpoints in addition to LAST + BEST. Set False to skip the
    # FIRST/Q* writes — useful for tiny experiments or huge models where
    # per-epoch checkpoint I/O dominates wall-clock time.
    save_phase_checkpoints: bool = True

    train_loader: Optional[DataLoader] = field(repr=False, default=None)
    val_loader: Optional[DataLoader] = field(repr=False, default=None)

    # Custom metrics: name -> callable(Y_true, Y_pred) -> float. Runtime-only
    # (functions don't round-trip through YAML), so this lives outside
    # state() / from_state() — like train_loader/val_loader. Each is invoked
    # on every train batch and on every evaluate() aggregate.
    extra_metrics: Optional[Mapping[str, Callable]] = field(repr=False, default=None)

    # Resume control. When `resume_from_run_id` is set, train() loads that
    # run's checkpoint of the named type and warm-restarts training from
    # its model weights AND optimizer state (when an .opt.pt sidecar exists).
    # Runtime-only (knowing the prior run id isn't part of *this* run's
    # configuration identity).
    resume_from_run_id: Optional[str] = field(repr=False, default=None)
    resume_from_checkpoint: Optional[str] = field(repr=False, default="last")

    def with_train_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, train_loader=value)

    def with_val_loader(self, value: DataLoader) -> NNTrainParams:
        return replace(self, val_loader=value)

    def __str__(self):
        return f"Train={{n_epochs={self.n_epochs}, seed={self.seed}, Optim={self.optim}, Scheduler={self.scheduler}}}"

    def state(self):
        d = dict(n_epochs=self.n_epochs, optim=self.optim.state(), scheduler=self.scheduler.state())
        # Only emit `seed` / `save_phase_checkpoints` into state() when they
        # diverge from their defaults so a NNTrainParams created without
        # them hashes to the same run.id as before these fields existed.
        # Existing on-disk runs without these keys are loadable via .get()
        # defaults in from_state below.
        if self.seed is not None:
            d["seed"] = self.seed
        if self.save_phase_checkpoints is not True:
            d["save_phase_checkpoints"] = self.save_phase_checkpoints
        return d

    @staticmethod
    def from_state(state: dict) -> NNTrainParams:
        return NNTrainParams(
            n_epochs=state["n_epochs"],
            optim=NNOptimParams.from_state(state["optim"]),
            scheduler=NNSchedulerParams.from_state(state["scheduler"]),
            seed=state.get("seed"),
            save_phase_checkpoints=state.get("save_phase_checkpoints", True),
        )
