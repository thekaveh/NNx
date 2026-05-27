"""Multi-optimizer Trainer — parallel to NNModel.train().

Built for scenarios where supervised forward → loss → backward → step
is the wrong abstraction:
  - GAN training (G/D alternation, separate optimizers + losses)
  - Actor–critic RL (policy + value optimizers stepping on different
    gradient sources within the same batch)
  - Energy-based models, contrastive multi-head setups, anything with
    multiple optimizers operating on different parameter subsets.

The Trainer takes ONE NNModel and a name-keyed dict of NNOptimParams.
Each entry produces a distinct torch.optim.Optimizer; each optimizer
can be scoped to a subset of the model's parameters via
NNOptimParams.param_groups (the fine-tuning hook,
`NNParamGroupSpec(name_pattern="G.*", lr=...)`) — that is how a single
NNModel wrapping a combined G+D nn.Module ends up with two disjoint
optimizers.

The user supplies a `trainer_step_fn(ctx) -> NNEvaluationDataPoint`
that runs whatever multi-step interaction the scenario requires.
There is no `default_trainer_step` — every paradigm's step is
scenario-specific, so requiring an explicit fn prevents accidentally
running the wrong update.

Saves NNRun + per-tag NNCheckpoint artifacts the same way
NNModel.train() does, with an extra `trainer` block in run.yaml
capturing the multi-optim config. Optimizer states are NOT sidecar'd
in this initial pass — trainer-mode warm-resume is a follow-up.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from .._metrics import _resolve_metric
from ..nn.enum.checkpoints import Checkpoints
from ..nn.nn_model import CallbackLike, NNModel, _CallbackContext
from ..nn.params.nn_checkpoint import NNCheckpoint
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..nn.params.nn_iteration_data_point import NNIterationDataPoint
from ..nn.params.nn_run import NNRun, _best_err
from ..nn.params.nn_scheduler_params import NNSchedulerParams
from ..nn.params.nn_train_params import NNTrainParams
from ..utils import Utils
from .params import NNTrainerParams


@dataclass(frozen=True, slots=True)
class TrainerStepContext:
    """Per-batch state passed into a trainer_step_fn.

    Mirrors TrainStepContext from NNModel.train() but with `optimizer`
    (singular) replaced by `optimizers` (name-keyed dict) and `schedulers`
    threaded through alongside — multi-optim hooks may want to reach
    into either at any point during a step (e.g., for warmup logic).

    `model` is the single NNModel the Trainer was constructed with;
    `model.net` carries the actual nn.Module (which may itself be a
    composite, e.g., a GAN-style wrapper exposing G and D as submodules).
    """

    model: NNModel
    batch: Any
    optimizers: Mapping[str, torch.optim.Optimizer]
    schedulers: Mapping[str, Any]
    extra_metrics: Optional[Mapping[str, Callable]]
    batch_idx: int
    epoch_idx: int


TrainerStepFn = Callable[[TrainerStepContext], NNEvaluationDataPoint]


# Same default as NNTrainParams — ReduceLROnPlateau with the patience /
# cooldown / factor knobs the existing NNModel.train() loop uses. Reused
# for any optim that doesn't have a sibling entry in `schedulers`.
_DEFAULT_SCHEDULER_PARAMS = NNSchedulerParams(
    patience=8,
    cooldown=2,
    factor=95e-2,
    threshold=1e-3,
    min_lr=1e-7,
)


def _primary_name(names) -> str:
    """Pick the 'primary' optimizer name for surfaces that only accept a
    single optimizer (callback `ctx.optimizer`, IDP `.lr`, tqdm postfix).
    Sorted-first key — deterministic across Python's dict-insertion-order
    semantics."""
    return sorted(names)[0]


def _representative_train_params(params: NNTrainerParams) -> NNTrainParams:
    """Synthesize an NNTrainParams that represents the trainer run for
    the existing NNRun.train slot.

    Uses the *primary* (sorted-first) optim + its matching scheduler
    so the surface signal — what does this run's overall LR / scheduler
    look like — points at a real, deterministic sub-config. The full
    multi-optim configuration is preserved in NNRun.trainer; this is
    just the legacy-shape view of it.
    """
    primary = _primary_name(params.optims.keys())
    sched = params.schedulers.get(primary, _DEFAULT_SCHEDULER_PARAMS)
    return NNTrainParams(
        n_epochs=params.n_epochs,
        optim=params.optims[primary],
        scheduler=sched,
        seed=params.seed,
        save_phase_checkpoints=params.save_phase_checkpoints,
    )


def _build_scheduler(opt, sched_params, n_epochs):
    """Same dispatch logic as NNModel._build_scheduler — duplicated rather
    than promoted to a shared helper because it's a small body and lifting
    it would expand the public surface."""
    kind = getattr(sched_params, "kind", None)
    if kind is None:
        return lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            min_lr=sched_params.min_lr,
            factor=sched_params.factor,
            cooldown=sched_params.cooldown,
            patience=sched_params.patience,
            threshold=sched_params.threshold,
        )
    return kind(optimizer=opt, params=sched_params, n_epochs=n_epochs)


def _step_scheduler(sched, val_edp, train_edp) -> None:
    """ReduceLROnPlateau wants a metric; other schedulers step on epoch.
    Uses the shared val→train, error→loss fallback resolver in
    nnx._metrics so the four call sites (NNModel + Trainer × scheduler
    + tqdm) can't drift."""
    if isinstance(sched, lr_scheduler.ReduceLROnPlateau):
        metric = _resolve_metric(val_edp, train_edp)
        if metric is None:
            return
        sched.step(metric)
    else:
        sched.step()


class Trainer:
    """Multi-optimizer training orchestrator.

    Constructed around a single NNModel. At train() time, builds one
    torch.optim.Optimizer per entry in NNTrainerParams.optims (each
    scoped to its sub-net via NNOptimParams.param_groups) and invokes
    the user-supplied trainer_step_fn for each batch.

    Same NNRun + per-tag NNCheckpoint cadence as NNModel.train(),
    with the extra `trainer` block on NNRun preserving the multi-optim
    configuration on disk.
    """

    def __init__(self, model: NNModel):
        if model is None:
            raise ValueError("Trainer requires a non-None model")
        self.model = model

    def train(
        self,
        params: NNTrainerParams,
        trainer_step_fn: TrainerStepFn,
        callbacks: Optional[list[CallbackLike]] = None,
    ) -> NNRun:
        """Run the multi-optimizer training loop and return the resulting NNRun.

        Args:
            params: NNTrainerParams — train_loader + n_epochs + optims dict +
                (optional) schedulers dict + (optional) val_loader, seed,
                save_phase_checkpoints, extra_metrics.
            trainer_step_fn: required. `Callable[[TrainerStepContext],
                NNEvaluationDataPoint]`. The function owns the entire per-batch
                update — including which optimizers to step, in what order, and
                with what loss(es). There is no supervised fallback.
            callbacks: optional list of Callback instances. The callback
                context exposes `ctx.optimizer` (primary, sorted-first), plus
                a `ctx.optimizers` dict and `ctx.trainer` reference for
                trainer-aware callbacks.

        Returns:
            NNRun with per-iteration idps, persisted under runs/<run.id>/
            alongside the standard FIRST/Q1/Q2/Q3/LAST/BEST checkpoints.

        Raises:
            ValueError: when params is None, trainer_step_fn is None, or
                any optim's NNOptimParams.is_valid() returns False.
        """
        if params is None:
            raise ValueError("trainer params must not be None")
        if trainer_step_fn is None:
            raise ValueError(
                "trainer_step_fn is required — Trainer has no default "
                "supervised step because multi-optim updates are inherently "
                "scenario-specific."
            )
        for name, opt_params in params.optims.items():
            if not opt_params.is_valid():
                raise ValueError(f"optim {name!r} has invalid config: {opt_params}")

        if params.seed is not None:
            from ..seeding import set_seed

            set_seed(params.seed)

        validate = params.val_loader is not None

        run = NNRun(
            train=_representative_train_params(params),
            trainer=params,
            model=self.model.params,
            # Use the model's stored NNParams rather than self.model.net.params
            # so callers who substitute a custom nn.Module post-construction
            # (the GAN composite idiom) still produce a saveable run.
            net=self.model.net_params,
        )

        # `strict_param_groups=True` is the multi-optim contract: each
        # optimizer owns only the parameters its specs explicitly match,
        # not also the default-bucket leftovers. Without this, opt_G
        # would also hold D's params (unmatched by G's specs) and the
        # two optimizers would silently fight over the same gradients.
        optimizers = {
            name: opt_params.name(
                net=self.model.net,
                lr_start=opt_params.max_lr,
                momentum=opt_params.momentum,
                weight_decay=opt_params.weight_decay,
                param_groups=opt_params.param_groups,
                strict_param_groups=True,
            )
            for name, opt_params in params.optims.items()
        }
        schedulers = {
            name: _build_scheduler(
                opt=optimizers[name],
                sched_params=params.schedulers.get(name, _DEFAULT_SCHEDULER_PARAMS),
                n_epochs=params.n_epochs,
            )
            for name in optimizers
        }

        normalized_callbacks = NNModel._normalize_callbacks(callbacks)

        primary = _primary_name(optimizers.keys())
        ctx = _CallbackContext(
            model=self.model,
            run=run,
            optimizer=optimizers[primary],
        )
        # Trainer-aware extensions: existing callbacks reading ctx.optimizer
        # see the primary; new callbacks can downcast through ctx.optimizers
        # or ctx.trainer for the multi-optim view.
        ctx.optimizers = optimizers
        ctx.trainer = self

        idps: list[NNIterationDataPoint] = []
        try:
            n_iter: Optional[int] = int(params.n_epochs * len(params.train_loader))
        except TypeError:
            n_iter = None
        best_checkpoint: Optional[NNCheckpoint] = NNCheckpoint.load(
            run=run.id,
            type=Checkpoints.BEST,
        )

        Utils.print_table(
            header=False,
            title="Trainer Run Details...",
            data=Utils.flatten_dict(data=run.state()),
        )

        for cb in normalized_callbacks:
            cb.on_train_begin(ctx)

        idx_iter = 0
        tqdm_disabled = os.environ.get("NNX_TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
        with (
            torch.set_grad_enabled(True),
            tqdm(colour="blue", total=n_iter, desc="Training", disable=tqdm_disabled) as tqdm_bar,
        ):
            for idx_epoch in range(params.n_epochs):
                ctx.epoch = idx_epoch
                for cb in normalized_callbacks:
                    cb.on_epoch_begin(ctx)

                for idx_batch, batch in enumerate(params.train_loader):
                    step_ctx = TrainerStepContext(
                        model=self.model,
                        batch=batch,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        extra_metrics=params.extra_metrics,
                        batch_idx=idx_batch,
                        epoch_idx=idx_epoch,
                    )
                    train_edp = trainer_step_fn(step_ctx)

                    idps.append(
                        NNIterationDataPoint(
                            iter_idx=idx_iter,
                            epoch_idx=idx_epoch,
                            batch_idx=idx_batch,
                            train_edp=train_edp,
                            lr=optimizers[primary].param_groups[0]["lr"],
                        )
                    )
                    idx_iter += 1
                    tqdm_bar.update(1)

                val_edp = (
                    self.model.evaluate(
                        loader=params.val_loader,
                        extra_metrics=params.extra_metrics,
                    )
                    if validate
                    else None
                )
                idps[-1] = idps[-1].with_val_edp(val_edp)

                checkpoint = self._save_checkpoint(
                    idp=idps[-1],
                    run_id=run.id,
                    idx_epoch=idx_epoch,
                    n_epochs=params.n_epochs,
                    best_checkpoint=best_checkpoint,
                    save_phase_checkpoints=params.save_phase_checkpoints,
                )
                if best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
                    best_checkpoint = checkpoint

                # Each scheduler steps on its own optimizer's signal.
                # We feed the SAME (val_edp, train_edp) pair to all of
                # them because IDPs aggregate over all optims — separating
                # per-optim metrics would require the step fn to return
                # multiple EDPs, which complicates the contract without
                # clear benefit. Users who need per-optim scheduler
                # signals can step their scheduler directly in the step fn.
                for sched in schedulers.values():
                    _step_scheduler(sched, val_edp, train_edp)

                self._update_tqdm_postfix(tqdm_bar, optimizers[primary], val_edp, train_edp)

                run.with_idps(idps).save()

                ctx.idp = idps[-1]
                ctx.idps = idps
                for cb in normalized_callbacks:
                    cb.on_epoch_end(ctx)

                if ctx.should_stop:
                    break

            for cb in normalized_callbacks:
                cb.on_train_end(ctx)

        print()
        runs_root_path = os.path.join(os.getcwd(), "runs", run.id)
        print(f"Run saved to {runs_root_path}")
        return run.with_idps(idps).save()

    def _save_checkpoint(
        self,
        idp: NNIterationDataPoint,
        run_id: str,
        idx_epoch: int,
        n_epochs: int,
        best_checkpoint: Optional[NNCheckpoint],
        save_phase_checkpoints: bool,
    ) -> NNCheckpoint:
        """Same FIRST/Q1/Q2/Q3/LAST/BEST cadence as NNModel._save_checkpoints,
        minus the optimizer-state sidecar. Trainer-mode warm-resume is a
        follow-up — saving multiple sidecars (`<tag>.opt.<name>.pt`) is the
        natural extension when that lands."""
        checkpoint = NNCheckpoint(
            idp=idp,
            model_params=self.model.params,
            net_params=self.model.net_params,
            net_state=self.model.net.state_dict(),
        )
        if save_phase_checkpoints:
            if idx_epoch == 0:
                checkpoint.save(run=run_id, type=Checkpoints.FIRST)
            elif idx_epoch == int(n_epochs * 1 / 4) - 1:
                checkpoint.save(run=run_id, type=Checkpoints.Q1)
            elif idx_epoch == int(n_epochs * 2 / 4) - 1:
                checkpoint.save(run=run_id, type=Checkpoints.Q2)
            elif idx_epoch == int(n_epochs * 3 / 4) - 1:
                checkpoint.save(run=run_id, type=Checkpoints.Q3)
        checkpoint.save(run=run_id, type=Checkpoints.LAST)
        if best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
            checkpoint.save(run=run_id, type=Checkpoints.BEST)
        return checkpoint

    def _update_tqdm_postfix(self, tqdm_bar, opt, val_edp, train_edp) -> None:
        lr = opt.param_groups[0]["lr"]
        err = _resolve_metric(val_edp, train_edp)
        err_str = f"{err:.4f}" if err is not None else "n/a"
        tqdm_bar.set_postfix_str(f"error={err_str}, lr={lr:.4f}")
