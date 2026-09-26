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
capturing the multi-optim config. Every checkpoint's training-state
sidecar carries each named optimizer's and scheduler's state, the RNG and
every registered component (FEAT-005), so
``NNTrainerParams(resume_from_run_id=...)`` (or
``NNTrainerParams.builder().resume_from(...)``) warm-resumes a
multi-optimizer run where it stopped.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from .._metrics import _resolve_scheduler_metric
from ..components import ComponentRegistry, ResumeStatus
from ..monitors import MonitorRecord, MonitorSpec, MonitorTracker, _TrainEpochSummary
from ..nn.enum.checkpoints import Checkpoints
from ..nn.nn_model import (
    CallbackLike,
    NNModel,
    _batch_sample_count,
    _CallbackContext,
    _CallbackFinalizer,
    _capture_rng_state,
    _check_plateau_resume,
    _check_provenance,
    _check_resume_horizon,
    _collect_checkpoint_transforms,
    _component_type,
    _dispatch_update,
    _enumerate_with_last,
    _load_resume_source,
    _loader_num_workers,
    _monitored_plateau,
    _monitoring_preflight,
    _named_training_state,
    _objective_engine,
    _objective_microbatch,
    _optimizer_topology,
    _plan_component_restore,
    _restore_rng_state,
    _restore_weights_only,
    _rollback_resume,
    _step_monitored_plateau,
    _with_attempt,
)
from ..nn.params.nn_checkpoint import NNCheckpoint, _snapshot_state_dict
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..nn.params.nn_iteration_data_point import NNIterationDataPoint
from ..nn.params.nn_run import NNRun, _best_err, _print_run_saved
from ..nn.params.nn_scheduler_params import NNSchedulerParams
from ..nn.params.nn_train_params import NNTrainParams
from ..provenance import ExperimentManifest
from ..utils import Utils
from .params import NNTrainerParams


@dataclass(frozen=True, slots=True)
class TrainerStepContext:
    """Per-batch state passed into a trainer_step_fn.

    Mirrors TrainStepContext from NNModel.train() but with `optimizer`
    (singular) replaced by `optimizers` (name-keyed dict) and `schedulers`
    threaded through alongside for inspection. Step functions should only
    call schedulers directly when ``auto_step_schedulers=False``.

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


def _objective_window(params: NNTrainerParams) -> int:
    """The update window of a Trainer objective run: every named optimizer
    commits together, so their ``accumulate_grad_batches`` must agree."""
    windows = {getattr(p, "accumulate_grad_batches", 1) for p in params.optims.values()}
    if len(windows) > 1:
        raise ValueError(
            "an objective commits every named optimizer together, so their accumulate_grad_batches must "
            f"agree; got {sorted(windows)}"
        )
    return windows.pop() if windows else 1


def _warn_full_precision_objective(model: Any) -> None:
    """``Trainer`` has no mixed-precision setting — its step functions own
    AMP — so the shared engine runs a Trainer objective in full precision.
    Say so when the model asks for mixed precision where it would apply."""
    if getattr(model.params, "mixed_precision", False) and model.device.type == "cuda":
        warnings.warn(
            "Trainer.train(objective=...) runs in full precision: Trainer has no mixed-precision setting, so "
            "the model's mixed_precision=True is not applied (NNModel.train(objective=...) applies it)",
            RuntimeWarning,
            stacklevel=3,
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
        data_id=params.data_id,
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


def _step_schedulers(scheds, val_edp, train_edp, *, epoch_idx: int, record: Optional[MonitorRecord] = None) -> None:
    """ReduceLROnPlateau wants a metric; other schedulers step on epoch.
    Uses the shared finite-only val→train, error→loss resolver in
    nnx._metrics so the NNModel and Trainer paths can't drift. The
    metric is resolved once per epoch, not once per scheduler, so a
    multi-optimizer run emits at most one rejection/skip warning per
    epoch; a None resolution skips every plateau scheduler."""
    plateau_metric: Optional[float] = None
    resolved = False
    for sched in scheds:
        if record is not None and isinstance(sched, lr_scheduler.ReduceLROnPlateau):
            # FEAT-003: monitor-aligned plateau schedulers step on the
            # epoch's monitor decision (see _step_monitored_plateau).
            _step_monitored_plateau(sched, record)
        elif isinstance(sched, lr_scheduler.ReduceLROnPlateau):
            if not resolved:
                plateau_metric = _resolve_scheduler_metric(val_edp, train_edp, epoch_idx=epoch_idx)
                resolved = True
            if plateau_metric is not None:
                sched.step(plateau_metric)
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
        trainer_step_fn: Optional[TrainerStepFn] = None,
        callbacks: Optional[list[CallbackLike]] = None,
        salt: Optional[str] = None,
        components: Optional[list[Any]] = None,
        objective: Optional[Callable[[Any], Any]] = None,
        provenance: Optional[ExperimentManifest] = None,
    ) -> NNRun:
        """Run the multi-optimizer training loop and return the resulting NNRun.

        Args:
            params: NNTrainerParams — train_loader + n_epochs + optims dict +
                (optional) schedulers dict + (optional) val_loader, seed,
                save_phase_checkpoints, extra_metrics. Schedulers step once
                per epoch by default; set auto_step_schedulers=False when the
                custom step function owns scheduler timing.
            trainer_step_fn: `Callable[[TrainerStepContext],
                NNEvaluationDataPoint]`. The function owns the entire per-batch
                update — including which optimizers to step, in what order, and
                with what loss(es). There is no supervised fallback: pass it or
                ``objective``.
            objective: an objective (FEAT-004, ``nnx.objectives``) in place of
                ``trainer_step_fn``: the shared update engine accumulates its
                loss terms over each window (the optimizers'
                ``accumulate_grad_batches``, which must agree), clips each
                optimizer's parameters with its own ``grad_clip_norm`` and
                steps every named optimizer once per committed update,
                announcing each to ``Callback.on_optimizer_update``.
            provenance: an optional ``nnx.provenance.ExperimentManifest``
                (FEAT-019): the declared intent, recorded with a fresh
                attempt exactly as ``NNModel.train`` records it.
            callbacks: optional list of Callback instances. The callback
                context exposes `ctx.optimizer` (primary, sorted-first), plus
                a `ctx.optimizers` dict and `ctx.trainer` reference for
                trainer-aware callbacks.
            salt: mirrors ``NNModel.train()``'s ``salt`` parameter — an
                optional string folded into the run.id hash so identical
                (model, net, train) configs run as distinct experiments
                without altering modeled params. ``None`` (the default)
                preserves existing run.id hashes exactly.
            components: extra checkpointable components (FEAT-005);
                callbacks and a step function that implement
                :class:`~nnx.StatefulComponent` register automatically.

        Returns:
            NNRun with per-iteration idps, persisted under runs/<run.id>/
            alongside the standard FIRST/Q1/Q2/Q3/LAST/BEST checkpoints.
            The printed completion line uses the same cwd-relative
            ``runs/<id>`` display path as ``NNModel.train``.

        Raises:
            ValueError: when params is None, params.train_loader is None,
                neither or both of trainer_step_fn and objective are given, or
                any optim's NNOptimParams.is_valid() returns False.
        """
        # One owner per optimizer update, decided before anything else.
        if trainer_step_fn is not None and objective is not None:
            raise ValueError(
                "pass trainer_step_fn or objective, not both: a step function owns its optimizer updates, an "
                "objective hands them to NNx's shared update engine"
            )
        if objective is not None and not callable(objective):
            raise TypeError(f"objective must be callable, got {type(objective).__name__}")
        _check_provenance(provenance)
        if params is None:
            raise ValueError("trainer params must not be None")
        if params.train_loader is None:
            raise ValueError(
                "params.train_loader is required — set it directly or via with_train_loader(...) before train()."
            )
        if trainer_step_fn is None and objective is None:
            raise ValueError(
                "trainer_step_fn is required (or pass an objective) — Trainer has no default "
                "supervised step because multi-optim updates are inherently "
                "scenario-specific."
            )
        objective_window = _objective_window(params) if objective is not None else 1
        for name, opt_params in params.optims.items():
            if not opt_params.is_valid():
                raise ValueError(f"optim {name!r} has invalid config: {opt_params}")
        if not any(p.requires_grad for p in self.model.net.parameters()):
            raise ValueError(
                "model has no trainable parameters — did you freeze('*')? Unfreeze something before train()."
            )

        if params.seed is not None:
            from ..seeding import set_seed

            set_seed(params.seed)

        # `strict_param_groups=True` is the multi-optim contract: each
        # optimizer owns only the parameters its specs explicitly match,
        # not also the default-bucket leftovers. Without this, opt_G
        # would also hold D's params (unmatched by G's specs) and the
        # two optimizers would silently fight over the same gradients.
        #
        # That contract only bites when each optimizer HAS specs: an optimizer
        # with `param_groups=None` routes to all `net.parameters()`, so two such
        # optimizers would each step the full set — silently double-stepping
        # every parameter. Require explicit scoping up front (the params object
        # itself stays constructible for serialization / builder round-trips).
        unscoped = sorted(n for n, op in params.optims.items() if op.param_groups is None)
        if len(params.optims) >= 2 and unscoped:
            raise ValueError(
                "Trainer with multiple optimizers requires every optimizer to scope its "
                "parameters via `param_groups` (NNParamGroupSpec); these have none and would "
                f"each grab all net parameters, double-stepping them: {unscoped}."
            )
        # Built through the shared hook (nnx.optimizers.build_optimizer)
        # before any run directory exists, so an unknown registered factory
        # or one returning a malformed optimizer fails with no run reserved.
        from ..optimizers import build_optimizer

        # FEAT-002: a declared task must be able to score the model's current
        # loss_fn — checked before any loader is iterated or run reserved.
        self.model._check_task_preflight()
        # FEAT-003: declared metrics and monitors are resolved before any run
        # is reserved; Trainer steps are custom, so only validation metrics
        # (computed by evaluate()) and training loss / error can be tracked.
        _monitoring_preflight(
            self.model,
            metrics=params.metrics,
            monitor=params.monitor,
            callbacks=callbacks,
            default_train_step=False,
            default_eval_step=True,
            has_val_loader=params.val_loader is not None,
            owner="NNTrainerParams",
        )

        optimizers = {
            name: build_optimizer(self.model.net, opt_params, strict_param_groups=True)
            for name, opt_params in params.optims.items()
        }
        if objective is not None:
            _warn_full_precision_objective(self.model)

        run = NNRun(
            train=_representative_train_params(params),
            trainer=params,
            model=self.model.params,
            # Use the model's stored NNParams rather than self.model.net.params
            # so callers who substitute a custom nn.Module post-construction
            # (the GAN composite idiom) still produce a saveable run.
            net=self.model.net_params,
            salt=salt,
        )
        with run.writable_lease(overwrite=params.overwrite_existing):
            return _with_attempt(
                run,
                provenance,
                params,
                lambda: self._train_impl(
                    params=params,
                    run=run,
                    optimizers=optimizers,
                    trainer_step_fn=trainer_step_fn,
                    callbacks=callbacks,
                    components=components,
                    objective=objective,
                    objective_window=objective_window,
                ),
            )

    def _train_impl(
        self,
        *,
        params: NNTrainerParams,
        run: NNRun,
        optimizers: dict[str, torch.optim.Optimizer],
        trainer_step_fn: Optional[TrainerStepFn],
        callbacks: Optional[list[CallbackLike]],
        components: Optional[list[Any]] = None,
        objective: Optional[Callable[[Any], Any]] = None,
        objective_window: int = 1,
    ) -> NNRun:
        """Execute a validated multi-optimizer training session."""
        assert params.train_loader is not None
        train_loader = params.train_loader
        validate = params.val_loader is not None

        monitor = params.monitor.resolve(params.metrics) if params.monitor is not None else None
        tracker = MonitorTracker(monitor, warn_missing=True) if monitor is not None else None
        schedulers = {
            name: _monitored_plateau(
                _build_scheduler(
                    opt=optimizers[name],
                    sched_params=params.schedulers.get(name, _DEFAULT_SCHEDULER_PARAMS),
                    n_epochs=params.n_epochs,
                ),
                optimizers[name],
                monitor,
            )
            for name in optimizers
        }

        from ..optimizers import optimizer_factory_state

        optimizer_factories = {name: optimizer_factory_state(params.optims[name]) for name in optimizers}
        normalized_callbacks = NNModel._normalize_callbacks(callbacks)
        summarize = (
            bool(params.metrics)
            or monitor is not None
            or any(isinstance(getattr(cb, "monitor", None), MonitorSpec) for cb in normalized_callbacks)
        )
        registry = ComponentRegistry.discover(
            normalized_callbacks, trainer_step_fn, objective, explicit=list(components or [])
        )
        if tracker is not None:
            registry.register(tracker)  # its best continues across a stateful resume
        # FEAT-004: an objective's updates belong to the shared engine, which
        # steps every named optimizer once per committed update; its counters
        # are component state, so they continue across a stateful resume.
        engine = None
        if objective is not None:
            engine = _objective_engine(
                objective,
                optimizers=optimizers,
                clip_norms={name: getattr(params.optims[name], "grad_clip_norm", None) for name in optimizers},
                scaler=None,  # Trainer has no mixed-precision setting (see _warn_full_precision_objective)
                device=self.model.device,
            )
            registry.register(engine)
        start_epoch, component_plan, resume_status, rollback = self._resume(
            params, optimizers, schedulers, registry, train_loader
        )

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
        if engine is not None:
            engine.listeners.append(lambda event: _dispatch_update(normalized_callbacks, ctx, event))
            ctx.update_count = engine.commits

        idps: list[NNIterationDataPoint] = []
        # `len()` is not defined on iterable-style DataLoaders (IterableDataset).
        # Fall back to None so tqdm renders without a total instead of crashing.
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

        idx_iter = 0
        tqdm_disabled = os.environ.get("NNX_TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
        with (
            torch.set_grad_enabled(True),
            tqdm(colour="blue", total=n_iter, desc="Training", disable=tqdm_disabled) as tqdm_bar,
            _CallbackFinalizer(normalized_callbacks, ctx) as callback_lifecycle,
        ):
            callback_lifecycle.start()
            # FEAT-005: reset hooks have run once; restore the validated
            # component states (all-or-nothing) before the first resumed epoch.
            if component_plan is not None:
                try:
                    restored = registry.restore(component_plan)
                except BaseException:
                    assert rollback is not None
                    rollback()
                    raise
                resume_status = replace(resume_status, restored_components=restored)
            if engine is not None:
                ctx.update_count = engine.commits  # continues after a stateful resume
            run = run.with_resume_status(resume_status)
            ctx.run = run
            pre_transform_net_state: Optional[dict[str, Any]] = None
            pre_transform_rng_state: Optional[dict[str, Any]] = None
            for local_epoch in range(params.n_epochs):
                idx_epoch = start_epoch + local_epoch
                ctx.epoch = idx_epoch
                for cb in normalized_callbacks:
                    cb.on_epoch_begin(ctx)

                n_idps_before_epoch = len(idps)
                # FEAT-003 whole-epoch summary of the step's records (no named
                # training metrics: Trainer steps are custom).
                epoch_summary = _TrainEpochSummary((), None, None) if summarize else None
                # Only an objective needs to know the epoch's last batch (to
                # close a short final window); step functions keep the plain
                # fetch order.
                batches = (
                    _enumerate_with_last(params.train_loader)
                    if engine is not None
                    else ((idx, batch, False) for idx, batch in enumerate(params.train_loader))
                )
                for idx_batch, batch, is_last_batch in batches:
                    if engine is not None:
                        assert objective is not None
                        train_edp = _objective_microbatch(
                            engine,
                            objective,
                            model=self.model,
                            batch=batch,
                            epoch_idx=idx_epoch,
                            batch_idx=idx_batch,
                            extra_metrics=params.extra_metrics,
                            close_window=idx_batch % objective_window == objective_window - 1 or is_last_batch,
                            epoch_summary=epoch_summary,
                        )
                    else:
                        assert trainer_step_fn is not None
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
                    if epoch_summary is not None:
                        epoch_summary.add(train_edp, _batch_sample_count(self.model.net, batch))

                    idps.append(
                        NNIterationDataPoint(
                            iter_idx=idx_iter,
                            epoch_idx=idx_epoch,
                            batch_idx=idx_batch,
                            train_edp=train_edp,
                            lr=optimizers[primary].param_groups[0]["lr"],
                            update_count=engine.commits if engine is not None else None,
                        )
                    )
                    idx_iter += 1
                    tqdm_bar.update(1)

                if len(idps) == n_idps_before_epoch:
                    # Same guard as NNModel.train: zero batches would
                    # crash on idps[-1] (first epoch) or corrupt the
                    # previous epoch's logged metrics (later epochs).
                    raise ValueError(
                        f"train_loader yielded no batches in epoch {idx_epoch} — check batch_size vs "
                        "dataset size with drop_last=True, or whether the loader is a one-shot iterable."
                    )

                if validate:
                    assert params.val_loader is not None
                    # `metrics=` only when declared, as in NNModel.train.
                    val_edp = (
                        self.model.evaluate(
                            loader=params.val_loader, extra_metrics=params.extra_metrics, metrics=params.metrics
                        )
                        if params.metrics
                        else self.model.evaluate(loader=params.val_loader, extra_metrics=params.extra_metrics)
                    )
                else:
                    val_edp = None
                idps[-1] = idps[-1].with_val_edp(val_edp)
                record: Optional[MonitorRecord] = None
                if epoch_summary is not None:
                    train_summary = epoch_summary.result()
                    if tracker is not None:
                        assert monitor is not None
                        value = monitor.value(train=train_summary or train_edp, val=val_edp)
                        record = tracker.observe(value, epoch=idx_epoch)
                    idps[-1] = idps[-1].with_epoch_summary(train_summary, record)

                # Each scheduler steps on its own optimizer's signal.
                # We feed the SAME (val_edp, train_edp) pair to all of
                # them because IDPs aggregate over all optims — separating
                # per-optim metrics would require the step fn to return
                # multiple EDPs, which complicates the contract without
                # clear benefit. Custom hooks can own scheduler timing by
                # setting auto_step_schedulers=False.
                if params.auto_step_schedulers:
                    _step_schedulers(schedulers.values(), val_edp, train_edp, epoch_idx=idx_epoch, record=record)

                ctx.idp = idps[-1]
                ctx.idps = idps
                ctx.deferred_checkpoint_writes.clear()
                for cb in normalized_callbacks:
                    cb.on_epoch_end(ctx)

                # Prepare history before the checkpoint commit marker so a
                # completed checkpoint can never outrun idps.csv.
                run.with_idps(idps).save(update_best=False)

                try:
                    checkpoint = self._save_checkpoint(
                        idp=idps[-1],
                        run_id=run.id,
                        idx_epoch=local_epoch,
                        n_epochs=params.n_epochs,
                        best_checkpoint=best_checkpoint,
                        save_phase_checkpoints=params.save_phase_checkpoints,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        completed_epoch=idx_epoch,
                        train_loader=train_loader,
                        components=registry.collect(),
                        optimizer_factories=optimizer_factories,
                        is_best=record.improved if record is not None else None,
                    )
                except BaseException:
                    committed = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
                    if committed is None or committed.idp.epoch_idx != idx_epoch:
                        run.with_idps(idps[:n_idps_before_epoch]).save(update_best=False)
                    raise
                for deferred_checkpoint in ctx.deferred_checkpoint_writes:
                    deferred_checkpoint()
                if record is not None:
                    if record.improved:
                        best_checkpoint = checkpoint
                elif best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
                    best_checkpoint = checkpoint

                # Verbatim-shared with the NNModel.train loop — one
                # implementation, so the postfix format can't drift.
                self.model._update_tqdm_postfix(tqdm_bar, optimizers[primary], val_edp, train_edp, record)

                if ctx.should_stop:
                    break

            # Resume state is the pre-on_train_end model (callbacks may
            # convert modules as the finalizer exits), like NNModel.train.
            pre_transform_net_state = _snapshot_state_dict(self.model.net.state_dict())
            pre_transform_rng_state = _capture_rng_state(train_loader)

        # on_train_end callbacks run as the finalizer exits above and may
        # mutate the net (for example, by converting modules). Refresh LAST
        # from the live model so it matches the state returned to the caller.
        # BEST remains the best state observed during training.
        if idps:
            final_transforms = (*self.model._topology_transforms, *_collect_checkpoint_transforms(normalized_callbacks))
            self.model._topology_transforms = final_transforms
            NNCheckpoint(
                idp=idps[-1],
                model_params=self.model.params,
                net_params=self.model.net_params,
                net_state=self.model.net.state_dict(),
                transforms=final_transforms,
            ).save(
                run=run.id,
                type=Checkpoints.LAST,
                # FEAT-005: the final post-callback LAST generation keeps every
                # named optimizer / scheduler and component, so a completed run
                # resumes the continuous states.
                **_named_training_state(self.model.net, optimizers, schedulers, optimizer_factories),
                rng_state=pre_transform_rng_state if final_transforms else _capture_rng_state(train_loader),
                completed_epoch=idps[-1].epoch_idx,
                resume_net_state=pre_transform_net_state if final_transforms else None,
                components=registry.collect(),
            )

        saved = run.with_idps(idps).save()
        _print_run_saved(run.id)
        return saved

    def _save_checkpoint(
        self,
        idp: NNIterationDataPoint,
        run_id: str,
        idx_epoch: int,
        n_epochs: int,
        best_checkpoint: Optional[NNCheckpoint],
        save_phase_checkpoints: bool,
        optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None,
        schedulers: Optional[Mapping[str, Any]] = None,
        completed_epoch: Optional[int] = None,
        train_loader: Optional[Any] = None,
        components: Optional[dict[str, Any]] = None,
        optimizer_factories: Optional[Mapping[str, Optional[dict[str, Any]]]] = None,
        is_best: Optional[bool] = None,
    ) -> NNCheckpoint:
        """Delegates to NNModel._save_checkpoints — the same
        FIRST/Q1/Q2/Q3/LAST/BEST cadence — with the named optimizers and
        schedulers, the RNG and the component states in one generation
        sidecar per tag (FEAT-005), so every tag is a warm-resume point."""
        return self.model._save_checkpoints(
            idp=idp,
            run_id=run_id,
            idx_epoch=idx_epoch,
            n_epochs=n_epochs,
            best_checkpoint=best_checkpoint,
            save_phase_checkpoints=save_phase_checkpoints,
            optimizer=None,
            completed_epoch=completed_epoch,
            train_loader=train_loader,
            components=components,
            optimizers=optimizers,
            schedulers=schedulers,
            optimizer_factories=optimizer_factories,
            is_best=is_best,
        )

    def _resume(
        self,
        params: NNTrainerParams,
        optimizers: Mapping[str, torch.optim.Optimizer],
        schedulers: Mapping[str, Any],
        registry: ComponentRegistry,
        train_loader: Any,
    ) -> tuple[int, Any, ResumeStatus, Optional[Callable[[], None]]]:
        """Warm-resume a multi-optimizer run (FEAT-005).

        Everything is validated before any state is mutated, with the same
        rules ``NNModel.train`` applies to its one optimizer: the optimizer
        and scheduler name sets, each optimizer's type, registered-factory
        identity and parameter topology, each scheduler's type and
        one-cycle horizon, and the component set. Returns ``(start_epoch,
        component_plan, status, rollback)``; ``rollback`` puts the model and
        RNG back if the later component restore fails.
        """
        if params.resume_from_run_id is None:
            return 0, None, ResumeStatus(), None
        from ..optimizers import _canonical_factory_state, optimizer_factory_state

        stateful = params.resume_mode != "weights_only"
        scheduler_params = {name: params.schedulers.get(name, _DEFAULT_SCHEDULER_PARAMS) for name in optimizers}
        if stateful:
            for name, sched_params in scheduler_params.items():
                _check_resume_horizon(sched_params, n_epochs=params.n_epochs, owner=f" for {name!r}")
        source = _load_resume_source(
            params.resume_from_run_id, params.resume_from_checkpoint, params.resume_mode, trainer=True
        )
        net = self.model.net
        training_state = source.training_state
        if training_state is None:
            _restore_weights_only(
                net,
                source,
                train_loader,
                params.resume_mode,
                fresh="optimizer, scheduler, RNG and component state",
                stacklevel=4,
            )
            status = ResumeStatus(
                mode="weights_only",
                source_run_id=params.resume_from_run_id,
                source_checkpoint=source.label,
                fresh_components=registry.names,
            )
            return source.checkpoint.idp.epoch_idx + 1, None, status, None

        saved_optimizers = training_state["optimizers"]
        saved_schedulers = training_state.get("schedulers") or {}
        for kind, saved in (("optimizer", saved_optimizers), ("scheduler", saved_schedulers)):
            if set(saved) != set(optimizers):
                raise ValueError(
                    f"resume {kind} names do not match: checkpoint has {sorted(saved)}, "
                    f"configuration builds {sorted(optimizers)}"
                )
        for kind, built, saved_types in (
            ("optimizer", optimizers, training_state.get("optimizer_types") or {}),
            ("scheduler", schedulers, training_state.get("scheduler_types") or {}),
        ):
            for name, component in built.items():
                expected = saved_types.get(name)
                if expected is not None and expected != _component_type(component):
                    raise ValueError(
                        f"resume {kind} type mismatch for {name!r}: checkpoint has {expected}, "
                        f"configuration builds {_component_type(component)}"
                    )
        saved_factories = training_state.get("optimizer_factories") or {}
        saved_topologies = training_state.get("optimizer_topologies") or {}
        for name, optimizer in optimizers.items():
            expected_factory = saved_factories.get(name)
            configured_factory = optimizer_factory_state(params.optims[name])
            if _canonical_factory_state(expected_factory) != _canonical_factory_state(configured_factory):
                raise ValueError(
                    f"resume optimizer factory mismatch for {name!r}: checkpoint has {expected_factory}, "
                    f"configuration builds {configured_factory}"
                )
            expected_topology = saved_topologies.get(name)
            if expected_topology is not None and expected_topology != _optimizer_topology(optimizer, net):
                raise ValueError(f"resume optimizer parameter topology for {name!r} does not match the checkpoint")
        monitor = params.monitor.resolve(params.metrics) if params.monitor is not None else None
        for name, scheduler in schedulers.items():
            _check_plateau_resume(saved_schedulers.get(name), scheduler, monitor)
        completed = training_state.get("completed_epoch")
        start_epoch = int(completed) + 1 if completed is not None else source.checkpoint.idp.epoch_idx + 1
        for name, sched_params in scheduler_params.items():
            _check_resume_horizon(
                sched_params, n_epochs=params.n_epochs, start_epoch=start_epoch, owner=f" for {name!r}"
            )
        component_plan = _plan_component_restore(registry, training_state)
        warn_worker_rng = training_state.get("rng") is not None and _loader_num_workers(train_loader) > 0

        previous_net_state = _snapshot_state_dict(net.state_dict())
        previous_rng_state = _capture_rng_state(train_loader)
        try:
            net.load_state_dict(source.net_state)
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(saved_optimizers[name])
            for name, scheduler in schedulers.items():
                scheduler.load_state_dict(saved_schedulers[name])
            if training_state.get("rng") is not None:
                _restore_rng_state(training_state["rng"], train_loader)
        except BaseException:
            net.load_state_dict(previous_net_state)
            _restore_rng_state(previous_rng_state, train_loader)
            raise
        if warn_worker_rng:
            warnings.warn(
                "exact warm-resume continuity requires train_loader.num_workers=0; "
                "worker-local RNG state cannot be reconstructed",
                RuntimeWarning,
                stacklevel=4,
            )

        def rollback() -> None:
            _rollback_resume(net, previous_net_state, previous_rng_state, train_loader)

        status = ResumeStatus(
            mode="stateful",
            source_run_id=params.resume_from_run_id,
            source_checkpoint=source.label,
            fresh_components=tuple(component_plan.fresh),
        )
        return start_epoch, component_plan, status, rollback
