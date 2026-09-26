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
from ..nn.enum.checkpoints import Checkpoints
from ..nn.nn_model import (
    CallbackLike,
    NNModel,
    _CallbackContext,
    _CallbackFinalizer,
    _capture_rng_state,
    _check_resume_horizon,
    _collect_checkpoint_transforms,
    _component_type,
    _load_resume_source,
    _loader_num_workers,
    _named_training_state,
    _optimizer_topology,
    _plan_component_restore,
    _restore_rng_state,
    _restore_weights_only,
    _rollback_resume,
)
from ..nn.params.nn_checkpoint import NNCheckpoint, _snapshot_state_dict
from ..nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from ..nn.params.nn_iteration_data_point import NNIterationDataPoint
from ..nn.params.nn_run import NNRun, _best_err, _print_run_saved
from ..nn.params.nn_scheduler_params import NNSchedulerParams
from ..nn.params.nn_train_params import NNTrainParams
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


def _step_schedulers(scheds, val_edp, train_edp, *, epoch_idx: int) -> None:
    """ReduceLROnPlateau wants a metric; other schedulers step on epoch.
    Uses the shared finite-only val→train, error→loss resolver in
    nnx._metrics so the NNModel and Trainer paths can't drift. The
    metric is resolved once per epoch, not once per scheduler, so a
    multi-optimizer run emits at most one rejection/skip warning per
    epoch; a None resolution skips every plateau scheduler."""
    plateau_metric: Optional[float] = None
    resolved = False
    for sched in scheds:
        if isinstance(sched, lr_scheduler.ReduceLROnPlateau):
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
        trainer_step_fn: TrainerStepFn,
        callbacks: Optional[list[CallbackLike]] = None,
        salt: Optional[str] = None,
        components: Optional[list[Any]] = None,
    ) -> NNRun:
        """Run the multi-optimizer training loop and return the resulting NNRun.

        Args:
            params: NNTrainerParams — train_loader + n_epochs + optims dict +
                (optional) schedulers dict + (optional) val_loader, seed,
                save_phase_checkpoints, extra_metrics. Schedulers step once
                per epoch by default; set auto_step_schedulers=False when the
                custom step function owns scheduler timing.
            trainer_step_fn: required. `Callable[[TrainerStepContext],
                NNEvaluationDataPoint]`. The function owns the entire per-batch
                update — including which optimizers to step, in what order, and
                with what loss(es). There is no supervised fallback.
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
                trainer_step_fn is None, or any optim's
                NNOptimParams.is_valid() returns False.
        """
        if params is None:
            raise ValueError("trainer params must not be None")
        if params.train_loader is None:
            raise ValueError(
                "params.train_loader is required — set it directly or via with_train_loader(...) before train()."
            )
        if trainer_step_fn is None:
            raise ValueError(
                "trainer_step_fn is required — Trainer has no default "
                "supervised step because multi-optim updates are inherently "
                "scenario-specific."
            )
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

        optimizers = {
            name: build_optimizer(self.model.net, opt_params, strict_param_groups=True)
            for name, opt_params in params.optims.items()
        }

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
            return self._train_impl(
                params=params,
                run=run,
                optimizers=optimizers,
                trainer_step_fn=trainer_step_fn,
                callbacks=callbacks,
                components=components,
            )

    def _train_impl(
        self,
        *,
        params: NNTrainerParams,
        run: NNRun,
        optimizers: dict[str, torch.optim.Optimizer],
        trainer_step_fn: TrainerStepFn,
        callbacks: Optional[list[CallbackLike]],
        components: Optional[list[Any]] = None,
    ) -> NNRun:
        """Execute a validated multi-optimizer training session."""
        assert params.train_loader is not None
        train_loader = params.train_loader
        validate = params.val_loader is not None

        schedulers = {
            name: _build_scheduler(
                opt=optimizers[name],
                sched_params=params.schedulers.get(name, _DEFAULT_SCHEDULER_PARAMS),
                n_epochs=params.n_epochs,
            )
            for name in optimizers
        }

        from ..optimizers import optimizer_factory_state

        optimizer_factories = {name: optimizer_factory_state(params.optims[name]) for name in optimizers}
        normalized_callbacks = NNModel._normalize_callbacks(callbacks)
        registry = ComponentRegistry.discover(normalized_callbacks, trainer_step_fn, explicit=list(components or []))
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
                    val_edp = self.model.evaluate(
                        loader=params.val_loader,
                        extra_metrics=params.extra_metrics,
                    )
                else:
                    val_edp = None
                idps[-1] = idps[-1].with_val_edp(val_edp)

                # Each scheduler steps on its own optimizer's signal.
                # We feed the SAME (val_edp, train_edp) pair to all of
                # them because IDPs aggregate over all optims — separating
                # per-optim metrics would require the step fn to return
                # multiple EDPs, which complicates the contract without
                # clear benefit. Custom hooks can own scheduler timing by
                # setting auto_step_schedulers=False.
                if params.auto_step_schedulers:
                    _step_schedulers(schedulers.values(), val_edp, train_edp, epoch_idx=idx_epoch)

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
                    )
                except BaseException:
                    committed = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
                    if committed is None or committed.idp.epoch_idx != idx_epoch:
                        run.with_idps(idps[:n_idps_before_epoch]).save(update_best=False)
                    raise
                for deferred_checkpoint in ctx.deferred_checkpoint_writes:
                    deferred_checkpoint()
                if best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
                    best_checkpoint = checkpoint

                # Verbatim-shared with the NNModel.train loop — one
                # implementation, so the postfix format can't drift.
                self.model._update_tqdm_postfix(tqdm_bar, optimizers[primary], val_edp, train_edp)

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
