from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import inspect
import json
import math
import os
import re
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, cast

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from .._confusion import LabelCounts, label_counts, label_kind, record_scores
from .._metrics import _resolve_metric, _resolve_scheduler_metric, classification_edp
from .._scheduler_clock import (
    HORIZON_KINDS,
    NO_UPDATE_REPORTER,
    SchedulerClock,
    listens,
    planned_updates,
    uses_update_clock,
)
from .._update_engine import gradients_finite, scaler_step
from ..compilation import CompileSpec, _CompileHolder, _CompileSession
from ..components import ComponentRegistry, ComponentRestoreError, ResumeStatus
from ..history import (
    HistoryJournal,
    _check_history,
    _dispatch_epoch_end,
    _lend_idps,
    _ReplicaHistory,
    _training_history,
)
from ..models import (
    BatchAdapter,
    MissingModelFactoryError,
    ModelSpec,
    RuntimeModule,
    _as_inputs,
    _UnpackBatch,
    build_module,
    check_state_schema,
    default_batch_adapter,
)
from ..monitors import (
    MetricSpec,
    MonitorRecord,
    MonitorSpec,
    MonitorTracker,
    _check_metric_inputs,
    _ignore_index,
    _metric_domain,
    _MetricSet,
    _TrainEpochSummary,
)
from ..precision import EVALUATE as _PRECISION_EVALUATE
from ..precision import FULL_PRECISION_ONLY as _FULL_PRECISION_ONLY
from ..precision import PREDICT as _PRECISION_PREDICT
from ..precision import TRAIN as _PRECISION_TRAIN
from ..precision import PrecisionPolicy, PrecisionUnsupportedError, ResolvedPrecision, resolve_precision
from ..provenance import ExperimentManifest
from ..seeding import _capture_rng_state, _restore_rng_state  # the loop's checkpointed RNG streams
from ..tasks import TaskAdapter, _bounded_extra_metrics_error, task_adapter
from ..transforms import _canonical_transforms, _recipe_transforms, _replayable, _snapshot_transforms, _state_shapes
from ..utils import Utils, _capture_training_modes, _restore_training_modes
from .enum.checkpoints import Checkpoints, phase_tag
from .enum.devices import Devices
from .enum.nets import Nets
from .params.nn_checkpoint import (
    _MODEL_CHECKPOINT_TAG,
    NNCheckpoint,
    NNCheckpointTransform,
    ResumePointError,
    _snapshot_state_dict,
    _tensor_state_dict,
)
from .params.nn_evaluation_data_point import NNEvaluationDataPoint
from .params.nn_iteration_data_point import NNIterationDataPoint
from .params.nn_model_params import NNModelParams
from .params.nn_params import NNParams
from .params.nn_run import NNRun, _best_err, _print_run_saved
from .params.nn_train_params import NNTrainParams

if TYPE_CHECKING:
    from ..prediction import PredictionResult, ProbabilitySpec
    from ..streaming import PredictionStream
    from .callbacks import Callback


# HuggingFace Hub integration — the mixin is OPTIONAL. We only import it
# at module load when the `thekaveh-nnx[hub]` extra is installed; otherwise we use
# a thin stub that defers errors to call time. This keeps `pip install thekaveh-nnx`
# working without huggingface_hub.
try:
    from huggingface_hub import PyTorchModelHubMixin as _HubMixinBase  # pyright: ignore[reportAssignmentType]

    _HUB_AVAILABLE = True
except ImportError:  # pragma: no cover — gated by optional dep

    class _HubMixinBase:
        """No-op stub installed when ``huggingface_hub`` is not available.

        Any attempt to call save_pretrained / from_pretrained / push_to_hub
        raises a clear ImportError pointing at the ``thekaveh-nnx[hub]`` extra.
        """

        def _hub_unavailable(self) -> NNModel:
            raise ImportError("HuggingFace Hub integration requires the `hub` extra: `pip install thekaveh-nnx[hub]`.")

        def save_pretrained(self, *args, **kwargs):
            self._hub_unavailable()

        def push_to_hub(self, *args, **kwargs):
            self._hub_unavailable()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("HuggingFace Hub integration requires the `hub` extra: `pip install thekaveh-nnx[hub]`.")

    _HUB_AVAILABLE = False


# Name of the safetensors file inside a save_pretrained directory. Matches
# the constant huggingface_hub publishes (SAFETENSORS_SINGLE_FILE) — kept
# duplicated here so the no-hub stub doesn't reach into hf_hub internals.
_HUB_MODEL_FILENAME = "model.safetensors"
_HUB_CONFIG_FILENAME = "config.json"


# Legacy callback signature retained for backwards compatibility with notebooks
# that pass `callbacks=[lambda idps: plot(...)]`. Adapted internally via
# _LegacyCallback (in callbacks.py).
LegacyCallback = Callable[[list[NNIterationDataPoint]], None]
CallbackLike = Union["Callback", LegacyCallback]


def _component_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _optimizer_topology(optimizer: torch.optim.Optimizer, net: torch.nn.Module) -> list[list[dict[str, Any]]]:
    """Describe optimizer groups by model parameter identity, not position.
    An uninitialized lazy parameter (a wrapped module's ``nn.LazyLinear``
    before its first forward, FEAT-006) has no shape yet: ``None``."""
    names = {id(param): name for name, param in net.named_parameters()}
    return [
        [
            {
                "name": names.get(id(param), "<external>"),
                "shape": None if torch.nn.parameter.is_lazy(param) else list(param.shape),
            }
            for param in group["params"]
        ]
        for group in optimizer.param_groups
    ]


def _named_training_state(
    net: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, Any],
    optimizer_factories: Optional[Mapping[str, Optional[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """A ``Trainer``'s name-keyed optimizer / scheduler bundle, as
    ``NNCheckpoint.save`` keyword arguments (FEAT-005): states, types,
    parameter topologies and registered-factory identities, so a resume can
    validate each optimizer as ``NNModel.train`` validates its one."""
    return {
        "optimizers_state": {name: opt.state_dict() for name, opt in optimizers.items()},
        "optimizer_types": {name: _component_type(opt) for name, opt in optimizers.items()},
        "optimizer_topologies": {name: _optimizer_topology(opt, net) for name, opt in optimizers.items()},
        "optimizer_factories": {name: (optimizer_factories or {}).get(name) for name in optimizers},
        "schedulers_state": {name: sch.state_dict() for name, sch in schedulers.items()},
        "scheduler_types": {name: _component_type(sch) for name, sch in schedulers.items()},
    }


def _loader_num_workers(train_loader: Any) -> int:
    """Worker count of a batch source, treating absent metadata as zero.

    Training accepts any re-iterable of batches, not only ``DataLoader``
    (FIX-012); only a real loader can spawn workers whose local RNG state
    is not reconstructible on resume. Never iterates the source.
    """
    workers = getattr(train_loader, "num_workers", 0)
    try:
        return int(workers or 0)
    except (TypeError, ValueError):
        return 0


def _resume_checkpoint_type(value: Any) -> Any:
    """The checkpoint a resume reads: a :class:`Checkpoints` tag, or a
    ``ModelCheckpoint`` file stem ``"<tag>_e<epoch>"`` such as
    ``"custom_e3"`` (FEAT-005). Anything else — ``"LAST"`` included — is
    rejected with the list of valid tags."""
    try:
        return Checkpoints(value)
    except ValueError:
        if isinstance(value, str) and re.fullmatch(rf"{_MODEL_CHECKPOINT_TAG}_e[0-9]+", value):
            return value
        raise ValueError(
            f"resume_from_checkpoint must be a Checkpoints tag ({', '.join(c.value for c in Checkpoints)}) "
            f"or a ModelCheckpoint file stem '<tag>_e<epoch>', got {value!r}"
        ) from None


def _check_resume_horizon(
    scheduler_params: Any, *, n_epochs: int, start_epoch: Optional[int] = None, owner: str = ""
) -> None:
    """A resumed one-cycle / warmup-decay schedule needs one explicit
    horizon covering the original and resumed epochs (checked before the
    checkpoint is read, and again against its completed epoch)."""
    kind = getattr(scheduler_params, "kind", None)
    if kind is None or str(kind) not in HORIZON_KINDS:
        return
    total_steps = scheduler_params.total_steps
    if total_steps is None:
        raise ValueError(
            f"resuming {kind}{owner} requires scheduler.total_steps to be set explicitly "
            "to one shared horizon covering the original and resumed epochs"
        )
    # An optimizer_update clock counts updates, not epochs: its horizon is
    # checked against the restored update count (SchedulerClock).
    if start_epoch is not None and not uses_update_clock(scheduler_params) and start_epoch + n_epochs > total_steps:
        raise ValueError(
            f"resumed {kind}{owner} would reach epoch {start_epoch + n_epochs}, beyond "
            f"scheduler.total_steps={total_steps}; configure one shared horizon covering the original and "
            "resumed epochs"
        )


def _manifest_lineage(manifest: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The resume point's generation and checkpoint id for ``ResumeStatus``
    (#394); nothing for a checkpoint without a manifest."""
    if manifest is None:
        return {}
    return {"source_generation": manifest.get("generation"), "source_checkpoint_id": manifest.get("checkpoint_id")}


def _session_epochs(params: Any, start_epoch: int, counters: Optional[Mapping[str, Any]], where: str) -> int:
    """How many epochs this session trains (#394). ``resume_epochs="additional"``:
    ``n_epochs`` more. ``"planned"``: up to the run's planned ``n_epochs`` in
    total — refused, before anything is restored, when the source planned a
    different total or has already completed it."""
    if getattr(params, "resume_epochs", "additional") != "planned":
        return params.n_epochs
    if counters is None:
        raise ValueError(
            f"{where}: resume_epochs='planned' continues a run's training state; this resume restores the weights "
            "only (resume_mode='weights_only', or a checkpoint without training state) — resume statefully, or "
            "use resume_epochs='additional'"
        )
    if counters.get("planned_n_epochs") is None or counters.get("global_step") is None:
        raise ValueError(
            f"{where}: its checkpoint predates resume counters (#394), so its plan and logical step are unknown — "
            "resume it with resume_epochs='additional'"
        )
    saved_plan = counters.get("planned_n_epochs")
    if saved_plan is not None and saved_plan != params.n_epochs:
        raise ValueError(
            f"{where} planned {saved_plan} epochs; a planned resume continues that plan — set n_epochs={saved_plan}, "
            "or resume with resume_epochs='additional' to train n_epochs more"
        )
    remaining = params.n_epochs - start_epoch
    if remaining <= 0:
        raise ValueError(
            f"{where} already completed its planned {params.n_epochs} epochs (through epoch {start_epoch - 1}): "
            "nothing is left to resume"
        )
    return remaining


@dataclass(frozen=True)
class _ResumeSource:
    checkpoint: NNCheckpoint
    # The training-state bundle after ``resume_mode`` was applied (None:
    # restore the weights only).
    training_state: Optional[dict[str, Any]]
    net_state: dict[str, Any]
    label: str
    # #394: the verified resume-point manifest (None for a checkpoint written
    # before manifests, or an in-memory source).
    manifest: Optional[dict[str, Any]] = None


# A checkpoint and its training state held in memory under (run id, label) for
# one train() call — a run bundle's (nnx.bundles), which never becomes a pickle.
_IN_MEMORY_RESUME: contextvars.ContextVar[Optional[tuple[str, str, NNCheckpoint, Optional[dict[str, Any]]]]] = (
    contextvars.ContextVar("nnx_in_memory_resume", default=None)
)


def _without_submodules(state: Mapping[str, Any], names: Sequence[str]) -> Mapping[str, Any]:
    """``state`` without the weights of the named top-level submodules
    (``from_checkpoint(exclude_submodules=...)``); every name must be present."""
    if isinstance(names, str):
        raise TypeError("exclude_submodules must be a sequence of submodule names, not a string")
    names = tuple(names or ())  # a one-shot iterable is read once
    if not names:
        return state
    held = {key.split(".", 1)[0] for key in state if "." in key}
    unknown = sorted(set(names) - held)
    if unknown:
        raise ValueError(
            f"exclude_submodules names {unknown}, which the checkpoint does not hold (its submodules: {sorted(held)})"
        )
    excluded = tuple(f"{name}." for name in names)
    kept: OrderedDict[str, Any] = OrderedDict(
        (key, value) for key, value in state.items() if not key.startswith(excluded)
    )
    metadata = getattr(state, "_metadata", None)  # per-module versions load_state_dict reads
    if metadata is not None:
        kept._metadata = OrderedDict(  # type: ignore[attr-defined]
            (key, value) for key, value in metadata.items() if not key.startswith(excluded) and key not in names
        )
    return kept


def _verified_resume_point(run_id: str, ckpt_type: Any, ddp: Any) -> Optional[dict[str, Any]]:
    """``NNCheckpoint.verify`` of the resume point, once (FEAT-046): under
    DDP the writer rank verifies it and broadcasts the manifest or its
    refusal, so the other ranks never hash it — and every rank refuses a
    torn, mixed or corrupted point together, naming the reason. Every rank
    must reach this alike: it is the first thing the "resuming the run" block
    can fail on."""
    if ddp is None:
        return NNCheckpoint.verify(run=run_id, type=ckpt_type)
    import torch.distributed as dist

    from ..distributed import DistributedFailure

    refusal: Optional[BaseException] = None
    verdict: Any = None
    if ddp.writer:
        try:
            verdict = ("verified", NNCheckpoint.verify(run=run_id, type=ckpt_type))
        except Exception as error:  # shared below, then raised on every rank
            refusal = error
            verdict = ("refused", type(error).__name__, str(error))
    shared = [verdict]
    dist.broadcast_object_list(shared, src=ddp.spec.writer_rank)
    verdict = shared[0]
    if verdict[0] == "verified":
        return verdict[1]
    if refusal is not None:
        raise refusal
    kind, message = verdict[1], verdict[2]
    if kind == ResumePointError.__name__:
        raise ResumePointError(message)
    raise DistributedFailure(f"verifying the resume point failed on rank {ddp.spec.writer_rank}: {kind}: {message}")


def _load_resume_source(
    run_id: str,
    checkpoint: Any,
    mode: str,
    *,
    trainer: bool,
    live_transforms: Sequence[Any] = (),
    ddp: Any = None,
) -> _ResumeSource:
    """Read the checkpoint a resume starts from — shared by ``NNModel.train``
    and ``Trainer.train`` — and reject, before anything is mutated, a
    missing checkpoint, a transformed one without pre-transform state, and
    a bundle ``resume_mode`` cannot use. Under DDP (``ddp``, the fit's
    session) only the writer rank verifies the point (FEAT-046)."""
    ckpt_type = _resume_checkpoint_type(checkpoint)
    in_memory = _IN_MEMORY_RESUME.get()
    manifest: Optional[dict[str, Any]] = None
    if in_memory is not None and in_memory[:2] == (run_id, str(ckpt_type)):
        ckpt, training_state = in_memory[2], in_memory[3]
    else:
        # #394: a torn, mixed or corrupted resume point is refused here, with
        # its reason, before anything is restored from it.
        manifest = _verified_resume_point(run_id, cast(Any, ckpt_type), ddp)
        ckpt, training_state = NNCheckpoint.load_with_training_state(run=run_id, type=cast(Any, ckpt_type))
    if ckpt is None:
        raise ValueError(f"resume_from_run_id={run_id!r}/{ckpt_type} not found on disk")
    resume_net_state = training_state.get("model") if training_state is not None else None
    # FEAT-016: the resuming model must carry exactly the recipe the source
    # was trained with (ids, versions, targets and config) — its weights
    # only fit that topology and that configuration.
    saved_recipe, live_recipe = _recipe_transforms(ckpt.transforms), _recipe_transforms(live_transforms)
    if saved_recipe != live_recipe:
        raise ValueError(
            f"resume_from_run_id={run_id!r}/{ckpt_type} was trained with the transformation recipe "
            f"{[t.state() for t in saved_recipe]}, but this model carries {[t.state() for t in live_recipe]}: "
            "materialize the same nnx.transforms.TransformRecipe on the model before resuming"
        )
    # A checkpoint whose transforms are exactly the ones the model already
    # carries (its recipe, or a recipe and the conversion of a converted
    # model trained again) holds weights of the live topology, which load
    # directly; any other transformed checkpoint needs its pre-transform
    # state.
    if ckpt.transforms and resume_net_state is None and tuple(ckpt.transforms) != tuple(live_transforms):
        raise ValueError(
            "this transformed checkpoint has no pre-transform training state and cannot be warm-resumed; "
            "use NNModel.from_checkpoint() for inference or resume from an untransformed checkpoint"
        )
    label = str(ckpt_type)
    training_state = _resume_training_state(training_state, mode, f"{run_id}/{label}", trainer=trainer)
    return _ResumeSource(ckpt, training_state, resume_net_state or ckpt.net_state, label, manifest)


def _restore_weights_only(
    net: torch.nn.Module, source: _ResumeSource, train_loader: Any, mode: str, *, fresh: str, stacklevel: int = 3
) -> None:
    """Load only the source weights (restoring the net on failure); warn
    unless ``resume_mode="weights_only"`` asked for exactly this. The
    warning is attributed ``stacklevel`` frames up (``train()`` by default)."""
    net_snapshot = _snapshot_state_dict(net.state_dict())
    rng_snapshot = _capture_rng_state(train_loader)
    try:
        net.load_state_dict(source.net_state)
    except BaseException:
        net.load_state_dict(net_snapshot)
        _restore_rng_state(rng_snapshot, train_loader)
        raise
    if mode != "weights_only":
        warnings.warn(
            f"checkpoint has no training-state sidecar; model weights were restored, but {fresh} restart "
            "from their configured defaults",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def _rollback_resume(
    net: torch.nn.Module, net_state: dict[str, Any], rng_state: dict[str, Any], train_loader: Any
) -> None:
    """Put the model and RNG back after a failed component restore, which
    runs after ``on_train_begin``. A callback may have changed the network
    there (QAT preparation swaps modules); a rollback that cannot load is
    reported as a warning instead of replacing the original error."""
    try:
        net.load_state_dict(net_state)
    except Exception as error:
        warnings.warn(
            f"component restore failed and the model could not be rolled back ({type(error).__name__}: {error})",
            RuntimeWarning,
            stacklevel=3,
        )
    _restore_rng_state(rng_state, train_loader)


def _resume_training_state(
    training_state: Optional[dict[str, Any]], mode: str, source: str, *, trainer: bool
) -> Optional[dict[str, Any]]:
    """Apply ``resume_mode`` to a loaded training-state bundle (FEAT-005),
    failing before anything is restored when it cannot be honoured."""
    if mode == "weights_only":
        return None
    if training_state is None:
        if mode == "stateful":
            raise ValueError(
                f"checkpoint {source} is weights-only (it carries no training state), so a stateful resume is "
                "impossible; resume from a checkpoint written by train(), or pass resume_mode='weights_only'"
            )
        return None
    written_by_trainer = training_state.get("optimizers") is not None and training_state.get("optimizer") is None
    if written_by_trainer and not trainer:
        raise ValueError(
            f"checkpoint {source} was written by Trainer (named optimizers); resume it with Trainer.train, "
            "or pass resume_mode='weights_only' to warm-start NNModel.train from its weights"
        )
    if trainer and not written_by_trainer:
        raise ValueError(
            f"checkpoint {source} was written by NNModel.train (one optimizer); resume it with NNModel.train, "
            "or pass resume_mode='weights_only' to warm-start the Trainer from its weights"
        )
    return training_state


def _plan_component_restore(registry: ComponentRegistry, training_state: Mapping[str, Any]) -> Any:
    """Validate saved component state against ``registry`` without mutating
    anything. Sidecars written before FEAT-005 carry no component state:
    every component then starts fresh (with a warning when there are any),
    unless a component says it cannot (an optimizer_update scheduler clock,
    FEAT-014): that is refused before anything is restored."""
    saved = training_state.get("components")
    if saved is None:
        problems = registry._legacy_problems(training_state)
        if problems:
            raise ComponentRestoreError(problems)
        if len(registry):
            warnings.warn(
                f"checkpoint predates component state (FEAT-005); {', '.join(registry.names)} start fresh",
                RuntimeWarning,
                stacklevel=3,
            )
        return registry.fresh_plan()
    return registry.plan(saved)


def _collect_checkpoint_transforms(callbacks: list[Callback]) -> tuple[NNCheckpointTransform, ...]:
    # on_train_end runs in reverse callback order, so persist transforms in
    # that same order for deterministic topology replay during reconstruction.
    return tuple(transform for callback in reversed(callbacks) for transform in callback.checkpoint_transforms())


def _apply_checkpoint_transform(model: NNModel, transform: NNCheckpointTransform) -> None:
    if _replayable(transform):
        # FEAT-016: a recorded recipe operation — its topology is rebuilt
        # (low-rank factors allocated, never re-factorized) before the
        # saved tensors are loaded.
        from ..transforms import _replay

        _replay(model, transform)
        return
    if transform.name == "torchao_qat" and transform.version == 1:
        from ..quantize.qat import _build_quantizer

        try:
            qat_config = transform.options["qat_config"]
            groupsize = transform.options["groupsize"]
        except KeyError as error:
            raise ValueError(f"invalid torchao_qat checkpoint transform: missing option {error.args[0]!r}") from error
        quantizer = _build_quantizer(qat_config, groupsize=groupsize)
        quantizer.prepare(model.net)
        quantizer.convert(model.net)
        return

    raise ValueError(
        f"unsupported checkpoint transform {transform.name!r} version {transform.version}; "
        "upgrade NNx or load the checkpoint with the producer's compatible version"
    )


def _unportable_transform(transforms: Sequence[NNCheckpointTransform]) -> Optional[str]:
    """Why the first recorded transform NNx cannot replay from data alone
    (``"topology transform <index> (<id> version <v>) ..."``), or ``None``.
    Recipe operations (FEAT-016) of a known version and the torchao QAT
    conversion replay; any other id is an operation only its producer can
    rebuild — a callback's own train-end transform, say."""
    from ..transforms import _VERSIONS, RecipeError, TransformOp

    for index, transform in enumerate(transforms):
        where = f"topology transform {index} ({transform.name!r} version {transform.version})"
        if _replayable(transform):
            if transform.version not in _VERSIONS[transform.name]:
                return f"{where} has a version this NNx does not know"
            try:
                TransformOp.from_checkpoint_transform(transform)
            except RecipeError as error:
                return f"{where} has malformed options: {error}"
        elif not (transform.name == "torchao_qat" and transform.version == 1):
            return f"{where} is not an operation NNx can replay; only its producer can rebuild it"
    return None


def _replay_transforms(model: NNModel, transforms: Sequence[NNCheckpointTransform]) -> None:
    """Replay a checkpoint's recorded transforms in order — before any saved
    tensor is loaded — naming the transform that cannot be replayed."""
    for index, transform in enumerate(transforms):
        try:
            _apply_checkpoint_transform(model, transform)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"topology transform {index} ({transform.name!r} version {transform.version}) cannot be replayed: "
                f"{error}"
            ) from error


def _replaced_layers(state_keys: Iterable[str], base_keys: Iterable[str]) -> tuple[list[str], list[str]]:
    """The base layers ``X`` a state holds as LoRA wrappers (``X.base.weight``
    or ``X.lora_A``) and as low-rank factors (``X.0.weight`` and
    ``X.1.weight``) instead of ``X.weight`` — the topology an nnx.transforms
    recipe (FEAT-016) records and unrecorded surgery leaves behind."""
    keys = set(state_keys)
    lora_layers: list[str] = []
    low_rank_layers: list[str] = []
    for key in base_keys:
        if not key.endswith(".weight") or key in keys:
            continue
        layer = key[: -len(".weight")]
        if f"{layer}.lora_A" in keys or f"{layer}.base.weight" in keys:
            lora_layers.append(layer)
        if f"{layer}.0.weight" in keys and f"{layer}.1.weight" in keys:
            low_rank_layers.append(layer)
    return lora_layers, low_rank_layers


def _check_trained_recipe(
    model: NNModel,
    trained_recipe: Sequence[NNCheckpointTransform],
    declared: Sequence[NNCheckpointTransform] = (),
) -> None:
    """Refuse, before a checkpoint is written, a model whose topology no
    checkpoint of this run could rebuild (FEAT-016): recipe operations a
    callback applied or declared during training (a recipe is materialized
    before training — the one the run id records), or unrecorded surgery
    on a recipe model. A train-end transform callbacks declared (QAT)
    rebuilds its own topology, so drift is not checked against it."""
    live = _recipe_transforms(model._topology_transforms)
    late = [t for t in declared if _replayable(t)]
    if live != tuple(trained_recipe) or late:
        added = live[len(trained_recipe) :] if live[: len(trained_recipe)] == tuple(trained_recipe) else live
        raise ValueError(
            "a transformation recipe is materialized before training, but recipe operations "
            f"{[t.state() for t in (*added, *late)]} were applied or declared during it, which no checkpoint could "
            "resume — materialize the nnx.transforms.TransformRecipe before calling train() (refused before this "
            "checkpoint was written; the run's earlier checkpoints are kept)"
        )
    if trained_recipe and all(_replayable(t) for t in declared):
        drift = model._topology_drift()
        if drift is not None:
            raise ValueError(
                f"{drift}; refused before the checkpoint is written (the run's earlier checkpoints are kept) — apply "
                "topology changes through nnx.transforms.TransformRecipe before training"
            )


def _final_transforms(
    model: NNModel, callbacks: list[Callback], trained_recipe: Sequence[NNCheckpointTransform]
) -> tuple[tuple[NNCheckpointTransform, ...], bool]:
    """The transforms the final LAST records — the model's own followed by
    those its callbacks applied at train end — and whether that LAST keeps
    the pre-transform state for resuming. A recipe recorded before training
    (FEAT-016) is the live topology already, so it alone keeps none; the
    model is checked first (:func:`_check_trained_recipe`)."""
    declared = _collect_checkpoint_transforms(callbacks)
    _check_trained_recipe(model, trained_recipe, declared)
    final = (*model._topology_transforms, *declared)
    return final, any(not _replayable(t) for t in final)


def _refuse_unrecorded_recipe_state(net_state: Mapping[str, Any], base_state: Mapping[str, Any]) -> None:
    if any(_replaced_layers(net_state, base_state)):
        raise ValueError(
            "the weights come from a transformed topology (LoRA wrappers or low-rank factors) but record no "
            "transformation recipe: a raw state dict or an adapter-only export cannot rebuild the topology alone — "
            "materialize the same nnx.transforms.TransformRecipe on a fresh model, then load the weights into it"
        )


def _looks_like_converted_qat_state(net_state: Mapping[str, Any]) -> bool:
    return any("scales" in key or "zeros" in key for key in net_state)


class _CallbackFinalizer:
    def __init__(self, callbacks: list[Any], ctx: Any):
        self._callbacks = callbacks
        self._ctx = ctx
        self._started: list[Any] = []

    def __enter__(self):
        return self

    def start(self) -> None:
        for cb in self._callbacks:
            cb.on_train_begin(self._ctx)
            self._started.append(cb)

    def __exit__(self, exc_type, exc, tb):
        cleanup_errors: list[BaseException] = []
        # FEAT-036: a journal run lends history_access="full" callbacks the
        # whole history (read back once); everyone else sees ctx.idps as is.
        history = getattr(self._ctx, "history_records", None)
        view = history.lender(tolerant=True) if history is not None else (lambda callback: None)
        for cb in reversed(self._started):
            try:
                _lend_idps(self._ctx, view(cb), lambda cb=cb: cb.on_train_end(self._ctx))
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)

        if exc is not None:
            for cleanup_error in cleanup_errors:
                warnings.warn(
                    f"on_train_end cleanup failed while handling {type(exc).__name__}: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return False

        if cleanup_errors:
            for cleanup_error in cleanup_errors[1:]:
                warnings.warn(
                    f"additional on_train_end cleanup failed: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raise cleanup_errors[0]
        return False


class PredictResult(NamedTuple):
    """Structured result of NNModel.predict().

    Unpacks positionally as ``(logits, classes)`` so callers doing
    ``log, hat = model.predict(X)`` keep working after the upgrade from
    the original 2-tuple. Field access (``result.logits``, ``result.classes``)
    is preferred for new code.
    """

    logits: np.ndarray
    classes: np.ndarray


@dataclass(slots=True)
class GradientAccumulationState:
    """Scalar loss-normalization state shared by consecutive step contexts."""

    normalization_weight: float = 0.0
    loss_numerator: float = 0.0
    normalization_required: bool = True


@dataclass(frozen=True, slots=True)
class TrainStepContext:
    """Frozen bundle of state passed into a training-step function.

    The default `default_train_step` runs the standard supervised
    forward/backward/step. Users can pass their own
    `train_step_fn: Callable[[TrainStepContext], NNEvaluationDataPoint]`
    to NNModel.train() for non-supervised paradigms (autoencoder, VAE,
    link prediction, recommendation, diffusion, etc.). The custom step
    is fully responsible for forward, backward, optimizer.step,
    gradient accumulation, AMP scale/unscale, grad clipping, and the
    NaN/Inf guard — the context tells it what knobs are set; honoring
    them is on the caller.
    """

    model: NNModel
    batch: Any
    optimizer: torch.optim.Optimizer
    scaler: Optional[torch.amp.GradScaler]
    grad_clip_norm: Optional[float]
    extra_metrics: Optional[Mapping[str, Callable]]
    accumulate_grad_batches: int
    batch_idx: int
    epoch_idx: int
    is_last_batch: bool = False
    accumulation_state: Optional[GradientAccumulationState] = None
    # FEAT-003: the whole-epoch training summary a run with declared metrics
    # or a monitor keeps; `default_train_step` reports each batch's outputs
    # and denominators to it. Custom steps may ignore it.
    epoch_summary: Optional[_TrainEpochSummary] = None
    # FEAT-028: the run's resolved precision — `precision.autocast()` is the
    # forward context, and `scaler` is set for fp16 only. A custom step owns
    # applying it; a context built by hand without it keeps the legacy rule
    # (autocast and the scaler only for a scaler on CUDA).
    precision: Optional[ResolvedPrecision] = None
    # FEAT-014: call (without a name) once after each optimizer step the
    # step function takes itself; an ``optimizer_update``-clock scheduler
    # steps on every report. ``default_train_step`` (never for a step the
    # AMP scaler skipped) and ``finalize_step`` report their own steps, so a
    # step that delegates to them does not report again.
    report_update: Callable[[], None] = NO_UPDATE_REPORTER


TrainStepFn = Callable[[TrainStepContext], NNEvaluationDataPoint]


@dataclass(frozen=True, slots=True)
class EvalStepContext:
    """Frozen bundle of state passed into a validation-step function (#86).

    Mirrors :class:`TrainStepContext` for the per-epoch VALIDATION pass: users
    can pass ``eval_step_fn: Callable[[EvalStepContext], NNEvaluationDataPoint]``
    to ``NNModel.train()`` to replace the built-in classification ``evaluate()``
    for non-classification paradigms (next-token LM perplexity, DPO margins,
    regression MAE, ...). The step runs under ``torch.no_grad()`` and its
    returned EDP becomes ``val_edp`` — recorded on the epoch's last idp and
    persisted through the incremental run save like any built-in val metric.
    """

    model: NNModel
    val_loader: Iterable[Any]
    extra_metrics: Optional[Mapping[str, Callable]]
    epoch_idx: int
    # The run's declared metrics (FEAT-003), for a step that reports them
    # (e.g. nnx.streaming.streaming_eval_step); () when none are declared.
    metrics: tuple[MetricSpec, ...] = ()


EvalStepFn = Callable[[EvalStepContext], NNEvaluationDataPoint]


def _finite_training_loss_value(train_loss: torch.Tensor) -> float:
    loss_value = float(train_loss.detach())
    # NaN/Inf guard: silent divergence leaves checkpoints full of garbage
    # weights. Raise before backward/step so non-finite gradients cannot
    # mutate model parameters.
    if not np.isfinite(loss_value):
        raise FloatingPointError(
            f"non-finite training loss ({loss_value!r}) — training diverged. "
            "Check learning rate, gradient clipping (NNOptimParams.grad_clip_norm), "
            "or input normalization."
        )
    return loss_value


_ELEMENTWISE_MEAN_LOSS_TYPES = (
    torch.nn.BCELoss,
    torch.nn.BCEWithLogitsLoss,
    torch.nn.GaussianNLLLoss,
    torch.nn.HuberLoss,
    torch.nn.KLDivLoss,
    torch.nn.L1Loss,
    torch.nn.MSELoss,
    torch.nn.PoissonNLLLoss,
    torch.nn.SmoothL1Loss,
    torch.nn.SoftMarginLoss,
)


def _is_native_nll(loss_fn: torch.nn.Module) -> bool:
    """True for a `torch.nn.NLLLoss` whose forward is the stock one.

    A subclass that overrides `forward` keeps its own supplied-input
    contract (it may already expect raw logits), so it is deliberately
    *not* treated as native here — see `_loss_input`.
    """
    return isinstance(loss_fn, torch.nn.NLLLoss) and type(loss_fn).forward is torch.nn.NLLLoss.forward


def _loss_input(loss_fn: torch.nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """Adapt the network's raw output to what `loss_fn` expects (FIX-001).

    Built-in nets emit unrestricted logits, but native `torch.nn.NLLLoss`
    requires *log probabilities*: feeding it raw logits optimizes an
    unnormalized objective (loss can go negative, the gradient lacks the
    competing-class term). For the exact native type this returns
    `log_softmax(logits, dim=1)` — callers pass logits already reshaped
    to `(rows, classes)` by `_fwd_pass`, so dim 1 is always the class
    axis. Every other loss (CE, BCE, MSE, custom modules and NLLLoss
    subclasses with their own forward) receives the raw output unchanged.
    Prediction paths never call this: `predict().logits` stays raw.
    """
    if _is_native_nll(loss_fn):
        return torch.nn.functional.log_softmax(logits, dim=1)
    return logits


def _loss_normalization_weight(
    loss_fn: torch.nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Optional[float]:
    """Return the native mean denominator, or None for additive sums."""
    reduction = getattr(loss_fn, "reduction", None)
    if reduction == "sum":
        return None
    if reduction != "mean":
        return float(target.size(0))

    native_ce = (
        isinstance(loss_fn, torch.nn.CrossEntropyLoss) and type(loss_fn).forward is torch.nn.CrossEntropyLoss.forward
    )
    native_nll = _is_native_nll(loss_fn)
    if native_ce or native_nll:
        classification_loss = cast(Union[torch.nn.CrossEntropyLoss, torch.nn.NLLLoss], loss_fn)
        if native_ce and target.is_floating_point():
            return float(target.numel() // target.size(1))

        valid_target = target[target != classification_loss.ignore_index]
        if classification_loss.weight is None:
            return float(valid_target.numel())
        if valid_target.numel() == 0:
            return 0.0
        return float(classification_loss.weight[valid_target].sum().detach())

    if type(loss_fn) in _ELEMENTWISE_MEAN_LOSS_TYPES:
        return float(math.prod(torch.broadcast_shapes(logits.shape, target.shape)))

    return float(target.size(0))


def _loss_terms(
    loss_fn: torch.nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Optional[float]]:
    """Return display loss, additive numerator, and its normalization weight."""
    loss = loss_fn(_loss_input(loss_fn, logits), target)
    normalization_weight = _loss_normalization_weight(loss_fn, logits, target)
    if normalization_weight is None:
        return loss, loss, None
    if normalization_weight == 0:
        # Native mean CE/NLL returns NaN when every target is ignored (or has
        # zero class weight). Preserve that display value for the finite-loss
        # guard, but contribute a differentiable zero to a larger valid cycle.
        return loss, logits.sum() * 0.0, 0.0
    return loss, loss * normalization_weight, normalization_weight


def _record_accumulation_loss(
    train_loss: torch.Tensor,
    normalization_weight: Optional[float],
    accumulation_state: Optional[GradientAccumulationState],
    *,
    should_step: bool,
) -> float:
    """Record a batch denominator and enforce finiteness at the right scope."""
    if accumulation_state is None:
        return _finite_training_loss_value(train_loss)

    if normalization_weight is None:
        accumulation_state.normalization_required = False
        return _finite_training_loss_value(train_loss)

    if normalization_weight != 0:
        loss_value = _finite_training_loss_value(train_loss)
        accumulation_state.loss_numerator += loss_value * normalization_weight
        accumulation_state.normalization_weight += normalization_weight
        return loss_value

    if should_step and accumulation_state.normalization_required and accumulation_state.normalization_weight == 0:
        return _finite_training_loss_value(train_loss)
    if accumulation_state.normalization_weight:
        return accumulation_state.loss_numerator / accumulation_state.normalization_weight
    return 0.0


def _classification_metric_tensors(
    loss_fn: torch.nn.Module,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove exact CE/NLL ignore targets before classification metrics."""
    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        return (target >= 0.5).to(dtype=torch.long), prediction
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss) and target.is_floating_point():
        return target.argmax(dim=1).reshape(-1), prediction.reshape(-1)
    if isinstance(loss_fn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)) and not target.is_floating_point():
        classification_loss = cast(Union[torch.nn.CrossEntropyLoss, torch.nn.NLLLoss], loss_fn)
        valid = target != classification_loss.ignore_index
        return target[valid], prediction[valid]
    return target, prediction


def _classification_edp_for_loss(
    *,
    loss_fn: torch.nn.Module,
    target: torch.Tensor,
    prediction: torch.Tensor,
    loss: float,
    extra_metrics: Optional[Mapping[str, Callable]],
) -> NNEvaluationDataPoint:
    metric_target, metric_prediction = _classification_metric_tensors(loss_fn, target, prediction)
    if metric_target.numel() == 0:
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=loss,
            error=None,
        )
    return classification_edp(
        Y=metric_target,
        Y_hat=metric_prediction,
        loss=loss,
        extra_metrics=extra_metrics,
    )


def _check_resume_precision(
    training_state: Mapping[str, Any], precision: ResolvedPrecision, scaler: Optional[Any]
) -> None:
    """Refuse a stateful resume into a different effective precision
    (FEAT-028), or with a scaler that appears or disappears, before anything
    is restored. A sidecar written before precision was recorded trained in
    fp16 exactly when it saved an enabled scaler (a disabled GradScaler saves
    an empty state); a weights-only warm start may switch."""
    record = training_state.get("precision")
    if isinstance(record, Mapping) and record.get("effective") is not None:
        saved = str(record["effective"])
    else:
        saved = "fp16" if training_state.get("scaler") else "fp32"
    if saved != precision.effective:
        raise ValueError(
            f"resume precision mismatch: the checkpoint trained in {saved}, this run resolves {precision.effective}; "
            "resume with the same precision, or start from its weights with resume_mode='weights_only' to switch"
        )
    if (training_state.get("scaler") is None) != (scaler is None):
        raise ValueError(
            "resume GradScaler presence mismatch: checkpoint and configuration must both use AMP or neither"
        )


def _precision_key(params: Any, device: Any) -> tuple[Any, bool, str, Optional[int]]:
    """What a cached resolution depends on: the whole policy, the legacy
    flag and the device (type and index: bf16 support is per CUDA device)."""
    torch_device = torch.device(device)
    return (
        getattr(params, "precision", None),
        bool(getattr(params, "mixed_precision", False)),
        torch_device.type,
        torch_device.index,
    )


def _inference_precision(model: Any) -> ResolvedPrecision:
    """The precision evaluation and prediction run in (FEAT-028): an
    explicit policy's; full precision for the legacy ``mixed_precision``
    flag (a training-only setting) and for stand-ins borrowing these
    methods."""
    resolved = model.resolved_precision if isinstance(model, NNModel) else None
    if resolved is not None and _PRECISION_EVALUATE in resolved.applies_to:
        return resolved
    return _FULL_PRECISION  # needs no record, so TF32 is not read on this hot path


def _check_finite_gradients(module: torch.nn.Module) -> None:
    """Raise before a scaler-free reduced-precision update applies a
    non-finite gradient (FEAT-028) — one host sync for the whole model."""
    if gradients_finite(module.parameters()):
        return
    name = next((n for n, p in module.named_parameters() if p.grad is not None and not gradients_finite([p])), "?")
    raise FloatingPointError(
        f"non-finite gradient for {name!r} in a reduced-precision update; nothing was stepped. Check the learning "
        "rate and loss scale, or train in fp32"
    )


def _scale_gradients(module: torch.nn.Module, factor: float) -> None:
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


@dataclass(frozen=True, slots=True)
class _StepLossTerms:
    """One batch's forward outputs and loss terms for `default_train_step`."""

    output: torch.Tensor
    target: torch.Tensor
    prediction: Optional[torch.Tensor]
    valid: Optional[torch.Tensor]
    train_loss: torch.Tensor
    backward_loss: torch.Tensor
    normalization_weight: Optional[float]


def _step_loss_terms(
    model: NNModel,
    batch: Any,
    accumulation_state: Optional[GradientAccumulationState],
    accumulate_grad_batches: int,
    precision: Optional[ResolvedPrecision] = None,
) -> _StepLossTerms:
    """Forward one batch and compute its loss terms.

    Legacy models decode by loss (`_fwd_pass`). A model with a task
    (FEAT-002) validates the batch through its adapter first — before any
    backward pass or optimizer update — and scores only the valid targets.
    Under a reduced ``precision`` (FEAT-028) the forward's output is taken
    in full precision before the task, the loss and the records see it
    (bf16 has no NumPy dtype; a task casts its targets to the output's).
    """
    adapter = getattr(model, "task_adapter", None)
    full = precision.output if precision is not None else _unchanged
    if adapter is None:
        _, Y, Y_hat_logits, Y_hat = model._fwd_pass(batch)
        output, target, prediction, valid = full(Y_hat_logits), Y, Y_hat, None
        if accumulation_state is None:
            train_loss = model.loss_fn(_loss_input(model.loss_fn, output), target)
            return _StepLossTerms(
                output, target, prediction, valid, train_loss, train_loss / accumulate_grad_batches, None
            )
        train_loss, backward_loss, weight = _loss_terms(model.loss_fn, output, target)
        return _StepLossTerms(output, target, prediction, valid, train_loss, backward_loss, weight)

    _, Y, logits = model._fwd_outputs(batch)
    output, target, valid = adapter.prepare(full(logits), Y)
    train_loss, backward_loss, weight = adapter.loss_terms(model.loss_fn, output, target, valid)
    if accumulation_state is None and weight != 0:
        backward_loss = train_loss / accumulate_grad_batches
    return _StepLossTerms(output, target, None, valid, train_loss, backward_loss, weight)


# The legacy rule for a step context without a precision (FEAT-028): FP16
# for a scaler on CUDA, full precision otherwise. TF32 is never read here.
_LEGACY_FP16 = ResolvedPrecision(requested="fp16", effective="fp16", device_type="cuda", source="legacy", tf32={})
_FULL_PRECISION = ResolvedPrecision(requested="fp32", effective="fp32", device_type="cpu", tf32={})


def _check_scaler_hook(precision: ResolvedPrecision, scaler: Any, device_type: str) -> None:
    """The policy, not ``_build_grad_scaler``, decides AMP (FEAT-028): an fp16
    run needs the hook's scaler, and a CUDA run that is not fp16 refuses one
    (a scaler there used to switch AMP on; it would now be ignored). Off CUDA
    a scaler never switched anything on, so it is left to the steps (the
    paradigm steps refuse it)."""
    if precision.uses_scaler and scaler is None:
        raise ValueError("fp16 trains through a GradScaler, and _build_grad_scaler returned none")
    if precision.uses_scaler and not getattr(scaler, "is_enabled", lambda: True)():
        # A disabled scaler scales nothing: the float16 backward would underflow.
        raise ValueError("fp16 trains through an enabled GradScaler, and _build_grad_scaler returned a disabled one")
    if scaler is not None and not precision.uses_scaler and device_type == "cuda":
        raise ValueError(
            f"_build_grad_scaler returned a GradScaler, but this run resolves to {precision.effective}: AMP is "
            "decided by NNModelParams.precision (PrecisionPolicy('fp16')), which builds the scaler itself"
        )


def _check_step_precision(train_step_fn: Optional[Callable[..., Any]], precision: ResolvedPrecision) -> None:
    """Refuse a step function marked full-precision-only (the built-in
    imperative paradigm steps) under a reduced precision (FEAT-028) —
    ``NNModel.train``'s rule, shared with plan validation."""
    if precision.reduced and getattr(train_step_fn, _FULL_PRECISION_ONLY, False):
        name = getattr(train_step_fn, "__qualname__", type(train_step_fn).__name__)
        raise PrecisionUnsupportedError(
            f"{name} runs in full precision only (an imperative paradigm step built on finalize_step does not "
            f"apply the {precision.effective} policy); train it in fp32, or express the loss as an objective"
        )


def _unchanged(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _record_step_loss(
    terms: _StepLossTerms,
    accumulation_state: Optional[GradientAccumulationState],
    *,
    should_step: bool,
) -> Optional[float]:
    """The batch's display loss, with the finite-loss guard. An all-masked
    task batch (weight 0) has no loss of its own."""
    if terms.valid is not None and terms.normalization_weight == 0:
        return None
    return _record_accumulation_loss(
        terms.train_loss,
        terms.normalization_weight,
        accumulation_state,
        should_step=should_step,
    )


def _agreed_step_loss(
    ddp: Any,
    terms: _StepLossTerms,
    accumulation_state: Optional[GradientAccumulationState],
    *,
    should_step: bool,
) -> Optional[float]:
    """``_record_step_loss`` decided on every rank together (FEAT-030): a
    non-finite loss on any rank raises ``FloatingPointError`` on all of
    them, before any gradient collective could leave the others waiting."""
    error: Optional[FloatingPointError] = None
    value: Optional[float] = None
    try:
        if terms.valid is None and terms.normalization_weight == 0:
            # This rank holds no valid target in the batch (all ignored): its
            # NaN display loss is not a divergence — whether the *global*
            # window has any is decided at the update (_global_window_check).
            value = None
        else:
            value = _record_step_loss(terms, accumulation_state, should_step=should_step)
    except FloatingPointError as caught:
        error = caught
    failed = [rank for rank, bad in enumerate(ddp.gather(error is not None)) if bad]
    if error is not None:
        raise error
    if failed:
        raise FloatingPointError(f"non-finite training loss on rank(s) {failed}: every rank stops before the update")
    return value


def _global_window_check(
    terms: _StepLossTerms, accumulation_state: Optional[GradientAccumulationState], global_weight: Optional[float]
) -> None:
    """A legacy (task-free) model whose whole *global* window holds no valid
    target has a NaN loss, which one process training on the union batch
    refuses: refuse it on every rank alike (FEAT-030)."""
    if (
        terms.valid is None
        and accumulation_state is not None
        and accumulation_state.normalization_required
        and global_weight == 0
    ):
        raise FloatingPointError(
            "non-finite training loss (nan): no rank's batch in this optimizer window has a valid target"
        )


def _cpu(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if value is None else value.detach().cpu()


def _global_records(
    ddp: Any, terms: _StepLossTerms, accumulation_state: Optional[GradientAccumulationState]
) -> tuple[_StepLossTerms, Optional[float]]:
    """The global batch's record terms and display loss (FEAT-030): every
    rank's outputs, targets, predictions and validity concatenated in rank
    order, with the loss recomputed from every rank's numerator and
    denominator — identical on every rank, so the records, summaries and
    monitors agree. A global batch without a valid target shows what one
    process would: the window's running mean so far (or 0.0), or no loss
    for a task model."""
    loss = float(terms.train_loss.detach())
    window = (
        (accumulation_state.loss_numerator, accumulation_state.normalization_weight)
        if accumulation_state is not None
        else (0.0, 0.0)
    )
    parts = ddp.gather(
        (
            _cpu(terms.output),
            _cpu(terms.target),
            _cpu(terms.prediction),
            _cpu(terms.valid),
            loss,
            terms.normalization_weight,
            window,
        )
    )

    def joined(index: int) -> Optional[torch.Tensor]:
        tensors = [part[index] for part in parts]
        return None if tensors[0] is None else torch.cat(tensors)

    weights = [part[5] for part in parts]
    display: Optional[float]
    if all(weight is None for weight in weights):  # an additive (sum) loss
        total: Optional[float] = None
        display = float(sum(part[4] for part in parts))
    else:
        total = float(sum(weight or 0.0 for weight in weights))
        if total:
            display = sum(part[4] * part[5] for part in parts if part[5]) / total
        elif terms.valid is not None:
            display = None  # an all-masked task batch has no loss of its own
        else:
            window_weight = sum(part[6][1] for part in parts)
            display = sum(part[6][0] for part in parts) / window_weight if window_weight else 0.0
    output, target = joined(0), joined(1)
    assert output is not None and target is not None
    loss_tensor = torch.tensor(float("nan") if display is None else display, dtype=torch.float64)
    return _StepLossTerms(output, target, joined(2), joined(3), loss_tensor, loss_tensor, total), display


def _window_is_masked(
    terms: _StepLossTerms,
    accumulation_state: Optional[GradientAccumulationState],
    *,
    window_is_this_batch: bool,
) -> bool:
    """True when a task model's whole optimizer window had no valid target.

    Without accumulation state (direct legacy callers) only a one-batch
    window can be judged; a longer window may hold valid gradients from
    earlier batches, so it is never discarded."""
    if terms.valid is None:
        return False
    if accumulation_state is None:
        return window_is_this_batch and terms.normalization_weight == 0
    return accumulation_state.normalization_required and accumulation_state.normalization_weight == 0


def _reset_accumulation(accumulation_state: Optional[GradientAccumulationState]) -> None:
    if accumulation_state is not None:
        accumulation_state.normalization_weight = 0.0
        accumulation_state.loss_numerator = 0.0
        accumulation_state.normalization_required = True


def _single_input_batch(model: Any, batch: Any, *, who: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``(X, Y)`` of a supervised batch with exactly one positional input,
    split by the model's batch adapter (a built-in net's own
    ``unpack_batch``, FEAT-006) and moved to the model's device. Shared by
    the paradigm steps that transform ``X`` (Mixup / CutMix, KD, MoE)."""
    args, kwargs, target = model._split_batch(batch)
    if kwargs or len(args) != 1 or target is None:
        raise ValueError(
            f"{who} needs batches of one positional input and a target; the model's batch adapter gave "
            f"{len(args)} positional and {len(kwargs)} keyword input(s)"
            f"{' and no target' if target is None else ''} — use a custom train_step_fn for other layouts"
        )
    return args[0].to(model.device), target.to(model.device)


_GRAPH_NETS = frozenset({Nets.GRAPH_ATT, Nets.GRAPH_CONV, Nets.GRAPH_SAGE})


def _compile_session(
    model: Any,
    spec: Optional[CompileSpec],
    train_step_fn: Any,
    objective: Any,
    callbacks: Optional[list[Any]],
    precision: ResolvedPrecision,
) -> Optional[_CompileSession]:
    """A fit's compile session (FEAT-029), or ``None`` — the scope is
    checked here, before any run is reserved: built-in non-graph nets, the
    default step, full precision and no topology-changing callback."""
    if spec is None:
        return None
    if not isinstance(spec, CompileSpec):
        raise TypeError(f"compile must be an nnx.compilation.CompileSpec or None, got {type(spec).__name__}")
    if objective is not None or train_step_fn not in (None, default_train_step):
        raise ValueError(
            "compile= covers the built-in train step only; drop train_step_fn / objective, or compile inside your own step"
        )
    net = model.params.net
    if not isinstance(net, Nets) or net in _GRAPH_NETS:
        raise ValueError(f"compile= covers the built-in non-graph nets only, got {net!r}")
    if precision.effective != "fp32":
        raise ValueError(
            f"compile= covers full-precision (fp32) forwards only; this run trains in {precision.effective}"
        )
    from .callbacks import Callback

    changing = [
        type(cb).__name__
        for cb in callbacks or ()
        if isinstance(cb, Callback) and type(cb).checkpoint_transforms is not Callback.checkpoint_transforms
    ]
    if changing:
        raise ValueError(
            f"callbacks {changing} change model.net's topology mid-run (e.g. QAT), which a compiled forward would "
            "not see; train without compile=, or without them"
        )
    return _CompileSession(model.net, spec)


def _record_of(session: Optional[_CompileSession]) -> Any:
    return None if session is None else session.record


def _state_of(session: Optional[_CompileSession]) -> Optional[dict[str, Any]]:
    return None if session is None else session.record.record()


def _collectively(what: str) -> Any:
    """``nnx.distributed.collectively`` (FEAT-030). Rule: never nest one in
    another, and keep any collective inside such a block reached by every
    rank alike — a rank failing before it would skip that gather and leave
    the ranks one collective apart."""
    from ..distributed import collectively

    return collectively(what)


def _writer_owned_callbacks(callbacks: Optional[list[Any]]) -> Optional[list[Any]]:
    """Without DDP, ``nnx.distributed.writer_only`` callbacks are simply built."""
    if not callbacks:
        return callbacks
    from .callbacks import _WriterOwned

    return [callback.build() if isinstance(callback, _WriterOwned) else callback for callback in callbacks]


def _distributed_scope(
    model: Any,
    params: Any,
    callbacks: Optional[list[Any]],
    *,
    train_step_fn: Any,
    eval_step_fn: Any,
    objective: Any,
    precision: ResolvedPrecision,
    history: Any,
    compile: Any,
    writer: bool,
) -> list[Any]:
    """What DDP covers (FEAT-030), checked on each rank: the callbacks this
    rank runs, or a ValueError."""
    from ..distributed import ShardedLoader, classify_callbacks
    from .callbacks import Callback

    if objective is not None or train_step_fn not in (None, default_train_step) or eval_step_fn is not None:
        raise ValueError(
            "distributed= covers the default train and validation steps only (no custom steps or objective)"
        )
    if compile is not None:
        raise ValueError("distributed= and compile= cannot be combined")
    if history is not None:
        raise ValueError("distributed= keeps the eager history; a HistoryJournal is not supported")
    if precision.effective != "fp32":
        raise ValueError(
            f"distributed= trains in full precision (fp32) only; this run resolves to {precision.effective}"
        )
    batch_norms = [
        type(m).__name__ for m in model.net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if batch_norms:
        raise ValueError(
            f"distributed= does not cover batch normalization ({sorted(set(batch_norms))}): its running statistics "
            "would differ per rank, so no update would equal the union batch's"
        )
    if model.device.type == "cuda":
        local = int(os.environ.get("LOCAL_RANK", "0"))
        index = model.device.index if model.device.index is not None else torch.cuda.current_device()
        if index != local:
            raise ValueError(
                f"this rank (LOCAL_RANK={local}) holds its model on cuda:{index}: build the model after "
                "nnx.distributed.init_process_group(), which selects cuda:LOCAL_RANK"
            )
    if isinstance(model.params.net, Nets) and model.params.net in _GRAPH_NETS:
        raise ValueError("distributed= does not cover graph nets (graph neighbour sampling is out of scope)")
    train_loader = params.train_loader
    if not isinstance(train_loader, ShardedLoader) or train_loader.partition.kind != "train":
        raise ValueError(
            "distributed= needs params.train_loader = nnx.distributed.train_loader(dataset, batch_size, ...) — "
            "a partition that declares its padding/drop policy and epoch seeding"
        )
    if params.val_loader is not None and (
        not isinstance(params.val_loader, ShardedLoader) or params.val_loader.partition.kind != "validation"
    ):
        raise ValueError(
            "distributed= needs params.val_loader = nnx.distributed.validation_loader(dataset, batch_size) — an "
            "unpadded shard that scores every row exactly once"
        )
    changing = [
        type(cb).__name__
        for cb in callbacks or ()
        if isinstance(cb, Callback) and type(cb).checkpoint_transforms is not Callback.checkpoint_transforms
    ]
    if changing:
        raise ValueError(f"callbacks {changing} change model.net's topology, which distributed= does not cover")
    return classify_callbacks(model._normalize_callbacks(callbacks), writer=writer)


def _distributed_session(
    model: Any,
    spec: Any,
    *,
    params: Any,
    callbacks: Optional[list[Any]],
    train_step_fn: Any,
    eval_step_fn: Any,
    objective: Any,
    precision: ResolvedPrecision,
    history: Any,
    compile: Any,
) -> tuple[Optional[Any], Optional[list[Any]]]:
    """``(session, this rank's callbacks)`` for ``distributed=`` (FEAT-030),
    decided collectively: every rank checks the scope and the partitions
    agree, then every rank builds the DDP wrapper (itself a collective)."""
    if spec is None:
        return None, callbacks  # writer_only(...) wrappers were built at the top of train()
    from ..distributed import DDP, _DDPSession, _world, agree

    if not isinstance(spec, DDP):
        raise TypeError(f"distributed must be an nnx.distributed.DDP or None, got {type(spec).__name__}")
    rank, world_size = _world()
    error: Optional[BaseException] = None
    kept: list[Any] = []
    descriptors: Any = None
    try:
        if spec.writer_rank >= world_size:
            raise ValueError(f"writer_rank {spec.writer_rank} is outside a world of {world_size}")
        kept = _distributed_scope(
            model,
            params,
            callbacks,
            train_step_fn=train_step_fn,
            eval_step_fn=eval_step_fn,
            objective=objective,
            precision=precision,
            history=history,
            compile=compile,
            writer=rank == spec.writer_rank,
        )
        partition = params.train_loader.partition
        if (partition.rank, partition.world_size) != (rank, world_size):
            raise ValueError(
                f"the train partition was built for rank {partition.rank} of {partition.world_size}, "
                f"this process is rank {rank} of {world_size}"
            )
        descriptors = (
            params.train_loader.descriptor(),
            None if params.val_loader is None else params.val_loader.descriptor(),
        )
    except Exception as caught:
        error = caught
    agree(error, "the distributed preflight")
    from ..distributed import _send

    if len({repr(item) for item in _send(descriptors)}) != 1:
        raise ValueError("the ranks' train / validation partitions differ (dataset size, policy, seed or batch size)")
    error = None
    session = None
    try:
        session = _DDPSession(model.net, spec, model.device)
    except Exception as caught:
        error = caught
    agree(error, "building the DDP wrapper")
    return session, kept


@contextlib.contextmanager
def _distributing(model: Any, session: Optional[Any]) -> Iterator[None]:
    """Route the train step through ``session``'s DDP wrapper for one fit."""
    previous = model._ddp_session
    model._ddp_session = session
    try:
        yield
    finally:
        model._ddp_session = previous


class _Replay:
    """A stand-in model for evaluating gathered outputs (FEAT-030): its
    forward returns the precomputed ``(X, Y, logits)`` a rank produced;
    everything else is the real model's."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.net = torch.nn.Module()  # no parameters, no modes to restore
        # A CPU copy: _evaluate moves the loss to the stand-in's device in
        # place, which must never move the real model's loss (buffers included).
        self.loss_fn = copy.deepcopy(model.loss_fn).to("cpu")
        self.device = torch.device("cpu")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _fwd_outputs(self, batch: Any) -> Any:
        return batch

    def _fwd_pass(self, batch: Any) -> Any:
        X, Y, logits = batch
        class_axis = -1 if self._model.params.net is Nets.TRANSFORMER else 1
        Y_hat = (
            (logits >= 0).to(dtype=torch.long)
            if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss)
            else logits.argmax(dim=class_axis)
        )
        return X, Y, logits, Y_hat


def _distributed_validation(model: Any, ddp: Any, params: Any) -> NNEvaluationDataPoint:
    """Validation under DDP (FEAT-030): each rank scores its unpadded shard
    with the canonical module — no collective inside the loop, so a rank
    with no rows simply has nothing to score — then one gather gives every
    rank the same global record. Every row id is scored exactly once."""
    loader = params.val_loader
    precision = _inference_precision(model)
    training_modes = _capture_training_modes(model.net)
    model.net.eval()
    produced: list[tuple[Any, Any, Any]] = []
    try:
        with torch.no_grad():
            for batch in loader:
                with precision.autocast():
                    X, Y, logits = model._fwd_outputs(batch)
                produced.append(((), Y.detach().cpu(), precision.output(logits).detach().cpu()))
    finally:
        _restore_training_modes(training_modes)
    gathered = ddp.gather((produced, loader.ids()))
    ids = [row for _, rows in gathered for row in rows]
    if len(ids) != len(set(ids)) or sorted(ids) != list(range(loader.partition.n)):
        raise RuntimeError("distributed validation did not score every row exactly once")  # pragma: no cover
    ddp.validation_ids = ids
    batches = [batch for part, _ in gathered for batch in part]
    replay = _Replay(model)
    if params.metrics:
        return _evaluate(replay, batches, params.extra_metrics, tuple(params.metrics), bounded=False, who="validation")
    return _evaluate(replay, batches, params.extra_metrics, (), bounded=False, who="validation")


def _distributed_state(ddp: Any, train_loader: Any, val_loader: Any, rank_rng: list[Any]) -> dict[str, Any]:
    return {
        "world_size": ddp.world_size,
        "writer_rank": ddp.spec.writer_rank,
        "train": train_loader.descriptor(),
        "validation": None if val_loader is None else val_loader.descriptor(),
        "rng_by_rank": rank_rng,
    }


def _check_distributed_resume(training_state: Mapping[str, Any], ddp: Any, train_loader: Any) -> Mapping[str, Any]:
    """A DDP resume needs the same world size and train partition (FEAT-030),
    checked before anything is restored; returns this rank's RNG state."""
    from ..distributed import rng_restore

    saved = training_state.get("distributed")
    if saved is None:
        raise ValueError(
            "this checkpoint was not written by a distributed run: a distributed resume needs the world size and "
            "partition it was trained with (resume_mode='weights_only' starts fresh from its weights)"
        )
    if saved["world_size"] != ddp.world_size:
        raise ValueError(
            f"the checkpoint was trained on {saved['world_size']} ranks, this world has {ddp.world_size}: "
            "resume with the same world size (or resume_mode='weights_only')"
        )
    if saved["train"] != train_loader.descriptor():
        raise ValueError(
            f"the train partition changed since the checkpoint: {saved['train']} vs {train_loader.descriptor()}"
        )
    state = rng_restore(saved.get("rng_by_rank"), ddp.rank, ddp.world_size)
    assert state is not None
    return state


@contextlib.contextmanager
def _compiling(model: Any, session: Optional[_CompileSession]) -> Iterator[None]:
    """Route the model's forward through ``session`` for one fit; the
    wrapper is dropped whatever the outcome (later calls are eager)."""
    previous = model._compile_session
    model._compile_session = session
    try:
        yield
    finally:
        # A nested train() (say, from a callback) restores the outer fit's session.
        model._compile_session = previous


def _check_provenance(provenance: Any) -> None:
    """Reject anything but an ExperimentManifest before a run is reserved."""
    if provenance is not None and not isinstance(provenance, ExperimentManifest):
        raise TypeError(f"provenance must be an nnx.provenance.ExperimentManifest, got {type(provenance).__name__}")


def _with_attempt(
    run: Any,
    provenance: Optional[ExperimentManifest],
    params: Any,
    fit: Callable[[], Any],
    execution: Optional[Callable[[], Optional[dict[str, Any]]]] = None,
) -> Any:
    """Run ``fit`` as one recorded attempt (FEAT-019) when a manifest is
    given — shared by ``NNModel.train`` and ``Trainer.train``. The attempt's
    final status (``completed``, ``failed``, or ``cancelled`` on
    ``KeyboardInterrupt``) and last committed checkpoint are recorded; a
    failure to record a failed attempt never masks the training error.
    ``execution`` returns the fit's compile record (FEAT-029) when it ends,
    whatever its outcome."""
    if provenance is None:
        return fit()
    from ..provenance import _AttemptRecorder

    recorder = _AttemptRecorder(
        run,
        provenance,
        parent_run_id=getattr(params, "resume_from_run_id", None),
        parent_checkpoint=getattr(params, "resume_from_checkpoint", None),
        execution=execution,
    )
    recorder.start()
    try:
        result = fit()
    except BaseException as error:
        recorder.fail(error)
        raise
    try:
        recorder.complete()
    except Exception as record_error:  # the fit succeeded and is saved; do not lose it
        warnings.warn(
            f"run {run.id} trained and saved, but its attempt could not be recorded as completed "
            f"({record_error}); attempt.json still reads 'running'",
            RuntimeWarning,
            stacklevel=3,
        )
    return result.with_provenance(recorder.record)


def _tensor_keys(state: Mapping[str, Any]) -> set[str]:
    return {key for key, value in state.items() if isinstance(value, torch.Tensor)}


def _to_device(value: Any, device: torch.device) -> Any:
    """Move a tensor (or anything with ``.to``, e.g. a graph batch) to the
    device; other values pass through."""
    to = getattr(value, "to", None)
    return to(device) if callable(to) else value


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    """Tell a training loader which epoch it is about to serve, when it
    defines ``set_epoch(epoch)`` (PyTorch's ``DistributedSampler``
    convention): a loader that draws per-epoch randomness from the epoch
    index — ``nnx.link_tasks`` training negatives — then draws the same
    batches in an uninterrupted run and in one resumed at that epoch."""
    set_epoch = getattr(loader, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)


def _enumerate_with_last(iterable: Iterable[Any]) -> Iterator[tuple[int, Any, bool]]:
    iterator = iter(iterable)
    try:
        current = next(iterator)
    except StopIteration:
        return
    index = 0
    while True:
        try:
            following = next(iterator)
        except StopIteration:
            yield index, current, True
            return
        yield index, current, False
        current = following
        index += 1


def _observe_epoch_summary(
    summary: _TrainEpochSummary, model: NNModel, terms: _StepLossTerms, adapter: Optional[TaskAdapter]
) -> None:
    """Report one default-step batch to the whole-epoch summary (FEAT-003):
    its outputs for the declared metrics, its loss denominator and the
    number of targets its error scores."""
    if adapter is not None:
        summary.observe(terms.target, terms.output, terms.valid, terms.normalization_weight)
        return
    assert terms.prediction is not None
    scored, _ = _classification_metric_tensors(model.loss_fn, terms.target, terms.prediction)
    summary.observe(terms.target, terms.output, None, terms.normalization_weight, float(scored.numel()))


def _batch_sample_count(net: Any, batch: Any) -> int:
    """Samples in a batch, for weighting custom-step records in the epoch
    summary: graph seed rows, else the leading size of the first tensor."""
    seed_count = getattr(net, "seed_count", None)
    if callable(seed_count):
        n_seed = seed_count(batch)
        if n_seed is not None:
            return int(cast(int, n_seed))
    sample_ids = getattr(net, "sample_ids", None)
    if callable(sample_ids):  # rows with their own identity (graph ids, FEAT-026): one sample per row
        return int(torch.as_tensor(sample_ids(batch)).numel())
    first = batch
    while (isinstance(first, (tuple, list)) and first) or (isinstance(first, Mapping) and first):
        # Mapping batches (keyword-input modules, FEAT-006): the first value.
        first = first[0] if isinstance(first, (tuple, list)) else next(iter(first.values()))
    if isinstance(first, torch.Tensor) and first.ndim:
        return int(first.shape[0])
    return 1


def _metric_context(model: Any) -> tuple[Optional[str], Optional[int], float, Optional[int]]:
    """How this model's outputs become metric inputs (FEAT-003): the output
    domain, the ignored class index, the multilabel decision threshold (in
    logit space, as the task decodes) and the class count."""
    return _metric_context_of(model.loss_fn, getattr(model, "task_adapter", None), getattr(model, "net_params", None))


def _metric_context_of(
    loss_fn: Any, adapter: Optional[TaskAdapter], net_params: Any
) -> tuple[Optional[str], Optional[int], float, Optional[int]]:
    """:func:`_metric_context` from the model's parts — the loss, the task
    adapter and the network parameters — so a plan can check metric inputs
    with no model built."""
    spec = adapter.spec if adapter is not None else None
    domain = _metric_domain(loss_fn, spec)
    threshold = float(getattr(adapter, "_logit_threshold", 0.0))
    n_classes = getattr(spec, "num_outputs", None) if spec is not None else None
    if n_classes is None:
        n_classes = getattr(net_params, "output_dim", None)
    return domain, _ignore_index(loss_fn), threshold, n_classes


def _named_metric_set(
    model: Any, metrics: tuple[MetricSpec, ...], *, where: str, bounded: bool = False
) -> Optional[_MetricSet]:
    """Accumulators for declared metrics (``None`` when there are none),
    after checking that the model can provide every metric's input.
    ``bounded=True`` builds mergeable, bounded accumulators (FEAT-020) and
    rejects a metric that needs every stored score."""
    if not metrics:
        return None
    domain, ignore_index, threshold, n_classes = _metric_context(model)
    _check_metric_inputs(metrics, domain, where=where, n_classes=n_classes)
    if not bounded:
        return _MetricSet(metrics, domain, ignore_index, threshold)
    from ..streaming import _bounded_accumulator

    return _MetricSet(metrics, domain, ignore_index, threshold, accumulator=_bounded_accumulator)


def _check_plateau_resume(saved: Optional[Mapping[str, Any]], scheduler: Any, monitor: Optional[MonitorSpec]) -> None:
    """A plateau scheduler resumes only from a plateau state saved under the
    improvement rule this run builds — direction, threshold mode and
    threshold — whether or not the run declares a monitor (FEAT-003).
    Loading the state replaces the built rule with the saved one, so a
    state saved while monitoring accuracy (``mode='max'``) would cut the LR
    of a run stepping on validation loss exactly when it improves. Checked
    before anything is restored."""
    if saved is None or not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        return
    fields = ("mode", "threshold_mode", "threshold")
    saved_rule = {name: saved.get(name) for name in fields}
    built_rule = {name: getattr(scheduler, name) for name in fields}
    differing = [name for name in fields if saved_rule[name] != built_rule[name]]
    if not differing:
        return

    def rule(values: Mapping[str, Any]) -> str:
        return ", ".join(f"{name}={values[name]!r}" for name in fields)

    decider = (
        f"this run's monitor {monitor.key!r} decides"
        if monitor is not None
        else "this run declares no monitor, so its scheduler decides"
    )
    raise ValueError(
        f"resume plateau scheduler was saved with {rule(saved_rule)}, but {decider} with {rule(built_rule)} "
        f"(differing: {', '.join(differing)}); resume with the monitor and scheduler threshold the checkpoint "
        "was trained with, or pass resume_mode='weights_only'"
    )


def _monitoring_preflight(
    model: Any,
    *,
    metrics: tuple[MetricSpec, ...],
    monitor: Optional[MonitorSpec],
    callbacks: Optional[list[Any]],
    default_train_step: bool,
    default_eval_step: bool,
    has_val_loader: bool,
    owner: str,
) -> Optional[MonitorSpec]:
    """FEAT-003: resolve declared metrics and monitors before any run is
    reserved or loader read. Unknown registrations fail here without any
    metric code running; so does a metric input the model cannot provide on
    a path NNx computes, and a monitor that can never have a value."""
    for spec in metrics:
        spec.check()
    if metrics and (default_train_step or default_eval_step):
        domain, _, _, n_classes = _metric_context(model)
        where = "the default training step" if default_train_step else "evaluate()"
        _check_metric_inputs(metrics, domain, where=where, n_classes=n_classes)
    resolved = monitor.resolve(metrics, owner=f"{owner}.monitor") if monitor is not None else None
    monitors = [resolved] if resolved is not None else []
    for callback in callbacks or ():
        bind = getattr(callback, "_bind_metrics", None)
        if callable(bind):
            bound = bind(metrics)
            if isinstance(bound, MonitorSpec):
                monitors.append(bound)
    for spec in monitors:
        problem = _monitor_problem(spec, has_val_loader=has_val_loader, default_train_step=default_train_step)
        if problem is not None:
            raise ValueError(problem)
    return resolved


def _monitor_problem(spec: MonitorSpec, *, has_val_loader: bool, default_train_step: bool) -> Optional[str]:
    """Why a resolved monitor can never have a value in this run, or
    ``None`` — shared by ``train()`` and ``ExperimentPlan.validate()``."""
    if spec.split == "val" and not has_val_loader:
        return f"monitor {spec.key!r} tracks the validation split, but no val_loader is configured"
    if spec.split == "train" and spec.metric not in ("loss", "error") and not default_train_step:
        return (
            f"monitor {spec.key!r} needs {spec.metric!r} over the full training epoch, which only the default "
            f"training step records; monitor 'val.{spec.metric}' or 'train.loss' instead"
        )
    return None


def _monitored_plateau(scheduler: Any, optimizer: torch.optim.Optimizer, monitor: Optional[MonitorSpec]) -> Any:
    """With a monitor, a ReduceLROnPlateau decides improvement exactly as the
    monitor does: its direction, and an absolute ``min_delta`` threshold
    (``a < best - min_delta`` / ``a > best + min_delta``, first finite value
    improves, ties never do). Other settings are kept."""
    if monitor is None or not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        return scheduler
    return lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cast(Any, monitor.mode),
        factor=scheduler.factor,
        patience=scheduler.patience,
        threshold=monitor.min_delta,
        threshold_mode="abs",
        cooldown=scheduler.cooldown,
        min_lr=list(scheduler.min_lrs),
        eps=scheduler.eps,
    )


def _step_monitored_plateau(scheduler: Any, record: MonitorRecord) -> None:
    """Step a monitor-aligned plateau scheduler on the epoch's decision: a
    missing value makes no decision; a non-finite one is an epoch without
    improvement."""
    if record.status == "missing":
        return
    if record.status == "nonfinite":
        scheduler.step(math.inf if record.monitor.mode == "min" else -math.inf)
        return
    scheduler.step(record.value)


def _objective_microbatch(
    engine: Any,
    objective: Callable[[Any], Any],
    *,
    model: Any,
    batch: Any,
    epoch_idx: int,
    batch_idx: int,
    extra_metrics: Optional[Mapping[str, Callable]],
    close_window: bool,
    epoch_summary: Optional[_TrainEpochSummary] = None,
) -> NNEvaluationDataPoint:
    """The shared objective primitive (FEAT-004), used by ``NNModel.train``
    and ``Trainer.train`` alike: run the objective under the engine's
    autocast, accumulate its loss terms and — at the window's end — commit
    one update. The terms also feed the whole-epoch summary (FEAT-003), so
    the epoch's training loss uses the objective's own denominators.
    Returns the microbatch's record."""
    from ..objectives import ObjectiveContext, ObjectiveResult

    with engine.autocast():
        result = objective(
            ObjectiveContext(
                model=model,
                batch=batch,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                extra_metrics=extra_metrics,
                precision=engine.precision,
            )
        )
    if not isinstance(result, ObjectiveResult):
        raise TypeError(f"an objective must return an ObjectiveResult, got {type(result).__name__}")
    engine.accumulate(result.terms, closes_window=close_window)
    if epoch_summary is not None:
        epoch_summary.observe_terms(result.terms)
    if close_window:
        engine.commit(epoch_idx=epoch_idx, batch_idx=batch_idx)
    record = result.record
    loss = result.loss()
    if record is None:
        return NNEvaluationDataPoint(loss=loss)
    return record if record.loss is not None or loss is None else record.with_loss(loss)


class _ObjectiveStep:
    """``NNModel.train``'s step for an objective: windows follow
    ``accumulate_grad_batches`` (cut short at the epoch's end), and the
    shared engine owns every update."""

    def __init__(self, objective: Callable[[Any], Any], engine: Any) -> None:
        self.objective = objective
        self.engine = engine

    def __call__(self, ctx: TrainStepContext) -> NNEvaluationDataPoint:
        window = ctx.accumulate_grad_batches
        return _objective_microbatch(
            self.engine,
            self.objective,
            model=ctx.model,
            batch=ctx.batch,
            epoch_idx=ctx.epoch_idx,
            batch_idx=ctx.batch_idx,
            extra_metrics=ctx.extra_metrics,
            close_window=ctx.batch_idx % window == window - 1 or ctx.is_last_batch,
            epoch_summary=ctx.epoch_summary,
        )


def _objective_engine(
    objective: Callable[[Any], Any],
    *,
    optimizers: Mapping[str, torch.optim.Optimizer],
    clip_norms: Mapping[str, Optional[float]],
    scaler: Optional[torch.amp.GradScaler],
    precision: ResolvedPrecision,
) -> Any:
    """The update engine for an objective run, in the run's precision
    (FEAT-028): autocast around the objective, the scaler for an fp16
    update."""
    from .._update_engine import UpdateEngine
    from ..objectives import Objective

    if precision.uses_scaler and scaler is None:
        # As in default_train_step: an unscaled float16 backward underflows.
        raise ValueError("fp16 trains through a GradScaler, and this objective run has none")
    after_update = getattr(objective, "after_update", None)
    if isinstance(objective, Objective) and type(objective).after_update is Objective.after_update:
        after_update = None  # the base class's no-op: no per-commit work to schedule

    return UpdateEngine(
        optimizers=optimizers,
        scaler=scaler if precision.uses_scaler else None,
        clip_norms=clip_norms,
        nonfinite=getattr(objective, "nonfinite", "fail"),
        autocast=precision.autocast if precision.reduced else None,
        precision=precision,
        # FEAT-040: the objective's own once-per-commit work (a JEPA EMA).
        commit_hooks=(cast(Callable[[tuple[Any, ...]], None], after_update),) if callable(after_update) else (),
    )


def default_train_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
    """Standard supervised training step: forward → loss → backward → step.

    This is the body that `NNModel.train()` runs when no custom
    `train_step_fn` is supplied. It honors:
      - gradient accumulation (zero_grad at cycle start, step at cycle
        end). A trailing partial cycle is stepped at the epoch boundary;
        gradients use each loss's effective normalization weight.
      - the run's precision (FEAT-028): autocast around the forward only,
        the backward outside it; fp16 unscales before grad clip and steps
        through the scaler, scaler-free bf16 checks its gradients are
        finite before clipping
      - grad clipping by L2 norm
      - the NaN/Inf guard (raises FloatingPointError on divergent loss)
      - extra_metrics injection on the returned NNEvaluationDataPoint

    Custom training-step functions can call this directly to layer on
    behavior (e.g., extra logging) without reimplementing the standard
    forward/backward dance.
    """
    model = ctx.model
    model.net.train()

    # Gradient accumulation: only zero grads at the start of a fresh cycle,
    # and only step the optimizer at the end. NNModel.train supplies explicit
    # shared state so uneven batches use the loss's effective denominator.
    # Direct legacy callers without that optional state retain prior weighting.
    accumulate_grad_batches = ctx.accumulate_grad_batches
    is_cycle_start = (ctx.batch_idx % accumulate_grad_batches) == 0
    cycle_size = (ctx.batch_idx % accumulate_grad_batches) + 1
    is_cycle_end = cycle_size == accumulate_grad_batches
    should_step = is_cycle_end or ctx.is_last_batch
    accumulation_state = ctx.accumulation_state
    if is_cycle_start:
        model.net.zero_grad()
        _reset_accumulation(accumulation_state)

    # FEAT-028: the run's resolved precision decides autocast and the
    # scaler. A context without one keeps the legacy rule: FP16 only for a
    # scaler on CUDA.
    precision = ctx.precision
    if precision is None:
        precision = _LEGACY_FP16 if ctx.scaler is not None and model.device.type == "cuda" else _FULL_PRECISION
    if precision.uses_scaler and ctx.scaler is None:
        # An unscaled float16 backward underflows small gradients to zero
        # silently (the finite check sees nothing wrong): refuse it.
        raise ValueError(
            "fp16 trains through a GradScaler, and this step context has none: pass "
            "scaler=model._build_grad_scaler() (NNModel.train always does)"
        )
    scaler = ctx.scaler if precision.uses_scaler else None
    reduced = precision.reduced
    autocast = precision.autocast()

    adapter = getattr(model, "task_adapter", None)
    # FEAT-030: under DDP the forward and backward run through the DDP
    # wrapper, which synchronizes gradients only at a committed update.
    ddp = getattr(model, "_ddp_session", None)
    with contextlib.nullcontext() if ddp is None else ddp.step(sync=should_step):
        with autocast:  # the forward and loss only; the backward runs outside autocast
            terms = _step_loss_terms(model, ctx.batch, accumulation_state, accumulate_grad_batches, precision)
        if ddp is None:
            loss_value = _record_step_loss(terms, accumulation_state, should_step=should_step)
        else:
            # Every rank decides together — before any gradient collective.
            loss_value = _agreed_step_loss(ddp, terms, accumulation_state, should_step=should_step)
        # FEAT-029: a compiled fit's backward may compile on first use; its
        # failure is recorded honestly (never retried).
        session = getattr(model, "_compile_session", None)
        with contextlib.nullcontext() if session is None else session.backward():
            if scaler is not None:
                scaler.scale(terms.backward_loss).backward()
            else:
                terms.backward_loss.backward()

    # The batch's records: this rank's, or under DDP the global batch's.
    if ddp is None:
        record_terms = terms
    else:
        record_terms, loss_value = _global_records(ddp, terms, accumulation_state)

    if ctx.epoch_summary is not None:
        _observe_epoch_summary(ctx.epoch_summary, model, record_terms, adapter)

    # FEAT-030: the window's global loss denominator (one collective per
    # committed update) normalizes the averaged gradient and decides a
    # masked window on every rank alike.
    global_weight = (
        ddp.sum(accumulation_state.normalization_weight)
        if ddp is not None and should_step and accumulation_state is not None
        else None
    )
    if ddp is not None and should_step:
        _global_window_check(terms, accumulation_state, global_weight)
    masked = (
        _window_is_masked(terms, accumulation_state, window_is_this_batch=cycle_size == 1)
        if ddp is None
        else terms.valid is not None
        and accumulation_state is not None
        and accumulation_state.normalization_required
        and global_weight == 0
    )
    if should_step and masked:
        # FEAT-002: every target in this optimizer window is masked — there
        # is nothing to learn from, so no update is taken.
        model.net.zero_grad()
        _reset_accumulation(accumulation_state)
    elif should_step:
        if scaler is not None:
            scaler.unscale_(ctx.optimizer)
        if ddp is not None:
            # DDP averaged the ranks' numerator gradients: world / W turns the
            # average into the global batch's (an additive loss: its sum).
            assert accumulation_state is not None and global_weight is not None
            if not accumulation_state.normalization_required:
                _scale_gradients(model.net, float(ddp.world_size))
            elif global_weight:
                _scale_gradients(model.net, ddp.world_size / global_weight)
        elif (
            accumulation_state is not None
            and accumulation_state.normalization_required
            and accumulation_state.normalization_weight
        ):
            _scale_gradients(model.net, 1.0 / accumulation_state.normalization_weight)
        elif accumulation_state is None and cycle_size < accumulate_grad_batches:
            _scale_gradients(model.net, accumulate_grad_batches / cycle_size)
        if reduced and scaler is None:
            # No scaler skips a non-finite reduced-precision update (bf16),
            # so the step checks its gradients itself: nothing non-finite is
            # ever applied.
            _check_finite_gradients(model.net)
        if ctx.grad_clip_norm is not None:
            # Under AMP the gradients were unscaled above, so the clip
            # threshold applies in the original gradient space.
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), ctx.grad_clip_norm)
        if scaler is not None:
            # Report only a step the scaler did not skip (a lowered scale,
            # fused optimizers included); judged only when a clock listens,
            # sparing the comparison's host syncs otherwise.
            committed = scaler_step(scaler, (ctx.optimizer,), judge=listens(ctx.report_update))
        else:
            ctx.optimizer.step()
            committed = True
        _reset_accumulation(accumulation_state)
        if committed:
            ctx.report_update()

    if adapter is not None:
        assert record_terms.valid is not None
        accumulator = adapter.accumulator(keep_arrays=bool(ctx.extra_metrics))
        accumulator.update(record_terms.output, record_terms.target, record_terms.valid)
        return accumulator.result(
            loss=loss_value if record_terms.normalization_weight != 0 else None,
            extra_metrics=ctx.extra_metrics,
        )
    assert record_terms.prediction is not None
    return _classification_edp_for_loss(
        loss_fn=model.loss_fn,
        target=record_terms.target,
        prediction=record_terms.prediction,
        loss=cast(float, loss_value),
        extra_metrics=ctx.extra_metrics,
    )


def _evaluate(
    model: Any,
    loader: Iterable[Any],
    extra_metrics: Optional[Mapping[str, Callable]],
    metrics: tuple[MetricSpec, ...],
    *,
    bounded: bool,
    who: str,
) -> NNEvaluationDataPoint:
    """``NNModel.evaluate()``'s loop, shared with the streaming validation
    step (FEAT-020). ``bounded=True`` keeps counts and sums instead of every
    target and prediction, so memory does not grow with the loader; it
    rejects ``extra_metrics`` and metrics that need every stored score before
    any batch is read. Module-level: legacy stand-ins borrow ``evaluate()``."""
    if bounded and extra_metrics:
        raise _bounded_extra_metrics_error(who)
    # Ensure loss_fn lives on the same device as the model — guards
    # against callers reassigning model.device after construction.
    model.loss_fn = model.loss_fn.to(model.device)
    named = _named_metric_set(model, metrics, where=who, bounded=bounded)
    # getattr: legacy stand-ins borrow evaluate() without the property.
    if getattr(model, "task_adapter", None) is not None:
        return _evaluate_task(model, loader, extra_metrics, named, bounded=bounded, who=who)
    # Snapshot training-mode for non-destructive restore (matches the
    # convention already used by `nnx.viz.activation_map` and
    # `nnx.lr_finder`). Without this, a caller doing the common
    # train → evaluate → train-more pattern silently leaves the net
    # in `.eval()` mode after evaluate(); BatchNorm / Dropout layers
    # would behave incorrectly on the next batch unless the caller
    # remembered to call `model.net.train()` themselves.
    precision = _inference_precision(model)  # before eval(): a failure leaves the modes alone
    training_modes = _capture_training_modes(model.net)
    model.net.eval()

    counts: Optional[LabelCounts] = None  # bounded: chosen by the first batch's label shape
    kind = ""
    all_Y: list[np.ndarray] = []
    all_Y_hat: list[np.ndarray] = []
    loss_numerator = 0.0
    loss_normalization_weight = 0.0
    loss_uses_sum_reduction = False
    n_samples = 0
    n_metric_samples = 0

    try:
        with torch.no_grad():
            for batch in loader:
                with precision.autocast():
                    _, Y, Y_hat_logits, Y_hat = model._fwd_pass(batch)
                Y_hat_logits = precision.output(Y_hat_logits)  # scored in full precision
                if named is not None:
                    named.update(Y, Y_hat_logits)
                batch_n = int(Y.size(0))
                # Aggregate predictions / labels across the entire loader so
                # metrics are computed on the full eval set, not per-batch.
                metric_Y, metric_Y_hat = _classification_metric_tensors(model.loss_fn, Y, Y_hat)
                if metric_Y.numel():
                    if bounded:
                        labels = metric_Y.cpu().numpy()
                        if counts is None:
                            counts, kind = label_counts(labels), label_kind(labels)
                        elif label_kind(labels) != kind:  # the eager path's concatenate refuses it too
                            raise ValueError(
                                f"{who}: label batches changed shape between class labels and indicator rows"
                            )
                        counts.update(labels, metric_Y_hat.cpu().numpy())
                    else:
                        all_Y.append(metric_Y.cpu().numpy())
                        all_Y_hat.append(metric_Y_hat.cpu().numpy())
                    n_metric_samples += int(metric_Y.numel())
                _, batch_loss_numerator, batch_normalization_weight = _loss_terms(model.loss_fn, Y_hat_logits, Y)
                loss_numerator += float(batch_loss_numerator.detach())
                if batch_normalization_weight is None:
                    loss_uses_sum_reduction = True
                else:
                    loss_normalization_weight += batch_normalization_weight
                n_samples += batch_n
    finally:
        _restore_training_modes(training_modes)

    if n_samples == 0:
        raise ValueError(f"{who} loader produced zero samples")
    if n_metric_samples == 0:
        raise ValueError(f"{who} loader produced zero non-ignored samples")

    if counts is not None:
        accuracy, precision, recall, f1 = record_scores(counts)
        edp = NNEvaluationDataPoint(accuracy=accuracy, f1=f1, recall=recall, precision=precision)
    else:
        edp = NNEvaluationDataPoint.of(
            Y=np.concatenate(all_Y), Y_hat=np.concatenate(all_Y_hat), extra_metrics=extra_metrics
        )
    accuracy = edp.accuracy
    assert accuracy is not None  # both paths compute the classification fields
    if named is not None:
        edp = replace(edp, metrics={**edp.metrics, **named.results()})
    return edp.with_loss(
        value=(
            loss_numerator
            if loss_uses_sum_reduction
            else loss_numerator / loss_normalization_weight
            if loss_normalization_weight
            else float("nan")
        )
    ).with_error(value=float(1 - accuracy))


_BY_KEYWORD = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def _bounded_task_accumulator(adapter: TaskAdapter, who: str) -> Any:
    """``adapter.accumulator(bounded=True)``, refusing an adapter that cannot
    build one (a ``TaskAdapter`` subclass written before FEAT-020) or returns
    one that keeps arrays — before any batch is read."""
    needs = f"{who} needs the model's task adapter ({type(adapter).__name__}) to build a bounded accumulator"
    try:
        parameters: Any = inspect.signature(adapter.accumulator).parameters
    except (TypeError, ValueError):
        parameters = None  # not introspectable: the call decides
    keyword = parameters is not None and getattr(parameters.get("bounded"), "kind", None) in _BY_KEYWORD
    if parameters is not None and not (keyword or any(p.kind is p.VAR_KEYWORD for p in parameters.values())):
        raise ValueError(f"{needs} with accumulator(bounded=True); its accumulator() takes no bounded keyword")
    try:
        accumulator = adapter.accumulator(bounded=True)
    except TypeError as exc:
        if keyword:
            raise  # it names bounded: an error of the adapter's own is its own
        # Opaque or **kwargs (perhaps forwarded to a pre-FEAT-020 base): a missing
        # keyword and the adapter's own error look alike, so say both.
        raise ValueError(f"{needs}, but accumulator(bounded=True) raised TypeError: {exc}") from exc
    if not getattr(accumulator, "bounded", False):
        raise ValueError(
            f"the model's task adapter ({type(adapter).__name__}) returned an accumulator that is not bounded from "
            f"accumulator(bounded=True); {who} would keep every target"
        )
    return accumulator


def _evaluate_task(
    model: Any,
    loader: Iterable[Any],
    extra_metrics: Optional[Mapping[str, Callable]],
    named: Optional[_MetricSet],
    *,
    bounded: bool,
    who: str,
) -> NNEvaluationDataPoint:
    """``evaluate()`` for a model with a task (FEAT-002): the adapter
    validates every batch, and loss and metrics are accumulated over
    the valid targets of the whole loader. Every target masked yields
    an ``"empty"`` record (no loss, no metrics) instead of raising."""
    adapter = model.task_adapter
    assert adapter is not None
    precision = _inference_precision(model)  # before eval(): a failure leaves the modes alone
    training_modes = _capture_training_modes(model.net)
    model.net.eval()
    # `bounded=` only when asked: a TaskAdapter subclass written before FEAT-020 keeps working.
    accumulator = (
        _bounded_task_accumulator(adapter, who)  # extra_metrics were refused above
        if bounded
        else adapter.accumulator(keep_arrays=bool(extra_metrics))
    )
    loss_numerator = 0.0
    loss_normalization_weight = 0.0
    loss_uses_sum_reduction = False
    n_batches = 0
    try:
        with torch.no_grad():
            for batch in loader:
                with precision.autocast():
                    _, Y, logits = model._fwd_outputs(batch)
                output, target, valid = adapter.prepare(precision.output(logits), Y)
                accumulator.update(output, target, valid)
                if named is not None:
                    named.update(target, output, valid)
                _, numerator, weight = adapter.loss_terms(model.loss_fn, output, target, valid)
                loss_numerator += float(numerator.detach())
                if weight is None:
                    loss_uses_sum_reduction = True
                else:
                    loss_normalization_weight += weight
                n_batches += 1
    finally:
        _restore_training_modes(training_modes)
    if n_batches == 0:
        raise ValueError(f"{who} loader produced zero samples")
    if not accumulator.count:
        loss: Optional[float] = None
    elif loss_uses_sum_reduction:
        loss = loss_numerator
    else:
        loss = loss_numerator / loss_normalization_weight if loss_normalization_weight else float("nan")
    edp = accumulator.result(loss=loss, extra_metrics=extra_metrics)
    if named is not None and edp.count:
        edp = replace(edp, metrics={**edp.metrics, **named.results()})
    return edp


def _is_streaming_eval_step(eval_step_fn: Any) -> bool:
    """Whether ``eval_step_fn`` is ``nnx.streaming.streaming_eval_step``
    itself, directly or through ``functools.partial`` — the step whose
    limits ``train()`` and ``ExperimentPlan.validate()`` check up front. Any
    other wrapper is a step of its own; the streaming step then still
    refuses what it cannot compute at its first call, before reading a
    validation batch."""
    if eval_step_fn is None:
        return False
    from ..streaming import streaming_eval_step

    while isinstance(eval_step_fn, functools.partial):
        eval_step_fn = eval_step_fn.func
    return eval_step_fn is streaming_eval_step


def _shuffles(X: Any) -> bool:
    """Whether ``X`` is a shuffling DataLoader, whose positional sample ids
    cannot be joined back to the dataset."""
    return isinstance(X, DataLoader) and isinstance(X.sampler, torch.utils.data.RandomSampler)


def _warn_positional_ids(caller: str) -> None:
    from .callbacks import _warn_at_user_frame  # at the first caller outside nnx, whoever called predict

    _warn_at_user_frame(
        f"{caller} over a shuffling DataLoader: sample_ids are iteration positions, not "
        "dataset indices, so they cannot be joined back to the dataset; use a non-shuffled "
        "loader (graph seed rows are exempt: their ids are global node indices)",
        UserWarning,
        once_per_location=True,  # like warnings.warn: a loop over one call site warns once
    )


def _resolve_net_descriptor(
    net_params: Optional[NNParams], params: NNModelParams, module: Optional[torch.nn.Module]
) -> NNModelParams:
    """Check how a model's network is described (FEAT-006) and return the
    params to keep — with a wrapped module's :class:`RuntimeModule`
    descriptor filled in."""
    net = params.net
    if module is not None:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"module must be a torch.nn.Module, got {type(module).__name__}")
        if net_params is not None:
            raise ValueError("pass net_params (a built-in net) or module= (a caller-owned module), not both")
        descriptor = RuntimeModule.of(module)
        if net is None:
            return replace(params, net=descriptor)
        if isinstance(net, RuntimeModule):
            if net != descriptor:
                raise ValueError(
                    f"module {descriptor.module} (topology {descriptor.topology}) does not match the descriptor "
                    f"{net.module} (topology {net.topology})"
                )
            return params
        raise ValueError(
            f"module= wraps a caller-owned module, so NNModelParams.net must be None (or that module's "
            f"RuntimeModule descriptor), got {net}"
        )
    if isinstance(net, Nets):
        if net_params is None:
            raise ValueError("net_params must not be None")
        return params
    if isinstance(net, ModelSpec):
        if net_params is not None:
            raise ValueError(f"a registered ModelSpec ({net}) builds from its own config; pass net_params=None")
        return params
    if isinstance(net, RuntimeModule):
        raise MissingModelFactoryError(
            f"{net} is a runtime-only module (reconstructible=False) that no factory can build; pass module= "
            f"(a {net.module} with the same topology)"
        )
    raise ValueError(
        "NNModelParams.net is required: a Nets member, a registered nnx.models.ModelSpec, or pass module= to "
        "wrap a torch.nn.Module"
    )


class NNModel(_HubMixinBase):
    """Top-level training/eval/predict wrapper around an ``nn.Module``.

    Inherits from :class:`huggingface_hub.PyTorchModelHubMixin` (when the
    ``thekaveh-nnx[hub]`` extra is installed) to gain ``save_pretrained`` /
    ``push_to_hub`` / ``from_pretrained``. Without the extra installed,
    those three methods raise a clear ImportError pointing at the extra;
    no other NNModel functionality is affected.
    """

    net: torch.nn.Module
    # FEAT-029: the current fit's compiled forward, beside (never replacing)
    # ``net``; set only while ``train(compile=...)`` runs.
    _compile_session: Optional[_CompileSession] = None
    # FEAT-030: the current fit's DDP session (wrapper and collectives),
    # beside ``net``; set only while ``train(distributed=...)`` runs.
    _ddp_session: Optional[Any] = None

    def __init__(
        self,
        net_params: Optional[NNParams] = None,
        params: Optional[NNModelParams] = None,
        *,
        module: Optional[torch.nn.Module] = None,
        batch_adapter: Optional[BatchAdapter] = None,
    ):
        """Build the network from ``params.net`` (FEAT-006):

        - a built-in ``Nets`` member builds from ``net_params`` (the default);
        - a registered :class:`~nnx.models.ModelSpec` builds through its
          factory (``net_params`` must be ``None``);
        - ``module=`` wraps a caller-owned ``nn.Module`` as is — never cloned
          or re-initialized — and fills ``params.net`` with its
          :class:`~nnx.models.RuntimeModule` descriptor.

        ``batch_adapter`` (runtime-only) says how a non-built-in module sees
        a batch; the default uses the module's ``unpack_batch`` when it has
        one, else one positional input (:mod:`nnx.models`).
        """
        # NOTE: we deliberately do NOT call super().__init__() — the
        # PyTorchModelHubMixin base has no __init__ of its own (it's a
        # mixin that only contributes class-level methods), and even if
        # it grew one in a future hub release, the only side effect we'd
        # want is config-attribute initialization which we handle below.
        if params is None:
            raise ValueError("params must not be None")
        params = _resolve_net_descriptor(net_params, params, module)
        if batch_adapter is not None and not isinstance(batch_adapter, BatchAdapter):
            raise TypeError(f"batch_adapter must be an nnx.models.BatchAdapter, got {type(batch_adapter).__name__}")

        # FEAT-002: a declared task is checked against the loss, net type and
        # output width before any module is built.
        self._task_adapter: Optional[TaskAdapter] = None
        if params.task is not None:
            self._task_adapter = task_adapter(params.task)
            self._task_adapter.check_model(
                net=params.net, loss=params.loss, output_dim=getattr(net_params, "output_dim", None)
            )

        self.net_params = net_params
        self.params = params
        self._topology_transforms: tuple[NNCheckpointTransform, ...] = ()

        self.device = self.params.device()
        # FEAT-028: the precision policy is resolved against this device
        # before anything is built, so an unsupported request fails here.
        self._precision = (_precision_key(self.params, self.device), resolve_precision(self.params, self.device))
        self.loss_fn = self.params.loss().to(self.device)
        net = self.params.net
        if module is not None:
            # Module.to moves in place and returns the same object.
            self.net = module.to(self.device)
        elif isinstance(net, ModelSpec):
            self.net = build_module(net).to(self.device)
        else:
            assert isinstance(net, Nets) and net_params is not None
            self.net = net(params=net_params).to(self.device)
        if module is None:
            # The tensors the descriptor rebuilds, recorded once from the
            # module just built — never a second construction (FEAT-006) —
            # for the reconstructibility checks (FEAT-016); a shape is None
            # where a rebuild leaves the tensor uninitialized (a lazy layer).
            self._reference_state = _state_shapes(self.net.state_dict())
        # Built-in nets keep their own unpack_batch (the legacy path); other
        # modules see batches through an adapter (FEAT-006).
        self._batch_adapter: Optional[BatchAdapter] = (
            batch_adapter
            if batch_adapter is not None
            else (None if isinstance(net, Nets) else default_batch_adapter(self.net))
        )

    @property
    def task_adapter(self) -> Optional[TaskAdapter]:
        """The adapter for ``params.task`` (FEAT-002), or ``None`` for a
        legacy classification model. Custom steps can call
        ``task_adapter.record(output, target, loss=...)`` to write the same
        task record the default step writes."""
        # getattr: subclasses and stand-ins that bypass __init__ are legacy models.
        return getattr(self, "_task_adapter", None)

    @property
    def resolved_precision(self) -> ResolvedPrecision:
        """The precision this model runs in on its current device
        (FEAT-028): ``params.precision`` resolved against ``device`` — re-
        resolved, never read from saved metadata, whenever the device or the
        policy changes (a loaded model resolves on its destination device)."""
        key = _precision_key(self.params, self.device)
        cached = getattr(self, "_precision", None)
        if cached is None or cached[0] != key:
            cached = (key, resolve_precision(self.params, self.device))
            self._precision = cached
        return cached[1]

    def _check_task_preflight(self) -> None:
        """Reject a runtime ``loss_fn`` the declared task cannot score —
        before any loader is iterated (NNModel.train and Trainer.train)."""
        adapter = getattr(self, "task_adapter", None)
        if adapter is not None:
            adapter.check_loss_fn(self.loss_fn)

    def _base_state(self) -> Optional[Mapping[str, Optional[tuple[int, ...]]]]:
        """``{key: shape}`` of the tensors the descriptor rebuilds, recorded
        when the model built its module (a shape is ``None`` for an
        uninitialized lazy parameter); ``None`` for a runtime module, which
        nothing rebuilds."""
        recorded = getattr(self, "_reference_state", None)
        if recorded is None:
            # An object that never ran this __init__ (a stand-in, an older
            # pickle) gets the reference develop used: a factory's recorded
            # names, or a rebuild of a built-in net off the global random
            # streams — computed once.
            net, net_params = self.params.net, getattr(self, "net_params", None)
            legacy_keys = getattr(self, "_reference_state_keys", None)
            if isinstance(net, ModelSpec) and legacy_keys is not None:
                recorded = dict.fromkeys(legacy_keys)
                self._reference_names_only = True  # shapes, lazy layers included, are unknown
            elif isinstance(net, Nets) and net_params is not None:
                from ..seeding import _global_rng_kept

                with _global_rng_kept():
                    recorded = _state_shapes(net(params=net_params).state_dict())
            else:
                return None
            self._reference_state = recorded
        return MappingProxyType(recorded)

    def _lazy_base_keys(self) -> frozenset[str]:
        """The base tensors a rebuild leaves uninitialized (lazy layers):
        those recorded without a shape — unknown for an older factory
        model whose reference holds names only."""
        base = self._base_state()
        if base is None or getattr(self, "_reference_names_only", False):
            return frozenset()
        return frozenset(key for key, shape in base.items() if shape is None)

    def _topology_drift(self) -> Optional[str]:
        """How the live topology differs from the descriptor plus the
        recorded recipe (FEAT-016) — names, shapes, each target's module and
        configuration — as one message; ``None`` when it matches, or when
        nothing could tell (a runtime module, a train-end transform that
        rebuilds its own topology)."""
        base_state = self._base_state()
        if base_state is None:
            return None
        from ..transforms import _topology_problems

        problems = _topology_problems(self.net, base_state, tuple(self._topology_transforms))
        if not problems:
            return None
        return (
            "the model's topology differs from its descriptor plus its recorded transformation recipe (unrecorded "
            "surgery?): " + "; ".join(problems[:5])
        )

    def _assert_reconstructible_topology(self) -> None:
        transforms = tuple(self._topology_transforms)
        if transforms and not all(_replayable(t) for t in transforms):
            return  # train-end transforms (QAT) rebuild their own topology
        base_state = self._base_state()
        if base_state is None:
            return  # a runtime module is marked reconstructible=False instead
        if transforms:
            # FEAT-016: the live topology must be exactly the base plus its
            # recorded recipe; surgery outside the recipe stays unrecorded.
            drift = self._topology_drift()
            if drift is not None:
                raise ValueError(
                    drift
                    + "; apply topology changes through nnx.transforms.TransformRecipe so checkpoints can rebuild "
                    "them"
                )
            return
        expected_keys = set(base_state)
        actual_keys = _tensor_keys(self.net.state_dict())
        if _replaced_layers(actual_keys, expected_keys)[1]:
            raise ValueError(
                "low-rank surgery topology has no reconstruction recipe; train before surgery, "
                "or use export_state_dict() for the modified module"
            )

    def to_onnx(
        self,
        path: str,
        example_input: Union[torch.Tensor, tuple, np.ndarray],
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_batch: bool = True,
        opset_version: int = 17,
        dynamo: bool = False,
    ) -> str:
        """Export the underlying network to ONNX format.

        Args:
            path: output filename (e.g., "model.onnx").
            example_input: a tensor (or tuple of tensors for multi-input
                nets) with realistic shape/dtype used to trace the network.
            input_names: optional list of human-readable input port names.
            output_names: optional list of human-readable output port names.
            dynamic_batch: when True (default), marks dim 0 as dynamic so
                the exported model accepts any batch size at inference.
            opset_version: ONNX opset to target. 17 is broadly supported
                by current runtimes.
            dynamo: when False (default), uses the legacy TorchScript-based
                `torch.onnx.export` path — plain `pip install onnx` is
                enough. When True, dispatches to PyTorch's new
                `torch.export`-based exporter (default in torch>=2.9,
                supports >2 GB models via external data, faster). The
                dynamo path requires `onnxscript`; install via
                `pip install thekaveh-nnx[onnx-dynamo]`.

        Returns the path written. Network is put in eval mode for tracing.
        A network that declares ``onnx_export_unsupported`` (a graph
        classifier, FEAT-026) is refused before anything is written.
        """
        unsupported = getattr(self.net, "onnx_export_unsupported", None)
        if unsupported:
            raise NotImplementedError(f"to_onnx(): {unsupported}")
        if dynamo:
            # Lazy-import: keep `onnxscript` out of NNx's required deps so
            # plain `pip install thekaveh-nnx[onnx]` (legacy path) still works. If
            # the user opted in to `dynamo=True` without the extra, give
            # an error that names the install command instead of letting
            # torch surface a less actionable failure.
            try:
                import onnxscript  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "to_onnx(dynamo=True) requires the `onnxscript` package. "
                    "Install via `pip install thekaveh-nnx[onnx-dynamo]` (or `pip install onnxscript`)."
                ) from e

        # Normalize a single Tensor / np.ndarray to a length-1 tuple, then
        # coerce each element. Without the np.ndarray case in the singleton
        # check, a 2-D array like ``np.zeros((2, 4))`` falls into the
        # iterable branch and is unpacked row-by-row — torch.onnx.export
        # then sees a model with `N = first-dim` separate inputs instead of
        # the one input the caller meant.
        if isinstance(example_input, (torch.Tensor, np.ndarray)):
            example_input = (example_input,)
        example_input = tuple(
            (e.to(self.device) if isinstance(e, torch.Tensor) else torch.from_numpy(np.asarray(e)).to(self.device))
            for e in example_input
        )

        in_names = input_names or [f"input_{i}" for i in range(len(example_input))]
        out_names = output_names or ["output"]

        # Dynamic-shape spec is exporter-specific: the legacy TorchScript path
        # takes `dynamic_axes` (string-keyed dict of dim -> name); the dynamo
        # path takes `dynamic_shapes` (a pytree mirroring `example_input` whose
        # leaves are `{dim: torch.export.Dim(...)}`). Passing dynamic_axes with
        # dynamo=True triggers a UserWarning and on newer torch/onnxscript can
        # surface a hard `ConversionError` because dynamo emits `aten.sym_size`
        # ops that onnxscript can't always dispatch.
        dynamic_axes = None
        dynamic_shapes: Optional[tuple[dict[int, Any], ...]] = None
        if dynamic_batch:
            if dynamo:
                from torch.export import Dim

                batch = Dim("batch", min=1)
                dynamic_shapes = tuple({0: batch} for _ in example_input)
            else:
                dynamic_axes = {n: {0: "batch"} for n in in_names + out_names}

        # Snapshot training mode so the train → to_onnx → train-more pattern
        # doesn't silently strand the caller in .eval() (BatchNorm running-
        # stats / Dropout masking would then stay disabled on the next train
        # step). Matches the non-destructive contract every sibling inference
        # helper enforces (predict / evaluate / generate / diffusion.sample /
        # embed_texts / lr_finder / viz.activation_map / viz.attribute /
        # viz.netron_export). The bare `self.net.eval()` here was the lone
        # exception.
        training_modes = _capture_training_modes(self.net)
        self.net.eval()
        # torch>=2.5 defaults torch.onnx.export to the dynamo-based exporter,
        # which requires `onnxscript`. We pass `dynamo` through explicitly so
        # the default (False) keeps the legacy tracing path regardless of
        # the installed torch version — plain `pip install onnx` is enough.
        try:
            export_accepts_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
            if dynamo:
                if not export_accepts_dynamo:
                    raise RuntimeError(
                        "to_onnx(dynamo=True) requires torch>=2.5 (the dynamo-based "
                        "ONNX exporter wasn't available before then). Upgrade torch or "
                        "call with dynamo=False."
                    )
                torch.onnx.export(
                    self.net,
                    example_input,
                    path,
                    input_names=in_names,
                    output_names=out_names,
                    dynamic_shapes=dynamic_shapes,
                    opset_version=opset_version,
                    dynamo=True,
                )
            elif export_accepts_dynamo:
                torch.onnx.export(
                    self.net,
                    example_input,
                    path,
                    input_names=in_names,
                    output_names=out_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    dynamo=False,
                )
            else:
                torch.onnx.export(
                    self.net,
                    example_input,
                    path,
                    input_names=in_names,
                    output_names=out_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                )
        finally:
            _restore_training_modes(training_modes)
        return path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: NNCheckpoint,
        device: Optional[Devices] = None,
        *,
        module: Optional[torch.nn.Module] = None,
        batch_adapter: Optional[BatchAdapter] = None,
        precision: Optional[PrecisionPolicy] = None,
        exclude_submodules: Sequence[str] = (),
        **model_kwargs: Any,
    ) -> Self:
        """Rebuild a model, replay topology transforms, and load its weights.

        ``exclude_submodules`` names top-level submodules the training run
        attached to the net that are not part of the rebuilt architecture —
        a JEPA predictor registered as ``model.net._jepa_predictor``, say —
        whose weights are left out; every other weight still loads strictly.
        A name the checkpoint does not hold raises ``ValueError``.

        Ordinary and legacy FP32 checkpoints have no transforms. Converted
        QAT checkpoints replay their persisted torchao recipe before state
        loading; unsupported recipes fail explicitly rather than constructing
        a model with the wrong topology.

        FEAT-006: a registered :class:`~nnx.models.ModelSpec` is rebuilt
        through its factory — an unregistered one raises
        :class:`~nnx.models.MissingModelFactoryError`, and a rebuilt topology
        that differs from the saved weights raises ``ValueError``, both
        before any weight is loaded. A runtime-only module
        (``reconstructible=False``) needs ``module=`` — a module of the same
        topology, into which the weights are loaded.

        FEAT-028: the precision policy is re-resolved on the destination
        device (the constructor resolves it) — never read from the
        checkpoint's recorded precision — so a policy that device cannot
        run fails here; ``precision=`` replaces the saved policy (and the
        legacy ``mixed_precision`` flag) for this model.
        """
        model_params = checkpoint.model_params if device is None else replace(checkpoint.model_params, device=device)
        if precision is not None:
            model_params = replace(model_params, precision=precision, mixed_precision=False)
        net = model_params.net
        transforms = tuple(getattr(checkpoint, "transforms", ()))
        if isinstance(net, RuntimeModule) and _recipe_transforms(transforms):
            # A recipe is refused on a runtime-only module (FEAT-016); refused
            # before module= is wrapped (or moved), so the caller's module is
            # left exactly as given.
            raise ValueError(
                f"this checkpoint of the runtime-only module {net} records a transformation recipe, which nothing can "
                "replay on a caller-owned module"
            )
        if batch_adapter is not None:
            # Only passed when set: subclasses keep their own constructors.
            model_kwargs["batch_adapter"] = batch_adapter
        if module is not None or isinstance(net, RuntimeModule):
            if not isinstance(net, RuntimeModule):
                raise ValueError(f"module= only applies to runtime-module checkpoints; this one is built from {net}")
            if module is None:
                raise MissingModelFactoryError(
                    f"the checkpoint holds weights of the runtime-only module {net} (reconstructible=False): no "
                    f"factory can rebuild it — pass module= (a {net.module} with the same topology)"
                )
            model = cls(params=model_params, module=module, **model_kwargs)
        elif isinstance(net, ModelSpec):
            model = cls(params=model_params, **model_kwargs)
        else:
            model = cls(params=model_params, net_params=checkpoint.net_params, **model_kwargs)

        _replay_transforms(model, transforms)
        model._topology_transforms = _canonical_transforms(transforms)
        net_state = _without_submodules(checkpoint.net_state, exclude_submodules)
        if not transforms:
            _refuse_unrecorded_recipe_state(net_state, model.net.state_dict())
        if not isinstance(net, Nets):
            check_state_schema(model.net, net_state, what=f"checkpoint of {net}")

        try:
            model.net.load_state_dict(net_state)
        except RuntimeError as error:
            if not transforms and _looks_like_converted_qat_state(net_state):
                raise ValueError(
                    "converted QAT checkpoint lacks reconstruction metadata; "
                    "recreate its torchao topology with the original qat_config and groupsize"
                ) from error
            children = {name for name, _ in model.net.named_children()}
            attached = sorted({key.split(".", 1)[0] for key in net_state if "." in key} - children)
            if attached:
                raise RuntimeError(
                    f"{error}\nThe checkpoint holds submodules the rebuilt {type(model.net).__name__} does not have "
                    f"({', '.join(attached)}): NNModel.from_checkpoint(checkpoint, exclude_submodules="
                    f"{tuple(attached)!r}) rebuilds the architecture without them"
                ) from error
            raise

        return model

    # ------------------------------------------------------------------
    # HuggingFace Hub integration (inherited save_pretrained / push_to_hub /
    # from_pretrained from PyTorchModelHubMixin dispatch into these two
    # overrides).
    #
    # We override both because NNModel is NOT itself an nn.Module — its
    # weights live on `self.net`, and the default PyTorchModelHubMixin
    # implementation would call `self.state_dict()` and miss them. We
    # also need full control over config.json since the default mixin
    # auto-encoder hits the is_dataclass branch and emits asdict(NNParams)
    # which leaks the internal `_dims` cache and emits raw enums that
    # break JSON. Using the public NNParams.state() / NNModelParams.state()
    # round-trip keeps the on-Hub config compatible with NNRun's
    # hash-grouping form.
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory, *args: Any, **kwargs: Any) -> Any:
        """Hub save (``huggingface_hub.PyTorchModelHubMixin.save_pretrained``).

        FEAT-006: a runtime-only module has no factory to rebuild it from,
        so it is rejected with :class:`~nnx.models.MissingModelFactoryError`
        before any file or directory is written."""
        self._require_portable("save_pretrained")
        if _recipe_transforms(self._topology_transforms):
            # FEAT-016: an artifact whose recipe cannot rebuild its weights
            # would fail only when someone loads it.
            drift = self._topology_drift()
            if drift is not None:
                raise ValueError(f"save_pretrained refused before writing anything: {drift}")
        return super().save_pretrained(save_directory, *args, **kwargs)

    def _require_portable(self, operation: str) -> None:
        net = self.params.net
        if isinstance(net, RuntimeModule):
            raise MissingModelFactoryError(
                f"{operation} needs a rebuildable module, but {net} is runtime-only (reconstructible=False); "
                "register a model factory (nnx.models.register_model_factory) and build the model from a "
                "ModelSpec to save it portably"
            )

    def _save_pretrained(self, save_directory) -> None:
        """Write the network weights + params config under ``save_directory``.

        Dispatched-to by :meth:`PyTorchModelHubMixin.save_pretrained`. The
        on-disk layout is the canonical Hub layout:

          - ``model.safetensors`` — ``self.net.state_dict()`` as safetensors.
          - ``config.json`` — ``{"net_params": <state>, "params": <state>}``,
            using the same ``.state()`` form NNRun uses for hashing.

        :meth:`PyTorchModelHubMixin.save_pretrained` additionally writes a
        ``README.md`` model card via ``generate_model_card()``; that's
        emitted on top of these two files by the base implementation.
        """
        from pathlib import Path

        self._require_portable("save_pretrained")
        try:
            from safetensors.torch import save_file
        except ImportError as e:  # pragma: no cover — gated by optional dep
            raise ImportError("save_pretrained requires the `hub` extra: `pip install thekaveh-nnx[hub]`.") from e

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Detach + contiguous + clone matches the hygiene NNCheckpoint
        # applies on its safetensors path: drop autograd hooks, ensure
        # C-contiguous storage, and BREAK STORAGE SHARING — safetensors
        # rejects tied tensors (tok_embed/lm_head share storage on every
        # default TransformerNN), and .contiguous() is a no-op on an
        # already-contiguous shared view. load_state_dict reassembles the
        # tie on reload by copying both identical keys into the shared
        # parameter.
        tensors = _tensor_state_dict(self.net.state_dict(), operation="Hugging Face Hub export")
        save_file(tensors, str(save_dir / _HUB_MODEL_FILENAME))

        config: dict[str, Any] = {
            "params": self.params.state(),
            "transforms": [transform.state() for transform in self._topology_transforms],
        }
        if self.net_params is not None:
            # A registered ModelSpec is fully described by `params.net`.
            config["net_params"] = self.net_params.state()
        config.update(self._hub_reconstruction_config(save_dir))
        # Explicit utf-8 — Hub config files round-trip through HuggingFace's
        # repo download path and can be read on any platform; relying on
        # the host locale's default encoding could mis-encode unicode
        # paths or non-ASCII tokenizer names round-tripped via
        # `params.state()`.
        with open(save_dir / _HUB_CONFIG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(config, f, sort_keys=True, indent=2)

    def _hub_reconstruction_config(self, save_directory) -> dict[str, Any]:
        """Persist subclass-owned artifacts and return their config fragment."""
        return {}

    @classmethod
    def _hub_reconstruction_kwargs(cls, config: Mapping[str, Any], config_directory) -> dict[str, Any]:
        """Rebuild subclass constructor arguments from a Hub artifact."""
        return {}

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision=None,
        cache_dir=None,
        force_download: bool = False,
        local_files_only: bool = False,
        token=None,
        map_location: str = "cpu",
        strict: bool = True,
        precision: Optional[PrecisionPolicy] = None,
        **model_kwargs,
    ) -> NNModel:
        """Rebuild an NNModel from a save_pretrained directory or Hub repo.

        Dispatched-to by :meth:`PyTorchModelHubMixin.from_pretrained`,
        which handles the remote-download path before calling this. Local
        paths skip the download. Either way, we read ``config.json`` to
        reconstruct the net params and ``NNModelParams`` via their public
        ``from_state`` constructors, then load ``model.safetensors`` into
        the freshly-built ``self.net``.

        ``map_location`` is forwarded as the safetensors ``device=`` (the
        net is then moved to ``self.device`` by ``NNModel.__init__``
        regardless). ``strict`` is forwarded to ``load_state_dict``; it
        defaults to True because the net is instantiated from the same
        config the weights were saved with, so any key mismatch indicates
        a corrupted or hand-edited artifact. Unrecognized ``model_kwargs``
        raise instead of being silently dropped — NNModel reconstructs
        entirely from ``config.json``.

        FEAT-028: the saved precision policy is re-resolved on the
        ``map_location`` device, never taken from saved metadata;
        ``precision=`` replaces it (and the legacy ``mixed_precision`` flag).
        """
        # The mixin inspects NNModel.__init__'s signature and auto-injects
        # matching config.json entries ("net_params"/"params") as kwargs.
        # Both are rebuilt from config.json below via from_state — the
        # raw dicts are dropped knowingly. Anything else is a caller
        # error (a typo'd knob would otherwise vanish silently).
        model_kwargs.pop("net_params", None)
        model_kwargs.pop("params", None)
        model_kwargs.pop("tokenizer", None)
        model_kwargs.pop("transforms", None)
        # FEAT-006: a runtime-only adapter for a registered module's batches.
        batch_adapter = model_kwargs.pop("batch_adapter", None)
        if (
            os.path.isdir(model_id)
            and not os.path.exists(os.path.join(model_id, _HUB_CONFIG_FILENAME))
            and os.path.exists(os.path.join(model_id, "bundle.json"))
        ):
            raise ValueError(
                f"{model_id!r} is an NNx run bundle, not a Hugging Face Hub distribution; rebuild it with "
                "nnx.bundles.reconstruct_bundle"
            )
        if model_kwargs:
            raise TypeError(
                f"from_pretrained got unexpected model kwargs {sorted(model_kwargs)!r} — "
                "NNModel reconstructs entirely from the repo's config.json."
            )
        try:
            from safetensors.torch import load_file
        except ImportError as e:  # pragma: no cover — gated by optional dep
            raise ImportError("from_pretrained requires the `hub` extra: `pip install thekaveh-nnx[hub]`.") from e

        if os.path.isdir(model_id):
            config_path = os.path.join(model_id, _HUB_CONFIG_FILENAME)
            weights_path = os.path.join(model_id, _HUB_MODEL_FILENAME)
        else:
            from huggingface_hub import snapshot_download

            snapshot_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
            config_path = os.path.join(snapshot_path, _HUB_CONFIG_FILENAME)
            weights_path = os.path.join(snapshot_path, _HUB_MODEL_FILENAME)

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        # We accept either form for back-compat with any future config
        # writer: nested under {"net_params": ..., "params": ...} (what
        # _save_pretrained writes today) or flat at the top level. The
        # nested form takes precedence.
        net_params_state = config.get("net_params", config)
        model_params_state = config.get("params", config)

        params = NNModelParams.from_state(model_params_state)
        net_params: Optional[NNParams] = None
        if isinstance(params.net, RuntimeModule):
            raise MissingModelFactoryError(
                f"{model_id!r} describes the runtime-only module {params.net} (reconstructible=False); "
                "it cannot be rebuilt"
            )
        if isinstance(params.net, ModelSpec):
            # Fail before anything is built when the factory is unknown.
            from ..models import resolve_model_factory

            resolve_model_factory(params.net)
        else:
            # resolve_from_state dispatches transformer configs to
            # NNTransformerParams so LM models round-trip through the Hub.
            net_params = NNParams.resolve_from_state(net_params_state)
        try:
            torch_load_device = torch.device(map_location)
            load_device = Devices(torch_load_device.type)
        except (TypeError, ValueError) as e:
            raise ValueError(f"unsupported Hub map_location {map_location!r}") from e
        if torch_load_device.index is not None:
            raise ValueError(f"indexed Hub map_location is unsupported: {map_location!r}")
        params = replace(params, device=load_device)
        if precision is not None:
            params = replace(params, precision=precision, mixed_precision=False)

        transforms = tuple(NNCheckpointTransform.from_state(item) for item in config.get("transforms", []))
        reconstruction_kwargs = cls._hub_reconstruction_kwargs(config, os.path.dirname(config_path))
        if batch_adapter is not None:
            reconstruction_kwargs["batch_adapter"] = batch_adapter
        model = cls(net_params=net_params, params=params, **reconstruction_kwargs)
        _replay_transforms(model, transforms)
        model._topology_transforms = _canonical_transforms(transforms)
        state_dict = load_file(weights_path, device=str(torch_load_device))
        if not transforms and strict:
            _refuse_unrecorded_recipe_state(state_dict, model.net.state_dict())
        if net_params is None and strict:
            check_state_schema(model.net, state_dict, what=f"Hub artifact of {params.net}")
        model.net.load_state_dict(state_dict, strict=strict)
        return model

    def freeze(self, *patterns: str) -> int:
        """Freeze parameters under ``self.net`` matching any of ``patterns``
        (fnmatch globs against the dotted parameter name). Returns the
        number of parameters newly frozen.

        Convenience wrapper around :func:`nnx.finetune.freezing.freeze`
        — use the standalone function when freezing a module that isn't
        ``self.net`` (e.g., a custom decoder hanging off this model).
        """
        from ..finetune.freezing import freeze as _freeze

        return _freeze(self.net, *patterns)

    def unfreeze(self, *patterns: str) -> int:
        """Mirror of :meth:`freeze` — set ``requires_grad=True`` on
        matching parameters."""
        from ..finetune.freezing import unfreeze as _unfreeze

        return _unfreeze(self.net, *patterns)

    def export_state_dict(self, path: str) -> str:
        """Save just ``self.net.state_dict()`` to ``path``.

        The file is a plain ``torch.save`` of a state-dict — loadable by
        any torch consumer without nnx installed, and by
        :func:`nnx.finetune.load_pretrained` for the fine-tuning round-trip.
        Companion to the NNCheckpoint format, which carries the params +
        idp wrapper alongside the weights; ``export_state_dict`` strips
        all of that and leaves just the weights. It records no
        transformation recipe (FEAT-016, ``nnx.transforms``): it cannot
        rebuild a recipe's topology alone — materialize the recipe on a
        fresh model before loading it, or keep a checkpoint or
        ``save_pretrained`` artifact, which carry the recipe.

        Returns ``path`` so calls can be chained.
        """
        torch.save(self.net.state_dict(), path)
        return path

    def train(
        self,
        params: NNTrainParams,
        callbacks: Optional[list[CallbackLike]] = None,
        train_step_fn: Optional[TrainStepFn] = None,
        eval_step_fn: Optional[EvalStepFn] = None,
        salt: Optional[str] = None,
        components: Optional[list[Any]] = None,
        objective: Optional[Callable[[Any], Any]] = None,
        provenance: Optional[ExperimentManifest] = None,
        history: Optional[HistoryJournal] = None,
        compile: Optional[CompileSpec] = None,
        distributed: Optional[Any] = None,
    ) -> NNRun:
        """Train the model and return its persisted run history.

        Args:
            params: Required loaders, optimizer/scheduler configuration,
                epoch count, persistence controls, and optional resume source.
            callbacks: Lifecycle callbacks invoked around training and epochs.
            train_step_fn: Optional per-batch override; the default performs
                supervised forward, loss, backward, and optimizer stepping.
            eval_step_fn: Optional once-per-epoch validation override that
                receives the complete validation loader.
            salt: Optional string folded into the run.id hash so identical
                (model, net, train) configs run as distinct experiments
                without altering modeled params. ``None`` (the default)
                preserves existing run.id hashes exactly.
            components: Extra checkpointable components (FEAT-005,
                :class:`~nnx.StatefulComponent`) whose state is saved with
                every checkpoint and restored on a stateful warm resume.
                Callbacks and step functions that implement the protocol
                (``EarlyStopping``, the JEPA step) are registered
                automatically; names must be unique.
            objective: Optional objective (FEAT-004, ``nnx.objectives``) —
                a callable returning loss terms with explicit denominators.
                NNx's shared update engine then owns backward,
                accumulation (``accumulate_grad_batches``, exact for uneven
                microbatches and masks), mixed precision, clipping and the
                optimizer step, firing ``Callback.on_optimizer_update`` once
                per committed update. Mutually exclusive with
                ``train_step_fn``.
            provenance: Optional :class:`~nnx.provenance.ExperimentManifest`
                (FEAT-019) — the declared intent. The fit records it
                (``runs/<id>/provenance.json``, with its fingerprint) and a
                fresh attempt (``attempt.json``: parent attempt and
                checkpoint generation on resume; final status and last
                committed checkpoint). Never part of the run id.
            history: Optional :class:`~nnx.history.HistoryJournal`
                (FEAT-036): keep only its ``retention`` most recent records
                in memory (``ctx.idps``, the returned ``NNRun.idps``) and
                append every record once to ``runs/<id>/history/`` instead
                of rewriting ``idps.csv`` each epoch. ``None`` (the default)
                keeps the eager in-memory list and CSV. Never part of the
                run id.
            compile: Optional :class:`~nnx.compilation.CompileSpec`
                (FEAT-029): run the built-in step's FP32 forward (and the
                fit's validation) through ``torch.compile``. The wrapper is
                built per call beside ``model.net``, never assigned to it, so
                checkpoints, export and optimizers see the eager module.
                Refused, before any run is reserved, with a custom
                ``train_step_fn`` or ``objective``, a graph or custom net, a
                reduced precision or a topology-changing callback (QAT).
                ``NNRun.compile`` records the request and what took effect.
                ``None`` (the default) trains eagerly. Never part of the run
                id.
            distributed: Optional :class:`nnx.distributed.DDP` (FEAT-030):
                train data-parallel over the process group ``torchrun``
                started (``nnx.distributed.init_process_group()``), with
                ``nnx.distributed.train_loader`` / ``validation_loader``
                partitions. Each update equals a single process training on
                the ranks' union batch; records are global and identical on
                every rank; only the writer rank holds the lease and writes
                artifacts, and every rank returns the same run. FP32, the
                default steps and one node only — refused, on every rank
                together, before any run is reserved otherwise. Never part
                of the run id.

        Returns:
            The completed :class:`NNRun`, persisted with run metadata,
            iteration history, and configured checkpoints under
            ``<cwd>/runs/<run.id>/``. The printed completion line names
            ``runs/<id>`` relative to the working directory: a display path
            with no absolute prefix, so captured notebook output stays
            portable; it is not artifact provenance.

        Raises:
            ValueError: If required training inputs are missing or invalid,
                the model is fully frozen, or resume state is incompatible.
            FileExistsError: If the content-addressed run already exists and
                ``overwrite_existing`` is false.
            FloatingPointError: If the default step encounters non-finite loss.

        The run lease prevents another process using ``overwrite_existing``
        from deleting or interleaving artifacts until final persistence ends.
        """
        from ..objectives import _check_objective_run, _check_update_owner

        # Checked before anything else: one owner per optimizer update.
        _check_update_owner(train_step_fn, objective)
        if distributed is None:
            # nnx.distributed.writer_only(...) without DDP: simply built, so
            # every later check sees the real callback (FEAT-030).
            callbacks = _writer_owned_callbacks(callbacks)
        _check_provenance(provenance)
        _check_history(history, callbacks)
        if train_step_fn is None or _recipe_transforms(self._topology_transforms):
            # NNx owns the update (default step or objective), or the model
            # carries a recipe (FEAT-016): the run's checkpoints must be
            # reconstructible from the params and recorded recipe.
            self._assert_reconstructible_topology()
        if params is None:
            raise ValueError("train params must be non-None")
        self._check_task_preflight()
        # FEAT-028: the precision policy is resolved once, before any run is
        # reserved; the loop, validation and prediction reuse it.
        precision = self._precision_for_training(train_step_fn)
        _monitoring_preflight(
            self,
            metrics=params.metrics,
            monitor=params.monitor,
            callbacks=callbacks,
            default_train_step=train_step_fn is None and objective is None,
            default_eval_step=eval_step_fn is None,
            has_val_loader=params.val_loader is not None,
            owner="NNTrainParams",
        )
        # NNx's own eval steps (nnx.streaming.streaming_eval_step) check what
        # they can compute before any run is reserved or loader read.
        if params.val_loader is not None and _is_streaming_eval_step(eval_step_fn):
            from ..streaming import _streaming_preflight

            _streaming_preflight(self, params)
        if params.train_loader is None:
            raise ValueError(
                "params.train_loader is required — set it directly or via with_train_loader(...) before train()."
            )
        if params.optim is None or not params.optim.is_valid():
            raise ValueError(f"train params has an invalid optim config: {params.optim!r}")
        if not any(p.requires_grad for p in self.net.parameters()):
            raise ValueError(
                "model has no trainable parameters — did you freeze('*')? Unfreeze something before train()."
            )

        if params.resume_epochs == "planned" and params.resume_from_run_id is None:
            raise ValueError(
                "resume_epochs='planned' continues a run's plan: set resume_from_run_id (it does nothing on a "
                "fresh run)"
            )
        if params.seed is not None:
            from ..seeding import set_seed

            set_seed(params.seed)

        from ..optimizers import build_optimizer

        # Build the optimizer before any run directory exists: a registered
        # factory that is unknown, or that returns something other than an
        # optimizer over exactly the resolved parameters, fails here with no
        # run reserved (nnx.optimizers.build_optimizer is the shared hook).
        optimizer = build_optimizer(self.net, params.optim)
        if objective is not None:
            # FEAT-040: an objective refuses what it cannot train (say, a JEPA
            # predictor the optimizer does not own) before any run exists.
            _check_objective_run(objective, self, optimizers={"default": optimizer}, callbacks=callbacks)
        # The fp16 scaler, through the override hook and checked against the
        # policy (FEAT-028) before any run is reserved.
        scaler = self._build_grad_scaler()
        _check_scaler_hook(precision, scaler, self.device.type)
        # FEAT-029: a compile request is checked (scope, backend) before any
        # run is reserved; the wrapper itself compiles lazily, on first use.
        compile_holder = _CompileHolder(_compile_session(self, compile, train_step_fn, objective, callbacks, precision))
        # FEAT-030: a DDP request is checked on every rank together (scope,
        # partitions, callbacks), then the DDP wrapper is built collectively.
        ddp_session, callbacks = _distributed_session(
            self,
            distributed,
            params=params,
            callbacks=callbacks,
            train_step_fn=train_step_fn,
            eval_step_fn=eval_step_fn,
            objective=objective,
            precision=precision,
            history=history,
            compile=compile,
        )
        run = NNRun(
            train=params,
            model=self.params,
            net=self.net_params,
            salt=salt,
            transforms=_recipe_transforms(self._topology_transforms),  # FEAT-016: part of the run id when present
        )
        with contextlib.ExitStack() as stack:
            if ddp_session is None:
                stack.enter_context(run.writable_lease(overwrite=params.overwrite_existing))
            else:
                from ..distributed import collectively

                # Only the writer rank holds the lease; every rank learns
                # whether it got it.
                with collectively("reserving the run"):
                    if ddp_session.writer:
                        stack.enter_context(run.writable_lease(overwrite=params.overwrite_existing))
            stack.enter_context(_compiling(self, compile_holder.session))
            stack.enter_context(_distributing(self, ddp_session))
            replica = ddp_session is not None and not ddp_session.writer
            fitted = _with_attempt(
                run,
                None if replica else provenance,  # the writer records the attempt
                params,
                execution=compile_holder.record,
                fit=lambda: self._train_impl(
                    params=params,
                    run=run,
                    optimizer=optimizer,
                    callbacks=callbacks,
                    train_step_fn=train_step_fn,
                    eval_step_fn=eval_step_fn,
                    components=components,
                    objective=objective,
                    history=history,
                    precision=precision,
                    scaler=scaler,
                ),
            )
            if ddp_session is not None:
                # Every rank returns the same run: the writer's provenance too.
                records = ddp_session.gather(fitted.provenance if ddp_session.writer else None)
                fitted = fitted.with_provenance(records[ddp_session.spec.writer_rank])
            return fitted

    def _precision_for_training(self, train_step_fn: Optional[TrainStepFn]) -> ResolvedPrecision:
        """Resolve the run's precision (FEAT-028) — afresh, so the record
        carries TF32 as it is now — and refuse a step function that cannot
        apply a reduced precision (the built-in imperative paradigm steps,
        which run in full precision) before any work is done."""
        # A custom step receives the precision (ctx.precision) but applies it
        # itself: the record claims training only where NNx applies it.
        trained = (_PRECISION_TRAIN,) if train_step_fn is None or train_step_fn is default_train_step else ()
        precision = self._resolve_run_precision((*trained, _PRECISION_EVALUATE, _PRECISION_PREDICT))
        _check_step_precision(train_step_fn, precision)
        return precision

    def _resolve_run_precision(self, covers: tuple[str, ...]) -> ResolvedPrecision:
        """Resolve the policy afresh for a run (TF32 as it is now), scoped to
        the surfaces the run applies it to — the legacy flag keeps its
        training-only scope — and cache it for evaluation and prediction
        (shared by ``NNModel.train`` and ``Trainer.train``)."""
        precision = resolve_precision(self.params, self.device)
        if precision.source != "legacy":
            precision = precision.scoped(covers)
        self._precision = (_precision_key(self.params, self.device), precision)
        return precision

    def _train_impl(
        self,
        params: NNTrainParams,
        run: NNRun,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[list[CallbackLike]] = None,
        train_step_fn: Optional[TrainStepFn] = None,
        eval_step_fn: Optional[EvalStepFn] = None,
        components: Optional[list[Any]] = None,
        objective: Optional[Callable[[Any], Any]] = None,
        history: Optional[HistoryJournal] = None,
        *,
        precision: ResolvedPrecision,
        scaler: Optional[Any],
    ) -> NNRun:
        """Run the training loop and return the resulting NNRun.

        Args:
            params: dataloaders + optim + scheduler + epochs + seed. The
                train_loader is required; val_loader is optional (skips the
                per-epoch evaluation when absent).
            callbacks: optional list of `Callback` instances (or legacy
                `Callable[[List[IDP]], None]` for back-compat). Each hook
                runs at the documented lifecycle point (on_train_begin,
                on_epoch_begin/end, on_train_end).
            train_step_fn: optional override for the per-batch training
                step. When None (default), runs `default_train_step` —
                supervised forward → loss_fn(net(X), Y) → backward → step.
            eval_step_fn: optional override for the per-epoch VALIDATION
                pass (#86), symmetric with train_step_fn. When set (and a
                val_loader is present), each epoch calls
                ``eval_step_fn(EvalStepContext(...))`` under no-grad and
                records its EDP as val_edp — instead of the built-in
                classification ``evaluate()`` (which argmaxes logits and is
                meaningless for LM/DPO/regression val). When None (default),
                behavior is unchanged. Ignored when val_loader is None.
                The hook owns iteration and aggregation across the complete
                validation loader carried by `EvalStepContext`. See
                `docs/concepts.md` and `examples/26_custom_eval_step.py`.

        Returns:
            An `NNRun` with per-iteration `idps`, persisted under
            `runs/<run.id>/` along with per-tag checkpoints. The same
            object is returned with the in-memory idps list attached.

        Raises:
            ValueError: if `params` is None, `params.train_loader` is
                None, or `params.optim` is invalid.
            FloatingPointError: from `default_train_step` if training
                loss becomes non-finite (custom `train_step_fn` hooks are
                responsible for their own divergence checks).
        """
        assert params.train_loader is not None
        train_loader = params.train_loader
        validate: bool = params.val_loader is not None
        compile_session = self._compile_session  # FEAT-029: None for an eager fit
        ddp = self._ddp_session  # FEAT-030: None for a single-process fit
        writes = ddp is None or ddp.writer  # only the writer rank persists anything
        from ..optimizers import _canonical_factory_state, optimizer_factory_state

        # FEAT-005: checkpointable components — callbacks, the step functions
        # and explicit ones — are registered (names checked for uniqueness)
        # before anything is restored or trained.
        normalized_callbacks = self._normalize_callbacks(callbacks)
        registry = ComponentRegistry.discover(
            normalized_callbacks, train_step_fn, eval_step_fn, objective, explicit=list(components or [])
        )
        # FEAT-003: declared metrics and the run's monitor (validated in
        # train()); the whole-epoch summary is kept when anything monitors.
        monitor = params.monitor.resolve(params.metrics) if params.monitor is not None else None
        tracker = MonitorTracker(monitor, warn_missing=True) if monitor is not None else None
        if tracker is not None:
            # Its best continues across a stateful resume (component state).
            registry.register(tracker)
        summarize = (
            bool(params.metrics)
            or monitor is not None
            or any(isinstance(getattr(cb, "monitor", None), MonitorSpec) for cb in normalized_callbacks)
        )
        metric_domain, metric_ignore_index, metric_threshold, _ = _metric_context(self)
        # Declared metrics over the training epoch need the default step's
        # outputs; step functions and objectives record them on validation.
        train_metrics = params.metrics if train_step_fn is None and objective is None else ()
        component_plan = None
        resume_status = ResumeStatus()
        previous_net_state: Optional[dict[str, Any]] = None
        previous_rng_state: Optional[dict[str, Any]] = None

        resume_optimizer_factory = optimizer_factory_state(params.optim)
        resume_optimizer_topology = _optimizer_topology(optimizer, self.net)
        # A weights-only resume starts a fresh schedule and needs no shared horizon.
        stateful_resume = params.resume_from_run_id is not None and params.resume_mode != "weights_only"
        if stateful_resume:
            _check_resume_horizon(params.scheduler, n_epochs=params.n_epochs)
        # FEAT-014: an optimizer_update clock steps on committed updates; its
        # default horizon is the planned updates when NNx owns the windows
        # (the default step or an objective) and the loader has a length.
        update_clock = uses_update_clock(params.scheduler)
        owns_windows = train_step_fn is None or train_step_fn is default_train_step
        n_updates = (
            planned_updates(params.train_loader, params.optim.accumulate_grad_batches, params.n_epochs)
            if update_clock and owns_windows
            else None
        )
        # The plan reaches _build_scheduler without a new argument, so a
        # subclass override of _build_scheduler(optimizer, params) that
        # calls super() still gets the default horizon.
        self._planned_scheduler_updates = n_updates
        try:
            built = self._build_scheduler(optimizer, params)
        finally:
            del self._planned_scheduler_updates
        scheduler = _monitored_plateau(built, optimizer, monitor)
        clock: Optional[SchedulerClock] = None
        if update_clock:
            clock = SchedulerClock.for_schedule("default", scheduler, params.scheduler, planned=n_updates)
            registry.register(clock)
        # FEAT-004: an objective's updates belong to the shared engine; its
        # committed-update counters are component state (nnx.update_engine),
        # so they continue across a stateful resume.
        engine = None
        if objective is not None:
            engine = _objective_engine(
                objective,
                optimizers={"default": optimizer},
                clip_norms={"default": params.optim.grad_clip_norm},
                scaler=scaler,
                precision=precision,
            )
            registry.register(engine)
        start_epoch = 0
        # #394: the epochs this session trains, and the logical position a
        # planned resume continues from.
        session_epochs = params.n_epochs
        restored_counters: Mapping[str, Any] = {}

        # Warm resume restores every stateful training component when the
        # source checkpoint has a versioned sidecar. Legacy optimizer-only
        # sidecars remain supported.
        # FEAT-030: under DDP every rank resumes — or fails — together: a
        # failure on any rank (a missing component, a mismatch) raises on all.
        try:
            with contextlib.nullcontext() if ddp is None else _collectively("resuming the run"):
                if params.resume_from_run_id is not None:
                    source = _load_resume_source(
                        params.resume_from_run_id,
                        params.resume_from_checkpoint,
                        params.resume_mode,
                        trainer=False,
                        live_transforms=self._topology_transforms,
                        ddp=ddp,
                    )
                    training_state = source.training_state
                    if training_state is not None:
                        expected_optimizer = training_state.get("optimizer_type")
                        expected_scheduler = training_state.get("scheduler_type")
                        if expected_optimizer is not None and expected_optimizer != _component_type(optimizer):
                            raise ValueError(
                                f"resume optimizer type mismatch: checkpoint has {expected_optimizer}, "
                                f"configuration builds {_component_type(optimizer)}"
                            )
                        if expected_scheduler is not None and expected_scheduler != _component_type(scheduler):
                            raise ValueError(
                                f"resume scheduler type mismatch: checkpoint has {expected_scheduler}, "
                                f"configuration builds {_component_type(scheduler)}"
                            )
                        # A registered factory is identified by id, version and config
                        # (None for a built-in, and for sidecars written before
                        # factories existed): any difference is a different optimizer.
                        expected_factory = training_state.get("optimizer_factory")
                        if _canonical_factory_state(expected_factory) != _canonical_factory_state(
                            resume_optimizer_factory
                        ):
                            raise ValueError(
                                f"resume optimizer factory mismatch: checkpoint has {expected_factory}, "
                                f"configuration builds {resume_optimizer_factory}"
                            )
                        expected_topology = training_state.get("optimizer_topology")
                        if expected_topology is not None and expected_topology != resume_optimizer_topology:
                            raise ValueError("resume optimizer parameter topology does not match the checkpoint")
                        _check_resume_precision(training_state, precision, scaler)
                        _check_plateau_resume(training_state.get("scheduler"), scheduler, monitor)
                        completed_epoch = training_state.get("completed_epoch")
                        if completed_epoch is not None:
                            start_epoch = int(completed_epoch) + 1
                        counters = training_state.get("counters") or {}
                        if (
                            source.manifest is not None
                            and source.manifest.get("checkpoint_id") != source.checkpoint.training_state_id
                        ):
                            raise ResumePointError(
                                f"resume_from_run_id={params.resume_from_run_id!r}/{source.label}: the resume point "
                                "changed while it was being read — retry once its writer has finished"
                            )
                        session_epochs = _session_epochs(
                            params, start_epoch, counters, f"resume_from_run_id={params.resume_from_run_id!r}"
                        )
                        if params.resume_epochs == "planned":
                            restored_counters = counters
                            if clock is not None:
                                # The scheduler's horizon is the whole plan; the
                                # clock's budget check needs the updates still to run.
                                clock.planned = (
                                    planned_updates(train_loader, params.optim.accumulate_grad_batches, session_epochs)
                                    if owns_windows
                                    else None
                                )
                        _check_resume_horizon(params.scheduler, n_epochs=session_epochs, start_epoch=start_epoch)
                        # Worker capability is decided BEFORE any state is restored:
                        # ordinary training accepts any re-iterable batch source (a
                        # list, NNGraphDataset's one-element full-batch list, ...),
                        # which has no `num_workers`. Absent metadata means "no
                        # worker-local RNG to worry about"; a real DataLoader with
                        # workers keeps its warning. Nothing here iterates the source.
                        # Components are validated against the checkpoint before any
                        # state is mutated (one report listing every problem).
                        component_plan = _plan_component_restore(registry, training_state)
                        warn_worker_rng = (
                            training_state.get("rng") is not None and _loader_num_workers(train_loader) > 0
                        )
                        # FEAT-030: same world size and partition, or nothing is restored.
                        rank_rng: Optional[Mapping[str, Any]] = None
                        if ddp is None and training_state.get("distributed") is not None:
                            raise ValueError(
                                f"this checkpoint was written by a {training_state['distributed']['world_size']}-rank "
                                "distributed run: resume it with train(distributed=DDP()) on the same world size, or "
                                "start from its weights with resume_mode='weights_only'"
                            )
                        if ddp is not None:
                            # Agreed by the enclosing "resuming the run" block: a
                            # collective block is never nested in another (a rank
                            # failing before the inner one would skip its gather).
                            rank_rng = _check_distributed_resume(training_state, ddp, train_loader)
                        previous_net_state = net_snapshot = _snapshot_state_dict(self.net.state_dict())
                        previous_rng_state = rng_snapshot = _capture_rng_state(train_loader)
                        try:
                            self.net.load_state_dict(source.net_state)
                            optimizer.load_state_dict(training_state["optimizer"])
                            if training_state.get("scheduler") is not None:
                                scheduler.load_state_dict(training_state["scheduler"])
                            if scaler is not None and training_state.get("scaler") is not None:
                                scaler.load_state_dict(training_state["scaler"])
                            if training_state.get("rng") is not None:
                                _restore_rng_state(training_state["rng"], train_loader)
                            if rank_rng is not None:  # this rank's own streams (FEAT-030)
                                _restore_rng_state(dict(rank_rng), train_loader)
                        except BaseException:
                            self.net.load_state_dict(net_snapshot)
                            _restore_rng_state(rng_snapshot, train_loader)
                            raise
                        if warn_worker_rng:
                            warnings.warn(
                                "exact warm-resume continuity requires train_loader.num_workers=0; "
                                "worker-local RNG state cannot be reconstructed",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                        resume_status = ResumeStatus(
                            mode="stateful",
                            source_run_id=params.resume_from_run_id,
                            source_checkpoint=source.label,
                            source_epoch=source.checkpoint.idp.epoch_idx,
                            fresh_components=tuple(component_plan.fresh),
                            **_manifest_lineage(source.manifest),
                        )
                    else:
                        # Epoch numbering continues after the checkpoint's epoch; the
                        # optimizer, scheduler, scaler and components start fresh.
                        start_epoch = source.checkpoint.idp.epoch_idx + 1
                        session_epochs = _session_epochs(
                            params, start_epoch, None, f"resume_from_run_id={params.resume_from_run_id!r}"
                        )
                        _restore_weights_only(
                            self.net,
                            source,
                            train_loader,
                            params.resume_mode,
                            fresh="optimizer, scheduler, scaler, RNG and component state",
                        )
                        resume_status = ResumeStatus(
                            mode="weights_only",
                            source_run_id=params.resume_from_run_id,
                            source_checkpoint=source.label,
                            source_epoch=source.checkpoint.idp.epoch_idx,
                            fresh_components=registry.names,
                            **_manifest_lineage(source.manifest),
                        )
        except BaseException:
            # Another rank's failure reaches a rank that restored cleanly only
            # through the agreement: undo its restore too, like the failing one.
            if ddp is not None and previous_net_state is not None and previous_rng_state is not None:
                _rollback_resume(self.net, previous_net_state, previous_rng_state, train_loader)
            raise

        # Every record in a list (idps.csv), or a bounded window plus the run's
        # history journal (FEAT-036); either way saved before LAST each epoch.
        records = _training_history(run, history) if writes else _ReplicaHistory()
        # `len()` is not defined on iterable-style DataLoaders (IterableDataset).
        # Fall back to None so tqdm renders without a total instead of crashing.
        try:
            n_iter: Optional[int] = int(session_epochs * len(cast(Sized, train_loader)))
        except TypeError:
            n_iter = None
        best_checkpoint: Optional[NNCheckpoint] = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)

        if writes:
            Utils.print_table(header=False, title="Run Details...", data=Utils.flatten_dict(data=run.state()))

        ctx = _CallbackContext(model=self, run=run, optimizer=optimizer)
        ctx.scheduler = scheduler  # read-only view for callbacks (#394)
        ctx.history_retention = history.retention if history is not None else None
        ctx.history_records = records
        # Default to the standard supervised step when the caller doesn't
        # override. Custom step gets dispatched from inside the batch loop
        # below so the rest of train() (scheduler, callbacks, checkpoint
        # cadence, val loop, incremental save) is identical either way.
        # Explicit None check (not `or`) so a hypothetical callable that
        # happens to be falsy by __bool__ doesn't silently fall back.
        step_fn: TrainStepFn = default_train_step if train_step_fn is None else train_step_fn
        # FEAT-014: a step's committed updates drive an optimizer_update
        # clock (an objective's engine reports to the clock directly).
        clock_report: Callable[[], None] = (
            clock.report_update if clock is not None and engine is None else NO_UPDATE_REPORTER
        )

        def _announce_update() -> None:
            # FEAT-033: count the committed update and tell the listeners.
            ctx.committed_updates += 1
            for listener in list(ctx.update_listeners):
                listener(ctx)

        class _ReportUpdate:
            """The step's ``report_update``: the clock's (refusing an
            optimizer name, as ever), then the update listeners. ``listening``
            says whether anything needs a judged step (``listens``)."""

            def __call__(self, *names: Any) -> None:
                clock_report(*names)
                _announce_update()

            @property
            def listening(self) -> bool:
                return listens(clock_report) or bool(ctx.update_listeners)

        report_update: Callable[..., None] = _ReportUpdate()

        if engine is not None:
            assert objective is not None
            # Committed updates are announced to every callback.
            engine.listeners.append(lambda event: _dispatch_update(normalized_callbacks, ctx, event))
            engine.listeners.append(lambda event: _announce_update())
            if clock is not None:
                # After the callbacks: they see the learning rate the update
                # was taken with; the clock then steps the schedule.
                engine.listeners.append(lambda event: clock.committed())
            step_fn = _ObjectiveStep(objective, engine)
            ctx.update_count = engine.commits

        # #394: a planned resume continues the logical step and update counters.
        idx_iter = int(restored_counters.get("global_step") or 0)
        ctx.committed_updates = int(restored_counters.get("committed_updates") or 0)
        stopped_mid_epoch = False
        pre_transform_net_state: Optional[dict[str, Any]] = None
        pre_transform_rng_state: Optional[dict[str, Any]] = None
        # Respect NNX_TQDM_DISABLE=1 in tests / CI / non-TTY environments so
        # the progress bar doesn't pollute output. Same env var works as
        # well in subprocess contexts where the user can't pass a flag.
        tqdm_disabled = os.environ.get("NNX_TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
        with (
            torch.set_grad_enabled(True),
            tqdm(colour="blue", total=n_iter, desc="Training", disable=tqdm_disabled) as tqdm_bar,
            _CallbackFinalizer(normalized_callbacks, ctx) as callback_lifecycle,
        ):
            callback_lifecycle.start()
            # FEAT-033: under DDP, whether any rank listens for updates (and
            # may request a stop at one) is agreed once, not per batch.
            ddp_listening = ddp is not None and any(ddp.gather(bool(ctx.update_listeners)))
            # FEAT-005: reset hooks (on_train_begin) have run once; now the
            # validated component states are restored, all-or-nothing, before
            # the first resumed epoch.
            if component_plan is not None:
                try:
                    with contextlib.nullcontext() if ddp is None else _collectively("restoring components"):
                        restored = registry.restore(component_plan)
                except BaseException:
                    assert previous_net_state is not None and previous_rng_state is not None
                    _rollback_resume(self.net, previous_net_state, previous_rng_state, train_loader)
                    raise
                resume_status = replace(resume_status, restored_components=restored)
            if engine is not None:
                ctx.update_count = engine.commits  # continues after a stateful resume
            run = (
                run.with_resume_status(resume_status)
                .with_precision(precision)
                .with_compile(_record_of(compile_session))
            )
            ctx.run = run
            for local_epoch in range(session_epochs):
                idx_epoch = start_epoch + local_epoch
                ctx.epoch = idx_epoch
                _set_loader_epoch(train_loader, idx_epoch)
                if ddp is not None:
                    # FEAT-030: unequal step counts fail on every rank, never hang.
                    ddp.check_steps(len(cast(Sized, train_loader)), idx_epoch)
                with contextlib.nullcontext() if ddp is None else _collectively(f"epoch {idx_epoch}'s start"):
                    for cb in normalized_callbacks:
                        cb.on_epoch_begin(ctx)

                records.begin_epoch()
                accumulation_state = GradientAccumulationState()
                if clock is not None:
                    clock.trace.clear()
                    updates_before_epoch = clock.count
                epoch_summary = (
                    _TrainEpochSummary(train_metrics, metric_domain, metric_ignore_index, metric_threshold)
                    if summarize
                    else None
                )
                for idx_batch, batch, is_last_batch in _enumerate_with_last(train_loader):
                    step_ctx = TrainStepContext(
                        model=self,
                        batch=batch,
                        optimizer=optimizer,
                        scaler=scaler,
                        grad_clip_norm=params.optim.grad_clip_norm,
                        extra_metrics=params.extra_metrics,
                        accumulate_grad_batches=params.optim.accumulate_grad_batches,
                        batch_idx=idx_batch,
                        precision=precision,
                        epoch_idx=idx_epoch,
                        is_last_batch=is_last_batch,
                        accumulation_state=accumulation_state,
                        epoch_summary=epoch_summary,
                        report_update=report_update,
                    )
                    # The rate this batch trains with, for an update clock
                    # (its scheduler may step inside the step function).
                    lr_used = float(optimizer.param_groups[0]["lr"]) if clock is not None else None
                    if compile_session is not None:
                        compile_session.where = {"phase": "train", "epoch": idx_epoch, "batch": idx_batch}
                    train_edp = step_fn(step_ctx)
                    if epoch_summary is not None:
                        epoch_summary.add(train_edp, _batch_sample_count(self.net, batch))

                    records.append(
                        NNIterationDataPoint(
                            iter_idx=idx_iter,
                            epoch_idx=idx_epoch,
                            batch_idx=idx_batch,
                            train_edp=train_edp,
                            lr=lr_used if lr_used is not None else optimizer.param_groups[0]["lr"],
                            update_count=engine.commits if engine is not None else None,
                        )
                    )

                    idx_iter += 1
                    tqdm_bar.update(1)

                    if ddp is not None and ddp_listening:
                        # FEAT-030: a stop request is decided on every rank alike
                        # (gathered only when some rank listens for updates).
                        ctx.stop_at_update = any(ddp.gather(bool(ctx.stop_at_update)))
                    # Under DDP a mid-epoch stop needs the agreed request (no
                    # rank listened: the epoch-end gather decides it instead).
                    if ctx.stop_at_update and (ddp is None or ddp_listening):
                        if is_last_batch:
                            ctx.should_stop = True  # the epoch is complete: commit it, then stop
                        else:
                            # FEAT-033: stop at this update boundary; the epoch in
                            # progress is discarded, never committed.
                            stopped_mid_epoch = True
                            records.discard_epoch()
                            break

                if stopped_mid_epoch:
                    break

                if records.epoch_is_empty():
                    # Zero batches this epoch: first epoch would crash on
                    # records.last below; later epochs would silently attach
                    # this epoch's val_edp to the PREVIOUS epoch's last
                    # idp and reuse its stale train_edp.
                    raise ValueError(
                        f"train_loader yielded no batches in epoch {idx_epoch} — check batch_size vs "
                        "dataset size with drop_last=True, or whether the loader is a one-shot iterable."
                    )

                if validate and compile_session is not None:
                    compile_session.where = {"phase": "validation", "epoch": idx_epoch}
                if validate and ddp is not None:
                    # FEAT-030: each rank scores its own shard; one gather
                    # gives every rank the same global record.
                    val_edp = _distributed_validation(self, ddp, params)
                elif validate and eval_step_fn is not None:
                    assert params.val_loader is not None
                    # #86: pluggable validation step (mirrors train_step_fn) —
                    # LM/DPO/regression val metrics computed INSIDE the loop so
                    # they persist through the incremental run save below.
                    with torch.no_grad():
                        val_edp = eval_step_fn(
                            EvalStepContext(
                                model=self,
                                val_loader=params.val_loader,
                                extra_metrics=params.extra_metrics,
                                epoch_idx=idx_epoch,
                                metrics=params.metrics,
                            )
                        )
                elif validate:
                    assert params.val_loader is not None
                    # `metrics=` only when declared: evaluate() overrides written
                    # before FEAT-003 keep their (loader, extra_metrics) call.
                    val_edp = (
                        self.evaluate(
                            loader=params.val_loader, extra_metrics=params.extra_metrics, metrics=params.metrics
                        )
                        if params.metrics
                        else self.evaluate(loader=params.val_loader, extra_metrics=params.extra_metrics)
                    )
                else:
                    val_edp = None
                records.replace_last(records.last.with_val_edp(val_edp))
                record: Optional[MonitorRecord] = None
                if epoch_summary is not None:
                    train_summary = epoch_summary.result()
                    if tracker is not None:
                        assert monitor is not None
                        value = monitor.value(train=train_summary or train_edp, val=val_edp)
                        record = tracker.observe(value, epoch=idx_epoch)
                    records.replace_last(records.last.with_epoch_summary(train_summary, record))

                if clock is not None:
                    # FEAT-014: stepped on committed updates, not here.
                    if clock.count == updates_before_epoch and not owns_windows and engine is None:
                        warnings.warn(
                            f"epoch {idx_epoch}: the step function reported no optimizer update, so the "
                            "optimizer_update-clock scheduler did not step; call ctx.report_update() after each "
                            "optimizer.step() the step function takes itself (default_train_step and finalize_step "
                            "report theirs, and no report is due for an epoch whose updates were all skipped)",
                            UserWarning,
                            stacklevel=4,
                        )
                elif record is not None and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    _step_monitored_plateau(scheduler, record)
                else:
                    self._step_scheduler(scheduler, val_edp, train_edp, epoch_idx=idx_epoch)
                # The epoch's per-update learning rates (empty on the epoch clock).
                ctx.update_lrs = list(clock.trace) if clock is not None else []

                ctx.idp = records.last
                ctx.deferred_checkpoint_writes.clear()
                # ctx.idps: the running list, or the journal's window (the whole
                # history, read back, for a callback declaring history_access="full").
                if ddp is None:
                    _dispatch_epoch_end(normalized_callbacks, ctx, records)
                else:
                    # FEAT-030: a writer-only callback failing (a full disk,
                    # say) stops every rank; and every rank stops together.
                    with _collectively(f"epoch {idx_epoch}'s callbacks"):
                        _dispatch_epoch_end(normalized_callbacks, ctx, records)
                    ctx.should_stop = any(ddp.gather(bool(ctx.should_stop or ctx.stop_at_update)))

                # Prepare run history first (idps.csv, or the journal's chunks
                # and manifest); the checkpoint is the epoch's commit marker and
                # is never allowed to get ahead of the history.
                # FEAT-029: the run and the checkpoint carry the compile
                # record as it stands now (an eager restart included).
                run = run.with_compile(_record_of(compile_session))
                ctx.run = run
                # FEAT-030: every rank's RNG streams go into the writer's
                # checkpoint (a collective, so every rank takes part).
                distributed_state = (
                    None
                    if ddp is None
                    else _distributed_state(
                        ddp, train_loader, params.val_loader, ddp.gather(_capture_rng_state(train_loader))
                    )
                )
                commit_error: Optional[BaseException] = None
                checkpoint: Optional[NNCheckpoint] = None
                try:
                    if writes:
                        checkpoint = self._commit_epoch(
                            records=records,
                            run=run,
                            ctx=ctx,
                            idx_epoch=idx_epoch,
                            # Phase tags follow the plan's logical epochs on a
                            # planned resume (#394), the session's otherwise.
                            local_epoch=idx_epoch if params.resume_epochs == "planned" else local_epoch,
                            params=params,
                            counters={
                                "global_step": idx_iter,
                                "committed_updates": ctx.committed_updates,
                                "planned_n_epochs": start_epoch + session_epochs,
                            },
                            best_checkpoint=best_checkpoint,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            precision=precision,
                            compile_session=compile_session,
                            train_loader=train_loader,
                            resume_optimizer_factory=resume_optimizer_factory,
                            registry=registry,
                            record=record,
                            distributed_state=distributed_state,
                        )
                except Exception as caught:
                    if ddp is None:
                        raise
                    commit_error = caught
                if ddp is not None:
                    from ..distributed import agree

                    # The writer's commit, agreed: a failure stops every rank,
                    # and replicas never run ahead of what is on disk.
                    agree(commit_error, f"committing epoch {idx_epoch}")
                # In-memory best_checkpoint tracking must use the same
                # comparison as the on-disk BEST write inside
                # _save_checkpoints (val→train, error→loss, +inf fall-through).
                # Without this, val_loader=None runs would silently overwrite
                # best_checkpoint every epoch (because checkpoint.idp.val_edp
                # is None there) while the on-disk BEST tracks training error,
                # diverging the two views of "best". A DDP replica writes no
                # checkpoint and tracks none.
                if checkpoint is None:
                    pass
                elif record is not None:
                    # FEAT-003: the monitor decides BEST (same rule as the
                    # on-disk BEST write above).
                    if record.improved:
                        best_checkpoint = checkpoint
                elif best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
                    best_checkpoint = checkpoint

                self._update_tqdm_postfix(tqdm_bar, optimizer, val_edp, train_edp, record)

                if ctx.should_stop or ctx.stop_at_update:
                    # A stop requested at the epoch boundary (on_epoch_end)
                    # stops here: the epoch is committed already.
                    break

            pre_transform_net_state = _snapshot_state_dict(self.net.state_dict())
            pre_transform_rng_state = _capture_rng_state(train_loader)

        # #87: on_train_end callbacks (fired in _CallbackFinalizer.__exit__ as
        # the with-block closed above) may mutate self.net — QAT's convert()
        # swaps Linear modules for quantized ones. Every in-loop LAST save
        # predates that, so re-save LAST from the live net so the on-disk
        # artifact matches the post-on_train_end model. Unconditional rather
        # than diffed: state_dict() returns live tensor references, so an
        # in-place value mutation would compare equal against the stale
        # in-memory checkpoint even though the disk copy is pre-mutation.
        # Costs one extra checkpoint write per training run. BEST is
        # deliberately untouched — it tracks the best *training-time* state.
        final_error: Optional[BaseException] = None
        # FEAT-030: every rank's RNG for the final LAST, gathered outside the
        # agreed block so a rank failing inside it can never skip the gather
        # (distributed runs have no topology transforms: the live streams).
        # FEAT-033: after a mid-epoch stop the live tensors belong to no
        # committed epoch: LAST is left as the last commit wrote it.
        refresh_last = bool(records) and not stopped_mid_epoch
        final_rng = ddp.gather(_capture_rng_state(train_loader)) if ddp is not None and refresh_last else None
        try:
            if refresh_last:
                self._save_final_last(
                    records=records,
                    run=run,
                    normalized_callbacks=normalized_callbacks,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    precision=precision,
                    compile_session=compile_session,
                    ddp=ddp,
                    writes=writes,
                    params=params,
                    train_loader=train_loader,
                    pre_transform_net_state=pre_transform_net_state,
                    pre_transform_rng_state=pre_transform_rng_state,
                    resume_optimizer_topology=resume_optimizer_topology,
                    resume_optimizer_factory=resume_optimizer_factory,
                    registry=registry,
                    final_rng=final_rng,
                    # No batch runs after the last commit, so its counters hold.
                    counters={
                        "global_step": idx_iter,
                        "committed_updates": ctx.committed_updates,
                        "planned_n_epochs": start_epoch + session_epochs,
                    },
                )
            saved = records.finish(run.with_compile(_record_of(compile_session)))
        except Exception as caught:
            if ddp is None:
                raise
            final_error = caught
        if ddp is not None:
            from ..distributed import agree

            agree(final_error, "the final commit")
        if writes:
            _print_run_saved(run.id)
        return saved

    def _save_final_last(
        self,
        *,
        records: Any,
        run: NNRun,
        normalized_callbacks: list[Callback],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        precision: ResolvedPrecision,
        compile_session: Optional[_CompileSession],
        ddp: Any,
        writes: bool,
        params: NNTrainParams,
        train_loader: Any,
        pre_transform_net_state: Optional[dict[str, Any]],
        pre_transform_rng_state: Optional[dict[str, Any]],
        resume_optimizer_topology: Any,
        resume_optimizer_factory: Any,
        registry: ComponentRegistry,
        final_rng: Optional[list[Any]] = None,
        counters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Re-save LAST from the live net after ``on_train_end`` (see #87)."""
        final_transforms, keeps_pre_transform = _final_transforms(self, normalized_callbacks, run.transforms)
        self._topology_transforms = final_transforms
        rng_state = pre_transform_rng_state if keeps_pre_transform else _capture_rng_state(train_loader)
        distributed_state = (
            None if ddp is None else _distributed_state(ddp, train_loader, params.val_loader, final_rng or [])
        )
        if not writes:
            return
        NNCheckpoint(
            idp=records.last,
            model_params=self.params,
            net_params=self.net_params,
            net_state=self.net.state_dict(),
            transforms=final_transforms,
        ).save(
            run=run.id,
            type=Checkpoints.LAST,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict() if scaler is not None else None,
            precision=precision.record(),
            compile=_state_of(compile_session),
            distributed=distributed_state,
            rng_state=rng_state,
            completed_epoch=records.last.epoch_idx,
            resume_net_state=pre_transform_net_state if keeps_pre_transform else None,
            optimizer_type=_component_type(optimizer),
            scheduler_type=_component_type(scheduler),
            optimizer_topology=resume_optimizer_topology,
            optimizer_factory=resume_optimizer_factory,
            components=registry.collect(),
            counters=counters,
        )

    def evaluate(
        self, loader: Iterable[Any], extra_metrics=None, metrics: Sequence[MetricSpec] = ()
    ) -> NNEvaluationDataPoint:
        """Aggregate predictions across all batches in `loader` and compute
        a single NNEvaluationDataPoint. Aggregating (rather than averaging
        per-batch metrics) gives correct sample-weighted f1/precision/recall
        when the final batch is short.

        `extra_metrics` ({name -> callable(y_true, y_pred) -> float}) are
        called once on the aggregate truth / decoded predictions.

        `metrics` (FEAT-003) are declared :class:`~nnx.MetricSpec`s,
        accumulated over every valid sample of the loader from the input
        each declares (decoded labels, probabilities or continuous outputs)
        and reported in the record's ``metrics`` under each spec's name. A
        metric whose input this model cannot provide raises before any
        batch is read.

        Raises ValueError if the loader yields zero batches — previously
        produced NaN metrics silently from np.mean over an empty list.
        """
        return _evaluate(self, loader, extra_metrics, tuple(metrics), bounded=False, who="evaluate()")

    def predict(self, X) -> PredictResult:
        """Run the network in eval mode and return logits + argmax classes.

        Accepts any of:

        - ``np.ndarray`` (single input tensor) — historical API.
        - ``tuple[np.ndarray, ...]`` — for multi-input networks.
        - ``torch.Tensor`` / ``tuple[torch.Tensor, ...]`` — skips the numpy
          conversion when callers already have tensors.
        - ``DataLoader`` — iterates the loader, runs predictions per batch,
          concatenates and returns the full result. Y labels in the batch
          (if present) are ignored.

        Returns a ``PredictResult`` (a ``NamedTuple`` of (logits, classes))
        that unpacks like the original 2-tuple. For probabilities, decoded
        labels and per-row sample ids, use :meth:`predict_proba`.

        Non-destructive: ``self.net.training`` is snapshotted before
        switching to ``eval()`` and restored on exit (matches
        ``NNModel.evaluate``, ``nnx.viz.activation_map``, and
        ``nnx.lr_finder``). Without this, a caller doing the common
        train → predict → train-more pattern silently leaves the net
        in ``.eval()`` mode.
        """
        logits, _ = self._predict_logits(X, caller="predict()")
        return PredictResult(logits=logits, classes=self._decode_classes(logits))

    def _decode_classes(self, logits: np.ndarray) -> np.ndarray:
        """``predict().classes`` for raw logits: the task's decoding, else
        ``logit >= 0`` for ``BCEWithLogitsLoss`` and the argmax over the class
        axis (class-last for a transformer's token logits). Row-wise, so a
        batch decodes like the rows of the whole."""
        adapter = getattr(self, "task_adapter", None)
        if adapter is not None:
            return adapter.decode_array(logits)
        class_axis = -1 if self.params.net is Nets.TRANSFORMER and logits.ndim > 2 else 1
        return (
            (logits >= 0).astype(np.int64)
            if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss)
            else logits.argmax(axis=class_axis)
        )

    def iter_predict(
        self, X: Iterable[Any], spec: Optional[ProbabilitySpec] = None, *, rich: bool = False
    ) -> PredictionStream:
        """Stream predictions one loader batch at a time (FEAT-020).

        Returns a :class:`~nnx.streaming.PredictionStream` — use it as a
        context manager — over ``X``, a ``DataLoader`` or another iterable of
        batches (in-memory arrays and tensors go to :meth:`predict`). Each
        item is a :class:`~nnx.streaming.PredictionBatch` of ``logits``,
        ``classes`` and ``sample_ids``, or, with a ``spec`` (or ``rich=True``
        for a model with a task), a :class:`~nnx.prediction.PredictionResult`
        as :meth:`predict_proba` builds it. The batches follow loader order,
        and concatenated they are exactly the eager result for the same
        ``DataLoader`` (the eager calls read other iterables as one in-memory
        input): ``predict(X)``'s logits and classes and ``predict_proba(X)``'s
        sample ids, graph seed-row slicing included.

        Each batch runs in eval mode under ``no_grad``, and every submodule's
        training mode is restored before the batch is yielded or its error
        raised. The stream holds only the batch in flight; closing it drops
        its references to the loader's iterator and the model and ends the
        iteration, and a closed or consumed stream cannot be iterated again.
        An empty loader yields no batches (the eager calls raise instead).
        Over a shuffling ``DataLoader``, the first batch whose sample ids are
        iteration positions warns (graph seed rows carry global node indices,
        graph-level rows their own graph ids).
        """
        from ..prediction import _check_spec_fits, prediction_from_logits
        from ..streaming import PredictionBatch, PredictionStream, _as_probability_spec, _check_stream_source

        _check_stream_source(X)
        explicit = _as_probability_spec(spec)
        adapter = getattr(self, "task_adapter", None)
        if explicit is None and rich and adapter is None:
            raise TypeError(
                "iter_predict(rich=True) needs a ProbabilitySpec for a model without a task "
                "(or declare NNModelParams(task=TaskSpec...))"
            )

        warn_as = "iter_predict()" if _shuffles(X) else None  # every batch carries sample_ids

        def batches() -> Iterator[Any]:
            if explicit is not None:
                declared = explicit
                for logits, ids in self._logit_batches(
                    X,
                    check_first=lambda first: _check_spec_fits(first, declared),
                    positional_warning=warn_as,
                    caller="iter_predict()",
                ):
                    yield prediction_from_logits(logits, declared, sample_ids=ids)
            elif rich:
                assert adapter is not None
                for logits, ids in self._logit_batches(
                    X, check_first=adapter.check_logits, positional_warning=warn_as, caller="iter_predict()"
                ):
                    yield adapter.prediction(logits, ids)
            else:
                for logits, ids in self._logit_batches(X, positional_warning=warn_as, caller="iter_predict()"):
                    yield PredictionBatch(logits=logits, classes=self._decode_classes(logits), sample_ids=ids)

        return PredictionStream(batches())

    def predict_proba(self, X, spec: Optional[ProbabilitySpec] = None) -> PredictionResult:
        """Probability-aware prediction declared by an explicit ``spec``.

        ``spec`` may be omitted for a model with a task (FEAT-002): the task
        supplies it — softmax over axis 1 for ``categorical``, sigmoid
        decoded at the task's ``threshold`` for ``multilabel`` — and a
        ``regression`` task returns its continuous values with
        ``probabilities=None`` and ``spec=None`` (``decoded`` holds the
        values). An explicit ``spec`` always wins.

        Accepts the same inputs as :meth:`predict` (arrays, tensors, tuples
        and ``DataLoader``s, including graph loaders whose rows are sliced
        to seed nodes) with the same non-destructive eval-mode contract,
        and returns a :class:`~nnx.prediction.PredictionResult`: the same
        raw logits ``predict()`` returns, probabilities (softmax over
        ``spec.class_axis`` for ``"categorical"``, element-wise sigmoid for
        ``"bernoulli"``), decoded values and ``sample_ids``: the row index
        for arrays, tensors and tuples; the position in iteration order for
        an ordinary ``DataLoader`` (the dataset index only when the loader
        does not shuffle — a shuffling loader warns); and the global node
        index for graph seed rows. A spec that does not fit the logits is
        rejected on the first loader batch.

        The task kind is never inferred from the loss or the output shape.
        For the default decoding rules, a categorical spec's ``decoded``
        equals ``predict().classes`` when the class axes agree, and a
        bernoulli spec's equals the ``BCEWithLogitsLoss`` threshold.
        Raises :class:`~nnx.prediction.PredictionValidationError` for a
        class axis or label count that does not fit the logits, or for
        non-finite logits. Parameters and gradients are never touched.
        """
        from ..prediction import _check_spec_fits, prediction_from_logits

        if spec is None:
            adapter = getattr(self, "task_adapter", None)
            if adapter is None:
                raise TypeError(
                    "predict_proba() needs a ProbabilitySpec for a model without a task "
                    "(or declare NNModelParams(task=TaskSpec...))"
                )
            logits, sample_ids = self._predict_logits(
                X, caller="predict_proba()", check_first=adapter.check_logits, warn_positional=True
            )
            return adapter.prediction(logits, sample_ids)
        explicit = spec
        logits, sample_ids = self._predict_logits(
            X,
            caller="predict_proba()",
            check_first=lambda first: _check_spec_fits(first, explicit),
            warn_positional=True,
        )
        return prediction_from_logits(logits, explicit, sample_ids=sample_ids)

    def _predict_logits(
        self,
        X,
        *,
        caller: str,
        check_first: Optional[Callable[[np.ndarray], object]] = None,
        batches: bool = False,
        warn_positional: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared inference path of :meth:`predict` / :meth:`predict_proba`:
        raw logits (numpy) plus an ``int64`` sample id per row, computed in
        eval mode under ``no_grad`` with every submodule's training mode
        restored on exit (success or failure). ``check_first`` sees the
        first loader batch's logits, so a caller can reject them before the
        rest of the loader is run. ``batches=True`` treats any iterable of
        batches like a ``DataLoader``."""
        precision = _inference_precision(self)  # before eval(): a failure leaves the modes alone
        training_modes = _capture_training_modes(self.net)
        self.net.eval()
        try:
            if batches or isinstance(X, DataLoader):
                logits_chunks: list[np.ndarray] = []
                id_chunks: list[np.ndarray] = []
                # Eval mode once for the whole call; a stream restores it per batch.
                warn_as = caller if warn_positional and _shuffles(X) else None
                for logits, ids in self._logit_batches(
                    X, check_first=check_first, restore_each_batch=False, positional_warning=warn_as, caller=caller
                ):
                    logits_chunks.append(logits)
                    id_chunks.append(ids)
                if not logits_chunks:
                    raise ValueError(f"{caller} loader produced zero batches")
                return np.concatenate(logits_chunks), np.concatenate(id_chunks)

            def _to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.to(self.device)
                # Fall through to numpy → tensor for arrays and array-likes.
                return torch.from_numpy(np.asarray(x)).to(self.device)

            # FEAT-006: a mapping holds a keyword-input module's inputs.
            if isinstance(X, Mapping):
                args_t: tuple[Any, ...] = ()
                kwargs_t = {name: _to_tensor(value) for name, value in X.items()}
            else:
                # Single input (any of: ndarray, Tensor, or a tuple thereof).
                if not isinstance(X, tuple):
                    X = (X,)
                args_t, kwargs_t = tuple(_to_tensor(x) for x in X), {}

            with torch.no_grad(), precision.autocast():
                output = self._net_forward(args_t, kwargs_t)
            Y_hat_logits = precision.output(output).cpu().numpy()
            return Y_hat_logits, np.arange(Y_hat_logits.shape[0], dtype=np.int64)
        finally:
            _restore_training_modes(training_modes)

    def _logit_batches(
        self,
        X: Iterable[Any],
        *,
        check_first: Optional[Callable[[np.ndarray], object]] = None,
        restore_each_batch: bool = True,
        positional_warning: Optional[str] = None,
        caller: str = "predict()",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """``(logits, sample_ids)`` for each batch of ``X``, in order — the
        one batch path of :meth:`predict`, :meth:`predict_proba` and
        :meth:`iter_predict`. Each forward pass runs under ``no_grad``; with
        ``restore_each_batch`` it also runs in eval mode with every
        submodule's training mode restored right after it, success or failure
        (a stream), otherwise the caller holds eval mode for the whole loop.
        ``check_first`` sees the first batch's logits, so a caller can reject
        them before the rest of the loader runs. The forward runs in the
        model's inference precision (FEAT-028), and the logits are returned in
        full precision."""
        precision = _inference_precision(self)  # before eval(): a failure leaves the modes alone
        offset = 0
        checked = check_first is None
        for batch in X:
            training_modes = _capture_training_modes(self.net) if restore_each_batch else None
            if training_modes is not None:
                self.net.eval()
            try:
                with torch.no_grad():
                    with precision.autocast():
                        output = self._inference_forward(batch)[0]  # predict discards labels
                    logits = precision.output(output).cpu().numpy()
            finally:
                if training_modes is not None:
                    _restore_training_modes(training_modes)
            ids: Optional[np.ndarray] = None
            # NeighborLoader subgraphs: only the leading seed rows are this
            # batch's nodes (see GraphNNBase.seed_count) — without the slice,
            # predictions for sampled neighbors pollute the output and the row
            # count exceeds the loader's node set. Their identity is the
            # global node index.
            seed_count = getattr(self.net, "seed_count", None)
            if seed_count is not None:
                n_seed = seed_count(batch)
                if n_seed is not None:
                    logits = logits[:n_seed]
                    # NeighborLoader's `n_id` holds the global ids of the
                    # subgraph's nodes, seeds first; its `input_id` is only
                    # global when input_nodes was a mask. NNGraphDataset's
                    # full-graph batches carry the global ids in `input_id`.
                    node_ids = getattr(batch, "n_id", None)
                    if node_ids is None:
                        node_ids = getattr(batch, "input_id", None)
                    if node_ids is not None:
                        ids = np.asarray(node_ids[:n_seed].cpu(), dtype=np.int64)
            ids_of = getattr(self.net, "sample_ids", None)
            if ids is None and callable(ids_of):
                # Rows with their own identity (graph ids, FEAT-026): stable
                # through shuffling and concatenation.
                ids = np.asarray(torch.as_tensor(ids_of(batch)).cpu(), dtype=np.int64).reshape(-1)
                if ids.shape[0] != logits.shape[0]:
                    raise ValueError(f"{caller}: the network gave {ids.shape[0]} sample ids for {logits.shape[0]} rows")
            if ids is None:
                ids = np.arange(offset, offset + logits.shape[0], dtype=np.int64)
                if positional_warning is not None:  # a shuffling loader: these positions cannot be joined back
                    _warn_positional_ids(positional_warning)
                    positional_warning = None
            offset += logits.shape[0]
            if not checked:
                assert check_first is not None
                check_first(logits)
                checked = True
            yield logits, ids

    def _inference_forward(self, batch: Any) -> tuple[torch.Tensor, Any]:
        """``(output, target)`` for a batch that may hold inputs only: split
        as :meth:`_split_inference_batch` does, moved to the model's device
        and run through the network — ``predict()``'s per-batch forward, and
        ``ExperimentPlan.probe``'s. The caller sets the mode and grad
        context."""
        args, kwargs, target = self._split_inference_batch(batch)
        inputs = tuple(_to_device(value, self.device) for value in args)
        keywords = {name: _to_device(value, self.device) for name, value in kwargs.items()}
        return self._net_forward(inputs, keywords), target

    def _split_inference_batch(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
        """``(args, kwargs, target)`` of a batch that may hold inputs only —
        how ``predict()`` reads a loader batch and ``ExperimentPlan.probe``
        an example batch. A positional / keyword adapter (FEAT-006) splits
        it; a bare tensor or a 1-tuple is inputs only (target ``None``);
        supervised tuples and graph batches keep their model-specific
        unpacking (:meth:`_split_batch`)."""
        adapter = getattr(self, "_batch_adapter", None)
        if adapter is not None and not isinstance(adapter, _UnpackBatch):
            return adapter.split(batch)
        if isinstance(batch, torch.Tensor):
            return (batch,), {}, None
        if isinstance(batch, (tuple, list)) and len(batch) == 1:
            return (batch[0],), {}, None
        return self._split_batch(batch)

    def _split_batch(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
        """``(args, kwargs, target)`` of a batch: a built-in net's own
        ``unpack_batch``, else the model's batch adapter (FEAT-006)."""
        if getattr(batch, "_nnx_teacher_records", False) is True:
            raise TypeError(
                "this batch holds stored teacher probabilities (a TeacherBatch, FEAT-022), not (inputs, labels): "
                "train it with TeacherDataset.objective() (nnx.paradigms.offline_distillation) and score it with "
                "evaluate_offline; the supervised and live-teacher steps cannot use it"
            )
        adapter = getattr(self, "_batch_adapter", None)
        if adapter is None:
            inputs, target = cast(Any, self.net).unpack_batch(batch)
            return _as_inputs(inputs), {}, target
        return adapter.split(batch)

    def _net_forward(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.Tensor:
        """Call the net on device-placed inputs; an adapter turns the raw
        return value into the output tensor."""
        session = getattr(self, "_compile_session", None) or getattr(self, "_ddp_session", None)
        raw = self.net(*args, **kwargs) if session is None else session.forward(*args, **kwargs)
        adapter = getattr(self, "_batch_adapter", None)
        return raw if adapter is None else adapter.output(raw)

    def _fwd_pass(self, batch):
        """Standard supervised forward pass: unpack batch, move to device,
        run net, take argmax over class logits. Used by `default_train_step`
        and `evaluate()`; custom train_step_fn's may call this directly
        or roll their own forward pass."""
        X, Y, Y_hat_logits = self._fwd_outputs(batch)
        class_axis = -1 if self.params.net is Nets.TRANSFORMER else 1
        Y_hat = (
            (Y_hat_logits >= 0).to(dtype=torch.long)
            if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss)
            else Y_hat_logits.argmax(dim=class_axis)
        )

        return X, Y, Y_hat_logits, Y_hat

    def _fwd_outputs(self, batch):
        """Unpack a supervised batch, move it to the device and run the net:
        ``(X, Y, logits)`` with transformer outputs flattened to rows and
        graph outputs sliced to their seed rows — no decoding. Shared by
        `_fwd_pass` and the task-adapter path (FEAT-002)."""
        args, kwargs, Y = self._split_batch(batch)
        if Y is None:
            raise ValueError("the batch has no target to train or evaluate against (see the model's batch adapter)")

        X = tuple(_to_device(x, self.device) for x in args)
        inputs = {name: _to_device(value, self.device) for name, value in kwargs.items()}
        Y = Y.to(self.device)

        Y_hat_logits = self._net_forward(X, inputs)
        X = (*X, *inputs.values())
        if self.params.net is Nets.TRANSFORMER and Y_hat_logits.ndim > 2:
            if tuple(Y_hat_logits.shape) == tuple(Y.shape):
                Y_hat_logits = Y_hat_logits.reshape(-1, Y_hat_logits.size(-1))
                Y = Y.reshape(-1, Y.size(-1))
            elif tuple(Y_hat_logits.shape[:-1]) == tuple(Y.shape):
                Y_hat_logits = Y_hat_logits.reshape(-1, Y_hat_logits.size(-1))
                Y = Y.reshape(-1)
        # Graph nets score every node in the sampled subgraph, but only
        # the leading seed rows belong to this batch's split — see
        # GraphNNBase.seed_count for the leakage this prevents.
        seed_count = getattr(self.net, "seed_count", None)
        if seed_count is not None:
            n_seed = seed_count(batch)
            if n_seed is not None:
                Y_hat_logits = Y_hat_logits[:n_seed]
                Y = Y[:n_seed]
        return X, Y, Y_hat_logits

    def _train_step(
        self,
        batch,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        grad_clip_norm: Optional[float] = None,
        extra_metrics=None,
        accumulate_grad_batches: int = 1,
        batch_idx: int = 0,
    ) -> NNEvaluationDataPoint:
        """Thin wrapper around :func:`default_train_step` kept for back-compat
        with any code that calls ``model._train_step(batch, ...)`` directly
        (e.g., a notebook that pre-dates the ``train_step_fn`` hook).

        **The :meth:`train` loop does NOT call this method.** It builds a
        :class:`TrainStepContext` and dispatches to
        ``train_step_fn or default_train_step`` directly. A subclass that
        overrides ``_train_step`` will therefore be ignored by ``train()`` —
        if you want a custom training step for ``train()``, pass it as the
        ``train_step_fn=`` kwarg instead.
        """
        return default_train_step(
            TrainStepContext(
                model=self,
                batch=batch,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=grad_clip_norm,
                extra_metrics=extra_metrics,
                accumulate_grad_batches=accumulate_grad_batches,
                batch_idx=batch_idx,
                epoch_idx=0,
                # An explicit policy (FEAT-028) applies here as in train();
                # the legacy flag keeps its scaler-on-CUDA rule.
                precision=self.resolved_precision if self.resolved_precision.source == "policy" else None,
            )
        )

    # The planned committed updates while train() builds an
    # optimizer_update-clock scheduler (FEAT-014); None otherwise.
    _planned_scheduler_updates: Optional[int] = None

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        params: NNTrainParams,
    ):
        # If params.scheduler has a `kind` attribute (set by the Schedulers
        # enum), dispatch on it; otherwise fall back to ReduceLROnPlateau
        # for backwards compatibility with existing notebook code.
        sched_params = params.scheduler
        kind = getattr(sched_params, "kind", None)

        if kind is None:
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                min_lr=sched_params.min_lr,
                factor=sched_params.factor,
                cooldown=sched_params.cooldown,
                patience=sched_params.patience,
                threshold=sched_params.threshold,
            )

        # When a `kind` is supplied, the params dataclass carries kind-specific
        # config. The enum's __call__ knows how to construct.
        # The run's planned committed updates (the default horizon of an
        # optimizer_update clock, FEAT-014) come from train() through
        # ``_planned_scheduler_updates``, keeping this signature unchanged.
        return kind(
            optimizer=optimizer,
            params=sched_params,
            n_epochs=params.n_epochs,
            n_updates=self._planned_scheduler_updates,
        )

    def _build_grad_scaler(self) -> Optional[torch.amp.GradScaler]:
        """The AMP loss scaler for this model, or ``None``.

        Built only when the model's precision resolves to fp16 on its device
        (FEAT-028): ``PrecisionPolicy("fp16")``, or the legacy
        ``mixed_precision=True``, on CUDA — CPU / MPS runs and bf16 never
        instantiate one. ``torch.amp.GradScaler(device)`` is the PyTorch >=
        2.3 factory; NNx's declared floor (``torch>=2.4``, the oldest release
        the full test suite passes on) guarantees it exists, so no legacy
        ``torch.cuda.amp`` fallback is needed (FIX-011). The returned object
        is used through the standard ``scale`` / ``unscale_`` / ``step`` /
        ``update`` / ``state_dict`` protocol.
        """
        resolved = self.resolved_precision if isinstance(self, NNModel) else resolve_precision(self.params, self.device)
        return resolved.build_scaler()

    def _commit_epoch(
        self,
        *,
        records: Any,
        run: NNRun,
        ctx: Any,
        idx_epoch: int,
        local_epoch: int,
        params: NNTrainParams,
        counters: Optional[dict[str, Any]],
        best_checkpoint: Optional[NNCheckpoint],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        precision: ResolvedPrecision,
        compile_session: Optional[_CompileSession],
        train_loader: Any,
        resume_optimizer_factory: Any,
        registry: ComponentRegistry,
        record: Any,
        distributed_state: Optional[dict[str, Any]],
    ) -> Optional[NNCheckpoint]:
        """Commit one epoch: history first, then LAST (the commit marker),
        phase / BEST checkpoints and the callbacks' deferred writes."""
        # Prepare run history first (idps.csv, or the journal's chunks and
        # manifest); the checkpoint is the epoch's commit marker and is never
        # allowed to get ahead of the history.
        records.save_epoch(run)
        try:
            checkpoint = self._save_checkpoints(
                idp=records.last,
                run_id=run.id,
                idx_epoch=local_epoch,
                n_epochs=params.n_epochs,
                best_checkpoint=best_checkpoint,
                save_phase_checkpoints=params.save_phase_checkpoints,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                precision=precision.record(),
                compile=_state_of(compile_session),
                distributed=distributed_state,
                completed_epoch=idx_epoch,
                train_loader=train_loader,
                optimizer_factory=resume_optimizer_factory,
                components=registry.collect(),
                is_best=record.improved if record is not None else None,
                trained_recipe=run.transforms,
                counters=counters,
            )
        except BaseException:
            # LAST is the epoch commit marker. If it cannot be published,
            # restore history to the preceding epoch.
            committed = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
            if committed is None or committed.idp.epoch_idx != idx_epoch:
                records.rollback_epoch(run)
            raise
        for deferred_checkpoint in ctx.deferred_checkpoint_writes:
            deferred_checkpoint()
        return checkpoint

    def _save_checkpoints(
        self,
        idp: NNIterationDataPoint,
        run_id: str,
        idx_epoch: int,
        n_epochs: int,
        best_checkpoint: Optional[NNCheckpoint],
        save_phase_checkpoints: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        scaler: Optional[torch.amp.GradScaler] = None,
        completed_epoch: Optional[int] = None,
        train_loader: Optional[Iterable[Any]] = None,
        optimizer_factory: Optional[dict[str, Any]] = None,
        components: Optional[dict[str, Any]] = None,
        precision: Optional[dict[str, Any]] = None,
        compile: Optional[dict[str, Any]] = None,
        distributed: Optional[dict[str, Any]] = None,
        optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None,
        schedulers: Optional[Mapping[str, Any]] = None,
        optimizer_factories: Optional[Mapping[str, Optional[dict[str, Any]]]] = None,
        is_best: Optional[bool] = None,
        trained_recipe: Optional[Sequence[NNCheckpointTransform]] = None,
        counters: Optional[dict[str, Any]] = None,
    ) -> NNCheckpoint:
        """Publish LAST, the due phase tag and — when this epoch is the best
        so far — BEST. ``is_best`` is the monitor's decision (FEAT-003);
        ``None`` keeps the legacy comparison. ``trained_recipe`` (the
        recipe the run id records, FEAT-016) refuses, before anything is
        written, a topology no checkpoint of the run could rebuild."""
        if trained_recipe is not None:
            _check_trained_recipe(self, trained_recipe)
        checkpoint = NNCheckpoint(
            idp=idp,
            model_params=self.params,
            net_params=self.net_params,
            net_state=self.net.state_dict(),
            # FEAT-016: a recipe recorded before training rebuilds every tag's
            # topology (none for a model without one, as before).
            transforms=_snapshot_transforms(self._topology_transforms),
        )
        # Every checkpoint tag is a valid resume point, so each carries the
        # same stateful training bundle as LAST/BEST.
        opt_state = optimizer.state_dict() if optimizer is not None else None
        scheduler_state = scheduler.state_dict() if scheduler is not None else None
        scaler_state = scaler.state_dict() if scaler is not None else None
        rng_state = _capture_rng_state(train_loader)
        optimizer_type = _component_type(optimizer) if optimizer is not None else None
        scheduler_type = _component_type(scheduler) if scheduler is not None else None
        optimizer_topology = _optimizer_topology(optimizer, self.net) if optimizer is not None else None
        # FEAT-005: a Trainer's named optimizers / schedulers and every
        # component travel in the same generation sidecar.
        # FEAT-028: the run's resolved precision record rides along, so a
        # stateful resume can refuse a different one.
        # FEAT-029: and so does the compile record (None for an eager run).
        # FEAT-030: a DDP run's world, partitions and every rank's RNG.
        stateful_extras: dict[str, Any] = {
            "components": components,
            "precision": precision,
            "compile": compile,
            "distributed": distributed,
            "counters": counters,
        }
        if optimizers is not None:
            stateful_extras.update(_named_training_state(self.net, optimizers, schedulers or {}, optimizer_factories))

        # LAST is the epoch commit marker, so publish it before ancillary
        # phase/BEST checkpoints.
        checkpoint.save(
            run=run_id,
            type=Checkpoints.LAST,
            optimizer_state=opt_state,
            scheduler_state=scheduler_state,
            scaler_state=scaler_state,
            rng_state=rng_state,
            completed_epoch=completed_epoch,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            optimizer_topology=optimizer_topology,
            optimizer_factory=optimizer_factory,
            **stateful_extras,
        )

        # Phase markers at epoch boundaries — fractions are nominal (1/4, 2/4,
        # 3/4 of the planned epoch count); off-by-one allowed when n_epochs
        # isn't divisible by 4. See `phase_tag` for the small-`n_epochs`
        # caveat. Opt-out via NNTrainParams.save_phase_checkpoints.
        if save_phase_checkpoints:
            tag = phase_tag(idx_epoch, n_epochs)
            if tag is not None:
                checkpoint.save(
                    run=run_id,
                    type=tag,
                    optimizer_state=opt_state,
                    scheduler_state=scheduler_state,
                    scaler_state=scaler_state,
                    rng_state=rng_state,
                    completed_epoch=completed_epoch,
                    optimizer_type=optimizer_type,
                    scheduler_type=scheduler_type,
                    optimizer_topology=optimizer_topology,
                    optimizer_factory=optimizer_factory,
                    **stateful_extras,
                )

        # BEST tracking goes through the same _best_err helper used by
        # NNRun.save's cross-run comparison and by Trainer._save_checkpoint
        # — single source of truth for "what's the comparable error here"
        # (val→train, error→loss, +inf fall-through, tolerating None EDP
        # or None .error from custom train_step_fn paradigms).
        if is_best is None:
            is_best = best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint)
        if is_best:
            checkpoint.save(
                run=run_id,
                type=Checkpoints.BEST,
                optimizer_state=opt_state,
                scheduler_state=scheduler_state,
                scaler_state=scaler_state,
                rng_state=rng_state,
                completed_epoch=completed_epoch,
                optimizer_type=optimizer_type,
                scheduler_type=scheduler_type,
                optimizer_topology=optimizer_topology,
                optimizer_factory=optimizer_factory,
                **stateful_extras,
            )

        return checkpoint

    def _step_scheduler(
        self,
        scheduler,
        val_edp: Optional[NNEvaluationDataPoint],
        train_edp: NNEvaluationDataPoint,
        *,
        epoch_idx: int,
    ) -> None:
        # ReduceLROnPlateau wants a metric; other schedulers step on epoch index.
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            # Custom train_step_fn hooks may leave .error unset, and a
            # diverged epoch may report NaN/inf; ReduceLROnPlateau.step(None)
            # crashes inside float() and a non-finite value poisons its
            # running best. The shared finite-only val→train, error→loss
            # resolver picks the signal (and warns about rejected
            # candidates with epoch context); None means "skip this step".
            metric = _resolve_scheduler_metric(val_edp, train_edp, epoch_idx=epoch_idx)
            if metric is None:
                return
            scheduler.step(metric)
        else:
            scheduler.step()

    def _update_tqdm_postfix(
        self,
        tqdm_bar,
        optimizer,
        val_edp: Optional[NNEvaluationDataPoint],
        train_edp: NNEvaluationDataPoint,
        record: Optional[MonitorRecord] = None,
    ) -> None:
        lr = optimizer.param_groups[0]["lr"]
        if record is not None:
            # FEAT-003: show what selection tracks, from the epoch summary.
            shown = f"{record.value:.4f}" if record.value is not None else "n/a"
            tqdm_bar.set_postfix_str(f"{record.monitor.key}={shown}, lr={lr:.4f}")
            return
        # Custom train_step_fn hooks may leave .error unset — fall back to
        # .loss for display so the progress bar doesn't crash mid-train on
        # an `f"{None:.4f}"` format error. Same shared fallback resolver
        # used by _step_scheduler above.
        err = _resolve_metric(val_edp, train_edp)
        err_str = f"{err:.4f}" if err is not None else "n/a"
        tqdm_bar.set_postfix_str(f"error={err_str}, lr={lr:.4f}")

    @staticmethod
    def _normalize_callbacks(
        callbacks: Optional[list[CallbackLike]],
    ) -> list[Callback]:
        # Lazy import to keep nn_model.py importable before callbacks module exists.
        from .callbacks import Callback, _LegacyCallback

        if callbacks is None:
            return []
        out: list[Callback] = []
        for cb in callbacks:
            if isinstance(cb, Callback):
                out.append(cb)
            else:
                out.append(_LegacyCallback(cb))
        return out


def _dispatch_update(callbacks: Sequence[Any], ctx: Any, event: Any) -> None:
    """Deliver one committed-update event (FEAT-004) to every callback."""
    ctx.update_count = event.update_idx
    for callback in callbacks:
        hook = getattr(callback, "on_optimizer_update", None)
        if callable(hook):
            hook(ctx, event)


class _CallbackContext:
    """Mutable state carried across callback invocations.

    Exposes the model, the run-in-progress, the optimizer, and per-epoch state
    (current idp, the running list of idps, an early-stop flag). Lives only for
    the duration of `train()`.
    """

    def __init__(self, model: NNModel, run: NNRun, optimizer):
        self.model = model
        self.run = run
        self.optimizer = optimizer
        self.epoch: int = 0
        self.idp: Optional[NNIterationDataPoint] = None
        self.idps: list[NNIterationDataPoint] = []
        # FEAT-036: the history journal's retention when the run keeps a
        # bounded window (built-in callbacks bound their own logs by it).
        self.history_retention: Optional[int] = None
        self.history_records: Any = None  # the loop's history (nnx.history), for on_train_end
        self.should_stop: bool = False
        self.optimizers: Any = None
        self.trainer: Any = None
        self.deferred_checkpoint_writes: list[Callable[[], None]] = []
        # FEAT-004: committed optimizer updates so far (objective runs only).
        self.update_count: Optional[int] = None
        # FEAT-014: the epoch's ``(update index, LR after its scheduler step)``
        # for the primary optimizer's optimizer_update clock (empty otherwise).
        self.update_lrs: list[tuple[int, float]] = []
        # FEAT-033: called with this context after every committed optimizer
        # update (default step and objective runs alike); a listener — or any
        # hook — may set ``stop_at_update`` to stop at that update boundary.
        # Mid-epoch, the epoch in progress is then discarded, never
        # committed: LAST, its tensors and the history keep the previous
        # epoch (no LAST at all before a first completed epoch).
        self.update_listeners: list[Callable[[Any], None]] = []
        self.stop_at_update: bool = False
        self.committed_updates: int = 0
        # #394: the training scheduler, read-only for callbacks (None outside
        # NNModel.train, e.g. a Trainer's named schedulers).
        self.scheduler: Any = None
