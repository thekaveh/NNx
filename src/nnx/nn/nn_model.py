from __future__ import annotations

import inspect
import json
import math
import os
import random
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, cast

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from .._metrics import _resolve_metric, _resolve_scheduler_metric, classification_edp
from ..tasks import TaskAdapter, task_adapter
from ..utils import Utils, _capture_training_modes, _restore_training_modes
from .enum.checkpoints import Checkpoints, phase_tag
from .enum.devices import Devices
from .enum.nets import Nets
from .params.nn_checkpoint import NNCheckpoint, NNCheckpointTransform, _snapshot_state_dict, _tensor_state_dict
from .params.nn_evaluation_data_point import NNEvaluationDataPoint
from .params.nn_iteration_data_point import NNIterationDataPoint
from .params.nn_model_params import NNModelParams
from .params.nn_params import NNParams
from .params.nn_run import NNRun, _best_err, _print_run_saved
from .params.nn_train_params import NNTrainParams

if TYPE_CHECKING:
    from ..prediction import PredictionResult, ProbabilitySpec
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
    """Describe optimizer groups by model parameter identity, not position."""
    names = {id(param): name for name, param in net.named_parameters()}
    return [
        [{"name": names.get(id(param), "<external>"), "shape": list(param.shape)} for param in group["params"]]
        for group in optimizer.param_groups
    ]


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


def _capture_rng_state(train_loader: Optional[Iterable[Any]] = None) -> dict[str, Any]:
    numpy_state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }
    if train_loader is not None:
        generators = (
            ("train_loader_generator", getattr(train_loader, "generator", None)),
            ("train_sampler_generator", getattr(getattr(train_loader, "sampler", None), "generator", None)),
            ("train_batch_sampler_generator", getattr(getattr(train_loader, "batch_sampler", None), "generator", None)),
            (
                "train_batch_sampler_sampler_generator",
                getattr(getattr(getattr(train_loader, "batch_sampler", None), "sampler", None), "generator", None),
            ),
        )
        generator_states: list[dict[str, Any]] = []
        seen: set[int] = set()
        for key, generator in generators:
            if isinstance(generator, torch.Generator):
                state[key] = generator.get_state()
                if id(generator) not in seen:
                    generator_states.append({"initial_seed": generator.initial_seed(), "state": generator.get_state()})
                    seen.add(id(generator))
        state["train_generators"] = generator_states
    return state


def _restore_rng_state(state: dict[str, Any], train_loader: Optional[Iterable[Any]] = None) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if train_loader is not None:
        generators = (
            ("train_loader_generator", getattr(train_loader, "generator", None)),
            ("train_sampler_generator", getattr(getattr(train_loader, "sampler", None), "generator", None)),
            ("train_batch_sampler_generator", getattr(getattr(train_loader, "batch_sampler", None), "generator", None)),
            (
                "train_batch_sampler_sampler_generator",
                getattr(getattr(getattr(train_loader, "batch_sampler", None), "sampler", None), "generator", None),
            ),
        )
        unique_generators: list[torch.Generator] = []
        seen: set[int] = set()
        for _, generator in generators:
            if isinstance(generator, torch.Generator) and id(generator) not in seen:
                unique_generators.append(generator)
                seen.add(id(generator))
        saved_generators = state.get("train_generators")
        if saved_generators is not None:
            if len(saved_generators) != len(unique_generators):
                raise ValueError("resume loader exposes a different number of torch.Generator instances")
            if saved_generators and isinstance(saved_generators[0], dict):
                if len(saved_generators) == 1:
                    unique_generators[0].set_state(saved_generators[0]["state"])
                    return
                saved_by_seed = {entry["initial_seed"]: entry["state"] for entry in saved_generators}
                current_seeds = [generator.initial_seed() for generator in unique_generators]
                if len(saved_by_seed) != len(saved_generators) or len(set(current_seeds)) != len(current_seeds):
                    raise ValueError("resume loader has ambiguous torch.Generator seeds")
                if set(saved_by_seed) != set(current_seeds):
                    raise ValueError("resume loader exposes different torch.Generator identities")
                for generator in unique_generators:
                    generator.set_state(saved_by_seed[generator.initial_seed()])
            else:
                # Version 2 sidecars recorded generators positionally.
                for generator, generator_state in zip(unique_generators, saved_generators, strict=True):
                    generator.set_state(generator_state)
        else:
            for key, generator in generators:
                if isinstance(generator, torch.Generator) and state.get(key) is not None:
                    generator.set_state(state[key])


def _collect_checkpoint_transforms(callbacks: list[Callback]) -> tuple[NNCheckpointTransform, ...]:
    # on_train_end runs in reverse callback order, so persist transforms in
    # that same order for deterministic topology replay during reconstruction.
    return tuple(transform for callback in reversed(callbacks) for transform in callback.checkpoint_transforms())


def _apply_checkpoint_transform(model: NNModel, transform: NNCheckpointTransform) -> None:
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
        for cb in reversed(self._started):
            try:
                cb.on_train_end(self._ctx)
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
) -> _StepLossTerms:
    """Forward one batch and compute its loss terms.

    Legacy models decode by loss (`_fwd_pass`). A model with a task
    (FEAT-002) validates the batch through its adapter first — before any
    backward pass or optimizer update — and scores only the valid targets.
    """
    adapter = getattr(model, "task_adapter", None)
    if adapter is None:
        _, Y, Y_hat_logits, Y_hat = model._fwd_pass(batch)
        output, target, prediction, valid = Y_hat_logits, Y, Y_hat, None
        if accumulation_state is None:
            train_loss = model.loss_fn(_loss_input(model.loss_fn, output), target)
            return _StepLossTerms(
                output, target, prediction, valid, train_loss, train_loss / accumulate_grad_batches, None
            )
        train_loss, backward_loss, weight = _loss_terms(model.loss_fn, output, target)
        return _StepLossTerms(output, target, prediction, valid, train_loss, backward_loss, weight)

    _, Y, logits = model._fwd_outputs(batch)
    output, target, valid = adapter.prepare(logits, Y)
    train_loss, backward_loss, weight = adapter.loss_terms(model.loss_fn, output, target, valid)
    if accumulation_state is None and weight != 0:
        backward_loss = train_loss / accumulate_grad_batches
    return _StepLossTerms(output, target, None, valid, train_loss, backward_loss, weight)


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


def default_train_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
    """Standard supervised training step: forward → loss → backward → step.

    This is the body that `NNModel.train()` runs when no custom
    `train_step_fn` is supplied. It honors:
      - gradient accumulation (zero_grad at cycle start, step at cycle
        end). A trailing partial cycle is stepped at the epoch boundary;
        gradients use each loss's effective normalization weight.
      - AMP (unscales before grad clip; scaler.step + update at cycle end)
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

    # Mixed precision is opt-in via NNModelParams.mixed_precision; only
    # takes effect on CUDA where autocast + GradScaler are meaningful.
    scaler = ctx.scaler
    amp_enabled = scaler is not None and model.device.type == "cuda"

    adapter = getattr(model, "task_adapter", None)
    if amp_enabled:
        assert scaler is not None
        with torch.amp.autocast(device_type="cuda"):
            terms = _step_loss_terms(model, ctx.batch, accumulation_state, accumulate_grad_batches)
        loss_value = _record_step_loss(terms, accumulation_state, should_step=should_step)
        scaler.scale(terms.backward_loss).backward()
    else:
        terms = _step_loss_terms(model, ctx.batch, accumulation_state, accumulate_grad_batches)
        loss_value = _record_step_loss(terms, accumulation_state, should_step=should_step)
        terms.backward_loss.backward()

    if should_step and _window_is_masked(terms, accumulation_state, window_is_this_batch=cycle_size == 1):
        # FEAT-002: every target in this optimizer window is masked — there
        # is nothing to learn from, so no update is taken.
        model.net.zero_grad()
        _reset_accumulation(accumulation_state)
    elif should_step:
        if amp_enabled:
            assert scaler is not None
            scaler.unscale_(ctx.optimizer)
        if (
            accumulation_state is not None
            and accumulation_state.normalization_required
            and accumulation_state.normalization_weight
        ):
            _scale_gradients(model.net, 1.0 / accumulation_state.normalization_weight)
        elif accumulation_state is None and cycle_size < accumulate_grad_batches:
            _scale_gradients(model.net, accumulate_grad_batches / cycle_size)
        if ctx.grad_clip_norm is not None:
            # Under AMP the gradients were unscaled above, so the clip
            # threshold applies in the original gradient space.
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), ctx.grad_clip_norm)
        if amp_enabled:
            assert scaler is not None
            scaler.step(ctx.optimizer)
            scaler.update()
        else:
            ctx.optimizer.step()
        _reset_accumulation(accumulation_state)

    if adapter is not None:
        assert terms.valid is not None
        accumulator = adapter.accumulator(keep_arrays=bool(ctx.extra_metrics))
        accumulator.update(terms.output, terms.target, terms.valid)
        return accumulator.result(
            loss=loss_value if terms.normalization_weight != 0 else None,
            extra_metrics=ctx.extra_metrics,
        )
    assert terms.prediction is not None
    return _classification_edp_for_loss(
        loss_fn=model.loss_fn,
        target=terms.target,
        prediction=terms.prediction,
        loss=cast(float, loss_value),
        extra_metrics=ctx.extra_metrics,
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

    def __init__(self, net_params: NNParams, params: NNModelParams):
        # NOTE: we deliberately do NOT call super().__init__() — the
        # PyTorchModelHubMixin base has no __init__ of its own (it's a
        # mixin that only contributes class-level methods), and even if
        # it grew one in a future hub release, the only side effect we'd
        # want is config-attribute initialization which we handle below.
        if net_params is None:
            raise ValueError("net_params must not be None")

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
        self.loss_fn = self.params.loss().to(self.device)
        self.net = self.params.net(params=net_params).to(self.device)

    @property
    def task_adapter(self) -> Optional[TaskAdapter]:
        """The adapter for ``params.task`` (FEAT-002), or ``None`` for a
        legacy classification model. Custom steps can call
        ``task_adapter.record(output, target, loss=...)`` to write the same
        task record the default step writes."""
        # getattr: subclasses and stand-ins that bypass __init__ are legacy models.
        return getattr(self, "_task_adapter", None)

    def _check_task_preflight(self) -> None:
        """Reject a runtime ``loss_fn`` the declared task cannot score —
        before any loader is iterated (NNModel.train and Trainer.train)."""
        adapter = getattr(self, "task_adapter", None)
        if adapter is not None:
            adapter.check_loss_fn(self.loss_fn)

    def _assert_reconstructible_topology(self) -> None:
        if self._topology_transforms:
            return
        rng_state = _capture_rng_state(None)
        try:
            expected = self.params.net(params=self.net_params).state_dict()
        finally:
            _restore_rng_state(rng_state, None)
        actual = self.net.state_dict()
        expected_schema = {
            key: tuple(value.shape) for key, value in expected.items() if isinstance(value, torch.Tensor)
        }
        actual_schema = {key: tuple(value.shape) for key, value in actual.items() if isinstance(value, torch.Tensor)}
        low_rank_replacements = [
            key
            for key in expected_schema
            if key.endswith(".weight")
            and key not in actual_schema
            and f"{key[:-7]}.0.weight" in actual_schema
            and f"{key[:-7]}.1.weight" in actual_schema
        ]
        if low_rank_replacements:
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
        """
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
    def from_checkpoint(cls, checkpoint: NNCheckpoint, device: Optional[Devices] = None, **model_kwargs: Any) -> Self:
        """Rebuild a model, replay topology transforms, and load its weights.

        Ordinary and legacy FP32 checkpoints have no transforms. Converted
        QAT checkpoints replay their persisted torchao recipe before state
        loading; unsupported recipes fail explicitly rather than constructing
        a model with the wrong topology.
        """
        model_params = checkpoint.model_params if device is None else replace(checkpoint.model_params, device=device)
        model = cls(params=model_params, net_params=checkpoint.net_params, **model_kwargs)

        transforms = getattr(checkpoint, "transforms", ())
        for transform in transforms:
            _apply_checkpoint_transform(model, transform)
        model._topology_transforms = tuple(transforms)

        try:
            model.net.load_state_dict(checkpoint.net_state)
        except RuntimeError as error:
            if not transforms and _looks_like_converted_qat_state(checkpoint.net_state):
                raise ValueError(
                    "converted QAT checkpoint lacks reconstruction metadata; "
                    "recreate its torchao topology with the original qat_config and groupsize"
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

        config = {
            "net_params": self.net_params.state(),
            "params": self.params.state(),
            "transforms": [transform.state() for transform in self._topology_transforms],
        }
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

        # resolve_from_state dispatches transformer configs to
        # NNTransformerParams so LM models round-trip through the Hub.
        net_params = NNParams.resolve_from_state(net_params_state)
        params = NNModelParams.from_state(model_params_state)
        try:
            torch_load_device = torch.device(map_location)
            load_device = Devices(torch_load_device.type)
        except (TypeError, ValueError) as e:
            raise ValueError(f"unsupported Hub map_location {map_location!r}") from e
        if torch_load_device.index is not None:
            raise ValueError(f"indexed Hub map_location is unsupported: {map_location!r}")
        params = replace(params, device=load_device)

        transforms = tuple(NNCheckpointTransform.from_state(item) for item in config.get("transforms", []))
        reconstruction_kwargs = cls._hub_reconstruction_kwargs(config, os.path.dirname(config_path))
        model = cls(net_params=net_params, params=params, **reconstruction_kwargs)
        for transform in transforms:
            _apply_checkpoint_transform(model, transform)
        model._topology_transforms = transforms
        state_dict = load_file(weights_path, device=str(torch_load_device))
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
        all of that and leaves just the weights.

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
        if train_step_fn is None:
            self._assert_reconstructible_topology()
        if params is None:
            raise ValueError("train params must be non-None")
        self._check_task_preflight()
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

        if params.seed is not None:
            from ..seeding import set_seed

            set_seed(params.seed)

        from ..optimizers import build_optimizer

        # Build the optimizer before any run directory exists: a registered
        # factory that is unknown, or that returns something other than an
        # optimizer over exactly the resolved parameters, fails here with no
        # run reserved (nnx.optimizers.build_optimizer is the shared hook).
        optimizer = build_optimizer(self.net, params.optim)
        run = NNRun(train=params, model=self.params, net=self.net_params, salt=salt)
        with run.writable_lease(overwrite=params.overwrite_existing):
            return self._train_impl(
                params=params,
                run=run,
                optimizer=optimizer,
                callbacks=callbacks,
                train_step_fn=train_step_fn,
                eval_step_fn=eval_step_fn,
            )

    def _train_impl(
        self,
        params: NNTrainParams,
        run: NNRun,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[list[CallbackLike]] = None,
        train_step_fn: Optional[TrainStepFn] = None,
        eval_step_fn: Optional[EvalStepFn] = None,
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
        from ..optimizers import _canonical_factory_state, optimizer_factory_state

        resume_optimizer_factory = optimizer_factory_state(params.optim)
        resume_optimizer_topology = _optimizer_topology(optimizer, self.net)
        scheduler_kind = getattr(params.scheduler, "kind", None)
        if (
            params.resume_from_run_id is not None
            and scheduler_kind is not None
            and str(scheduler_kind) in {"one_cycle", "linear_warmup_decay"}
            and params.scheduler.total_steps is None
        ):
            raise ValueError(
                f"resuming {scheduler_kind} requires scheduler.total_steps to be set explicitly "
                "to one shared horizon covering the original and resumed epochs"
            )
        scheduler = self._build_scheduler(optimizer, params)
        scaler = self._build_grad_scaler()
        start_epoch = 0

        # Warm resume restores every stateful training component when the
        # source checkpoint has a versioned sidecar. Legacy optimizer-only
        # sidecars remain supported.
        if params.resume_from_run_id is not None:
            ckpt_type = Checkpoints(params.resume_from_checkpoint)
            resume_ckpt, training_state = NNCheckpoint.load_with_training_state(
                run=params.resume_from_run_id,
                type=ckpt_type,
            )
            if resume_ckpt is None:
                raise ValueError(f"resume_from_run_id={params.resume_from_run_id!r}/{ckpt_type} not found on disk")
            resume_net_state = training_state.get("model") if training_state is not None else None
            if resume_ckpt.transforms and resume_net_state is None:
                raise ValueError(
                    "this transformed checkpoint has no pre-transform training state and cannot be warm-resumed; "
                    "use NNModel.from_checkpoint() for inference or resume from an untransformed checkpoint"
                )
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
                if _canonical_factory_state(expected_factory) != _canonical_factory_state(resume_optimizer_factory):
                    raise ValueError(
                        f"resume optimizer factory mismatch: checkpoint has {expected_factory}, "
                        f"configuration builds {resume_optimizer_factory}"
                    )
                expected_topology = training_state.get("optimizer_topology")
                if expected_topology is not None and expected_topology != resume_optimizer_topology:
                    raise ValueError("resume optimizer parameter topology does not match the checkpoint")
                if (training_state.get("scaler") is None) != (scaler is None):
                    raise ValueError(
                        "resume GradScaler presence mismatch: checkpoint and configuration must both use AMP or neither"
                    )
                completed_epoch = training_state.get("completed_epoch")
                if completed_epoch is not None:
                    start_epoch = int(completed_epoch) + 1
                if (
                    scheduler_kind is not None
                    and str(scheduler_kind) in {"one_cycle", "linear_warmup_decay"}
                    and params.scheduler.total_steps is not None
                    and start_epoch + params.n_epochs > params.scheduler.total_steps
                ):
                    raise ValueError(
                        f"resumed {scheduler_kind} would reach epoch {start_epoch + params.n_epochs}, "
                        f"beyond scheduler.total_steps={params.scheduler.total_steps}; configure one shared "
                        "horizon covering the original and resumed epochs"
                    )
                # Worker capability is decided BEFORE any state is restored:
                # ordinary training accepts any re-iterable batch source (a
                # list, NNGraphDataset's one-element full-batch list, ...),
                # which has no `num_workers`. Absent metadata means "no
                # worker-local RNG to worry about"; a real DataLoader with
                # workers keeps its warning. Nothing here iterates the source.
                warn_worker_rng = training_state.get("rng") is not None and _loader_num_workers(train_loader) > 0
                previous_net_state = _snapshot_state_dict(self.net.state_dict())
                previous_rng_state = _capture_rng_state(train_loader)
                try:
                    self.net.load_state_dict(resume_net_state or resume_ckpt.net_state)
                    optimizer.load_state_dict(training_state["optimizer"])
                    if training_state.get("scheduler") is not None:
                        scheduler.load_state_dict(training_state["scheduler"])
                    if scaler is not None and training_state.get("scaler") is not None:
                        scaler.load_state_dict(training_state["scaler"])
                    if training_state.get("rng") is not None:
                        _restore_rng_state(training_state["rng"], train_loader)
                except BaseException:
                    self.net.load_state_dict(previous_net_state)
                    _restore_rng_state(previous_rng_state, train_loader)
                    raise
                if warn_worker_rng:
                    warnings.warn(
                        "exact warm-resume continuity requires train_loader.num_workers=0; "
                        "worker-local RNG state cannot be reconstructed",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                start_epoch = resume_ckpt.idp.epoch_idx + 1
                previous_net_state = _snapshot_state_dict(self.net.state_dict())
                previous_rng_state = _capture_rng_state(train_loader)
                try:
                    self.net.load_state_dict(resume_ckpt.net_state)
                except BaseException:
                    self.net.load_state_dict(previous_net_state)
                    _restore_rng_state(previous_rng_state, train_loader)
                    raise
                warnings.warn(
                    "checkpoint has no training-state sidecar; model weights were restored, but optimizer, "
                    "scheduler, scaler, and RNG state restart from their configured defaults",
                    RuntimeWarning,
                    stacklevel=2,
                )

        normalized_callbacks = self._normalize_callbacks(callbacks)

        idps: list[NNIterationDataPoint] = []
        # `len()` is not defined on iterable-style DataLoaders (IterableDataset).
        # Fall back to None so tqdm renders without a total instead of crashing.
        try:
            n_iter: Optional[int] = int(params.n_epochs * len(cast(Sized, train_loader)))
        except TypeError:
            n_iter = None
        best_checkpoint: Optional[NNCheckpoint] = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)

        Utils.print_table(header=False, title="Run Details...", data=Utils.flatten_dict(data=run.state()))

        ctx = _CallbackContext(model=self, run=run, optimizer=optimizer)
        # Default to the standard supervised step when the caller doesn't
        # override. Custom step gets dispatched from inside the batch loop
        # below so the rest of train() (scheduler, callbacks, checkpoint
        # cadence, val loop, incremental save) is identical either way.
        # Explicit None check (not `or`) so a hypothetical callable that
        # happens to be falsy by __bool__ doesn't silently fall back.
        step_fn: TrainStepFn = default_train_step if train_step_fn is None else train_step_fn

        idx_iter = 0
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
            for local_epoch in range(params.n_epochs):
                idx_epoch = start_epoch + local_epoch
                ctx.epoch = idx_epoch
                for cb in normalized_callbacks:
                    cb.on_epoch_begin(ctx)

                n_idps_before_epoch = len(idps)
                accumulation_state = GradientAccumulationState()
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
                        epoch_idx=idx_epoch,
                        is_last_batch=is_last_batch,
                        accumulation_state=accumulation_state,
                    )
                    train_edp = step_fn(step_ctx)

                    idps.append(
                        NNIterationDataPoint(
                            iter_idx=idx_iter,
                            epoch_idx=idx_epoch,
                            batch_idx=idx_batch,
                            train_edp=train_edp,
                            lr=optimizer.param_groups[0]["lr"],
                        )
                    )

                    idx_iter += 1
                    tqdm_bar.update(1)

                if len(idps) == n_idps_before_epoch:
                    # Zero batches this epoch: first epoch would crash on
                    # idps[-1] below; later epochs would silently attach
                    # this epoch's val_edp to the PREVIOUS epoch's last
                    # idp and reuse its stale train_edp.
                    raise ValueError(
                        f"train_loader yielded no batches in epoch {idx_epoch} — check batch_size vs "
                        "dataset size with drop_last=True, or whether the loader is a one-shot iterable."
                    )

                if validate and eval_step_fn is not None:
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
                            )
                        )
                elif validate:
                    assert params.val_loader is not None
                    val_edp = self.evaluate(loader=params.val_loader, extra_metrics=params.extra_metrics)
                else:
                    val_edp = None
                idps[-1] = idps[-1].with_val_edp(val_edp)

                self._step_scheduler(scheduler, val_edp, train_edp, epoch_idx=idx_epoch)

                ctx.idp = idps[-1]
                ctx.idps = idps
                ctx.deferred_checkpoint_writes.clear()
                for cb in normalized_callbacks:
                    cb.on_epoch_end(ctx)

                # Prepare run history first; the checkpoint is the epoch's
                # commit marker and is never allowed to get ahead of idps.csv.
                run.with_idps(idps).save(update_best=False)
                try:
                    checkpoint = self._save_checkpoints(
                        idp=idps[-1],
                        run_id=run.id,
                        idx_epoch=local_epoch,
                        n_epochs=params.n_epochs,
                        best_checkpoint=best_checkpoint,
                        save_phase_checkpoints=params.save_phase_checkpoints,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        completed_epoch=idx_epoch,
                        train_loader=train_loader,
                        optimizer_factory=resume_optimizer_factory,
                    )
                except BaseException:
                    # LAST is the epoch commit marker. If it cannot be
                    # published, restore history to the preceding epoch.
                    committed = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
                    if committed is None or committed.idp.epoch_idx != idx_epoch:
                        run.with_idps(idps[:n_idps_before_epoch]).save(update_best=False)
                    raise
                for deferred_checkpoint in ctx.deferred_checkpoint_writes:
                    deferred_checkpoint()
                # In-memory best_checkpoint tracking must use the same
                # comparison as the on-disk BEST write inside
                # _save_checkpoints (val→train, error→loss, +inf fall-through).
                # Without this, val_loader=None runs would silently overwrite
                # best_checkpoint every epoch (because checkpoint.idp.val_edp
                # is None there) while the on-disk BEST tracks training error,
                # diverging the two views of "best".
                if best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
                    best_checkpoint = checkpoint

                self._update_tqdm_postfix(tqdm_bar, optimizer, val_edp, train_edp)

                if ctx.should_stop:
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
        if idps:
            final_transforms = (*self._topology_transforms, *_collect_checkpoint_transforms(normalized_callbacks))
            self._topology_transforms = final_transforms
            NNCheckpoint(
                idp=idps[-1],
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
                rng_state=(pre_transform_rng_state if final_transforms else _capture_rng_state(train_loader)),
                completed_epoch=idps[-1].epoch_idx,
                resume_net_state=pre_transform_net_state if final_transforms else None,
                optimizer_type=_component_type(optimizer),
                scheduler_type=_component_type(scheduler),
                optimizer_topology=resume_optimizer_topology,
                optimizer_factory=resume_optimizer_factory,
            )

        saved = run.with_idps(idps).save()
        _print_run_saved(run.id)
        return saved

    def evaluate(self, loader: Iterable[Any], extra_metrics=None) -> NNEvaluationDataPoint:
        """Aggregate predictions across all batches in `loader` and compute
        a single NNEvaluationDataPoint. Aggregating (rather than averaging
        per-batch metrics) gives correct sample-weighted f1/precision/recall
        when the final batch is short.

        `extra_metrics` ({name -> callable(y_true, y_pred) -> float}) are
        called once on the aggregate truth / decoded predictions.

        Raises ValueError if the loader yields zero batches — previously
        produced NaN metrics silently from np.mean over an empty list.
        """
        # Ensure loss_fn lives on the same device as the model — guards
        # against callers reassigning self.device after construction.
        self.loss_fn = self.loss_fn.to(self.device)
        # getattr: legacy stand-ins borrow these methods without the property.
        if getattr(self, "task_adapter", None) is not None:
            return self._evaluate_task(loader, extra_metrics)
        # Snapshot training-mode for non-destructive restore (matches the
        # convention already used by `nnx.viz.activation_map` and
        # `nnx.lr_finder`). Without this, a caller doing the common
        # train → evaluate → train-more pattern silently leaves the net
        # in `.eval()` mode after evaluate(); BatchNorm / Dropout layers
        # would behave incorrectly on the next batch unless the caller
        # remembered to call `self.net.train()` themselves.
        training_modes = _capture_training_modes(self.net)
        self.net.eval()

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
                    _, Y, Y_hat_logits, Y_hat = self._fwd_pass(batch)
                    batch_n = int(Y.size(0))
                    # Aggregate predictions / labels across the entire loader so
                    # metrics are computed on the full eval set, not per-batch.
                    metric_Y, metric_Y_hat = _classification_metric_tensors(self.loss_fn, Y, Y_hat)
                    if metric_Y.numel():
                        all_Y.append(metric_Y.cpu().numpy())
                        all_Y_hat.append(metric_Y_hat.cpu().numpy())
                        n_metric_samples += int(metric_Y.numel())
                    _, batch_loss_numerator, batch_normalization_weight = _loss_terms(self.loss_fn, Y_hat_logits, Y)
                    loss_numerator += float(batch_loss_numerator.detach())
                    if batch_normalization_weight is None:
                        loss_uses_sum_reduction = True
                    else:
                        loss_normalization_weight += batch_normalization_weight
                    n_samples += batch_n
        finally:
            _restore_training_modes(training_modes)

        if n_samples == 0:
            raise ValueError("evaluate() loader produced zero samples")
        if n_metric_samples == 0:
            raise ValueError("evaluate() loader produced zero non-ignored samples")

        Y_concat = np.concatenate(all_Y)
        Y_hat_concat = np.concatenate(all_Y_hat)

        edp = NNEvaluationDataPoint.of(Y=Y_concat, Y_hat=Y_hat_concat, extra_metrics=extra_metrics)
        assert edp.accuracy is not None  # `of` always computes the classification fields
        return edp.with_loss(
            value=(
                loss_numerator
                if loss_uses_sum_reduction
                else loss_numerator / loss_normalization_weight
                if loss_normalization_weight
                else float("nan")
            )
        ).with_error(value=float(1 - edp.accuracy))

    def _evaluate_task(self, loader: Iterable[Any], extra_metrics=None) -> NNEvaluationDataPoint:
        """``evaluate()`` for a model with a task (FEAT-002): the adapter
        validates every batch, and loss and metrics are accumulated over
        the valid targets of the whole loader. Every target masked yields
        an ``"empty"`` record (no loss, no metrics) instead of raising."""
        adapter = self.task_adapter
        assert adapter is not None
        training_modes = _capture_training_modes(self.net)
        self.net.eval()
        accumulator = adapter.accumulator(keep_arrays=bool(extra_metrics))
        loss_numerator = 0.0
        loss_normalization_weight = 0.0
        loss_uses_sum_reduction = False
        n_batches = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    _, Y, logits = self._fwd_outputs(batch)
                    output, target, valid = adapter.prepare(logits, Y)
                    accumulator.update(output, target, valid)
                    _, numerator, weight = adapter.loss_terms(self.loss_fn, output, target, valid)
                    loss_numerator += float(numerator.detach())
                    if weight is None:
                        loss_uses_sum_reduction = True
                    else:
                        loss_normalization_weight += weight
                    n_batches += 1
        finally:
            _restore_training_modes(training_modes)
        if n_batches == 0:
            raise ValueError("evaluate() loader produced zero samples")
        if not accumulator.count:
            loss: Optional[float] = None
        elif loss_uses_sum_reduction:
            loss = loss_numerator
        else:
            loss = loss_numerator / loss_normalization_weight if loss_normalization_weight else float("nan")
        return accumulator.result(loss=loss, extra_metrics=extra_metrics)

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
        adapter = getattr(self, "task_adapter", None)
        if adapter is not None:
            return PredictResult(logits=logits, classes=adapter.decode_array(logits))
        class_axis = -1 if self.params.net is Nets.TRANSFORMER and logits.ndim > 2 else 1
        classes = (
            (logits >= 0).astype(np.int64)
            if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss)
            else logits.argmax(axis=class_axis)
        )
        return PredictResult(logits=logits, classes=classes)

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

        if isinstance(X, DataLoader) and isinstance(X.sampler, torch.utils.data.RandomSampler):
            warnings.warn(
                "predict_proba() over a shuffling DataLoader: sample_ids are iteration positions, not "
                "dataset indices, so they cannot be joined back to the dataset; use a non-shuffled "
                "loader (graph loaders are exempt: their ids are global node indices)",
                UserWarning,
                stacklevel=2,
            )
        if spec is None:
            adapter = getattr(self, "task_adapter", None)
            if adapter is None:
                raise TypeError(
                    "predict_proba() needs a ProbabilitySpec for a model without a task "
                    "(or declare NNModelParams(task=TaskSpec...))"
                )
            logits, sample_ids = self._predict_logits(X, caller="predict_proba()", check_first=adapter.check_logits)
            return adapter.prediction(logits, sample_ids)
        explicit = spec
        logits, sample_ids = self._predict_logits(
            X, caller="predict_proba()", check_first=lambda first: _check_spec_fits(first, explicit)
        )
        return prediction_from_logits(logits, explicit, sample_ids=sample_ids)

    def _predict_logits(
        self, X, *, caller: str, check_first: Optional[Callable[[np.ndarray], object]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared inference path of :meth:`predict` / :meth:`predict_proba`:
        raw logits (numpy) plus an ``int64`` sample id per row, computed in
        eval mode under ``no_grad`` with every submodule's training mode
        restored on exit (success or failure). ``check_first`` sees the
        first loader batch's logits, so a caller can reject them before the
        rest of the loader is run."""
        training_modes = _capture_training_modes(self.net)
        self.net.eval()

        try:
            if isinstance(X, DataLoader):
                logits_chunks: list[np.ndarray] = []
                id_chunks: list[np.ndarray] = []
                offset = 0
                with torch.no_grad():
                    for batch in X:
                        if isinstance(batch, torch.Tensor):
                            X_in = (batch,)
                        elif isinstance(batch, (tuple, list)) and len(batch) == 1:
                            X_in = (batch[0],)
                        else:
                            # Supervised tuples and graph batches retain their
                            # model-specific unpacking; predict discards labels.
                            X_in, _ = cast(Any, self.net).unpack_batch(batch)
                        X_in = tuple(x.to(self.device) for x in X_in)
                        logits = self.net(*X_in).cpu().numpy()
                        ids: Optional[np.ndarray] = None
                        # NeighborLoader subgraphs: only the leading seed
                        # rows are this batch's nodes (see
                        # GraphNNBase.seed_count) — without the slice,
                        # predictions for sampled neighbors pollute the
                        # output and the row count exceeds the loader's
                        # node set. Their identity is the global node index.
                        seed_count = getattr(self.net, "seed_count", None)
                        if seed_count is not None:
                            n_seed = seed_count(batch)
                            if n_seed is not None:
                                logits = logits[:n_seed]
                                # NeighborLoader's `n_id` holds the global ids
                                # of the subgraph's nodes, seeds first; its
                                # `input_id` is only global when input_nodes
                                # was a mask. NNGraphDataset's full-graph
                                # batches carry the global ids in `input_id`.
                                node_ids = getattr(batch, "n_id", None)
                                if node_ids is None:
                                    node_ids = getattr(batch, "input_id", None)
                                if node_ids is not None:
                                    ids = np.asarray(node_ids[:n_seed].cpu(), dtype=np.int64)
                        if ids is None:
                            ids = np.arange(offset, offset + logits.shape[0], dtype=np.int64)
                        offset += logits.shape[0]
                        if check_first is not None and not logits_chunks:
                            check_first(logits)
                        logits_chunks.append(logits)
                        id_chunks.append(ids)
                if not logits_chunks:
                    raise ValueError(f"{caller} loader produced zero batches")
                return np.concatenate(logits_chunks), np.concatenate(id_chunks)

            # Single input (any of: ndarray, Tensor, or a tuple thereof).
            if not isinstance(X, tuple):
                X = (X,)

            def _to_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.to(self.device)
                # Fall through to numpy → tensor for arrays and array-likes.
                return torch.from_numpy(np.asarray(x)).to(self.device)

            X_t = tuple(_to_tensor(x) for x in X)

            with torch.no_grad():
                Y_hat_logits = self.net(*X_t).cpu().numpy()
            return Y_hat_logits, np.arange(Y_hat_logits.shape[0], dtype=np.int64)
        finally:
            _restore_training_modes(training_modes)

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
        X, Y = cast(Any, self.net).unpack_batch(batch)

        X = tuple(x.to(self.device) for x in X)
        Y = Y.to(self.device)

        Y_hat_logits = self.net(*X)
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
            )
        )

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
        return kind(optimizer=optimizer, params=sched_params, n_epochs=params.n_epochs)

    def _build_grad_scaler(self) -> Optional[torch.amp.GradScaler]:
        """The AMP loss scaler for this model, or ``None``.

        Built only for ``mixed_precision=True`` on a CUDA device — CPU / MPS
        runs never instantiate one. ``torch.amp.GradScaler(device)`` is the
        PyTorch >= 2.3 factory; NNx's declared floor (``torch>=2.4``, the
        oldest release the full test suite passes on) guarantees it exists,
        so no legacy ``torch.cuda.amp`` fallback is needed (FIX-011). The
        returned object is used through the standard ``scale`` /
        ``unscale_`` / ``step`` / ``update`` / ``state_dict`` protocol.
        """
        if getattr(self.params, "mixed_precision", False) and self.device.type == "cuda":
            return torch.amp.GradScaler("cuda")
        return None

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
    ) -> NNCheckpoint:
        checkpoint = NNCheckpoint(
            idp=idp, model_params=self.params, net_params=self.net_params, net_state=self.net.state_dict()
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
                )

        # BEST tracking goes through the same _best_err helper used by
        # NNRun.save's cross-run comparison and by Trainer._save_checkpoint
        # — single source of truth for "what's the comparable error here"
        # (val→train, error→loss, +inf fall-through, tolerating None EDP
        # or None .error from custom train_step_fn paradigms).
        if best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
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
    ) -> None:
        lr = optimizer.param_groups[0]["lr"]
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
        self.should_stop: bool = False
        self.optimizers: Any = None
        self.trainer: Any = None
        self.deferred_checkpoint_writes: list[Callable[[], None]] = []
