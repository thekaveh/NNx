from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .._metrics import _resolve_metric, classification_edp
from ..utils import Utils
from .enum.checkpoints import Checkpoints, phase_tag
from .params.nn_checkpoint import NNCheckpoint
from .params.nn_evaluation_data_point import NNEvaluationDataPoint
from .params.nn_iteration_data_point import NNIterationDataPoint
from .params.nn_model_params import NNModelParams
from .params.nn_params import NNParams
from .params.nn_run import NNRun, _best_err
from .params.nn_train_params import NNTrainParams

if TYPE_CHECKING:
    from .callbacks import Callback


# HuggingFace Hub integration — the mixin is OPTIONAL. We only import it
# at module load when the `thekaveh-nnx[hub]` extra is installed; otherwise we use
# a thin stub that defers errors to call time. This keeps `pip install thekaveh-nnx`
# working without huggingface_hub.
try:
    from huggingface_hub import PyTorchModelHubMixin as _HubMixinBase

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


class PredictResult(NamedTuple):
    """Structured result of NNModel.predict().

    Unpacks positionally as ``(logits, classes)`` so callers doing
    ``log, hat = model.predict(X)`` keep working after the upgrade from
    the original 2-tuple. Field access (``result.logits``, ``result.classes``)
    is preferred for new code.
    """

    logits: np.ndarray
    classes: np.ndarray


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


TrainStepFn = Callable[[TrainStepContext], NNEvaluationDataPoint]


def default_train_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
    """Standard supervised training step: forward → loss → backward → step.

    This is the body that `NNModel.train()` runs when no custom
    `train_step_fn` is supplied. It honors:
      - gradient accumulation (zero_grad at cycle start, step at cycle
        end). Caveat: when an epoch's batch count isn't a multiple of
        ``accumulate_grad_batches``, the trailing partial cycle's grads
        are zeroed at the next epoch's first batch (or dropped at the
        final epoch's end) without an optimizer step — size your loader
        or accumulation factor accordingly.
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

    # Gradient accumulation: only zero grads at the start of a fresh
    # accumulation cycle, and only step the optimizer at the end of one.
    accumulate_grad_batches = ctx.accumulate_grad_batches
    is_cycle_start = (ctx.batch_idx % accumulate_grad_batches) == 0
    is_cycle_end = ((ctx.batch_idx + 1) % accumulate_grad_batches) == 0
    if is_cycle_start:
        model.net.zero_grad()

    # Mixed precision is opt-in via NNModelParams.mixed_precision; only
    # takes effect on CUDA where autocast + GradScaler are meaningful.
    scaler = ctx.scaler
    amp_enabled = scaler is not None and model.device.type == "cuda"

    if amp_enabled:
        with torch.amp.autocast(device_type="cuda"):
            X, Y, Y_hat_logits, Y_hat = model._fwd_pass(ctx.batch)
            train_loss = model.loss_fn(Y_hat_logits, Y)
        # Scale loss by 1/N so accumulated grads = mean across batches.
        scaler.scale(train_loss / accumulate_grad_batches).backward()
        if is_cycle_end:
            if ctx.grad_clip_norm is not None:
                # Unscale before clipping so the clip threshold applies
                # in the original gradient space, not the scaled one.
                scaler.unscale_(ctx.optimizer)
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), ctx.grad_clip_norm)
            scaler.step(ctx.optimizer)
            scaler.update()
    else:
        X, Y, Y_hat_logits, Y_hat = model._fwd_pass(ctx.batch)
        train_loss = model.loss_fn(Y_hat_logits, Y)
        (train_loss / accumulate_grad_batches).backward()
        if is_cycle_end:
            if ctx.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), ctx.grad_clip_norm)
            ctx.optimizer.step()

    loss_value = float(train_loss.detach())
    # NaN/Inf guard: silent divergence leaves checkpoints full of garbage
    # weights. Raise so the training session terminates loudly.
    if not np.isfinite(loss_value):
        raise FloatingPointError(
            f"non-finite training loss ({loss_value!r}) — training diverged. "
            "Check learning rate, gradient clipping (NNOptimParams.grad_clip_norm), "
            "or input normalization."
        )

    return classification_edp(
        Y=Y,
        Y_hat=Y_hat,
        loss=loss_value,
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

    def __init__(self, net_params: NNParams, params: NNModelParams):
        # NOTE: we deliberately do NOT call super().__init__() — the
        # PyTorchModelHubMixin base has no __init__ of its own (it's a
        # mixin that only contributes class-level methods), and even if
        # it grew one in a future hub release, the only side effect we'd
        # want is config-attribute initialization which we handle below.
        if net_params is None:
            raise ValueError("net_params must not be None")

        self.net_params = net_params
        self.params = params

        self.device = self.params.device()
        self.loss_fn = self.params.loss().to(self.device)
        self.net = self.params.net(params=net_params).to(self.device)

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
        was_training = self.net.training
        self.net.eval()
        # torch>=2.5 defaults torch.onnx.export to the dynamo-based exporter,
        # which requires `onnxscript`. We pass `dynamo` through explicitly so
        # the default (False) keeps the legacy tracing path regardless of
        # the installed torch version — plain `pip install onnx` is enough.
        try:
            try:
                if dynamo:
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
                else:
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
            except TypeError:
                # Older torch versions don't accept the `dynamo` kwarg — they
                # already use the legacy path by default. If the caller asked
                # for `dynamo=True` on such a torch, fail loudly rather than
                # silently falling back to legacy.
                if dynamo:
                    raise RuntimeError(
                        "to_onnx(dynamo=True) requires torch>=2.5 (the dynamo-based "
                        "ONNX exporter wasn't available before then). Upgrade torch or "
                        "call with dynamo=False."
                    ) from None
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
            if was_training:
                self.net.train()
        return path

    @staticmethod
    def from_checkpoint(checkpoint: NNCheckpoint) -> NNModel:
        model = NNModel(params=checkpoint.model_params, net_params=checkpoint.net_params)

        model.net.load_state_dict(checkpoint.net_state)

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
        tensors = {k: v.detach().contiguous().clone() for k, v in self.net.state_dict().items()}
        save_file(tensors, str(save_dir / _HUB_MODEL_FILENAME))

        config = {
            "net_params": self.net_params.state(),
            "params": self.params.state(),
        }
        # Explicit utf-8 — Hub config files round-trip through HuggingFace's
        # repo download path and can be read on any platform; relying on
        # the host locale's default encoding could mis-encode unicode
        # paths or non-ASCII tokenizer names round-tripped via
        # `params.state()`.
        with open(save_dir / _HUB_CONFIG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(config, f, sort_keys=True, indent=2)

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
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_id,
                filename=_HUB_CONFIG_FILENAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename=_HUB_MODEL_FILENAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )

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

        model = cls(net_params=net_params, params=params)
        state_dict = load_file(weights_path, device=map_location)
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
                Pass a `Callable[[TrainStepContext], NNEvaluationDataPoint]`
                for non-supervised paradigms (autoencoder, VAE, link
                prediction, recommendation, diffusion). The custom function
                is responsible for forward/backward/step and honoring the
                grad-clip/accumulation/AMP knobs the context carries. See
                `docs/concepts.md` and `examples/05_*.py`.

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
        if params is None:
            raise ValueError("train params must be non-None")
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

        # V1: seed every RNG before constructing the run so dataset shuffling,
        # weight init, dropout — anything stochastic — is reproducible. The
        # `seed` field only affects state() (and run.id) when explicitly set,
        # so back-compat for no-seed callers is preserved.
        if params.seed is not None:
            from ..seeding import set_seed

            set_seed(params.seed)

        validate: bool = params.val_loader is not None
        # Use self.net_params (always set in __init__) rather than
        # self.net.params: the latter is FeedFwdNN-specific and fails when
        # the caller substitutes a custom nn.Module post-construction
        # (the same idiom Trainer's GAN demo uses and that diffusion
        # demos use for DiffusionMLP). They're identical for the
        # standard supervised path; the rename is a back-compat-safe
        # robustness fix.
        run: NNRun = NNRun(train=params, model=self.params, net=self.net_params)

        optimizer = params.optim.name(
            net=self.net,
            lr_start=params.optim.max_lr,
            momentum=params.optim.momentum,
            weight_decay=params.optim.weight_decay,
            param_groups=params.optim.param_groups,
        )

        # Warm resume: load weights + optimizer state from a prior run's
        # checkpoint. The .opt.pt sidecar is best-effort — pre-resume
        # checkpoints don't have it, in which case we still load weights
        # but the optimizer starts fresh.
        if params.resume_from_run_id is not None:
            ckpt_type = Checkpoints(params.resume_from_checkpoint)
            resume_ckpt = NNCheckpoint.load(run=params.resume_from_run_id, type=ckpt_type)
            if resume_ckpt is None:
                raise ValueError(f"resume_from_run_id={params.resume_from_run_id!r}/{ckpt_type} not found on disk")
            self.net.load_state_dict(resume_ckpt.net_state)
            opt_state = NNCheckpoint.load_optimizer_state(
                run=params.resume_from_run_id,
                type=ckpt_type,
            )
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

        scheduler = self._build_scheduler(optimizer, params)
        scaler = self._build_grad_scaler()

        normalized_callbacks = self._normalize_callbacks(callbacks)

        idps: list[NNIterationDataPoint] = []
        # `len()` is not defined on iterable-style DataLoaders (IterableDataset).
        # Fall back to None so tqdm renders without a total instead of crashing.
        try:
            n_iter: Optional[int] = int(params.n_epochs * len(params.train_loader))
        except TypeError:
            n_iter = None
        best_checkpoint: Optional[NNCheckpoint] = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)

        Utils.print_table(header=False, title="Run Details...", data=Utils.flatten_dict(data=run.state()))

        ctx = _CallbackContext(model=self, run=run, optimizer=optimizer)
        for cb in normalized_callbacks:
            cb.on_train_begin(ctx)

        # Default to the standard supervised step when the caller doesn't
        # override. Custom step gets dispatched from inside the batch loop
        # below so the rest of train() (scheduler, callbacks, checkpoint
        # cadence, val loop, incremental save) is identical either way.
        # Explicit None check (not `or`) so a hypothetical callable that
        # happens to be falsy by __bool__ doesn't silently fall back.
        step_fn: TrainStepFn = default_train_step if train_step_fn is None else train_step_fn

        idx_iter = 0
        # Respect NNX_TQDM_DISABLE=1 in tests / CI / non-TTY environments so
        # the progress bar doesn't pollute output. Same env var works as
        # well in subprocess contexts where the user can't pass a flag.
        tqdm_disabled = os.environ.get("NNX_TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
        with (
            torch.set_grad_enabled(True),
            tqdm(colour="blue", total=n_iter, desc="Training", disable=tqdm_disabled) as tqdm_bar,
        ):
            for idx_epoch in range(params.n_epochs):
                ctx.epoch = idx_epoch
                for cb in normalized_callbacks:
                    cb.on_epoch_begin(ctx)

                n_idps_before_epoch = len(idps)
                for idx_batch, batch in enumerate(params.train_loader):
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

                val_edp = (
                    self.evaluate(loader=params.val_loader, extra_metrics=params.extra_metrics) if validate else None
                )
                idps[-1] = idps[-1].with_val_edp(val_edp)

                checkpoint = self._save_checkpoints(
                    idp=idps[-1],
                    run_id=run.id,
                    idx_epoch=idx_epoch,
                    n_epochs=params.n_epochs,
                    best_checkpoint=best_checkpoint,
                    save_phase_checkpoints=params.save_phase_checkpoints,
                    optimizer=optimizer,
                )
                # In-memory best_checkpoint tracking must use the same
                # comparison as the on-disk BEST write inside
                # _save_checkpoints (val→train, error→loss, +inf fall-through).
                # Without this, val_loader=None runs would silently overwrite
                # best_checkpoint every epoch (because checkpoint.idp.val_edp
                # is None there) while the on-disk BEST tracks training error,
                # diverging the two views of "best".
                if best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
                    best_checkpoint = checkpoint

                self._step_scheduler(scheduler, val_edp, train_edp)
                self._update_tqdm_postfix(tqdm_bar, optimizer, val_edp, train_edp)

                # Incremental persistence: write idps.csv + run.yaml after
                # every epoch. KeyboardInterrupt / OOM during training now
                # leaves a partial-but-loadable run on disk. The extra
                # writes are O(idps so far) per epoch — negligible vs the
                # checkpoint write that already happens.
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

    def evaluate(self, loader: DataLoader, extra_metrics=None) -> NNEvaluationDataPoint:
        """Aggregate predictions across all batches in `loader` and compute
        a single NNEvaluationDataPoint. Aggregating (rather than averaging
        per-batch metrics) gives correct sample-weighted f1/precision/recall
        when the final batch is short.

        Raises ValueError if the loader yields zero batches — previously
        produced NaN metrics silently from np.mean over an empty list.
        """
        # Ensure loss_fn lives on the same device as the model — guards
        # against callers reassigning self.device after construction.
        self.loss_fn = self.loss_fn.to(self.device)
        # Snapshot training-mode for non-destructive restore (matches the
        # convention already used by `nnx.viz.activation_map` and
        # `nnx.lr_finder`). Without this, a caller doing the common
        # train → evaluate → train-more pattern silently leaves the net
        # in `.eval()` mode after evaluate(); BatchNorm / Dropout layers
        # would behave incorrectly on the next batch unless the caller
        # remembered to call `self.net.train()` themselves.
        was_training = self.net.training
        self.net.eval()

        all_Y: list[np.ndarray] = []
        all_Y_hat: list[np.ndarray] = []
        loss_sum = 0.0
        n_samples = 0

        try:
            with torch.no_grad():
                for batch in loader:
                    _, Y, Y_hat_logits, Y_hat = self._fwd_pass(batch)
                    batch_n = int(Y.size(0))
                    # Aggregate predictions / labels across the entire loader so
                    # metrics are computed on the full eval set, not per-batch.
                    all_Y.append(Y.cpu().numpy())
                    all_Y_hat.append(Y_hat.cpu().numpy())
                    # Sum-weight the loss by samples; divide once at the end.
                    loss_sum += float(self.loss_fn(Y_hat_logits, Y).detach()) * batch_n
                    n_samples += batch_n
        finally:
            if was_training:
                self.net.train()

        if n_samples == 0:
            raise ValueError("evaluate() loader produced zero samples")

        Y_concat = np.concatenate(all_Y)
        Y_hat_concat = np.concatenate(all_Y_hat)

        return (
            NNEvaluationDataPoint.of(Y=Y_concat, Y_hat=Y_hat_concat, extra_metrics=extra_metrics)
            .with_loss(value=loss_sum / n_samples)
            .with_error(value=float(1 - (Y_concat == Y_hat_concat).sum() / n_samples))
        )

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
        that unpacks like the original 2-tuple.

        Non-destructive: ``self.net.training`` is snapshotted before
        switching to ``eval()`` and restored on exit (matches
        ``NNModel.evaluate``, ``nnx.viz.activation_map``, and
        ``nnx.lr_finder``). Without this, a caller doing the common
        train → predict → train-more pattern silently leaves the net
        in ``.eval()`` mode.
        """
        was_training = self.net.training
        self.net.eval()

        try:
            if isinstance(X, DataLoader):
                logits_chunks: list[np.ndarray] = []
                classes_chunks: list[np.ndarray] = []
                with torch.no_grad():
                    for batch in X:
                        # net.unpack_batch handles both (X, Y) tuples and PyG Data,
                        # returning the X-tuple. The label is discarded for predict.
                        X_in, _ = self.net.unpack_batch(batch)
                        X_in = tuple(x.to(self.device) for x in X_in)
                        logits = self.net(*X_in).cpu().numpy()
                        # NeighborLoader subgraphs: only the leading seed
                        # rows are this batch's nodes (see
                        # GraphNNBase.seed_count) — without the slice,
                        # predictions for sampled neighbors pollute the
                        # output and the row count exceeds the loader's
                        # node set.
                        seed_count = getattr(self.net, "seed_count", None)
                        if seed_count is not None:
                            n_seed = seed_count(batch)
                            if n_seed is not None:
                                logits = logits[:n_seed]
                        logits_chunks.append(logits)
                        classes_chunks.append(logits.argmax(axis=1))
                return PredictResult(
                    logits=np.concatenate(logits_chunks),
                    classes=np.concatenate(classes_chunks),
                )

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
                Y_hat = Y_hat_logits.argmax(axis=1)
                return PredictResult(logits=Y_hat_logits, classes=Y_hat)
        finally:
            if was_training:
                self.net.train()

    def _fwd_pass(self, batch):
        """Standard supervised forward pass: unpack batch, move to device,
        run net, take argmax over class logits. Used by `default_train_step`
        and `evaluate()`; custom train_step_fn's may call this directly
        or roll their own forward pass."""
        X, Y = self.net.unpack_batch(batch)

        X = tuple(x.to(self.device) for x in X)
        Y = Y.to(self.device)

        Y_hat_logits = self.net(*X)
        # Graph nets score every node in the sampled subgraph, but only
        # the leading seed rows belong to this batch's split — see
        # GraphNNBase.seed_count for the leakage this prevents.
        seed_count = getattr(self.net, "seed_count", None)
        if seed_count is not None:
            n_seed = seed_count(batch)
            if n_seed is not None:
                Y_hat_logits = Y_hat_logits[:n_seed]
                Y = Y[:n_seed]
        Y_hat = Y_hat_logits.argmax(dim=1)

        return X, Y, Y_hat_logits, Y_hat

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
    ) -> NNCheckpoint:
        checkpoint = NNCheckpoint(
            idp=idp, model_params=self.params, net_params=self.net_params, net_state=self.net.state_dict()
        )
        # The optimizer state-dict goes only into LAST and BEST sidecars
        # (resume points). Phase markers (FIRST/Q*) don't carry one.
        opt_state = optimizer.state_dict() if optimizer is not None else None

        # Phase markers at epoch boundaries — fractions are nominal (1/4, 2/4,
        # 3/4 of the planned epoch count); off-by-one allowed when n_epochs
        # isn't divisible by 4. See `phase_tag` for the small-`n_epochs`
        # caveat. Opt-out via NNTrainParams.save_phase_checkpoints.
        if save_phase_checkpoints:
            tag = phase_tag(idx_epoch, n_epochs)
            if tag is not None:
                checkpoint.save(run=run_id, type=tag)

        checkpoint.save(run=run_id, type=Checkpoints.LAST, optimizer_state=opt_state)

        # BEST tracking goes through the same _best_err helper used by
        # NNRun.save's cross-run comparison and by Trainer._save_checkpoint
        # — single source of truth for "what's the comparable error here"
        # (val→train, error→loss, +inf fall-through, tolerating None EDP
        # or None .error from custom train_step_fn paradigms).
        if best_checkpoint is None or _best_err(checkpoint) < _best_err(best_checkpoint):
            checkpoint.save(run=run_id, type=Checkpoints.BEST, optimizer_state=opt_state)

        return checkpoint

    def _step_scheduler(
        self,
        scheduler,
        val_edp: Optional[NNEvaluationDataPoint],
        train_edp: NNEvaluationDataPoint,
    ) -> None:
        # ReduceLROnPlateau wants a metric; other schedulers step on epoch index.
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            # Custom train_step_fn hooks may leave .error unset;
            # ReduceLROnPlateau.step(None) crashes inside float(). Use the
            # shared val→train, error→loss fallback resolver so the four
            # call sites (NNModel + Trainer × scheduler + tqdm) can't drift.
            metric = _resolve_metric(val_edp, train_edp)
            if metric is None:
                # No signal to feed the scheduler — skip the step. The user
                # picked a metric-driven scheduler without producing a metric;
                # better to no-op than to crash mid-train.
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
