from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import Utils
from .enum.checkpoints import Checkpoints
from .params.nn_checkpoint import NNCheckpoint
from .params.nn_evaluation_data_point import NNEvaluationDataPoint
from .params.nn_iteration_data_point import NNIterationDataPoint
from .params.nn_model_params import NNModelParams
from .params.nn_params import NNParams
from .params.nn_run import NNRun
from .params.nn_train_params import NNTrainParams

if TYPE_CHECKING:
    from .callbacks import Callback


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


class NNModel:
    def __init__(
        self
        , net_params: NNParams
        , params    : NNModelParams
    ):
        if net_params is None:
            raise ValueError("net_params must not be None")

        self.net_params = net_params
        self.params     = params

        self.device     = self.params.device()
        self.loss_fn    = self.params.loss().to(self.device)
        self.net        = self.params.net(params=net_params).to(self.device)

    def to_onnx(
        self,
        path: str,
        example_input: Union[torch.Tensor, tuple, np.ndarray],
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_batch: bool = True,
        opset_version: int = 17,
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

        Returns the path written. Network is put in eval mode for tracing.
        """
        if isinstance(example_input, torch.Tensor):
            example_input = (example_input.to(self.device),)
        else:
            example_input = tuple(
                (e.to(self.device) if isinstance(e, torch.Tensor)
                 else torch.from_numpy(np.asarray(e)).to(self.device))
                for e in example_input
            )

        in_names = input_names or [f"input_{i}" for i in range(len(example_input))]
        out_names = output_names or ["output"]

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {n: {0: "batch"} for n in in_names + out_names}

        self.net.eval()
        # torch>=2.5 defaults torch.onnx.export to the dynamo-based exporter,
        # which requires `onnxscript`. We use the legacy tracing exporter
        # (dynamo=False) so plain `pip install onnx` is enough.
        try:
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
            # already use the legacy path by default.
            torch.onnx.export(
                self.net,
                example_input,
                path,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
            )
        return path

    @staticmethod
    def from_checkpoint(checkpoint: NNCheckpoint) -> NNModel:
        model = NNModel(
            params=checkpoint.model_params
            , net_params=checkpoint.net_params
        )

        model.net.load_state_dict(checkpoint.net_state)

        return model

    def train(
        self,
        params: NNTrainParams,
        callbacks: Optional[list[CallbackLike]] = None,
    ) -> NNRun:
        if params is None or params.optim is None or not params.optim.is_valid():
            raise ValueError("train params must be non-None and have a valid optim config")

        # V1: seed every RNG before constructing the run so dataset shuffling,
        # weight init, dropout — anything stochastic — is reproducible. The
        # `seed` field only affects state() (and run.id) when explicitly set,
        # so back-compat for no-seed callers is preserved.
        if params.seed is not None:
            from ..seeding import set_seed
            set_seed(params.seed)

        validate    : bool  = params.val_loader is not None
        run         : NNRun = NNRun(
            train   = params
            , model = self.params
            , net   = self.net.params
        )

        optimizer = params.optim.name(
            net=self.net
            , lr_start=params.optim.max_lr
            , momentum=params.optim.momentum
            , weight_decay=params.optim.weight_decay
        )

        # Warm resume: load weights + optimizer state from a prior run's
        # checkpoint. The .opt.pt sidecar is best-effort — pre-resume
        # checkpoints don't have it, in which case we still load weights
        # but the optimizer starts fresh.
        if params.resume_from_run_id is not None:
            ckpt_type = Checkpoints(params.resume_from_checkpoint)
            resume_ckpt = NNCheckpoint.load(run=params.resume_from_run_id, type=ckpt_type)
            if resume_ckpt is None:
                raise ValueError(
                    f"resume_from_run_id={params.resume_from_run_id!r}/"
                    f"{ckpt_type} not found on disk"
                )
            self.net.load_state_dict(resume_ckpt.net_state)
            opt_state = NNCheckpoint.load_optimizer_state(
                run=params.resume_from_run_id, type=ckpt_type,
            )
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

        scheduler = self._build_scheduler(optimizer, params)
        scaler = self._build_grad_scaler()

        normalized_callbacks = self._normalize_callbacks(callbacks)

        idps        : list[NNIterationDataPoint] = []
        # `len()` is not defined on iterable-style DataLoaders (IterableDataset).
        # Fall back to None so tqdm renders without a total instead of crashing.
        try:
            n_iter: Optional[int] = int(params.n_epochs * len(params.train_loader))
        except TypeError:
            n_iter = None
        best_checkpoint : Optional[NNCheckpoint]   = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)

        Utils.print_table(
            header=False
            , title="Run Details..."
            , data=Utils.flatten_dict(data=run.state())
        )

        ctx = _CallbackContext(model=self, run=run, optimizer=optimizer)
        for cb in normalized_callbacks:
            cb.on_train_begin(ctx)

        idx_iter = 0
        # Respect NNX_TQDM_DISABLE=1 in tests / CI / non-TTY environments so
        # the progress bar doesn't pollute output. Same env var works as
        # well in subprocess contexts where the user can't pass a flag.
        tqdm_disabled = os.environ.get("NNX_TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
        with (
            torch.set_grad_enabled(True)
            , tqdm(colour="blue", total=n_iter, desc="Training", disable=tqdm_disabled) as tqdm_bar
        ):
            for idx_epoch in range(params.n_epochs):
                ctx.epoch = idx_epoch
                for cb in normalized_callbacks:
                    cb.on_epoch_begin(ctx)

                for idx_batch, batch in enumerate(params.train_loader):
                    train_edp = self._train_step(
                        batch, optimizer, scaler,
                        grad_clip_norm=params.optim.grad_clip_norm,
                        extra_metrics=params.extra_metrics,
                        accumulate_grad_batches=params.optim.accumulate_grad_batches,
                        batch_idx=idx_batch,
                    )

                    idps.append(NNIterationDataPoint(
                        iter_idx    = idx_iter
                        , epoch_idx = idx_epoch
                        , batch_idx = idx_batch
                        , train_edp = train_edp
                        , lr        = optimizer.param_groups[0]['lr']
                    ))

                    idx_iter += 1
                    tqdm_bar.update(1)

                val_edp = self.evaluate(loader=params.val_loader, extra_metrics=params.extra_metrics) if validate else None
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
                if best_checkpoint is None or checkpoint.idp.val_edp is None or (
                    best_checkpoint.idp.val_edp is None
                    or checkpoint.idp.val_edp.error < best_checkpoint.idp.val_edp.error
                ):
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
        self.net.eval()

        all_Y: list[np.ndarray] = []
        all_Y_hat: list[np.ndarray] = []
        loss_sum = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in loader:
                _, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                batch_n = int(Y.size(0))
                # Aggregate predictions / labels across the entire loader so
                # metrics are computed on the full eval set, not per-batch.
                all_Y.append(Y.cpu().numpy())
                all_Y_hat.append(Y_hat.cpu().numpy())
                # Sum-weight the loss by samples; divide once at the end.
                loss_sum += float(self.loss_fn(Y_hat_log, Y).detach()) * batch_n
                n_samples += batch_n

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
        """
        self.net.eval()

        if isinstance(X, DataLoader):
            logits_chunks: list[np.ndarray] = []
            classes_chunks: list[np.ndarray] = []
            with torch.no_grad():
                for batch in X:
                    # net.unpack_batch handles both (X, Y) tuples and PyG Data,
                    # returning the X-tuple. The label is discarded for predict.
                    X_in, _ = self.net.unpack_batch(batch)
                    X_in = tuple(x.to(self.device) for x in X_in)
                    log = self.net(*X_in).cpu().numpy()
                    logits_chunks.append(log)
                    classes_chunks.append(log.argmax(axis=1))
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
            Y_hat_log = self.net(*X_t).cpu().numpy()
            Y_hat = Y_hat_log.argmax(axis=1)
            return PredictResult(logits=Y_hat_log, classes=Y_hat)

    def __fwd_pass(self, batch):
        X, Y = self.net.unpack_batch(batch)

        X = tuple(x.to(self.device) for x in X)
        Y = Y.to(self.device)

        Y_hat_log = self.net(*X)
        Y_hat = Y_hat_log.argmax(dim=1)

        return X, Y, Y_hat_log, Y_hat

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
        self.net.train()

        # Gradient accumulation: only zero grads at the start of a fresh
        # accumulation cycle, and only step the optimizer at the end of one.
        is_cycle_start = (batch_idx % accumulate_grad_batches) == 0
        is_cycle_end = ((batch_idx + 1) % accumulate_grad_batches) == 0
        if is_cycle_start:
            self.net.zero_grad()

        # Mixed precision is opt-in via NNModelParams.mixed_precision; only
        # takes effect on CUDA where autocast + GradScaler are meaningful.
        amp_enabled = scaler is not None and self.device.type == "cuda"

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda"):
                X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                train_loss = self.loss_fn(Y_hat_log, Y)
            # Scale loss by 1/N so accumulated grads = mean across batches.
            scaler.scale(train_loss / accumulate_grad_batches).backward()
            if is_cycle_end:
                if grad_clip_norm is not None:
                    # Unscale before clipping so the clip threshold applies
                    # in the original gradient space, not the scaled one.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
            train_loss = self.loss_fn(Y_hat_log, Y)
            (train_loss / accumulate_grad_batches).backward()
            if is_cycle_end:
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip_norm)
                optimizer.step()

        loss_value = float(train_loss.detach())
        # NaN/Inf guard: silent divergence leaves checkpoints full of garbage
        # weights. Raise so the training session terminates loudly.
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"non-finite training loss ({loss_value!r}) — training diverged. "
                "Check learning rate, gradient clipping (NNOptimParams.grad_clip_norm), "
                "or input normalization."
            )

        return (
            NNEvaluationDataPoint.of(
                Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy(),
                extra_metrics=extra_metrics,
            )
                .with_loss(value=loss_value)
                .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
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
                mode='min',
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
            idp           = idp
            , model_params= self.params
            , net_params  = self.net_params
            , net_state   = self.net.state_dict()
        )
        # The optimizer state-dict goes only into LAST and BEST sidecars
        # (resume points). Phase markers (FIRST/Q*) don't carry one.
        opt_state = optimizer.state_dict() if optimizer is not None else None

        # Phase markers at epoch boundaries — fractions are nominal (1/4, 2/4,
        # 3/4 of the planned epoch count); off-by-one allowed when n_epochs
        # isn't divisible by 4. Opt-out via NNTrainParams.save_phase_checkpoints.
        if save_phase_checkpoints:
            if idx_epoch == 0:
                checkpoint.save(run=run_id, type=Checkpoints.FIRST)
            elif idx_epoch == int(n_epochs * 1 / 4) - 1:
                checkpoint.save(run=run_id, type=Checkpoints.Q1)
            elif idx_epoch == int(n_epochs * 2 / 4) - 1:
                checkpoint.save(run=run_id, type=Checkpoints.Q2)
            elif idx_epoch == int(n_epochs * 3 / 4) - 1:
                checkpoint.save(run=run_id, type=Checkpoints.Q3)

        checkpoint.save(run=run_id, type=Checkpoints.LAST, optimizer_state=opt_state)

        # BEST is decided by validation error if available, else training error.
        def _err(c: NNCheckpoint) -> float:
            edp = c.idp.val_edp if c.idp.val_edp is not None else c.idp.train_edp
            return edp.error if edp is not None else float("inf")

        if best_checkpoint is None or _err(checkpoint) < _err(best_checkpoint):
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
            metric = val_edp.error if val_edp is not None else train_edp.error
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
        lr = optimizer.param_groups[0]['lr']
        err = val_edp.error if val_edp is not None else train_edp.error
        tqdm_bar.set_postfix_str(f"error={err:.4f}, lr={lr:.4f}")

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
