from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Union

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

        scheduler = self._build_scheduler(optimizer, params)
        scaler = self._build_grad_scaler()

        normalized_callbacks = self._normalize_callbacks(callbacks)

        idps        : list[NNIterationDataPoint] = []
        n_iter      : int                          = int(params.n_epochs * len(params.train_loader))
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
        with (
            torch.set_grad_enabled(True)
            , tqdm(colour="blue", total=n_iter, desc="Training") as tqdm_bar
        ):
            for idx_epoch in range(params.n_epochs):
                ctx.epoch = idx_epoch
                for cb in normalized_callbacks:
                    cb.on_epoch_begin(ctx)

                for idx_batch, batch in enumerate(params.train_loader):
                    train_edp = self._train_step(batch, optimizer, scaler)

                    idps.append(NNIterationDataPoint(
                        iter_idx    = idx_iter
                        , epoch_idx = idx_epoch
                        , batch_idx = idx_batch
                        , train_edp = train_edp
                        , lr        = optimizer.param_groups[0]['lr']
                    ))

                    idx_iter += 1
                    tqdm_bar.update(1)

                val_edp = self.evaluate(loader=params.val_loader) if validate else None
                idps[-1] = idps[-1].with_val_edp(val_edp)

                checkpoint = self._save_checkpoints(
                    idp=idps[-1],
                    run_id=run.id,
                    idx_epoch=idx_epoch,
                    n_epochs=params.n_epochs,
                    best_checkpoint=best_checkpoint,
                )
                if best_checkpoint is None or checkpoint.idp.val_edp is None or (
                    best_checkpoint.idp.val_edp is None
                    or checkpoint.idp.val_edp.error < best_checkpoint.idp.val_edp.error
                ):
                    best_checkpoint = checkpoint

                self._step_scheduler(scheduler, val_edp, train_edp)
                self._update_tqdm_postfix(tqdm_bar, optimizer, val_edp, train_edp)

                ctx.idp = idps[-1]
                ctx.idps = idps
                for cb in normalized_callbacks:
                    cb.on_epoch_end(ctx)

                if ctx.should_stop:
                    break

            for cb in normalized_callbacks:
                cb.on_train_end(ctx)

        print()
        return run.with_idps(idps).save()

    def evaluate(self, loader: DataLoader) -> NNEvaluationDataPoint:
        self.net.eval()
        edps = []

        with torch.no_grad():
            for batch in loader:
                _, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)

                edps.append(
                    NNEvaluationDataPoint.of(Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy())
                        .with_loss(value=float(self.loss_fn(Y_hat_log, Y)))
                        .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
                )

        return NNEvaluationDataPoint.mean_of(edps)

    def predict(self, X: np.ndarray):
        if not isinstance(X, tuple):
            X = (X,)

        X = tuple(torch.from_numpy(x).to(self.device) for x in X)

        self.net.eval()
        with torch.no_grad():
            Y_hat_log = self.net(*X).cpu().numpy()
            Y_hat = Y_hat_log.argmax(axis=1)

            return Y_hat_log, Y_hat

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
    ) -> NNEvaluationDataPoint:
        self.net.train()
        self.net.zero_grad()

        # Mixed precision is opt-in via NNModelParams.mixed_precision; only
        # takes effect on CUDA where autocast + GradScaler are meaningful.
        amp_enabled = scaler is not None and self.device.type == "cuda"

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda"):
                X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                train_loss = self.loss_fn(Y_hat_log, Y)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
            train_loss = self.loss_fn(Y_hat_log, Y)
            train_loss.backward()
            optimizer.step()

        return (
            NNEvaluationDataPoint.of(Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy())
                .with_loss(value=float(train_loss))
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
    ) -> NNCheckpoint:
        checkpoint = NNCheckpoint(
            idp           = idp
            , model_params= self.params
            , net_params  = self.net_params
            , net_state   = self.net.state_dict()
        )

        # Phase markers at epoch boundaries — fractions are nominal (1/4, 2/4,
        # 3/4 of the planned epoch count); off-by-one allowed when n_epochs
        # isn't divisible by 4.
        if idx_epoch == 0:
            checkpoint.save(run=run_id, type=Checkpoints.FIRST)
        elif idx_epoch == int(n_epochs * 1 / 4) - 1:
            checkpoint.save(run=run_id, type=Checkpoints.Q1)
        elif idx_epoch == int(n_epochs * 2 / 4) - 1:
            checkpoint.save(run=run_id, type=Checkpoints.Q2)
        elif idx_epoch == int(n_epochs * 3 / 4) - 1:
            checkpoint.save(run=run_id, type=Checkpoints.Q3)

        checkpoint.save(run=run_id, type=Checkpoints.LAST)

        # BEST is decided by validation error if available, else training error.
        def _err(c: NNCheckpoint) -> float:
            edp = c.idp.val_edp if c.idp.val_edp is not None else c.idp.train_edp
            return edp.error if edp is not None else float("inf")

        if best_checkpoint is None or _err(checkpoint) < _err(best_checkpoint):
            checkpoint.save(run=run_id, type=Checkpoints.BEST)

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
