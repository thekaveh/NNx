from __future__ import annotations

import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from dataclasses import asdict
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from IPython.display import clear_output
from typing import Callable, List, Optional

from .enum.losses import Losses
from .enum.devices import Devices
from .enum.checkpoints import Checkpoints

from ..utils import Utils
from .params.nn_run import NNRun
from .params.nn_params import NNParams
from .params.nn_checkpoint import NNCheckpoint
from .params.nn_train_params import NNTrainParams
from .params.nn_model_params import NNModelParams
from .params.nn_iteration_data_point import NNIterationDataPoint
from .params.nn_evaluation_data_point import NNEvaluationDataPoint

class NNModel():
    def __init__(
        self
        , net_params: NNParams
        , params    : NNModelParams
    ):
        assert net_params is not None
        
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

    def train(self, params: NNTrainParams, callbacks: Optional[List[Callable[[List[NNIterationDataPoint]], None]]] = None) -> NNRun:
        assert (
            params is not None
            and params.optim is not None
            and params.optim.is_valid()
        )

        validate    : bool  = params.val_loader is not None
        run         : NNRun = NNRun(
            train   = params
            , model = self.params
            , net   = self.net.params
        )
            
        optimizer   = params.optim.name(
            net=self.net
            , lr_start=params.optim.max_lr
            , momentum=params.optim.momentum
            , weight_decay=params.optim.weight_decay
        )

        scheduler   = lr_scheduler.ReduceLROnPlateau(
            optimizer
            , mode='min'
            , min_lr=params.scheduler.min_lr
            , factor=params.scheduler.factor
            , cooldown=params.scheduler.cooldown
            , patience=params.scheduler.patience
            , threshold=params.scheduler.threshold
        )
        
        idx_iter        : int                           = 0
        idps            : List[NNIterationDataPoint]    = []
        n_iter          : int                           = int(params.n_epochs * len(params.train_loader))
        best_checkpoint : Optional[NNCheckpoint]        = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
        
        Utils.print_table(
            header=False
            , title="Run Details..."
            , data=Utils.flatten_dict(
                data=run.state()
            )
        )

        with (
            torch.set_grad_enabled(True)
            , tqdm(
                colour="blue"
                , total=n_iter
                , desc="Training"
            ) as tqdm_bar
        ):
            for idx_epoch in range(params.n_epochs):
                for idx_batch, batch in enumerate(params.train_loader):
                    self.net.train()
                    self.net.zero_grad()
                        
                    X, Y, Y_hat_log, Y_hat = self.__fwd_pass(batch)
                    
                    train_loss = self.loss_fn(Y_hat_log, Y)
                    
                    train_edp = (
                        NNEvaluationDataPoint.of(Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy())
                            .with_loss(value=float(train_loss))
                            .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
                    )

                    idps.append(
                        NNIterationDataPoint(
                            iter_idx    = idx_iter
                            , epoch_idx = idx_epoch
                            , batch_idx = idx_batch
                            , train_edp = train_edp
                            , lr        = optimizer.param_groups[0]['lr']
                        )
                    )

                    train_loss.backward()
                    optimizer.step()

                    idx_iter += 1
                    tqdm_bar.update(1)

                val_edp     = self.evaluate(loader=params.val_loader) if validate else None           
                idps[-1]    = idps[-1].with_val_edp(val_edp)           
                checkpoint  = NNCheckpoint(
                    idp             = idps[-1]
                    , model_params= self.params
                    , net_params  = self.net_params
                    , net_state   = self.net.state_dict()
                )
                
                if idx_epoch == 0:
                    checkpoint.save(run=run.id, type=Checkpoints.FIRST)     
                elif idx_epoch == int(params.n_epochs * 1 / 4) - 1:
                    checkpoint.save(run=run.id, type=Checkpoints.Q1)     
                elif idx_epoch == int(params.n_epochs * 2 / 4) - 1:
                    checkpoint.save(run=run.id, type=Checkpoints.Q2)  
                elif idx_epoch == int(params.n_epochs * 3 / 4) - 1:
                    checkpoint.save(run=run.id, type=Checkpoints.Q3)
                    
                checkpoint.save(run=run.id, type=Checkpoints.LAST)
                
                if best_checkpoint is None or checkpoint.idp.val_edp.error < best_checkpoint.idp.val_edp.error:
                    best_checkpoint = checkpoint
                    checkpoint.save(run=run.id, type=Checkpoints.BEST)
                
                scheduler.step(val_edp.error if val_edp is not None else train_edp)
                
                tqdm_bar.set_postfix_str(
                    "error={error}, lr={lr}"
                        .format(
                            lr      = f"{optimizer.param_groups[0]['lr']:.4f}"
                            , error = f"{val_edp.error if val_edp is not None else train_edp.error:.4f}"
                        )
                )
                
                if callbacks is not None:
                    clear_output(wait=True)
                    
                    for callback in callbacks:
                        callback(idps)

        print()
        return run.with_idps(idps).save()

    def evaluate(self, loader: DataLoader):
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

    def __fwd_pass(self, batch):
        X, Y = self.net.unpack_batch(batch)
        
        X = tuple(x.to(self.device) for x in X)
        Y = Y.to(self.device)
        
        Y_hat_log = self.net(*X)
        Y_hat = Y_hat_log.argmax(dim=1)
        
        return X, Y, Y_hat_log, Y_hat
    
    def predict(self, X: np.ndarray):
        if not isinstance(X, tuple):
            X = (X,)
        
        X = tuple(torch.from_numpy(x).to(self.device) for x in X)
        
        self.net.eval()
        with torch.no_grad():
            Y_hat_log = self.net(*X).cpu().numpy()
            Y_hat = Y_hat_log.argmax(axis=1)
            
            return Y_hat_log, Y_hat