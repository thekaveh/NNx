"""Train a feed-forward classifier on synthetic 3-class data.

Demonstrates the core NNModel flow: build params → train with callbacks →
inspect the resulting NNRun → reload the BEST checkpoint and predict.

Run:
    python examples/01_synthetic_classification.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    EarlyStopping,
    Losses,
    LRMonitor,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    set_seed,
)


def main():
    set_seed(0)

    # 1. Build a tiny synthetic dataset. Labels derive from the inputs
    #    through a fixed random projection so the task is LEARNABLE —
    #    with random labels the val error would sit at 3-class chance
    #    and the BEST checkpoint would be selecting noise.
    n_train, n_val = 256, 64
    proj = torch.randn(8, 3)
    X_train = torch.randn(n_train, 8)
    y_train = (X_train @ proj).argmax(dim=1)
    X_val = torch.randn(n_val, 8)
    y_val = (X_val @ proj).argmax(dim=1)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # 2. Model.
    net_params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[32, 16],
        dropout_prob=0.1,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    model = NNModel(net_params=net_params, params=model_params)

    # 3. Train.
    train_params = NNTrainParams(
        n_epochs=20,
        seed=0,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
            momentum=(0.9, 0.999),
            weight_decay=5e-5,
            grad_clip_norm=1.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=3,
            cooldown=1,
            threshold=1e-3,
        ),
    )

    lr_monitor = LRMonitor()
    run = model.train(
        params=train_params,
        callbacks=[EarlyStopping(patience=8), lr_monitor],
    )

    # 4. Inspect.
    print(f"\nrun.id = {run.id}")
    print(f"completed iterations: {len(run.idps)}")
    last = run.idps[-1]
    print(f"final train loss: {last.train_edp.loss:.4f}, error: {last.train_edp.error:.4f}")
    if last.val_edp is not None:
        print(f"final val   loss: {last.val_edp.loss:.4f}, error: {last.val_edp.error:.4f}")
    print(f"LR trajectory: {[f'{lr:.4f}' for lr in lr_monitor.history]}")

    # 5. Reload the BEST checkpoint and run prediction.
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    best_model = NNModel.from_checkpoint(checkpoint=ckpt)
    result = best_model.predict(X=X_val)
    print(f"predicted classes for {len(result.classes)} val samples; first 8: {result.classes[:8]}")


if __name__ == "__main__":
    main()
