"""Born-Again Networks — iterated self-distillation across G generations.

Pipeline:

  1. Train a baseline classifier (Generation 0).
  2. For each subsequent generation, treat the previous generation
     as a frozen teacher and train a fresh student against it via
     `nnx.paradigms.kd_train_step_factory`.
  3. Print per-generation val accuracy.

The Furlanello et al. ICML 2018 result is that successive generations
often match or slightly outperform Generation 0 — the soft targets
act as an implicit regularizer.

Run:
    pip install nnx
    python examples/23_born_again_distillation.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    born_again_train,
    set_seed,
)


def main() -> None:
    set_seed(42)
    X = torch.randn(1024, 16)
    proj = torch.randn(16, 3)
    y = (X @ proj).argmax(dim=1)
    train_loader = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[800:], y[800:]), batch_size=64)

    net_params = NNParams(
        input_dim=16,
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
    train_params = NNTrainParams(
        n_epochs=5,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-6,
            factor=0.5,
            patience=2,
            cooldown=1,
            threshold=1e-3,
        ),
    )

    model = NNModel(net_params=net_params, params=model_params)
    runs = born_again_train(
        model,
        generations=3,
        train_params=train_params,
        alpha=0.5,
        temperature=4.0,
    )

    print(f"{len(runs)} generation runs returned.")
    for g, run in enumerate(runs):
        val_idps = [idp for idp in run.idps if idp.val_edp is not None]
        val_acc = 1.0 - val_idps[-1].val_edp.error
        print(f"  Gen {g}: val accuracy = {val_acc:.3f}")


if __name__ == "__main__":
    main()
