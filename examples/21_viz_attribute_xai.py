"""Captum-backed input attribution demo — ``nnx.viz.attribute``.

Pipeline:

  1. Train a tiny classifier on synthetic data.
  2. Run integrated_gradients via `nnx.viz.attribute(model.net, X, method="integrated_gradients", target=0)`.
  3. Print the attribution tensor's shape + the figure's basic info.
  4. Try four other Captum methods to demonstrate the dispatcher.

The Captum API surface is large; NNx's wrapper folds the six most
useful methods (integrated_gradients, gradient_shap, deep_lift,
saliency, input_x_gradient, occlusion) behind a single string-keyed
dispatch with sensible per-method defaults.

Requires the ``viz`` optional extra (for ``captum`` — the input-
attribution backend ``nnx.viz.attribute`` dispatches into):

    pip install 'thekaveh-nnx[viz]'

Run:
    python examples/21_viz_attribute_xai.py
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
    set_seed,
)
from nnx.viz import attribute


def main() -> None:
    set_seed(42)
    X = torch.randn(256, 8)
    proj = torch.randn(8, 3)
    y = (X @ proj).argmax(dim=1)
    train_loader = DataLoader(TensorDataset(X[:200], y[:200]), batch_size=32, shuffle=True)

    model = NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[32, 16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    model.train(
        params=NNTrainParams(
            n_epochs=5,
            train_loader=train_loader,
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
        ),
    )

    sample = X[:4]
    target_class = 0

    for method in ("integrated_gradients", "saliency", "input_x_gradient", "deep_lift"):
        attr, fig = attribute(model.net, sample, method=method, target=target_class)
        print(f"{method:>22} → attribution shape: {tuple(attr.shape)}, figure type: {type(fig).__name__}")

    print("\nNote: pass `target=class_idx` to attribute the class the explanation should target.")
    print("`baselines=` is auto-zeroed for gradient_shap; `sliding_window_shapes=` is auto-set for occlusion.")


if __name__ == "__main__":
    main()
