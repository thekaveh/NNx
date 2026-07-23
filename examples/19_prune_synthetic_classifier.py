"""Magnitude pruning demo — sparsify a tiny classifier, verify accuracy holds.

Pipeline:

  1. Build a small FeedFwdNN classifier on a synthetic 3-class dataset
     (no real MNIST download; keeps the example CPU-runnable + offline).
  2. Train to convergence.
  3. Snapshot the FP32 val accuracy.
  4. Apply `nnx.prune.magnitude_prune(net, sparsity=0.5)`. With
     `bake=True` (default), the state_dict keys stay identical to the
     pre-prune network — the checkpoint loads back into stock code
     under `strict=True`.
  5. Compare:
     - Sparsity (% of weight entries set to zero).
     - Validation accuracy before vs. after.
  6. Briefly fine-tune, then RE-prune to 50% and report the accuracy
     of the network that actually ships sparse. With `bake=True` the
     mask is dropped, so plain fine-tuning regrows the zeroed weights —
     an iterative prune→tune schedule would use `bake=False` masks.

Magnitude pruning sets the smallest-magnitude weight entries to zero
based on their L1 norm. At 50% sparsity on a small classifier this is
typically nondestructive; the tune→re-prune step recovers most of any
drop while keeping the final network genuinely 50% sparse.

Run:
    pip install thekaveh-nnx
    python examples/19_prune_mnist.py
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
from nnx.prune import magnitude_prune


def _make_data(n_samples: int = 1024, n_features: int = 16) -> tuple[DataLoader, DataLoader]:
    # No torch.manual_seed here — the caller does set_seed(42) in main()
    # before calling us. Re-seeding torch inside this helper would
    # silently override the caller's seed (the same bug PR #31's review
    # originally caught in this very file).
    X = torch.randn(n_samples, n_features)
    # Three Gaussian-mixture classes separated by a learned random projection.
    proj = torch.randn(n_features, 3)
    y = (X @ proj).argmax(dim=1)
    n_train = int(0.8 * n_samples)
    train = TensorDataset(X[:n_train], y[:n_train])
    val = TensorDataset(X[n_train:], y[n_train:])
    return DataLoader(train, batch_size=64, shuffle=True), DataLoader(val, batch_size=64)


def _sparsity(net: torch.nn.Module) -> float:
    """% of weight entries equal to zero across all Linear weights."""
    total, zeros = 0, 0
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            total += module.weight.numel()
            zeros += (module.weight == 0).sum().item()
    return zeros / max(total, 1)


def main() -> None:
    set_seed(42)
    train_loader, val_loader = _make_data()

    net_params = NNParams(
        input_dim=16,
        output_dim=3,
        hidden_dims=[64, 32],
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

    print("─── Phase 1: FP32 baseline ───")
    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=train_params)
    val_idps = [idp for idp in run.idps if idp.val_edp is not None]
    fp32_acc = 1.0 - val_idps[-1].val_edp.error
    print(f"FP32 val accuracy: {fp32_acc:.3f}; sparsity: {_sparsity(model.net):.1%}")

    print("─── Phase 2: magnitude prune at 50% sparsity ───")
    n_pruned = magnitude_prune(model.net, sparsity=0.5)
    print(f"Layers pruned: {n_pruned}")
    print(f"Sparsity after prune: {_sparsity(model.net):.1%}")

    # Eval accuracy with the pruned weights (no fine-tune yet).
    model.net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            preds = model.net(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    pruned_acc = correct / total
    print(f"Pruned val accuracy (pre-finetune): {pruned_acc:.3f}")

    print("─── Phase 3: brief fine-tune ───")
    finetune_params = NNTrainParams(
        n_epochs=3,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=5e-3,
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
    final_run = model.train(params=finetune_params)
    final_val_idps = [idp for idp in final_run.idps if idp.val_edp is not None]
    print(f"Fine-tuned val accuracy: {1.0 - final_val_idps[-1].val_edp.error:.3f}")
    # With bake=True the mask is gone, so fine-tuning regrows the
    # zeroed weights (sparsity drifts back toward dense). Re-prune to
    # the target sparsity and report the accuracy that actually ships
    # — an iterative prune→tune→prune schedule would use bake=False.
    print(f"Sparsity after fine-tune (regrown): {_sparsity(model.net):.1%}")
    magnitude_prune(model.net, sparsity=0.5)
    model.net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            preds = model.net(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    print(f"Re-pruned val accuracy: {correct / total:.3f}")
    print(f"Final sparsity: {_sparsity(model.net):.1%}")


if __name__ == "__main__":
    main()
