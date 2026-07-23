"""Model surgery demo — low-rank factorize a Linear, retrain at lower rank.

Pipeline:

  1. Build + train a small classifier with a wide Linear layer.
  2. Snapshot FP32 val accuracy + Linear parameter count.
  3. Apply `nnx.surgery.low_rank_factorize(layer, rank=8)`.
     Replaces the named Linear with two stacked Linears (out×r, r×in).
     At max rank the factorization is exact; below max rank it is the
     SVD truncation — the bottleneck.
  4. Compare:
     - Total param count before vs after.
     - Validation accuracy before vs after.
  5. Briefly fine-tune the surgically modified model and verify accuracy
     recovers.

Note: `low_rank_factorize` takes a `nn.Linear` directly and returns a
`nn.Sequential` of two Linears. The caller is responsible for swapping
the layer back into the network's ModuleList.

Run:
    pip install thekaveh-nnx
    python examples/20_low_rank_surgery_ffn.py
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
from nnx.surgery import low_rank_factorize


def _make_data():
    # No torch.manual_seed here — the caller does set_seed(42) before
    # calling us. Re-seeding torch inside this helper would silently
    # override the caller's seed (the same bug that PR #31's review
    # caught in examples 19 / 21 / 23).
    X = torch.randn(1024, 16)
    proj = torch.randn(16, 3)
    y = (X @ proj).argmax(dim=1)
    train = TensorDataset(X[:800], y[:800])
    val = TensorDataset(X[800:], y[800:])
    return DataLoader(train, batch_size=64, shuffle=True), DataLoader(val, batch_size=64)


def _param_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _val_acc(net: torch.nn.Module, loader: DataLoader) -> float:
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            preds = net(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total


def main() -> None:
    set_seed(42)
    train_loader, val_loader = _make_data()

    # Wide hidden_dims so the factorization has compressible structure.
    # FeedFwdNN with hidden_dims=[64, 128, 32] yields:
    #   layers.0: Linear(16 → 64)
    #   layers.1: Linear(64 → 128)   ← widest; factorize this one
    #   layers.2: Linear(128 → 32)
    #   layers.3: Linear(32 → 3)
    net_params = NNParams(
        input_dim=16,
        output_dim=3,
        hidden_dims=[64, 128, 32],
        dropout_prob=0.0,
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

    print("─── Phase 1: train the wide net ───")
    model = NNModel(net_params=net_params, params=model_params)
    model.train(params=train_params)
    print(f"FP32 val accuracy: {_val_acc(model.net, val_loader):.3f}")
    print(f"FP32 params:       {_param_count(model.net):,}")

    print("─── Phase 2: low-rank factorize the widest Linear at rank=8 ───")
    # FeedFwdNN stores Linears in a ModuleList; layers.1 is the 64→128 Linear.
    # low_rank_factorize takes the nn.Linear directly and returns nn.Sequential.
    target_linear = model.net.layers[1]
    factored = low_rank_factorize(target_linear, rank=8)
    model.net.layers[1] = factored
    print(f"Surgically reduced rank: 8 (max was {min(target_linear.in_features, target_linear.out_features)})")
    print(f"Post-surgery params:    {_param_count(model.net):,}")
    print(f"Post-surgery val acc:    {_val_acc(model.net, val_loader):.3f}  # expect drop before refinement")

    print("─── Phase 3: refine to recover accuracy ───")
    refine_params = NNTrainParams(
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
    model.train(params=refine_params)
    print(f"Refined val accuracy:   {_val_acc(model.net, val_loader):.3f}")


if __name__ == "__main__":
    main()
