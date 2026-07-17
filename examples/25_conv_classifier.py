"""Convolutional classifier — ``NNConvParams`` + ``Nets.CONV`` (LeNet-style).

Demonstrates:

  1. Building a conv net straight from params: :class:`NNConvParams` (an
     :class:`NNParams` subclass) carries the conv-stack knobs —
     ``conv_channels`` per block, plus LeNet-5 defaults for
     ``in_channels``/``kernel_size``/``stride``/``padding``/``pool_size``
     (1/5/1/0/2). ``Nets.CONV`` instantiates a ``ConvNN``:
     Conv→activation→MaxPool blocks, then an FC head that reuses
     ``hidden_dims``. v1 targets square images — the spatial side is
     derived as ``sqrt(input_dim / in_channels)``.
  2. The params helpers: ``image_side()``, ``spatial_sizes()`` (feature-map
     side after each block) and ``flatten_dim()`` (first FC width). A
     conv/pool stack that would collapse below 1×1 fails at construction,
     not deep inside the first forward.
  3. Per-layer FC overrides: the FC head honors the optional
     ``activations``/``dropout_probs`` lists (one entry per hidden layer),
     so different FC layers can use different activations/dropout while
     the conv blocks use the net-wide scalar activation.
  4. Input flexibility: forward accepts ``(B, C, H, W)`` images or the
     flattened ``(B, input_dim)`` rows a generic loader may produce —
     both reshape to the same images and yield identical logits.
  5. Checkpoint round-trip: ``resolve_from_state`` dispatches on the
     always-emitted ``conv_channels`` key, so the reloaded model is a
     conv net with identical logits.

The task is synthetic 16×16 imagery (horizontal stripes vs vertical
stripes vs checkerboard, plus noise) — spatially-structured classes a
convolution solves easily, with no dataset download.

Run:
    python examples/25_conv_classifier.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNConvParams,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    set_seed,
)

SIDE = 16  # image side; input_dim = 1 * SIDE * SIDE = 256


def _make_loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    """3-class synthetic imagery: horizontal stripes / vertical stripes /
    checkerboard (period-4 patterns in ±1), plus Gaussian noise."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.arange(SIDE).unsqueeze(1).expand(SIDE, SIDE)
    cols = torch.arange(SIDE).unsqueeze(0).expand(SIDE, SIDE)
    stripes_h = ((rows // 4) % 2 * 2 - 1).float()
    stripes_v = ((cols // 4) % 2 * 2 - 1).float()
    checker = stripes_h * stripes_v
    patterns = torch.stack([stripes_h, stripes_v, checker])  # (3, SIDE, SIDE)

    def make(n: int):
        cls = torch.randint(0, 3, (n,), generator=g)
        X = patterns[cls].unsqueeze(1) + 0.6 * torch.randn(n, 1, SIDE, SIDE, generator=g)
        return X, cls

    X_train, y_train = make(512)
    X_val, y_val = make(256)
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    return train, val


def main() -> None:
    set_seed(0)
    train_loader, val_loader = _make_loaders(seed=0)

    # Two conv blocks with the LeNet defaults (kernel 5, pool 2), then a
    # two-layer FC head with PER-LAYER overrides: tanh+25% dropout on the
    # first FC layer, relu+no dropout on the second.
    net_params = NNConvParams(
        input_dim=SIDE * SIDE,
        output_dim=3,
        hidden_dims=[32, 16],
        dropout_prob=0.0,
        activation=Activations.RELU,
        activations=[Activations.TANH, Activations.RELU],
        dropout_probs=[0.25, 0.0],
        conv_channels=[8, 16],
    )
    print("=" * 60)
    print("Conv stack arithmetic (from NNConvParams)")
    print("=" * 60)
    print(f"image side:     {net_params.image_side()}  ({SIDE}×{SIDE}, 1 channel)")
    print(f"spatial sizes:  {net_params.spatial_sizes()}  (per Conv→Pool block)")
    print(f"flatten dim:    {net_params.flatten_dim()}  (first FC layer width)")

    model = NNModel(
        net_params=net_params,
        params=NNModelParams(net=Nets.CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    total_params = sum(p.numel() for p in model.net.parameters())
    print(f"\nnet: {type(model.net).__name__}, {total_params} parameters")

    run = model.train(
        params=NNTrainParams(
            n_epochs=8,
            train_loader=train_loader,
            val_loader=val_loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
    )
    print(f"\nfinal val error: {run.idps[-1].val_edp.error:.4f}")

    # ---- Image input and flattened input yield identical logits.
    X_imgs = next(iter(val_loader))[0][:8]  # (8, 1, 16, 16)
    model.net.eval()
    with torch.no_grad():
        same = torch.equal(model.net(X_imgs), model.net(X_imgs.view(8, -1)))
    print(f"(B,C,H,W) vs flat (B,{SIDE * SIDE}) input — identical logits: {same}")

    # ---- Checkpoint round-trip via resolve_from_state.
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert ckpt is not None and isinstance(ckpt.net_params, NNConvParams)
    reloaded = NNModel.from_checkpoint(ckpt)
    reloaded.net.eval()
    with torch.no_grad():
        same = torch.allclose(model.net(X_imgs), reloaded.net(X_imgs))
    print(f"checkpoint round-trip: net_params={type(ckpt.net_params).__name__}, ")
    print(f"reloaded net={type(reloaded.net).__name__}, logits identical: {same}")


if __name__ == "__main__":
    main()
