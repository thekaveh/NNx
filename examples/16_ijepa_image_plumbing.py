"""I-JEPA on CIFAR-10-shaped images — verify-the-plumbing tutorial.

Demonstrates:

  1. Building a small ViT-S with :class:`ViTNN` (32x32 inputs,
     4x4 patches → 64 tokens per image, 4 layers, 4 heads).
  2. Constructing the EMA target encoder via
     :func:`build_target_encoder` and the predictor via
     :class:`JEPAPredictor`, attached to ``model.net`` so the
     optimizer picks up the predictor's params jointly with the
     encoder.
  3. Training for a handful of epochs with
     :func:`jepa_train_step_factory` and the bundled
     :func:`random_block_mask` sampler.

**Reality check.** Training I-JEPA to convergence on CIFAR is hours
on CPU and the resulting linear-probe accuracy depends heavily on
hyperparameters that this short demo does not tune. The example
runs on **synthetic 32x32 noise** by default so it stays a fast
verify-the-plumbing exercise rather than masquerading as a SOTA
reproduction. Pass ``--cifar`` to download the real CIFAR-10
training set (requires ``torchvision``) — the loss will still
decrease, but don't expect linear-probe accuracy on the order
of the I-JEPA paper.

Run:
    python examples/16_ijepa_image_plumbing.py
    python examples/16_ijepa_image_plumbing.py --cifar  # download + use CIFAR-10
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    JEPAPredictor,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    ViTNN,
    build_target_encoder,
    jepa_train_step_factory,
    random_block_mask,
    set_seed,
)


def _make_synthetic_loader(n: int = 256, batch_size: int = 32) -> DataLoader:
    """Synthetic 32x32 RGB noise. Same shape as CIFAR-10 so the demo's
    plumbing is identical."""
    # No torch.manual_seed here — the caller does set_seed(0) in main()
    # before calling this helper, and a redundant seed call here would
    # silently override the caller's chosen seed if main() ever picks
    # a different value (the seed-helper override anti-pattern that bit
    # examples 19 / 20 / 21 / 22 / 23 / 24 in PRs #37 + #38).
    x = torch.randn(n, 3, 32, 32)
    y = torch.zeros(n, dtype=torch.long)  # ignored by JEPA
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _make_cifar_loader(batch_size: int = 32) -> DataLoader:
    """CIFAR-10 training set (downloads on first run via torchvision)."""
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError("--cifar requires torchvision. Install via `pip install torchvision`.") from e

    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.CIFAR10(root="data/cifar10", train=True, download=True, transform=tx)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cifar",
        action="store_true",
        help="Download and use CIFAR-10 instead of synthetic noise.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(0)
    loader = _make_cifar_loader(args.batch_size) if args.cifar else _make_synthetic_loader(batch_size=args.batch_size)

    # The base NNParams is a placeholder; the real net is the ViT
    # swapped onto model.net below. The placeholder mirrors the input
    # surface so the run.yaml stays readable.
    model = NNModel(
        net_params=NNParams(
            input_dim=3 * 32 * 32,
            output_dim=128,
            hidden_dims=[128],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,  # unused by JEPA but required by NNModelParams
        ),
    )
    # The trainable context encoder.
    model.net = ViTNN(
        image_size=32,
        patch_size=4,
        in_channels=3,
        d_model=64,
        n_layers=4,
        n_heads=4,
    ).to(model.device)

    # EMA target encoder + predictor.
    target_encoder = build_target_encoder(model.net)
    predictor = JEPAPredictor(
        embed_dim=model.net.d_model,
        n_patches=model.net.n_patches,
        predictor_dim=32,
        n_layers=2,
        n_heads=2,
    ).to(model.device)
    # Register the predictor under model.net so the optimizer (which
    # walks model.net.parameters()) picks up its weights jointly with
    # the encoder. The EMA update is name-keyed against the target's
    # parameters, so the extra predictor params are skipped during
    # the EMA step.
    model.net.add_module("_jepa_predictor", predictor)

    # Mask sampler. Single random block per step; 8x8 grid for the
    # patch_size=4 image_size=32 configuration.
    grid_size = 32 // 4

    def mask_fn(n_p, device):
        ctx, tgt = random_block_mask(n_patches=n_p, grid_size=grid_size, device=device)
        return ctx, tgt

    step_fn = jepa_train_step_factory(
        target_encoder=target_encoder,
        predictor=predictor,
        mask_fn=mask_fn,
        ema_momentum=0.996,
    )

    run = model.train(
        params=NNTrainParams(
            n_epochs=args.epochs,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=5e-4,
                momentum=(0.9, 0.999),
                weight_decay=1e-4,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    losses = [idp.train_edp.loss for idp in run.idps]
    first_loss = losses[0]
    last_loss = losses[-1]
    print()
    print(f"first-step loss: {first_loss:.4f}")
    print(f"last-step  loss: {last_loss:.4f}")
    print(f"steps run      : {len(losses)}")
    # Quick sanity check — on synthetic noise the loss is small from the
    # start (the predictor learns a near-zero mapping); the assertion is
    # for the verify-the-plumbing demo to flag if something goes wrong.
    assert all(torch.isfinite(torch.tensor(lo)).item() for lo in losses), "non-finite loss"


if __name__ == "__main__":
    main()
