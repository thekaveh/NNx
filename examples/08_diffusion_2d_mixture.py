"""Tiny DDPM-style diffusion on a 2D Gaussian mixture.

Demonstrates `nnx.diffusion.{NoiseSchedulers, DiffusionMLP,
diffusion_train_step_factory, sample}` end-to-end:

  1. Build a tiny denoiser (`DiffusionMLP`) and a `NoiseSchedule`.
  2. Train via `NNModel.train(train_step_fn=...)` — the diffusion step
     factory makes the noise-prediction loop the framework's standard
     train_step_fn hook.
  3. Sample by running the reverse-diffusion loop with `sample(...)`.

Source distribution: a 2D mixture of four isotropic Gaussians at
(±2, ±2). After training, sampled points should cluster around those
four modes; we print summary stats to verify without needing matplotlib.

This is a *teaching* diffusion — small net, short training, low T —
intentionally minimal so the train/sample plumbing is visible. For
image-space diffusion, swap `DiffusionMLP` for a U-Net of your choice;
the schedule / train step / sampler are architecture-agnostic.

Run:
    python examples/08_diffusion_2d_mixture.py
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    DiffusionMLP,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    NoiseSchedulers,
    Optims,
    diffusion_train_step_factory,
    sample,
    set_seed,
)


def make_mixture_loader(n: int = 1024, batch_size: int = 64) -> DataLoader:
    """4 isotropic Gaussians at (±2, ±2). DataLoader yields (x, dummy_y)
    so the standard (X, Y) batch contract holds — Y is ignored."""
    centers = torch.tensor([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=torch.float32)
    idx = torch.randint(0, 4, (n,))
    means = centers[idx]
    X = means + 0.3 * torch.randn(n, 2)
    y_dummy = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(X, y_dummy), batch_size=batch_size, shuffle=True)


def main():
    set_seed(0)
    loader = make_mixture_loader()

    # An NNModel with placeholder NNParams; the real network is the
    # DiffusionMLP swapped in below. The placeholder mirrors the
    # diffusion net's surface dim so the run.yaml stays readable.
    model = NNModel(
        net_params=NNParams(
            input_dim=2, output_dim=2, hidden_dims=[16],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    # The FeedFwdNN built by Nets.FEED_FWD has forward(X) → logits — wrong
    # shape for diffusion (which needs forward(x_t, t) → ε). Swap it for
    # the DiffusionMLP. NNModel.train() reaches model.net.parameters()
    # and model.net_params (stored on the model itself, not the net),
    # so this substitution works without further setup.
    model.net = DiffusionMLP(
        input_dim=2, hidden_dims=[64, 64], time_embed_dim=16,
    ).to(model.device)

    # T=200 is enough for this toy problem and keeps sampling fast.
    schedule = NoiseSchedulers.LINEAR(T=200)
    step_fn = diffusion_train_step_factory(schedule)

    run = model.train(
        params=NNTrainParams(
            n_epochs=20,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=2e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=4, cooldown=1, threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    first = run.idps[0].train_edp.loss
    last = run.idps[-1].train_edp.loss
    print(f"\nDiffusion noise-prediction loss: {first:.4f} → {last:.4f}")
    print(f"trained {len(run.idps)} iterations; saved under runs/{run.id}/")

    # Sample and report the rough mode coverage. With 4 modes evenly
    # distributed, ~25% of well-trained samples should land near each
    # mode (within a small radius); pure noise would be ~uniform around
    # the origin.
    n_samples = 256
    samples = sample(model, schedule, shape=(n_samples, 2))
    centers = torch.tensor([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=torch.float32)
    nearest = torch.cdist(samples, centers).argmin(dim=1)
    print("samples per mode (target ~64 each):")
    for i, c in enumerate(centers.tolist()):
        count = int((nearest == i).sum())
        print(f"  near ({int(c[0]):+d}, {int(c[1]):+d}): {count}")


if __name__ == "__main__":
    main()
