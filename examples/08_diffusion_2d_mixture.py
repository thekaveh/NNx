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

**Objective mode (FEAT-040).** ``diffusion_objective(schedule)`` describes
the same noise-prediction loss as an objective: NNx's shared update engine
then owns backward, gradient accumulation (exact for uneven microbatches),
clipping and the optimizer step, and the timesteps and noise come from the
objective's own checkpointed generator — so a ``sample`` preview with its
own generator never perturbs training. ``--objective`` trains the demo that
way; ``objective_mode()`` below is a bounded check of it.

Run:
    python examples/08_diffusion_2d_mixture.py
    python examples/08_diffusion_2d_mixture.py --objective  # the objective adapter
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Callback,
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
    diffusion_objective,
    diffusion_train_step_factory,
    sample,
    set_seed,
)


def make_mixture_loader(n: int = 1024, batch_size: int = 64) -> DataLoader:
    """4 isotropic Gaussians at (±2, ±2). DataLoader yields (x, dummy_y)
    so the standard (X, Y) batch contract holds — Y is ignored."""
    # No torch.manual_seed here — the caller does set_seed(0) in main()
    # before calling us. Re-seeding torch inside this helper would
    # silently override the caller's seed (the recurring bug PR #31's
    # review caught in examples 19 / 21 / 23).
    centers = torch.tensor([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=torch.float32)
    idx = torch.randint(0, 4, (n,))
    means = centers[idx]
    X = means + 0.3 * torch.randn(n, 2)
    y_dummy = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(X, y_dummy), batch_size=batch_size, shuffle=True)


def objective_mode() -> dict:
    """Bounded demonstration of diffusion as an objective (FEAT-040).

    Uneven microbatches accumulated two at a time, and a ``sample`` preview
    with its own generator at every epoch's end: the preview leaves the
    objective's generator and the net's train mode exactly as it found
    them, and every record carries the named loss with no classification
    fields.
    """
    set_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.net = DiffusionMLP(input_dim=2, hidden_dims=[32, 32], time_embed_dim=16).to(model.device)
    schedule = NoiseSchedulers.LINEAR(T=50)
    objective = diffusion_objective(schedule, seed=0)
    untouched: list[bool] = []

    class Preview(Callback):
        def on_epoch_end(self, ctx) -> None:
            before = objective.generator.get_state()
            sample(ctx.model, schedule, shape=(16, 2), generator=torch.Generator().manual_seed(1))
            untouched.append(torch.equal(before, objective.generator.get_state()) and ctx.model.net.training)

    points = make_mixture_loader(n=40, batch_size=40).dataset.tensors[0]
    loader = [(chunk, torch.zeros(len(chunk))) for chunk in points.split([16, 16, 8])]
    run = model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=2e-3, momentum=(0.9, 0.999), weight_decay=0.0, accumulate_grad_batches=2
            ),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=4, cooldown=1, threshold=1e-3),
        ),
        objective=objective,
        callbacks=[Preview()],
    )
    assert untouched == [True, True], "a preview must not perturb the objective's generator or the net's mode"
    assert run.idps[-1].update_count == 4, "two committed updates per epoch ([16, 16] and the short [8])"
    assert all(set(idp.train_edp.metrics) == {"noise_mse"} and idp.train_edp.accuracy is None for idp in run.idps)
    summary = {"committed_updates": run.idps[-1].update_count, "noise_mse": round(run.idps[-1].train_edp.loss, 6)}
    print(f"objective mode: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        action="store_true",
        help="Train with the diffusion_objective adapter (FEAT-040) instead of the imperative step.",
    )
    args = parser.parse_args()

    set_seed(0)
    loader = make_mixture_loader()

    # An NNModel with placeholder NNParams; the real network is the
    # DiffusionMLP swapped in below. The placeholder mirrors the
    # diffusion net's surface dim so the run.yaml stays readable.
    model = NNModel(
        net_params=NNParams(
            input_dim=2,
            output_dim=2,
            hidden_dims=[16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    # The FeedFwdNN built by Nets.FEED_FWD has forward(X) → logits — wrong
    # shape for diffusion (which needs forward(x_t, t) → ε). Swap it for
    # the DiffusionMLP. NNModel.train() reaches model.net.parameters()
    # and model.net_params (stored on the model itself, not the net),
    # so this substitution works without further setup.
    model.net = DiffusionMLP(
        input_dim=2,
        hidden_dims=[64, 64],
        time_embed_dim=16,
    ).to(model.device)

    # T=200 is enough for this toy problem and keeps sampling fast.
    schedule = NoiseSchedulers.LINEAR(T=200)
    # The imperative step owns its own update; the objective hands it to
    # NNx's shared update engine and draws from its own generator.
    step_fn = None if args.objective else diffusion_train_step_factory(schedule)
    objective = diffusion_objective(schedule) if args.objective else None

    run = model.train(
        params=NNTrainParams(
            n_epochs=20,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=2e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=4,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
        objective=objective,
        salt="objective" if args.objective else None,  # a distinct run from the imperative mode
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
