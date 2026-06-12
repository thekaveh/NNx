"""Tiny GAN trained via the multi-optimizer Trainer.

Demonstrates `nnx.trainer.Trainer` + `NNTrainerParams` on a small
generator/discriminator pair packed into a single nn.Module
(`MiniGAN.G` and `MiniGAN.D`). Two distinct optimizers — one scoped to
G via `NNParamGroupSpec(name_pattern="G.*")`, one to D via `"D.*"` —
own disjoint subsets of the model's parameters, which is what enables
the GAN's alternating update pattern without optimizers stepping on
each other.

Real distribution: a 1D mixture of N(-3, 0.5) and N(3, 0.5). After
training, G should produce samples concentrated near ±3. The training
loop prints the average combined loss decreasing as G learns the
distribution.

This is a *teaching* GAN — small, CPU-fast, intentionally minimal —
not a production setup. Spectral norm, EMA, R1 regularization, larger
nets, and longer schedules are all things you'd add for a real run.

Run:
    python examples/09_gan_with_trainer.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParamGroupSpec,
    NNParams,
    NNTrainerParams,
    Optims,
    Trainer,
    TrainerStepContext,
    set_seed,
)

LATENT_DIM = 4


class MiniGAN(nn.Module):
    """Generator + Discriminator inside one nn.Module so a single
    NNModel can hold both. The named children — ``self.G`` and
    ``self.D`` — give us the dotted parameter names ``G.*`` / ``D.*``
    that the NNParamGroupSpec globs target."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.G = nn.Sequential(
            nn.Linear(LATENT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.D = nn.Sequential(
            nn.Linear(1, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # The Trainer's custom step bypasses .forward() entirely — it
        # reaches into .G and .D directly. This default exists so
        # NNModel.predict() etc. don't crash if called by mistake.
        return self.G(x)


def sample_real(n: int) -> torch.Tensor:
    """1D mixture of N(-3, 0.5) and N(3, 0.5). Bimodal, so covering the
    real distribution requires G to learn distinct modes — though this
    deliberately minimal GAN often collapses onto one of them, which is
    itself a classic GAN failure mode worth observing."""
    # No torch.manual_seed here — the caller does set_seed(0) in main()
    # before calling us. Re-seeding torch inside this helper would
    # silently override the caller's seed (the recurring bug PR #31's
    # review caught in examples 19 / 21 / 23).
    mix = torch.randint(0, 2, (n, 1)).float()
    means = mix * 3 - (1 - mix) * 3
    return means + 0.5 * torch.randn(n, 1)


def main():
    set_seed(0)

    real = sample_real(2048)
    # The y labels are unused — Trainer's step fn ignores them. We
    # include them only so the DataLoader's (X, Y) tuple contract holds.
    loader = DataLoader(
        TensorDataset(real, torch.zeros(real.size(0), dtype=torch.long)),
        batch_size=64,
        shuffle=True,
    )

    # NNModel is happy with a placeholder NNParams here; the real net is
    # swapped in after construction. The placeholder dims (LATENT_DIM→1)
    # echo G's surface shape so the run.yaml stays interpretable.
    model = NNModel(
        net_params=NNParams(
            input_dim=LATENT_DIM,
            output_dim=1,
            hidden_dims=[],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )
    # Swap the FeedFwdNN built by Nets.FEED_FWD for the GAN composite.
    # The Trainer never calls model.net(...) — it walks
    # model.net.named_parameters() and dispatches the step fn — so the
    # only thing this substitution affects is the optimizer's view of
    # the parameters.
    model.net = MiniGAN().to(model.device)

    trainer = Trainer(model=model)

    def gan_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
        net: MiniGAN = ctx.model.net  # type: ignore[assignment]
        opt_G = ctx.optimizers["G"]
        opt_D = ctx.optimizers["D"]
        device = ctx.model.device

        X_real, _ = ctx.batch
        X_real = X_real.to(device)
        n = X_real.size(0)

        # --- Discriminator step (real vs fake).
        # detach() the fake samples here so the D step's backward()
        # doesn't accumulate gradients into G's params — only opt_D's
        # parameters move on this step.
        opt_D.zero_grad()
        z = torch.randn(n, LATENT_DIM, device=device)
        X_fake = net.G(z).detach()
        d_real_logits = net.D(X_real)
        d_fake_logits = net.D(X_fake)
        d_loss = F.binary_cross_entropy_with_logits(
            d_real_logits, torch.ones_like(d_real_logits)
        ) + F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
        d_loss.backward()
        opt_D.step()

        # --- Generator step (fool D into calling fakes "real").
        # No detach() here: gradients flow from D's logits back into G's
        # parameters. opt_D doesn't step on this pass, so D's params
        # don't move even though its gradients are populated.
        opt_G.zero_grad()
        z = torch.randn(n, LATENT_DIM, device=device)
        g_fake_logits = net.D(net.G(z))
        g_loss = F.binary_cross_entropy_with_logits(g_fake_logits, torch.ones_like(g_fake_logits))
        g_loss.backward()
        opt_G.step()

        avg = float((d_loss + g_loss).detach()) / 2
        return NNEvaluationDataPoint(
            f1=0.0,
            recall=0.0,
            accuracy=0.0,
            precision=0.0,
            loss=avg,
            # Use g_loss as the "error" so BEST tracking favors checkpoints
            # where G is fooling D well.
            error=float(g_loss.detach()),
        )

    # Per-optim param scoping: NNParamGroupSpec with strict semantics
    # (enforced by Trainer) means opt_G owns ONLY G.* params and opt_D
    # owns ONLY D.* params. Without the strict contract, opt_G would
    # also carry D's params in a default bucket and the two optimizers
    # would silently update the same weights.
    g_optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=2e-4,
        momentum=(0.5, 0.999),
        weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=2e-4)],
    )
    d_optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=2e-4,
        momentum=(0.5, 0.999),
        weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="D.*", lr=2e-4)],
    )

    run = trainer.train(
        params=NNTrainerParams(
            n_epochs=10,
            train_loader=loader,
            optims={"G": g_optim, "D": d_optim},
        ),
        trainer_step_fn=gan_step,
    )

    first = run.idps[0].train_edp.loss
    last = run.idps[-1].train_edp.loss
    print(f"\nGAN combined loss: {first:.4f} → {last:.4f}")
    print(f"trained {len(run.idps)} iterations; saved under runs/{run.id}/")

    # Quick "sanity look": G's samples should now cluster near ±3.
    z = torch.randn(8, LATENT_DIM)
    samples = model.net.G(z).detach().squeeze(-1).tolist()
    rendered = ", ".join(f"{s:+.2f}" for s in samples)
    print(f"sample G(z): {rendered}")


if __name__ == "__main__":
    main()
