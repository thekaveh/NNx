"""Custom train_step_fn — a tiny linear autoencoder.

Demonstrates the train_step_fn hook on NNModel.train(). The supervised
forward → loss_fn(net(X), Y) → backward path doesn't fit autoencoders
(there are no labels; loss is reconstruction error against the input
itself). The hook lets the user supply their own step body while
NNModel still owns the rest of the loop (scheduler, callbacks,
checkpoint cadence, val loop, incremental save).

Trick: a `FeedFwdNN` with `input_dim == output_dim` and a smaller
`hidden_dims` is structurally an autoencoder (d → bottleneck → d).
No new architecture or optimizer plumbing needed.

Run:
    python examples/05_custom_train_step_autoencoder.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Nets,
    Optims,
    TrainStepContext,
    set_seed,
)


def main():
    set_seed(0)

    # Synthetic 16-dim inputs; "labels" exist only to satisfy the
    # standard (X, Y) loader contract — the autoencoder step ignores them.
    n_samples, d = 256, 16
    X = torch.randn(n_samples, d)
    y_dummy = torch.zeros(n_samples, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y_dummy), batch_size=32, shuffle=True)

    # Autoencoder shape via FeedFwdNN: 16 → 4 (bottleneck) → 16.
    # No custom architecture, no monkey-patching; the optimizer built by
    # NNModel.train naturally sees all parameters.
    model = NNModel(
        net_params=NNParams(
            input_dim=d, output_dim=d, hidden_dims=[4],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        # loss=CROSS_ENTROPY is unused — our hook computes its own loss.
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )

    def autoencoder_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        """Forward through the autoencoder, MSE reconstruction loss against
        the input itself, backward, single optimizer step. NaN guard and
        grad clipping are skipped here for brevity; copy them from
        `default_train_step` if your training is unstable."""
        m = ctx.model
        m.net.train()
        m.net.zero_grad()

        X_in, _ = m.net.unpack_batch(ctx.batch)
        X_in = tuple(x.to(m.device) for x in X_in)
        reconstructed = m.net(*X_in)
        loss = F.mse_loss(reconstructed, X_in[0])
        loss.backward()
        ctx.optimizer.step()

        loss_val = float(loss.detach())
        return NNEvaluationDataPoint(
            f1=0.0, recall=0.0, accuracy=0.0, precision=0.0,
            loss=loss_val,
            error=loss_val,  # use loss as the "error" so BEST tracking is meaningful
        )

    train_params = NNTrainParams(
        n_epochs=5,
        train_loader=loader,
        optim=NNOptimParams(
            name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3,
        ),
    )

    run = model.train(params=train_params, train_step_fn=autoencoder_step)

    first = run.idps[0].train_edp.loss
    last = run.idps[-1].train_edp.loss
    print(f"\nautoencoder reconstruction loss: {first:.4f} → {last:.4f}")
    print(f"trained {len(run.idps)} iterations; saved under runs/{run.id}/")


if __name__ == "__main__":
    main()
