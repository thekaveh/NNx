"""Custom train_step_fn — a tiny linear autoencoder.

Demonstrates the train_step_fn hook on NNModel.train(). The supervised
forward → loss_fn(net(X), Y) → backward path doesn't fit autoencoders
(there are no labels; loss is reconstruction error against the input
itself). The hook lets the user supply their own step body while
NNModel still owns the rest of the loop (scheduler, callbacks,
checkpoint cadence, val loop, incremental save).

Run:
    python examples/05_custom_train_step_autoencoder.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
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
    # standard NNDataset (X, Y) loader contract — we ignore them.
    n_samples, d = 256, 16
    X = torch.randn(n_samples, d)
    y_dummy = torch.zeros(n_samples, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y_dummy), batch_size=32, shuffle=True)

    # Encoder: 16 → 8 → 4. We reuse FeedFwdNN as the encoder; the decoder
    # lives outside model.net and is registered with the same optimizer
    # below. (A real-project autoencoder PR would land a proper
    # AutoencoderNN under nnx.nn.net so the whole thing is one module.)
    encoder = NNModel(
        net_params=NNParams(
            input_dim=d, output_dim=4, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        # loss=CROSS_ENTROPY is unused — our hook computes its own loss.
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    decoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, d))
    # Attach the decoder so the train_step_fn can reach it via ctx.model.
    encoder.decoder = decoder  # type: ignore[attr-defined]


    def autoencoder_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        """Forward through encoder + decoder, MSE reconstruction loss,
        backward, single optimizer step. NaN guard + grad clipping are
        skipped here for brevity; real code would copy the relevant
        sections from `default_train_step`."""
        m = ctx.model
        m.net.train()
        m.decoder.train()
        m.net.zero_grad()
        m.decoder.zero_grad()

        X_batch, _ = m.net.unpack_batch(ctx.batch)
        X_batch = tuple(x.to(m.device) for x in X_batch)
        encoded = m.net(*X_batch)
        decoded = m.decoder(encoded)
        loss = torch.nn.functional.mse_loss(decoded, X_batch[0])
        loss.backward()
        ctx.optimizer.step()

        loss_val = float(loss.detach())
        return NNEvaluationDataPoint(
            f1=0.0, recall=0.0, accuracy=0.0, precision=0.0,
            loss=loss_val,
            error=loss_val,  # use loss as "error" so BEST tracking is meaningful
        )

    # Adam is built from encoder.net's parameters by NNModel. Add the
    # decoder's parameters into the same optimizer manually after train
    # constructs it. (Cleaner long-term: a proper AutoencoderNN owns
    # both halves and the optimizer sees one .parameters() call.)
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

    # Monkey-patch the optimizer factory so it sees both encoder + decoder.
    original_optim_call = Optims.ADAM.__call__

    def _combined_adam(self, *, net, lr_start, momentum, weight_decay):
        return torch.optim.Adam(
            list(net.parameters()) + list(decoder.parameters()),
            lr=lr_start, betas=momentum, weight_decay=weight_decay,
        )
    # Bind the override only for this run; restore afterward.
    Optims.ADAM.__class__.__call__ = _combined_adam  # type: ignore[method-assign]
    try:
        run = encoder.train(params=train_params, train_step_fn=autoencoder_step)
    finally:
        Optims.ADAM.__class__.__call__ = original_optim_call  # type: ignore[method-assign]

    first_epoch_loss = run.idps[0].train_edp.loss
    last_epoch_loss = run.idps[-1].train_edp.loss
    print(f"\nautoencoder loss: {first_epoch_loss:.4f} → {last_epoch_loss:.4f}")
    print(f"trained {len(run.idps)} iterations; saved under runs/{run.id}/")


if __name__ == "__main__":
    main()
