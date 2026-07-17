"""Custom validation with ``eval_step_fn`` — replace the built-in
classification val pass for paradigms it can't score.

``NNModel.train`` computes argmax + sklearn classification metrics in
BOTH its default train step and its built-in val pass. For regression
(or LM perplexity, DPO margins, ...) those crash or produce garbage on
continuous targets, so a non-classification paradigm needs the pair:

  1. A custom ``train_step_fn`` (:class:`TrainStepContext` in, batch EDP
     out) — here the standard MSE forward/backward with MAE riding in
     ``extra``. Before ``eval_step_fn`` existed this was the END of the
     story: the built-in val pass still ran classification metrics, so
     the only workaround was training with ``val_loader=None`` and
     losing validation entirely.
  2. A custom ``eval_step_fn`` (:class:`EvalStepContext` in — frozen
     bundle of ``model``, ``val_loader``, ``extra_metrics``,
     ``epoch_idx`` — one :class:`NNEvaluationDataPoint` out). It runs
     INSIDE the epoch loop under ``torch.no_grad()``, so the returned
     metrics land on the epoch's ``val_edp`` and PERSIST through the
     incremental run save — real run history, not display-only numbers.

The task is 1-D synthetic regression (y = sin(3x) + noise) on a small
feed-forward net trained with MSE.

Run:
    python examples/26_custom_eval_step.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    EvalStepContext,
    Losses,
    Nets,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    TrainStepContext,
    set_seed,
)


def _make_loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(seed)

    def make(n: int):
        X = torch.rand(n, 1, generator=g) * 4 - 2
        Y = torch.sin(3 * X) + 0.1 * torch.randn(n, 1, generator=g)
        return X, Y

    X_train, y_train = make(512)
    X_val, y_val = make(256)
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    return train, val


def regression_train_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
    """MSE training step. Mirrors default_train_step's backward/step
    protocol (zero grads at the start of each accumulation cycle; clip +
    step at cycle end) but skips its per-batch classification metrics —
    the crash point for continuous targets."""
    model = ctx.model
    model.net.train()
    if (ctx.batch_idx % ctx.accumulate_grad_batches) == 0:
        model.net.zero_grad()
    X, Y = ctx.batch
    X, Y = X.to(model.device), Y.to(model.device)
    pred = model.net(X)
    mse = F.mse_loss(pred, Y)
    (mse / ctx.accumulate_grad_batches).backward()
    if ((ctx.batch_idx + 1) % ctx.accumulate_grad_batches) == 0:
        if ctx.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), ctx.grad_clip_norm)
        ctx.optimizer.step()
    mse_val = float(mse.detach())
    with torch.no_grad():
        mae = float(F.l1_loss(pred.detach(), Y))
    return NNEvaluationDataPoint(
        f1=0.0,
        recall=0.0,
        accuracy=0.0,
        precision=0.0,
        loss=mse_val,
        error=mse_val,
        extra={"mae": mae},
    )


def regression_eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
    """Sample-weighted val MSE + MAE over the full val loader.

    Runs under ``torch.no_grad()`` (NNModel.train wraps the call), but we
    still flip eval mode so dropout/batch-norm behave, and restore train
    mode after — the next epoch continues training.
    """
    net = ctx.model.net
    was_training = net.training
    net.eval()
    se_sum, ae_sum, n = 0.0, 0.0, 0
    for X, Y in ctx.val_loader:
        pred = net(X)
        se_sum += float(F.mse_loss(pred, Y, reduction="sum"))
        ae_sum += float(F.l1_loss(pred, Y, reduction="sum"))
        n += Y.numel()
    if was_training:
        net.train()
    mse, mae = se_sum / n, ae_sum / n
    # The classification fields are meaningless for regression — zero them
    # and carry the real numbers in loss/error/extra.
    return NNEvaluationDataPoint(
        f1=0.0,
        recall=0.0,
        accuracy=0.0,
        precision=0.0,
        loss=mse,
        error=mse,
        extra={"mae": mae},
    )


def main() -> None:
    set_seed(0)
    train_loader, val_loader = _make_loaders(seed=0)

    model = NNModel(
        net_params=NNParams(
            input_dim=1,
            output_dim=1,
            hidden_dims=[64, 64],
            dropout_prob=0.0,
            activation=Activations.TANH,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR),
    )

    run = model.train(
        params=NNTrainParams(
            n_epochs=15,
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
                patience=3,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=regression_train_step,
        eval_step_fn=regression_eval_step,
    )

    # The custom metrics are part of the persisted run history — one
    # val_edp per epoch, MAE riding in `extra`.
    print("=" * 60)
    print("per-epoch val history (custom eval step)")
    print("=" * 60)
    val_edps = [idp.val_edp for idp in run.idps if idp.val_edp is not None]
    for i, edp in enumerate(val_edps):
        print(f"epoch {i + 1:2d}:  val MSE {edp.loss:.5f}   val MAE {edp.extra['mae']:.5f}")

    first, last = val_edps[0], val_edps[-1]
    print(f"\nval MSE {first.loss:.5f} → {last.loss:.5f}; val MAE {first.extra['mae']:.5f} → {last.extra['mae']:.5f}")
    assert last.loss < first.loss, "val MSE should decrease on this toy task"


if __name__ == "__main__":
    main()
