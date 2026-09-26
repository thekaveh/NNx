"""Custom validation with ``eval_step_fn`` — replace the built-in
classification val pass for paradigms it can't score.

Without a task, ``NNModel.train`` computes argmax + sklearn
classification metrics in BOTH its default train step and its built-in val
pass. Plain regression and multilabel problems no longer need custom
steps: declare ``NNModelParams(task=TaskSpec.regression(...))`` (or
``.multilabel``) and the default step, ``evaluate()`` and ``predict()``
score continuous values with MSE / MAE — see
``examples/regression_task.py``. Paradigms no task adapter covers (LM
perplexity, DPO margins, a bespoke regression schedule like the one below)
still use the pair:

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

A custom evaluator also owns the *control signal* NNx derives from its
result: BEST-checkpoint selection, ``runs/best`` and ``ReduceLROnPlateau``
all read the first **finite** value in the order validation error →
validation loss → training error → training loss. NaN and ±inf are
skipped (with a per-epoch warning), ``None`` means "absent", and an epoch
with no finite signal at all skips the plateau step and compares as an
unavailable BEST baseline. ``nonfinite_metric_workflow`` below is a
bounded, self-checking demonstration of that contract.

Run:
    python examples/26_custom_eval_step.py
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    EvalStepContext,
    Losses,
    Nets,
    NNCheckpoint,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
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
    accumulation = ctx.accumulate_grad_batches
    cycle_size = (ctx.batch_idx % accumulation) + 1
    should_step = cycle_size == accumulation or ctx.is_last_batch
    if cycle_size == 1:
        model.net.zero_grad()
    X, Y = ctx.batch
    X, Y = X.to(model.device), Y.to(model.device)
    amp_enabled = ctx.scaler is not None and model.device.type == "cuda"
    with torch.amp.autocast(device_type=model.device.type, enabled=amp_enabled):
        pred = model.net(X)
        mse = F.mse_loss(pred, Y)
    if not torch.isfinite(mse):
        raise FloatingPointError(f"non-finite regression loss: {float(mse.detach())!r}")
    if ctx.scaler is not None:
        ctx.scaler.scale(mse / accumulation).backward()
    else:
        (mse / accumulation).backward()
    if should_step:
        if ctx.scaler is not None:
            ctx.scaler.unscale_(ctx.optimizer)
        if cycle_size < accumulation:
            for parameter in model.net.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(accumulation / cycle_size)
        if ctx.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), ctx.grad_clip_norm)
        if ctx.scaler is not None:
            ctx.scaler.step(ctx.optimizer)
            ctx.scaler.update()
        else:
            ctx.optimizer.step()
    mse_val = float(mse.detach())
    with torch.no_grad():
        mae = float(F.l1_loss(pred.detach(), Y))
    # Classification fields are optional: a regression record leaves them
    # None instead of fabricating zeros.
    return NNEvaluationDataPoint(loss=mse_val, error=mse_val, extra={"mae": mae})


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
    # The classification fields are meaningless for regression — leave them
    # None and carry the real numbers in loss/error/extra.
    return NNEvaluationDataPoint(loss=mse, error=mse, extra={"mae": mae})


def _make_model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=1,
            output_dim=1,
            hidden_dims=[64, 64],
            dropout_prob=0.0,
            activation=Activations.TANH,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR),
    )


def nonfinite_metric_workflow() -> dict:
    """Bounded demonstration of the finite-metric fallback (writes ``runs/``
    under the current working directory; the smoke test runs it in a
    temporary one).

    Four scripted epochs on the regression task:

    0. validation ``error=NaN, loss=0.9`` → the plateau scheduler steps on
       the finite validation *loss*; BEST is provisionally epoch 0.
    1. validation ``error=inf, loss=-inf`` → both rejected; the finite
       *training* error is the fallback.
    2. validation and training ``error=loss=None`` (the update still
       happened — only the reported signal is unavailable) → the plateau
       step is skipped and the epoch compares as an unavailable baseline.
    3. validation ``error=loss=1e-3`` → finite improvement; BEST moves here.

    Raw observations are kept as diagnostics: the live history and the
    FIRST checkpoint carry the NaN, while CSV readback keeps its existing
    NaN → ``None`` normalization. Returns a small summary dict.
    """
    scripted_val = {
        0: (float("nan"), 0.9),
        1: (float("inf"), float("-inf")),
        2: (None, None),
        3: (1e-3, 1e-3),
    }

    def scripted_eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
        error, loss = scripted_val[ctx.epoch_idx]
        return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=loss, error=error)

    train_signal: dict[int, float] = {}

    def scripted_train_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        edp = regression_train_step(ctx)
        if ctx.epoch_idx == 2:
            return replace(edp, loss=None, error=None)
        assert edp.error is not None
        train_signal[ctx.epoch_idx] = edp.error  # last batch of the epoch wins
        return edp

    # Instrument the scheduler so the exact plateau inputs are observable.
    plateau_inputs: list[float] = []
    original_step = ReduceLROnPlateau.step

    def recording_step(self, metrics, epoch=None):
        plateau_inputs.append(float(metrics))
        return original_step(self, metrics, epoch)

    set_seed(0)
    train_loader, val_loader = _make_loaders(seed=0)
    ReduceLROnPlateau.step = recording_step  # type: ignore[method-assign]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = _make_model().train(
                params=NNTrainParams(
                    n_epochs=4,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
                    scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=0, threshold=1e-3),
                ),
                train_step_fn=scripted_train_step,
                eval_step_fn=scripted_eval_step,
            )
    finally:
        ReduceLROnPlateau.step = original_step  # type: ignore[method-assign]

    # Control signal: finite fallback per epoch, skipped when nothing is finite.
    assert plateau_inputs == [0.9, train_signal[1], 1e-3], plateau_inputs
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 3, best
    plateau_warnings = [str(w.message) for w in caught if "ReduceLROnPlateau" in str(w.message)]
    assert len(plateau_warnings) == 3, plateau_warnings  # one bounded warning per affected epoch
    assert "no metric available" in plateau_warnings[2], plateau_warnings

    # Diagnostics: raw observations retained live and in the checkpoint payload …
    epoch_end = {idp.epoch_idx: idp for idp in run.idps if idp.val_edp is not None}
    assert epoch_end[0].val_edp is not None and math.isnan(epoch_end[0].val_edp.error)
    assert epoch_end[2].val_edp is not None and epoch_end[2].val_edp.loss is None
    first = NNCheckpoint.load(run=run.id, type=Checkpoints.FIRST)
    assert first is not None and first.idp.val_edp is not None and math.isnan(first.idp.val_edp.error)
    # … while CSV readback keeps its existing NaN → None normalization.
    reloaded = {idp.epoch_idx: idp for idp in NNRun.load(run.id).idps if idp.val_edp is not None}
    assert reloaded[0].val_edp is not None and reloaded[0].val_edp.error is None
    assert reloaded[0].val_edp.loss == 0.9

    summary = {"best_epoch": best.idp.epoch_idx, "plateau_inputs": plateau_inputs, "n_warnings": len(plateau_warnings)}
    print(f"finite-fallback workflow: {summary}")
    return summary


def main() -> None:
    set_seed(0)
    train_loader, val_loader = _make_loaders(seed=0)

    model = _make_model()

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
