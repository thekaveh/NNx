"""Pluggable ``eval_step_fn`` on ``NNModel.train`` (#86), mirroring ``train_step_fn``.

When provided, the per-epoch validation pass calls ``eval_step_fn(ctx)`` (no-grad,
with a val-flavored :class:`EvalStepContext`) and uses its returned
``NNEvaluationDataPoint`` as ``val_edp`` — instead of the built-in classification
``evaluate()``. Omitted → byte-identical current behavior.

The persisted run (idps/run.yaml) then carries the custom val metrics naturally,
because the val pass runs INSIDE the epoch loop before the incremental save —
this is what lets downstream consumers (nnx-studio's LM val-perplexity) drop
their inject-via-callback workaround, whose values never persisted.
"""

from __future__ import annotations

import pytest
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
    NNSchedulerParams,
    NNTrainParams,
    Optims,
)
from nnx.nn.nn_model import EvalStepContext
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_run import NNRun


def _tiny_model() -> NNModel:
    return NNModel(
        net_params=__import__("nnx").NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _loaders() -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(0)
    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    train = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
    val = DataLoader(TensorDataset(X[:8], y[:8]), batch_size=8, shuffle=False)
    return train, val


def _params(train: DataLoader, val: DataLoader | None, n_epochs: int = 2) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=train,
        val_loader=val,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    )


def _const_eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
    """A recognizable custom val metric: loss = 0.125, accuracy = 0.5 — values the
    built-in classification evaluate() would essentially never produce exactly."""
    assert ctx.model is not None
    assert ctx.val_loader is not None
    assert ctx.epoch_idx >= 0
    # prove no-grad is active inside the eval step
    assert not torch.is_grad_enabled()
    return NNEvaluationDataPoint(loss=0.125, error=0.125, accuracy=0.5, f1=0.5, precision=0.5, recall=0.5)


def test_eval_step_fn_drives_val_edp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    train, val = _loaders()
    run = model.train(params=_params(train, val), eval_step_fn=_const_eval_step)

    # Every epoch's val_edp came from the custom step, not classification evaluate().
    val_edps = [idp.val_edp for idp in run.idps if idp.val_edp is not None]
    assert len(val_edps) == 2  # one per epoch
    for edp in val_edps:
        assert edp.loss == 0.125
        assert edp.accuracy == 0.5


def test_eval_step_fn_persists_to_saved_run(tmp_path, monkeypatch):
    """The custom val metrics land in the PERSISTED run (the whole point of #86 —
    the callback workaround's injected values never survived the save)."""
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    train, val = _loaders()
    run = model.train(params=_params(train, val), eval_step_fn=_const_eval_step)

    reloaded = NNRun.load(run.id)
    assert reloaded is not None
    saved_val = [idp.val_edp for idp in reloaded.idps if idp.val_edp is not None]
    assert len(saved_val) == 2
    assert all(edp.loss == 0.125 for edp in saved_val)


def test_omitted_eval_step_fn_uses_builtin_evaluate(tmp_path, monkeypatch):
    """Back-compat: no eval_step_fn → the classification evaluate() path, with
    genuine (non-sentinel) metrics."""
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    train, val = _loaders()
    run = model.train(params=_params(train, val))

    val_edps = [idp.val_edp for idp in run.idps if idp.val_edp is not None]
    assert len(val_edps) == 2
    # Built-in evaluate over an 8-sample binary split: accuracy is k/8 — never
    # exactly the 0.125-loss sentinel pair the custom step returns.
    assert not any(edp.loss == 0.125 and edp.accuracy == 0.5 for edp in val_edps)


def test_eval_step_fn_without_val_loader_is_never_called(tmp_path, monkeypatch):
    """No val_loader → no val pass, custom step included (mirrors current gating)."""
    monkeypatch.chdir(tmp_path)
    model = _tiny_model()
    train, _ = _loaders()
    calls = []

    def spy(ctx: EvalStepContext) -> NNEvaluationDataPoint:
        calls.append(ctx.epoch_idx)
        return NNEvaluationDataPoint(loss=0.0, error=0.0, accuracy=0.0, f1=0.0, precision=0.0, recall=0.0)

    run = model.train(params=_params(train, None), eval_step_fn=spy)
    assert calls == []
    assert all(idp.val_edp is None for idp in run.idps)


def test_nonfinite_first_epoch_best_recovers(tmp_path, monkeypatch):
    """FIX-009: a NaN validation error at epoch 0 must not freeze BEST there.
    Under the finite fallback the epoch-0 signal is its finite loss (1.0),
    so epoch 1's finite error (0.2) replaces it. The raw NaN observation is
    retained in the live run history; CSV readback keeps mapping NaN → None."""
    import math

    from nnx import Checkpoints, NNCheckpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    train, val = _loaders()
    scripted_error = {0: float("nan"), 1: 0.2}

    def eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
        return NNEvaluationDataPoint(
            loss=1.0, error=scripted_error[ctx.epoch_idx], accuracy=0.5, f1=0.5, precision=0.5, recall=0.5
        )

    with pytest.warns(RuntimeWarning, match="epoch 0: ignoring non-finite metric"):
        run = _tiny_model().train(params=_params(train, val, n_epochs=2), eval_step_fn=eval_step)

    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    last = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert best is not None and last is not None
    assert (best.idp.epoch_idx, last.idp.epoch_idx) == (1, 1)
    assert best.idp.val_edp is not None and best.idp.val_edp.error == 0.2

    epoch0 = [idp for idp in run.idps if idp.epoch_idx == 0 and idp.val_edp is not None][-1]
    assert epoch0.val_edp is not None
    assert math.isnan(epoch0.val_edp.error) and epoch0.val_edp.loss == 1.0
    reloaded = [idp for idp in NNRun.load(run.id).idps if idp.epoch_idx == 0 and idp.val_edp is not None][-1]
    assert reloaded.val_edp is not None
    assert reloaded.val_edp.error is None and reloaded.val_edp.loss == 1.0


def test_plateau_never_receives_nonfinite_metric(tmp_path, monkeypatch):
    """FIX-009: ReduceLROnPlateau only ever sees finite values. Scripted
    epochs: (nan, 1.0) → steps on val loss; (inf, -inf) → steps on the
    finite train error; everything unavailable → no step at all, with a
    warning that says *absent* rather than *non-finite*; then a finite
    improvement steps on val error."""
    import warnings
    from dataclasses import replace

    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from nnx import TrainStepContext, default_train_step

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    steps: list[float] = []
    original_step = ReduceLROnPlateau.step

    def spy_step(self, metrics, epoch=None):
        steps.append(metrics)
        return original_step(self, metrics, epoch)

    monkeypatch.setattr(ReduceLROnPlateau, "step", spy_step)

    val_script = {
        0: (float("nan"), 1.0),
        1: (float("inf"), float("-inf")),
        2: (None, None),
        3: (0.1, 0.5),
    }

    def eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
        error, loss = val_script[ctx.epoch_idx]
        return NNEvaluationDataPoint(loss=loss, error=error, accuracy=0.5, f1=0.5, precision=0.5, recall=0.5)

    last_train_error: dict[int, float] = {}

    def train_step(ctx: TrainStepContext) -> NNEvaluationDataPoint:
        edp = default_train_step(ctx)
        if ctx.epoch_idx == 2:
            # The update happened; the *reported* signal is simply unavailable.
            return replace(edp, loss=None, error=None)
        assert edp.error is not None
        last_train_error[ctx.epoch_idx] = edp.error
        return edp

    train, val = _loaders()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _tiny_model().train(params=_params(train, val, n_epochs=4), train_step_fn=train_step, eval_step_fn=eval_step)

    assert steps == [1.0, last_train_error[1], 0.1]

    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    rejected = [m for m in messages if "non-finite" in m]
    absent = [m for m in messages if "no metric available" in m]
    assert any("epoch 0" in m and "val_edp.error=nan" in m and "val_edp.loss" in m for m in rejected), messages
    assert any("epoch 1" in m and "val_edp.error=inf" in m and "val_edp.loss=-inf" in m for m in rejected), messages
    assert len(absent) == 1 and "epoch 2" in absent[0] and "non-finite" not in absent[0], messages
    assert not any("epoch 3" in m for m in messages), messages
