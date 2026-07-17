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
