"""Tests for the Callback protocol and standard callbacks."""
from __future__ import annotations

from types import SimpleNamespace

from nnx.nn.callbacks import (
    Callback,
    EarlyStopping,
    LRMonitor,
    _LegacyCallback,
)


def _make_ctx(epoch=0, val_error=None, train_error=0.5, lr=1e-3):
    """A minimal _CallbackContext-shaped object."""
    val_edp = SimpleNamespace(error=val_error) if val_error is not None else None
    train_edp = SimpleNamespace(error=train_error)
    idp = SimpleNamespace(epoch_idx=epoch, val_edp=val_edp, train_edp=train_edp, lr=lr)
    optimizer = SimpleNamespace(param_groups=[{"lr": lr}])
    return SimpleNamespace(
        model=None, run=None, optimizer=optimizer,
        epoch=epoch, idp=idp, idps=[idp], should_stop=False,
    )


def test_callback_base_class_hooks_are_no_op():
    cb = Callback()
    ctx = _make_ctx()
    cb.on_train_begin(ctx)
    cb.on_epoch_begin(ctx)
    cb.on_epoch_end(ctx)
    cb.on_train_end(ctx)
    assert ctx.should_stop is False


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(monitor="val_edp.error", patience=2, mode="min")
    ctx = _make_ctx(epoch=0, val_error=0.5)
    es.on_epoch_end(ctx)
    assert not ctx.should_stop

    # No improvement for `patience` epochs → should_stop
    ctx = _make_ctx(epoch=1, val_error=0.5)
    es.on_epoch_end(ctx)
    assert not ctx.should_stop
    ctx = _make_ctx(epoch=2, val_error=0.5)
    es.on_epoch_end(ctx)
    assert ctx.should_stop


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(monitor="val_edp.error", patience=2, mode="min")
    for epoch, err in [(0, 0.5), (1, 0.5), (2, 0.4)]:
        ctx = _make_ctx(epoch=epoch, val_error=err)
        es.on_epoch_end(ctx)
        assert not ctx.should_stop


def test_early_stopping_max_mode():
    es = EarlyStopping(monitor="val_edp.error", patience=1, mode="max")
    ctx = _make_ctx(epoch=0, val_error=0.7)
    es.on_epoch_end(ctx)
    ctx = _make_ctx(epoch=1, val_error=0.7)
    es.on_epoch_end(ctx)
    assert ctx.should_stop


def test_early_stopping_invalid_mode():
    import pytest
    with pytest.raises(ValueError):
        EarlyStopping(mode="middle")


def test_lr_monitor_records_history():
    mon = LRMonitor()
    for ep, lr in [(0, 1e-3), (1, 5e-4), (2, 1e-4)]:
        ctx = _make_ctx(epoch=ep, lr=lr)
        mon.on_epoch_end(ctx)
    assert mon.history == [1e-3, 5e-4, 1e-4]


def test_legacy_callable_adapter_fires_on_epoch_end():
    seen = []
    legacy = _LegacyCallback(lambda idps: seen.append(len(idps)))
    ctx = _make_ctx(epoch=0)
    ctx.idps = ["idp0", "idp1", "idp2"]
    legacy.on_epoch_end(ctx)
    assert seen == [3]


def test_model_checkpoint_writes_at_matched_epochs(tmp_path, monkeypatch):
    """ModelCheckpoint must actually save a checkpoint at matched epochs.
    Previously this callback was a no-op stub; the audit caught it and
    we wired it through to NNCheckpoint.to_file."""
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
        NNParams,
        NNSchedulerParams,
        NNTrainParams,
        Optims,
    )
    from nnx.nn.callbacks import ModelCheckpoint
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
    model = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    cb = ModelCheckpoint(epochs=[0, 2], tag="snap")
    run = model.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
            ),
        ),
        callbacks=[cb],
    )
    # Both matched epochs must have produced files; epoch 1 (unmatched) must not.
    ckpt_dir = tmp_path / "runs" / run.id / "checkpoints"
    assert (ckpt_dir / "snap_e0.pt").is_file()
    assert (ckpt_dir / "snap_e2.pt").is_file()
    assert not (ckpt_dir / "snap_e1.pt").exists()
    # File contents must be a loadable NNCheckpoint.
    ckpt = NNCheckpoint.from_file(str(ckpt_dir / "snap_e0.pt"))
    assert ckpt is not None
    assert ckpt.idp.epoch_idx == 0


def test_model_checkpoint_no_matching_epochs_is_noop(tmp_path, monkeypatch):
    """When `epochs` is empty / None, ModelCheckpoint must NEVER write —
    the callback is just inert, not creating empty files."""
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
        NNParams,
        NNSchedulerParams,
        NNTrainParams,
        Optims,
    )
    from nnx.nn.callbacks import ModelCheckpoint

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    loader = DataLoader(
        TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))),
        batch_size=4, shuffle=False,
    )
    model = NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )
    cb = ModelCheckpoint()   # no epochs argument
    run = model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
            ),
        ),
        callbacks=[cb],
    )
    ckpt_dir = tmp_path / "runs" / run.id / "checkpoints"
    # No custom_e*.pt files — only the standard cycle wrote anything.
    custom_files = list(ckpt_dir.glob("custom*"))
    assert custom_files == []
