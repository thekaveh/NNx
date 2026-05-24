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
