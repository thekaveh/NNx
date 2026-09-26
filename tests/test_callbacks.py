"""Tests for the Callback protocol and standard callbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nnx.nn.callbacks import (
    Callback,
    EarlyStopping,
    LRMonitor,
    _LegacyCallback,
)
from nnx.nn.nn_model import _CallbackFinalizer


def _make_ctx(epoch=0, val_error=None, train_error=0.5, lr=1e-3):
    """A minimal _CallbackContext-shaped object."""
    val_edp = SimpleNamespace(error=val_error) if val_error is not None else None
    train_edp = SimpleNamespace(error=train_error)
    idp = SimpleNamespace(epoch_idx=epoch, val_edp=val_edp, train_edp=train_edp, lr=lr)
    optimizer = SimpleNamespace(param_groups=[{"lr": lr}])
    return SimpleNamespace(
        model=None,
        run=None,
        optimizer=optimizer,
        epoch=epoch,
        idp=idp,
        idps=[idp],
        should_stop=False,
    )


def test_callback_base_class_hooks_are_no_op():
    cb = Callback()
    ctx = _make_ctx()
    cb.on_train_begin(ctx)
    cb.on_epoch_begin(ctx)
    cb.on_epoch_end(ctx)
    cb.on_train_end(ctx)
    assert ctx.should_stop is False


def test_callback_finalizer_runs_every_cleanup_after_failure():
    events: list[str] = []

    class _Failing(Callback):
        def on_train_end(self, ctx):  # noqa: ARG002
            events.append("failing")
            raise RuntimeError("cleanup failed")

    class _Following(Callback):
        def on_train_end(self, ctx):  # noqa: ARG002
            events.append("following")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        with _CallbackFinalizer([_Following(), _Failing()], _make_ctx()) as lifecycle:
            lifecycle.start()

    assert events == ["failing", "following"]


def test_callback_finalizer_runs_every_cleanup_after_keyboard_interrupt():
    events: list[str] = []

    class _Interrupting(Callback):
        def on_train_end(self, ctx):  # noqa: ARG002
            events.append("interrupting")
            raise KeyboardInterrupt

    class _Following(Callback):
        def on_train_end(self, ctx):  # noqa: ARG002
            events.append("following")

    with pytest.raises(KeyboardInterrupt):
        with _CallbackFinalizer([_Following(), _Interrupting()], _make_ctx()) as lifecycle:
            lifecycle.start()

    assert events == ["interrupting", "following"]


def test_callback_finalizer_preserves_training_exception():
    class _Failing(Callback):
        def on_train_end(self, ctx):  # noqa: ARG002
            raise RuntimeError("cleanup failed")

    with pytest.warns(RuntimeWarning, match="cleanup failed"):
        with pytest.raises(ValueError, match="training failed"):
            with _CallbackFinalizer([_Failing()], _make_ctx()) as lifecycle:
                lifecycle.start()
                raise ValueError("training failed")


def test_callback_finalizer_unwinds_started_callbacks_after_begin_failure():
    events: list[str] = []

    class _Started(Callback):
        def on_train_begin(self, ctx):  # noqa: ARG002
            events.append("started.begin")

        def on_train_end(self, ctx):  # noqa: ARG002
            events.append("started.end")

    class _BeginFailure(Callback):
        def on_train_begin(self, ctx):  # noqa: ARG002
            events.append("failing.begin")
            raise RuntimeError("begin failed")

        def on_train_end(self, ctx):  # noqa: ARG002
            events.append("failing.end")

    class _NeverStarted(Callback):
        def on_train_begin(self, ctx):  # noqa: ARG002
            events.append("never.begin")

    with pytest.raises(RuntimeError, match="begin failed"):
        with _CallbackFinalizer([_Started(), _BeginFailure(), _NeverStarted()], _make_ctx()) as lifecycle:
            lifecycle.start()

    assert events == ["started.begin", "failing.begin", "started.end"]


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


def test_legacy_callable_adapter_works_without_ipython(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    seen = []
    legacy = _LegacyCallback(lambda idps: seen.append(len(idps)))
    ctx = _make_ctx(epoch=0)
    ctx.idps = ["idp0"]

    legacy.on_epoch_end(ctx)

    assert seen == [1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"monitor": "validation.loss"}, "monitor"),
        ({"patience": -1}, "patience"),
        ({"min_delta": -0.1}, "min_delta"),
    ],
)
def test_early_stopping_rejects_invalid_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EarlyStopping(**kwargs)


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
    cb = ModelCheckpoint(epochs=[0, 2], tag="snap")
    run = model.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=1,
                cooldown=1,
                threshold=1e-3,
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


def test_model_checkpoint_ignores_epoch_without_completed_idp(tmp_path, monkeypatch):
    from nnx.nn.callbacks import ModelCheckpoint

    monkeypatch.chdir(tmp_path)
    ctx = _make_ctx(epoch=0)
    ctx.idp = None

    ModelCheckpoint(epochs=[0]).on_epoch_end(ctx)

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("tag", ["", ".", "..", "../escaped", "nested/tag", "bad tag"])
def test_model_checkpoint_rejects_unsafe_tags(tag):
    from nnx.nn.callbacks import ModelCheckpoint

    with pytest.raises(ValueError, match="tag"):
        ModelCheckpoint(epochs=[0], tag=tag)


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
        batch_size=4,
        shuffle=False,
    )
    model = NNModel(
        net_params=NNParams(
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
    cb = ModelCheckpoint()  # no epochs argument
    run = model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-3,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=1,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        callbacks=[cb],
    )
    ckpt_dir = tmp_path / "runs" / run.id / "checkpoints"
    # No custom_e*.pt files — only the standard cycle wrote anything.
    custom_files = list(ckpt_dir.glob("custom*"))
    assert custom_files == []


def test_early_stopping_resets_state_on_train_begin():
    """Reusing one EarlyStopping instance across train() calls must
    start the second run with fresh best/patience state — pre-fix the
    previous run's best leaked in and could stop the new run after
    `patience` epochs even while it was improving."""
    es = EarlyStopping(monitor="train_edp.loss", patience=2)
    es._best = 0.0001  # simulate a finished prior run
    es._wait = 1
    es.on_train_begin(ctx=None)  # the hook reads nothing from ctx
    assert es._best is None
    assert es._wait == 0


# --- #189: default monitor selection and missing-field diagnostics ---------


def _edp_ctx(epoch=0, *, val=None, train=None):
    """A context whose data points carry explicit ``error`` / ``loss`` pairs.

    ``val`` / ``train`` are ``(error, loss)`` tuples, or ``None`` for an
    absent data point (no validation loader configured).
    """
    val_edp = None if val is None else SimpleNamespace(error=val[0], loss=val[1])
    train_edp = None if train is None else SimpleNamespace(error=train[0], loss=train[1])
    idp = SimpleNamespace(epoch_idx=epoch, val_edp=val_edp, train_edp=train_edp, lr=1e-3)
    return SimpleNamespace(epoch=epoch, idp=idp, idps=[idp], should_stop=False)


def _drive(es, epochs, *, split="val"):
    """Feed ``(error, loss)`` pairs to a fresh run; return the stop epoch."""
    es.on_train_begin(ctx=None)
    for epoch, pair in enumerate(epochs):
        ctx = _edp_ctx(epoch, **{split: pair})
        es.on_epoch_end(ctx)
        if ctx.should_stop:
            return epoch
    return None


def test_early_stopping_default_falls_back_to_val_loss_for_regression():
    """Regression evaluators leave ``error`` as None; the default monitor
    must then track ``val_edp.loss`` instead of silently never stopping."""
    es = EarlyStopping(patience=2)
    assert es.monitor is None  # automatic selection, not an explicit field
    stop = _drive(es, [(None, 0.5), (None, 0.5), (None, 0.5), (None, 0.5)])
    assert stop == 2
    assert es.selected_monitor == "val_edp.loss"


def test_early_stopping_default_prefers_val_error_for_classification():
    """With both fields present the default follows ``error``: an improving
    error keeps training alive even while the loss is flat, and a flat
    error stops it even while the loss improves."""
    es = EarlyStopping(patience=2)
    assert _drive(es, [(0.5, 0.9), (0.4, 0.9), (0.3, 0.9), (0.2, 0.9)]) is None
    assert es.selected_monitor == "val_edp.error"
    assert _drive(es, [(0.5, 0.9), (0.5, 0.8), (0.5, 0.7), (0.5, 0.6)]) == 2
    assert es.selected_monitor == "val_edp.error"


@pytest.mark.parametrize(
    ("monitor", "split", "index"),
    [
        ("val_edp.error", "val", 0),
        ("val_edp.loss", "val", 1),
        ("train_edp.error", "train", 0),
        ("train_edp.loss", "train", 1),
    ],
)
def test_early_stopping_explicit_monitor_reads_exactly_that_field(monitor, split, index):
    """An explicit monitor never falls back: only the named field decides."""
    es = EarlyStopping(monitor=monitor, patience=2)
    flat, improving = 0.5, [0.9, 0.8, 0.7, 0.6]
    epochs = []
    for value in improving:
        pair = [value, value]
        pair[index] = flat
        epochs.append(tuple(pair))
    assert _drive(es, epochs, split=split) == 2
    assert es.selected_monitor == monitor


@pytest.mark.parametrize(
    ("monitor", "val", "reason"),
    [
        ("val_edp.error", (None, 0.5), "has no 'error' value"),
        ("val_edp.loss", (0.5, None), "has no 'loss' value"),
        ("val_edp.loss", None, "no validation data point"),
    ],
)
def test_early_stopping_explicit_missing_field_warns_once(monitor, val, reason):
    """A missing explicitly requested field is reported, not silently
    ignored — once per run, and the epoch is not counted."""
    import warnings

    es = EarlyStopping(monitor=monitor, patience=1)
    es.on_train_begin(ctx=None)
    ctx = _edp_ctx(0, val=val, train=(0.5, 0.5))
    with pytest.warns(RuntimeWarning, match=reason) as record:
        es.on_epoch_end(ctx)
    assert monitor in str(record[0].message)
    assert not ctx.should_stop
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for epoch in (1, 2, 3):
            ctx = _edp_ctx(epoch, val=val, train=(0.5, 0.5))
            es.on_epoch_end(ctx)  # already reported this run → no repeat
            assert not ctx.should_stop
    es.on_train_begin(ctx=None)
    with pytest.warns(RuntimeWarning, match=reason):
        es.on_epoch_end(_edp_ctx(0, val=val, train=(0.5, 0.5)))


def test_early_stopping_default_warns_without_validation_data_point():
    """The default monitor reads validation only; without a validation
    loader it says so instead of silently doing nothing."""
    es = EarlyStopping(patience=1)
    es.on_train_begin(ctx=None)
    ctx = _edp_ctx(0, val=None, train=(0.5, 0.5))
    with pytest.warns(RuntimeWarning, match="no validation data point"):
        es.on_epoch_end(ctx)
    assert not ctx.should_stop
    assert es.selected_monitor is None


def test_early_stopping_default_warns_when_validation_has_no_error_or_loss():
    es = EarlyStopping(patience=1)
    es.on_train_begin(ctx=None)
    with pytest.warns(RuntimeWarning, match="neither an 'error' nor a 'loss'"):
        es.on_epoch_end(_edp_ctx(0, val=(None, None)))
    assert es.selected_monitor is None


def test_early_stopping_default_skips_nonfinite_error_when_loss_is_finite():
    """Selection follows the same finiteness rule as BEST / ReduceLROnPlateau:
    a NaN validation error must not lock the run onto an unusable field."""
    import warnings

    nan = float("nan")
    es = EarlyStopping(patience=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the NaN error is never compared
        stop = _drive(es, [(nan, 1.0), (nan, 0.5), (nan, 0.2), (nan, 0.1)])
    assert es.selected_monitor == "val_edp.loss"
    assert stop is None


def test_early_stopping_nonfinite_value_never_becomes_best():
    """A NaN first epoch must not poison ``_best`` (every later comparison
    against NaN is False); later finite improvements still count."""
    nan = float("nan")
    es = EarlyStopping(monitor="val_edp.loss", patience=2)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        stop = _drive(es, [(None, nan), (None, 1.0), (None, 0.9), (None, 0.8), (None, 0.7)])
    assert stop is None
    assert es._best == 0.7


def test_early_stopping_diverged_run_still_stops():
    """Non-finite epochs count as epochs without improvement, including
    under the default before any finite value allowed a selection."""
    nan = float("nan")
    es = EarlyStopping(patience=2)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        stop = _drive(es, [(nan, nan), (nan, nan), (nan, nan)])
    assert stop == 1
    assert es.selected_monitor is None


def test_early_stopping_warning_repeats_for_each_run_under_default_filters():
    """Python's default filter shows a message from one location only once
    per process; each run's diagnostic must still reach the user, and it is
    attributed to the caller rather than to nnx internals."""
    import warnings

    es = EarlyStopping(patience=1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for _ in range(2):
            es.on_train_begin(ctx=None)
            es.on_epoch_end(_edp_ctx(0, val=None, train=(0.5, 0.5)))
            es.on_epoch_end(_edp_ctx(1, val=None, train=(0.5, 0.5)))
    messages = [w for w in record if issubclass(w.category, RuntimeWarning)]
    assert len(messages) == 2
    assert all(w.filename == __file__ for w in messages)


def test_early_stopping_default_rejects_max_mode():
    """The default only ever selects error or loss, which improve downward."""
    with pytest.raises(ValueError, match="explicit monitor"):
        EarlyStopping(mode="max")
    assert EarlyStopping(monitor="val_edp.error", mode="max").mode == "max"


def test_early_stopping_default_selection_stays_fixed_for_the_run():
    """Once the default picks ``val_edp.loss`` it keeps comparing loss even
    when a later epoch starts reporting ``error``."""
    es = EarlyStopping(patience=2)
    stop = _drive(es, [(None, 0.5), (0.4, 0.5), (0.3, 0.5), (0.2, 0.5)])
    assert es.selected_monitor == "val_edp.loss"
    assert stop == 2


def test_early_stopping_default_reselects_per_train_call():
    """One instance reused across train() calls picks its field afresh:
    a regression run (loss) followed by a classification run (error)."""
    es = EarlyStopping(patience=2)
    assert _drive(es, [(None, 0.5), (None, 0.5), (None, 0.5)]) == 2
    assert es.selected_monitor == "val_edp.loss"
    assert _drive(es, [(0.5, 0.9), (0.4, 0.9), (0.3, 0.9)]) is None
    assert es.selected_monitor == "val_edp.error"


def test_early_stopping_default_stops_regression_training(tmp_path, monkeypatch):
    """End to end: a loss-only ``eval_step_fn`` (regression style, ``error``
    left as None) now stops a real ``train()`` under the default monitor."""
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
        NNTrainParams,
        Optims,
    )
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )

    def loss_only_eval(ctx):  # noqa: ARG001
        return NNEvaluationDataPoint(f1=0.0, recall=0.0, accuracy=0.0, precision=0.0, loss=0.25)

    es = EarlyStopping(patience=1)
    run = model.train(
        params=NNTrainParams(
            n_epochs=6,
            train_loader=loader,
            val_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0),
        ),
        callbacks=[es],
        eval_step_fn=loss_only_eval,
    )
    epochs = sorted({idp.epoch_idx for idp in run.idps})
    assert epochs == [0, 1]
    assert es.selected_monitor == "val_edp.loss"


# --- FEAT-002: task records through the metric writers ----------------------


class _FakeSummaryWriter:
    """Stands in for torch.utils.tensorboard.SummaryWriter (injected)."""

    instances: list[_FakeSummaryWriter] = []

    def __init__(self, log_dir=None):
        self.scalars: list[tuple[str, float, int]] = []
        _FakeSummaryWriter.instances.append(self)

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), step))

    def flush(self):
        pass

    def close(self):
        pass


class _FakeWandbRun:
    def __init__(self):
        self.logs: list[tuple[dict, int]] = []

    def log(self, values, step):
        self.logs.append((dict(values), step))

    def finish(self):  # pragma: no cover - the callback does not own this run
        raise AssertionError("an injected run must not be finished by the callback")


def test_regression_task_records_reach_tensorboard_and_wandb_without_classification_fields(tmp_path, monkeypatch):
    import math
    import sys

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import (
        Activations,
        Losses,
        Nets,
        NNModel,
        NNModelParams,
        NNOptimParams,
        NNParams,
        NNRun,
        NNTrainParams,
        TaskSpec,
        set_seed,
    )
    from nnx.nn.callbacks import TensorBoardCallback, WandbCallback

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=_FakeSummaryWriter))
    _FakeSummaryWriter.instances.clear()

    set_seed(0)
    model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)),
    )
    x = torch.randn(10, 3)
    y = x.sum(dim=1, keepdim=True)
    y[3, 0] = math.nan
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    wandb_run = _FakeWandbRun()
    run = model.train(
        params=NNTrainParams(n_epochs=2, optim=NNOptimParams.builder().sgd(max_lr=0.01).build())
        .with_train_loader(loader)
        .with_val_loader(loader),
        callbacks=[TensorBoardCallback(), WandbCallback(wandb_run=wandb_run)],
    )

    classification = {"accuracy", "f1", "precision", "recall", "error"}
    (writer,) = _FakeSummaryWriter.instances
    tb = {(name, step): value for name, value, step in writer.scalars}
    for epoch, idp in enumerate(idp for idp in run.idps if idp.val_edp is not None):
        assert tb[("val/mse", epoch)] == pytest.approx(idp.val_edp.metrics["mse"])
        assert tb[("val/mae", epoch)] == pytest.approx(idp.val_edp.metrics["mae"])
        assert tb[("train/mse", epoch)] == pytest.approx(idp.train_edp.metrics["mse"])
    assert not {name.split("/")[-1] for name, _ in tb} & classification
    assert all(value != 0.0 for (name, _), value in tb.items() if name.endswith(("mse", "mae")))

    logged = [values for values, _ in wandb_run.logs]
    assert len(logged) == 2 and all({"val/mse", "val/mae", "train/mse", "train/loss"} <= set(v) for v in logged)
    assert not {key.split("/")[-1] for values in logged for key in values} & classification

    reloaded = NNRun.load(run.id)
    for rendered in (str(run), str(reloaded)):
        assert "task=regression[1]" in rendered
    html = reloaded._repr_html_()
    assert "regression[1]" in html and "val_err" not in html and "train_err" not in html
    _assert_same_records(reloaded.idps, run.idps)


def _assert_same_records(loaded, expected):
    """Reloaded idps.csv records equal the originals up to pandas' float
    parsing (the CSV reader is not bit-exact for every double)."""
    assert len(loaded) == len(expected)
    for got, want in zip(loaded, expected, strict=True):
        for edp_got, edp_want in ((got.train_edp, want.train_edp), (got.val_edp, want.val_edp)):
            assert (edp_got is None) == (edp_want is None)
            if edp_got is None:
                continue
            state_got, state_want = edp_got.state(), edp_want.state()
            assert state_got.keys() == state_want.keys()
            for key, value in state_want.items():
                if isinstance(value, float):
                    assert state_got[key] == pytest.approx(value, rel=1e-12), key
                elif isinstance(value, dict):
                    assert state_got[key] == pytest.approx(value, rel=1e-12), key
                else:
                    assert state_got[key] == value, key
