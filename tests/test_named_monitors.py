"""FEAT-003: named monitors shared by BEST selection, early stopping and
plateau scheduling, whole-epoch training summaries, and their records.
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from nnx import (
    Activations,
    Callback,
    Checkpoints,
    Devices,
    EarlyStopping,
    EvalStepContext,
    Losses,
    MetricSpec,
    MonitorRecord,
    MonitorSpec,
    MonitorUnavailableError,
    Nets,
    NNCheckpoint,
    NNEvaluationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainerParams,
    NNTrainParams,
    Trainer,
    TrainerStepContext,
    set_seed,
)
from nnx.monitors import MonitorTracker


def _model(seed: int = 0) -> NNModel:
    set_seed(seed)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[6], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _data(n: int = 5, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4, generator=generator), torch.randint(0, 2, (n,), generator=generator)


def _batches(X: torch.Tensor, Y: torch.Tensor, sizes: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out, start = [], 0
    for size in sizes:
        out.append((X[start : start + size], Y[start : start + size]))
        start += size
    return out


def _scripted_eval(values: list):
    """An eval step reporting a scripted validation NLL (None: absent)."""

    def eval_step(ctx: EvalStepContext) -> NNEvaluationDataPoint:
        value = values[ctx.epoch_idx]
        return NNEvaluationDataPoint(loss=1.0, metrics={} if value is None else {"nll": value})

    return eval_step


class _Spy(Callback):
    """Records, after every other callback and the scheduler, what each
    epoch decided."""

    def __init__(self, stopper: EarlyStopping):
        self.stopper = stopper
        self.rows: list[tuple] = []

    def on_epoch_end(self, ctx):
        self.rows.append(
            (
                ctx.idp.selection.improved if ctx.idp.selection is not None else None,
                self.stopper._wait,
                ctx.optimizer.param_groups[0]["lr"],
                ctx.idp.selection,
                ctx.idp.train_summary,
            )
        )


def _plateau() -> NNSchedulerParams:
    # threshold=0.5 would call 0.8 → 0.7 "no improvement" under the legacy
    # relative rule; with a monitor the plateau uses the monitor's rule.
    return NNSchedulerParams(patience=0, cooldown=0, factor=0.5, threshold=0.5, min_lr=0.0)


def _decisions(tmp_path, monkeypatch, values: list, min_delta: float):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    spec = MonitorSpec(metric="nll", min_delta=min_delta)
    X, Y = _data()
    stopper = EarlyStopping(monitor=spec, patience=10)
    spy = _Spy(stopper)
    run = _model().train(
        params=NNTrainParams(
            n_epochs=len(values),
            train_loader=_batches(X, Y, [2, 2, 1]),
            val_loader=_batches(X, Y, [5]),
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            scheduler=_plateau(),
            metrics=[MetricSpec("nll")],
            monitor=spec,
        ),
        eval_step_fn=_scripted_eval(values),
        callbacks=[stopper, spy],
    )
    return run, spy.rows


def test_best_early_stopping_and_plateau_make_identical_decisions(tmp_path, monkeypatch):
    run, rows = _decisions(tmp_path, monkeypatch, [0.8, 0.7, 0.695, 0.71], min_delta=0.01)
    improved = [row[0] for row in rows]
    assert improved == [True, True, False, False]  # two improvements: 0.695 is within min_delta
    assert [row[1] for row in rows] == [0, 0, 1, 2]  # EarlyStopping waited exactly on the same epochs
    assert [row[2] for row in rows] == pytest.approx([0.1, 0.1, 0.05, 0.025])  # plateau reduced on them too
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 1 and best.idp.selection.value == pytest.approx(0.7)


def test_exact_ties_never_improve(tmp_path, monkeypatch):
    run, rows = _decisions(tmp_path, monkeypatch, [0.8, 0.8, 0.8], min_delta=0.0)
    assert [row[0] for row in rows] == [True, False, False]
    assert [row[1] for row in rows] == [0, 1, 2]
    assert [row[2] for row in rows] == pytest.approx([0.1, 0.05, 0.025])
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 0


def test_monitor_spec_contract():
    spec = MonitorSpec(metric="nll", split="train", mode="min", min_delta=0.01, on_missing="error")
    assert spec.key == "train.nll" and MonitorSpec.from_state(spec.state()) == spec
    assert MonitorSpec().mode == "min" and MonitorSpec(metric="error").mode == "min"
    assert MonitorSpec(metric="nll").mode is None  # a declared metric's direction is resolved at train()
    assert MonitorSpec(metric="nll").resolve([MetricSpec("nll")]).mode == "min"
    assert MonitorSpec(metric="accuracy").resolve([MetricSpec("accuracy")]).mode == "max"
    for bad in (
        dict(split="test"),
        dict(mode="up"),
        dict(min_delta=-0.1),
        dict(min_delta=math.nan),
        dict(min_delta=True),
        dict(on_missing="ignore"),
        dict(on_nonfinite="warn"),
        dict(metric="val nll"),
    ):
        with pytest.raises(ValueError):
            MonitorSpec(**bad)
    maximize = MonitorSpec(metric="accuracy", mode="max", min_delta=0.05)
    assert maximize.improved(0.5, None) and maximize.improved(0.56, 0.5) and not maximize.improved(0.55, 0.5)
    assert not spec.improved(math.nan, None) and not spec.improved(math.inf, 1.0)


def test_unknown_or_unreachable_monitors_fail_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    X, Y = _data()
    loader = _batches(X, Y, [5])
    with pytest.raises(ValueError, match="undeclared metric; declared: loss, error"):
        NNTrainParams(n_epochs=1, monitor=MonitorSpec(metric="nll"))
    base = NNTrainParams(n_epochs=1, train_loader=loader, val_loader=loader)
    with pytest.raises(ValueError, match="EarlyStopping monitor 'val.nll' names an undeclared metric"):
        _model().train(params=base, callbacks=[EarlyStopping(monitor=MonitorSpec(metric="nll"))])
    with pytest.raises(ValueError, match="no val_loader is configured"):
        _model().train(params=NNTrainParams(n_epochs=1, train_loader=loader, monitor=MonitorSpec()))
    with pytest.raises(ValueError, match="only the default training step records"):
        _model().train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=loader,
                metrics=[MetricSpec("nll")],
                monitor=MonitorSpec(metric="nll", split="train"),
            ),
            train_step_fn=lambda ctx: NNEvaluationDataPoint(loss=1.0),
        )
    with pytest.raises(ValueError, match="mode and min_delta"):
        EarlyStopping(monitor=MonitorSpec(), min_delta=0.1)
    assert not os.path.exists("runs")


def test_missing_and_nonfinite_policies(tmp_path, monkeypatch):
    run, rows = _decisions(tmp_path, monkeypatch, [0.8, None, math.nan, 0.5], min_delta=0.0)
    records = [row[3] for row in rows]
    assert [r.status for r in records] == ["ok", "missing", "nonfinite", "ok"]
    assert [r.improved for r in records] == [True, False, False, True]
    # missing: no decision (not counted, no plateau step); non-finite: an epoch without improvement.
    assert [row[1] for row in rows] == [0, 0, 1, 0]
    assert [row[2] for row in rows] == pytest.approx([0.1, 0.1, 0.05, 0.05])

    strict = MonitorTracker(MonitorSpec(on_missing="error", on_nonfinite="error"))
    with pytest.raises(MonitorUnavailableError, match="has no value"):
        strict.observe(None, epoch=3)
    with pytest.raises(MonitorUnavailableError, match="non-finite"):
        strict.observe(math.inf, epoch=3)


def test_training_monitors_use_full_epoch_denominators(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X, Y = _data()
    frozen_sgd = NNOptimParams.builder().sgd(max_lr=0.0).build()  # weights never move within the epoch
    metrics = [MetricSpec("nll"), MetricSpec("accuracy"), MetricSpec("f1")]
    summaries = {}
    for sizes in ([2, 2, 1], [5]):
        run = _model().train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=_batches(X, Y, sizes),
                optim=frozen_sgd,
                metrics=metrics,
                monitor=MonitorSpec(metric="loss", split="train"),
                data_id=f"batches-{sizes}",
            )
        )
        summaries[len(sizes)] = (run, run.idps[-1].train_summary)
    (uneven_run, uneven), (_, full) = summaries[3], summaries[1]
    reference = _model().evaluate([(X, Y)], metrics=metrics)  # the same weights over the full sample
    assert uneven.loss == pytest.approx(full.loss) == pytest.approx(reference.loss)
    assert uneven.error == pytest.approx(full.error) == pytest.approx(reference.error)
    for name in ("nll", "accuracy", "f1"):
        assert uneven.metrics[name] == pytest.approx(full.metrics[name]) == pytest.approx(reference.metrics[name])
    batch_mean = sum(idp.train_edp.loss for idp in uneven_run.idps) / 3
    assert batch_mean != pytest.approx(uneven.loss)  # the unweighted batch mean is what we must not report
    assert uneven_run.idps[-1].selection.value == pytest.approx(uneven.loss)


def test_custom_step_records_are_weighted_by_their_batch_sizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X, Y = _data()
    losses = iter([1.0, 1.0, 4.0])

    def step(ctx) -> NNEvaluationDataPoint:
        return NNEvaluationDataPoint(loss=next(losses))

    run = _model().train(
        params=NNTrainParams(
            n_epochs=1, train_loader=_batches(X, Y, [2, 2, 1]), monitor=MonitorSpec(metric="loss", split="train")
        ),
        train_step_fn=step,
    )
    assert run.idps[-1].train_summary.loss == pytest.approx((2 * 1 + 2 * 1 + 1 * 4) / 5)  # not (1 + 1 + 4) / 3
    assert run.idps[-1].selection.value == pytest.approx(1.6)


class _FakeSummaryWriter:
    instances: list[_FakeSummaryWriter] = []

    def __init__(self, log_dir=None):
        self.scalars: dict[tuple[str, int], float] = {}
        _FakeSummaryWriter.instances.append(self)

    def add_scalar(self, name, value, step):
        self.scalars[(name, step)] = float(value)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeWandbRun:
    def __init__(self):
        self.logs: list[dict] = []

    def log(self, values, step):
        self.logs.append(dict(values))


def test_displays_callbacks_and_reloaded_charts_show_the_selection_summary(tmp_path, monkeypatch):
    from nnx.nn.callbacks import TensorBoardCallback, WandbCallback

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter=_FakeSummaryWriter))
    _FakeSummaryWriter.instances.clear()
    X, Y = _data()
    wandb_run = _FakeWandbRun()
    seen = []

    class Injected(Callback):
        def on_epoch_end(self, ctx):
            seen.append((ctx.idp.train_summary, ctx.idp.selection))

    run = _model().train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=_batches(X, Y, [2, 2, 1]),
            val_loader=_batches(X, Y, [3, 2]),
            metrics=[MetricSpec("nll")],
            monitor=MonitorSpec(metric="nll"),
        ),
        callbacks=[TensorBoardCallback(), WandbCallback(wandb_run=wandb_run), Injected()],
    )
    epoch_records = [idp for idp in run.idps if idp.selection is not None]
    assert [(s, r) for s, r in seen] == [(idp.train_summary, idp.selection) for idp in epoch_records]
    (writer,) = _FakeSummaryWriter.instances
    for epoch, idp in enumerate(epoch_records):
        assert writer.scalars[("train_epoch/loss", epoch)] == pytest.approx(idp.train_summary.loss)
        assert writer.scalars[("train_epoch/nll", epoch)] == pytest.approx(idp.train_summary.metrics["nll"])
        assert writer.scalars[("monitor/val.nll", epoch)] == pytest.approx(idp.selection.value)
        assert writer.scalars[("monitor/improved", epoch)] == float(idp.selection.improved)
        assert wandb_run.logs[epoch]["monitor/val.nll"] == pytest.approx(idp.selection.value)

    class Bar:
        text = ""

        def set_postfix_str(self, text):
            Bar.text = text

    optimizer = torch.optim.SGD(_model().net.parameters(), lr=0.01)
    _model()._update_tqdm_postfix(Bar(), optimizer, None, NNEvaluationDataPoint(loss=1.0), epoch_records[-1].selection)
    assert Bar.text.startswith(f"val.nll={epoch_records[-1].selection.value:.4f}")

    reloaded = NNRun.load(run.id)
    series = reloaded._epoch_series()
    assert series["train_loss"] == pytest.approx([idp.train_summary.loss for idp in epoch_records])
    assert series["monitor"] == pytest.approx([idp.selection.value for idp in epoch_records])
    html = reloaded._repr_html_()
    assert "monitor: val.nll" in html and "val.nll (min, min_delta=0.0)" in html
    assert "monitor=val.nll (min, min_delta=0.0)" in str(reloaded)


def test_records_keep_monitor_identity_direction_config_and_status(tmp_path, monkeypatch):
    from nnx.nn.params.nn_checkpoint import _idp_from_nested_state

    run, rows = _decisions(tmp_path, monkeypatch, [0.8, None, math.nan, 0.5], min_delta=0.0)
    expected = [row[3] for row in rows]
    reloaded = [idp.selection for idp in NNRun.load(run.id).idps if idp.selection is not None]
    assert [r.monitor for r in reloaded] == [r.monitor for r in expected]
    assert [r.status for r in reloaded] == ["ok", "missing", "nonfinite", "ok"]
    assert [r.improved for r in reloaded] == [r.improved for r in expected]
    assert reloaded[1].value is None and math.isnan(reloaded[2].value)
    assert reloaded[3].value == pytest.approx(0.5)
    for idp in (i for i in run.idps if i.selection is not None):
        assert _idp_from_nested_state(idp.state()) == idp  # the checkpoint-metadata reader
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 3
    assert best.idp.selection == MonitorRecord(MonitorSpec(metric="nll", mode="min"), 0.5, "ok", True)


def test_named_monitor_runs_are_never_elected_into_runs_best(tmp_path, monkeypatch):
    from nnx.nn.params.nn_run import _elect_best, _read_best_pointer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X, Y = _data()
    loader = _batches(X, Y, [5])
    legacy = _model().train(params=NNTrainParams(n_epochs=1, train_loader=loader, val_loader=loader))
    assert _read_best_pointer(os.path.join("runs", "best")) == legacy.id
    named = _model().train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            val_loader=loader,
            monitor=MonitorSpec(metric="loss", min_delta=0.0),
            data_id="named",
        )
    )
    assert NNCheckpoint.load(run=named.id, type=Checkpoints.BEST) is not None
    assert _read_best_pointer(os.path.join("runs", "best")) == legacy.id  # pointer unchanged
    assert _elect_best("runs", None) == legacy.id
    assert _elect_best("runs", None, exclude=legacy.id) is None  # the named run is never a candidate

    only_named = tmp_path / "only_named"
    only_named.mkdir()
    monkeypatch.chdir(only_named)
    _model().train(params=NNTrainParams(n_epochs=1, train_loader=loader, val_loader=loader, monitor=MonitorSpec()))
    assert not os.path.lexists(os.path.join("runs", "best"))  # pointer absent


def test_early_stopping_named_monitor_state_round_trips():
    spec = MonitorSpec(metric="nll", min_delta=0.01)
    stopper = EarlyStopping(monitor=spec, patience=3)
    assert stopper._bind_metrics((MetricSpec("nll"),)) == MonitorSpec(metric="nll", mode="min", min_delta=0.01)
    stopper._best, stopper._wait = 0.7, 1
    state = stopper.component_state()
    assert state["monitor"] == {**spec.state(), "mode": "min"}
    fresh = EarlyStopping(monitor=spec, patience=3)
    fresh._bind_metrics((MetricSpec("nll"),))
    assert fresh.check_component_state(state, version=1) == []
    other = EarlyStopping(monitor=MonitorSpec(metric="nll", min_delta=0.02), patience=3)
    other._bind_metrics((MetricSpec("nll"),))
    assert other.check_component_state(state, version=1)
    assert fresh.selected_monitor is None or fresh.selected_monitor == "val.nll"


def _trainer_step(ctx: TrainerStepContext) -> NNEvaluationDataPoint:
    model = ctx.model
    model.net.train()
    (optimizer,) = ctx.optimizers.values()
    optimizer.zero_grad()
    (x,), y = model.net.unpack_batch(ctx.batch)
    loss = model.loss_fn(model.net(x), y)
    loss.backward()
    optimizer.step()
    return NNEvaluationDataPoint(loss=float(loss.detach()))


def test_trainer_monitors_validation_metrics_with_the_same_rule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X, Y = _data()
    spec = MonitorSpec(metric="nll", min_delta=0.0)
    builder = (
        NNTrainerParams.builder()
        .n_epochs(3)
        .train_loader(_batches(X, Y, [2, 2, 1]))
        .val_loader(_batches(X, Y, [5]))
        .optimizer("main", NNOptimParams.builder().sgd(max_lr=0.05).build())
        .scheduler("main", _plateau())
        .metrics(MetricSpec("nll"))
        .monitor(spec)
    )
    params = builder.build()
    assert NNTrainerParams.from_state(params.state()).state() == params.state()
    run = Trainer(_model()).train(params=params, trainer_step_fn=_trainer_step)
    records = [idp.selection for idp in run.idps if idp.selection is not None]
    assert len(records) == 3 and all(r.monitor.key == "val.nll" for r in records)
    values = [idp.val_edp.metrics["nll"] for idp in run.idps if idp.selection is not None]
    assert [r.value for r in records] == pytest.approx(values)
    tracker = MonitorTracker(spec.resolve([MetricSpec("nll")]))
    assert [r.improved for r in records] == [tracker.observe(v, epoch=i).improved for i, v in enumerate(values)]
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == tracker.best_epoch
    summaries = [idp.train_summary for idp in run.idps if idp.train_summary is not None]
    assert len(summaries) == 3 and all(s.loss is not None for s in summaries)

    with pytest.raises(ValueError, match="only the default training step records"):
        Trainer(_model()).train(
            params=builder.copy().monitor(MonitorSpec(metric="nll", split="train")).build(),
            trainer_step_fn=_trainer_step,
        )


def test_a_resumed_run_keeps_every_decision_of_the_uninterrupted_one(tmp_path, monkeypatch):
    """The run's MonitorTracker is checkpointed component state: after a
    stateful split, BEST, EarlyStopping and the plateau scheduler still
    agree with the continuous run."""
    from dataclasses import replace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    values = [0.8, 0.5, 0.6, 0.4]
    spec = MonitorSpec(metric="nll")
    X, Y = _data()

    def params(n_epochs, **extra):
        return NNTrainParams(
            n_epochs=n_epochs,
            train_loader=_batches(X, Y, [2, 2, 1]),
            val_loader=_batches(X, Y, [5]),
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            scheduler=_plateau(),
            metrics=[MetricSpec("nll")],
            monitor=spec,
            **extra,
        )

    def fit(model, p):
        stopper = EarlyStopping(monitor=spec, patience=10)
        spy = _Spy(stopper)
        run = model.train(params=p, eval_step_fn=_scripted_eval(values), callbacks=[stopper, spy])
        return run, spy.rows

    _, continuous = fit(_model(), params(4))
    first, head = fit(_model(), params(2, data_id="split"))
    resumed, tail = fit(_model(seed=5), replace(params(2), resume_from_run_id=first.id))
    assert resumed.resume_status is not None and "nnx.monitor" in resumed.resume_status.restored_components
    rows = head + tail
    assert [r[0] for r in rows] == [r[0] for r in continuous] == [True, True, False, True]
    assert [r[1] for r in rows] == [r[1] for r in continuous]
    assert [r[2] for r in rows] == pytest.approx([r[2] for r in continuous])
    best = NNCheckpoint.load(run=resumed.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.epoch_idx == 3  # epoch 2 (0.6) did not beat the source's 0.5


def test_a_plateau_saved_under_another_rule_fails_before_restoring(tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X, Y = _data()
    loader = _batches(X, Y, [5])
    legacy = _model().train(
        params=NNTrainParams(n_epochs=1, train_loader=loader, val_loader=loader, scheduler=_plateau())
    )
    model = _model(seed=3)
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    monitored = NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        val_loader=loader,
        scheduler=_plateau(),
        monitor=MonitorSpec(metric="loss"),
    )
    with pytest.raises(ValueError, match="resume plateau scheduler was saved with mode='min'"):
        model.train(params=replace(monitored, resume_from_run_id=legacy.id))
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, before[key]), key
    # weights-only warm starts are unaffected
    warm = model.train(params=replace(monitored, resume_from_run_id=legacy.id, resume_mode="weights_only"))
    assert warm.resume_status is not None and warm.resume_status.mode == "weights_only"


def test_a_validation_monitor_a_custom_evaluator_never_feeds_warns_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    X, Y = _data()
    with pytest.warns(RuntimeWarning, match="monitor 'val.nll' has no value") as caught:
        _model().train(
            params=NNTrainParams(
                n_epochs=3,
                train_loader=_batches(X, Y, [5]),
                val_loader=_batches(X, Y, [5]),
                metrics=[MetricSpec("nll")],
                monitor=MonitorSpec(metric="nll"),
            ),
            eval_step_fn=lambda ctx: NNEvaluationDataPoint(loss=1.0),
        )
    assert sum("monitor 'val.nll' has no value" in str(w.message) for w in caught) == 1


def test_a_reused_early_stopping_resolves_each_run_from_its_declared_spec():
    stopper = EarlyStopping(monitor=MonitorSpec(metric="score"))
    assert stopper._bind_metrics((MetricSpec("mae", name="score"),)).mode == "min"
    assert stopper._bind_metrics((MetricSpec("accuracy", name="score"),)).mode == "max"
