"""FEAT-002: task identity and task records survive every persistence path.

The task spec rides on ``NNModelParams`` (run YAML, checkpoints, Hub
config) as a versioned mapping that legacy models never emit; task
records — including an all-masked validation record — survive the
flattened ``idps.csv`` reader and the nested checkpoint-metadata reader
with their values, counts, status and kind.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNEvaluationDataPoint,
    NNIterationDataPoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    TaskSpec,
    set_seed,
)
from nnx.nn.params.nn_checkpoint import _idp_from_nested_state


def _regression_record(**overrides) -> NNEvaluationDataPoint:
    fields = dict(kind="regression", count=4, status="ok", loss=2.5, metrics={"mse": 2.5, "mae": 1.5})
    fields.update(overrides)
    return NNEvaluationDataPoint(**fields)  # type: ignore[arg-type]


def _empty_record() -> NNEvaluationDataPoint:
    return NNEvaluationDataPoint(kind="regression", count=0, status="empty")


def _net() -> NNParams:
    return NNParams(input_dim=3, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU)


def _run(idps, task=TaskSpec.regression(2)) -> NNRun:
    return NNRun(
        net=_net(),
        train=NNTrainParams(n_epochs=1),
        model=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=task),
        idps=idps,
    )


# --------------------------------------------------------- task identity


def test_model_params_carry_a_versioned_task_and_legacy_state_is_unchanged():
    legacy = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    assert legacy.state() == {"net": "feed_fwd", "loss": "cross_entropy", "device": "cpu"}
    assert NNModelParams.from_state(legacy.state()) == legacy and legacy.task is None

    task = NNModelParams(
        net=Nets.FEED_FWD,
        loss=Losses.CROSS_ENTROPY,
        task=TaskSpec.categorical(3, ignore_index=-100, labels=["a", "b", "c"]),
    )
    state = task.state()
    assert state["task"] == {
        "version": 1,
        "kind": "categorical",
        "num_outputs": 3,
        "ignore_index": -100,
        "labels": ["a", "b", "c"],
    }
    assert NNModelParams.from_state(state) == task


def test_the_task_participates_in_run_identity_only_when_set():
    legacy = _run([], task=None)
    assert "task" not in legacy.state()["model"]
    assert _run([], task=None).id == legacy.id
    assert _run([]).id != legacy.id
    assert _run([], task=TaskSpec.regression(2, labels=["a", "b"])).id != _run([]).id


def test_legacy_evaluation_records_keep_their_serialization():
    edp = NNEvaluationDataPoint(f1=0.5, recall=0.5, accuracy=0.5, precision=0.5, loss=1.0, error=0.5)
    assert list(edp.state()) == ["f1", "recall", "accuracy", "precision", "loss", "error"]
    assert NNEvaluationDataPoint.from_state(edp.state()) == edp


# --------------------------------------------------------- task records


def test_task_record_validation():
    with pytest.raises(ValueError, match="need a kind"):
        NNEvaluationDataPoint(count=3, status="ok")
    with pytest.raises(ValueError, match="status must be 'empty'"):
        NNEvaluationDataPoint(kind="regression", count=0, status="ok")
    with pytest.raises(ValueError, match="non-negative integer"):
        NNEvaluationDataPoint(kind="regression", count=-1, status="ok")
    assert NNEvaluationDataPoint(kind="regression", count=3.0, status="ok").count == 3  # CSV floats


@pytest.mark.parametrize("record", [_regression_record(), _empty_record()], ids=["ok", "empty"])
def test_task_records_round_trip_through_the_nested_checkpoint_reader(record):
    idp = NNIterationDataPoint(
        lr=0.1, iter_idx=0, epoch_idx=0, batch_idx=0, train_edp=_regression_record(), val_edp=record
    )
    assert _idp_from_nested_state(idp.state()) == idp


def test_task_records_round_trip_through_run_csv_including_an_all_masked_validation_record(tmp_path):
    idps = [
        NNIterationDataPoint(lr=0.1, iter_idx=0, epoch_idx=0, batch_idx=0, train_edp=_regression_record()),
        NNIterationDataPoint(
            lr=0.1, iter_idx=1, epoch_idx=0, batch_idx=1, train_edp=_regression_record(count=3), val_edp=_empty_record()
        ),
        NNIterationDataPoint(
            lr=0.1,
            iter_idx=2,
            epoch_idx=1,
            batch_idx=0,
            train_edp=_regression_record(),
            val_edp=_regression_record(count=7, loss=0.25, metrics={"mse": 0.25, "mae": 0.5}),
        ),
    ]
    run = _run(idps)
    run.save(root=str(tmp_path))
    loaded = NNRun.load(run.id, root=str(tmp_path))
    assert loaded.model.task == TaskSpec.regression(2)
    assert loaded.idps == idps
    empty = loaded.idps[1].val_edp
    assert empty is not None and (empty.kind, empty.count, empty.status, empty.loss) == ("regression", 0, "empty", None)
    assert isinstance(loaded.idps[2].val_edp.count, int)  # type: ignore[union-attr]


def test_a_trained_task_model_round_trips_through_its_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    set_seed(0)
    model = NNModel(
        net_params=_net(),
        params=NNModelParams(net=Nets.FEED_FWD, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(2)),
    )
    x, y = torch.randn(8, 3), torch.randn(8, 2)
    y[0, 0] = math.nan
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    params = (
        NNTrainParams(n_epochs=1, optim=NNOptimParams.builder().sgd(max_lr=0.01).build())
        .with_train_loader(loader)
        .with_val_loader(loader)
    )
    run = model.train(params=params)
    checkpoint = NNCheckpoint.load(run=run.id, type="last")  # type: ignore[arg-type]
    assert checkpoint is not None
    assert checkpoint.model_params.task == TaskSpec.regression(2)
    assert checkpoint.idp == run.idps[-1]
    restored = NNModel.from_checkpoint(checkpoint)
    assert restored.task_adapter is not None and restored.task_adapter.spec == TaskSpec.regression(2)
