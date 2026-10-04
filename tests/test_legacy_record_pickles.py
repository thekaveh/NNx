"""Training records pickled by NNx 0.2.3 restore whole.

Every pickle checkpoint carries an ``NNIterationDataPoint`` and its
``NNEvaluationDataPoint`` records. Both are frozen ``slots`` dataclasses,
and the dataclass-generated pickling stores their fields by position: a
0.2.3 pickle holds the 7 / 6 fields that release had. Restoring one with
the generated ``__setstate__`` left every field added since (``kind``,
``count``, ``status``, ``metrics``; ``train_summary``, ``selection``,
``update_count``) unset, so ``repr``, ``==``, ``deepcopy``, re-pickling,
safetensors export and bundle export raised ``AttributeError`` — and the
``runs/best`` election, which reads ``idp.selection``, silently skipped
every 0.2.3 run, so re-saving an upgraded best run removed the pointer.

The 0.2.3 layout is reproduced by truncating the current state to the
fields 0.2.3 had, exactly what its generated ``__getstate__`` wrote.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
import pickle
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import nnx.nn.params.nn_run as nn_run
from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNConvParams,
    NNModel,
    NNModelParams,
    NNMoEParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    NNTransformerParams,
    Optims,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint, NNCheckpointTransform
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint

# Field counts of the 0.2.3 classes: f1, recall, accuracy, precision, loss,
# error, extra / lr, iter_idx, epoch_idx, batch_idx, train_edp, val_edp.
EDP_FIELDS_0_2_3 = 7
IDP_FIELDS_0_2_3 = 6
_PARAMS_0_2_3 = (
    "dropout_prob",
    "n_heads",
    "activation",
    "activations",
    "dropout_probs",
    "input_dim",
    "output_dim",
    "hidden_dims",
    "_dims",
)


@contextlib.contextmanager
def _pickled_as_0_2_3():
    edp_state = NNEvaluationDataPoint.__getstate__
    idp_state = NNIterationDataPoint.__getstate__
    with (
        mock.patch.object(NNEvaluationDataPoint, "__getstate__", lambda self: edp_state(self)[:EDP_FIELDS_0_2_3]),
        mock.patch.object(NNIterationDataPoint, "__getstate__", lambda self: idp_state(self)[:IDP_FIELDS_0_2_3]),
    ):
        yield


def _record(**extra: float) -> NNEvaluationDataPoint:
    return NNEvaluationDataPoint(f1=0.5, recall=0.5, accuracy=0.75, precision=0.5, loss=0.3, error=0.25, extra=extra)


def _idp() -> NNIterationDataPoint:
    return NNIterationDataPoint(
        lr=1e-3, iter_idx=3, epoch_idx=1, batch_idx=1, train_edp=_record(), val_edp=_record(top3=0.9)
    )


def test_a_0_2_3_record_restores_every_field_added_since():
    idp = _idp()
    with _pickled_as_0_2_3():
        blob = pickle.dumps(idp)
    restored = pickle.loads(blob)

    assert restored == idp
    assert (restored.train_summary, restored.selection, restored.update_count) == (None, None, None)
    for record in (restored.train_edp, restored.val_edp):
        assert (record.kind, record.count, record.status) == (None, None, None)
        assert dict(record.metrics) == {}
    assert repr(restored).startswith("NNIterationDataPoint(lr=0.001, iter_idx=3")  # raised on a half-built record
    assert copy.deepcopy(restored) == idp
    assert pickle.loads(pickle.dumps(restored)) == idp
    assert restored.state() == idp.state()
    assert hash(restored.val_edp) == hash(idp.val_edp)


def test_a_0_2_3_record_gets_its_own_empty_metrics_mapping():
    with _pickled_as_0_2_3():
        blob = pickle.dumps([_record(), _record()])
    first, second = pickle.loads(blob)
    assert first.metrics == second.metrics == {}
    assert first.metrics is not second.metrics  # a default_factory field is built per record


def test_a_current_record_round_trips_unchanged():
    idp = NNIterationDataPoint(
        lr=1e-3,
        iter_idx=3,
        epoch_idx=1,
        batch_idx=1,
        train_edp=NNEvaluationDataPoint(loss=0.5, kind="regression", count=4, status="ok", metrics={"mse": 0.5}),
        update_count=2,
    )
    restored = pickle.loads(pickle.dumps(idp))
    assert restored == idp and restored.update_count == 2 and restored.train_edp.metrics == {"mse": 0.5}


def test_a_record_lacking_a_required_field_is_refused():
    restored = NNIterationDataPoint.__new__(NNIterationDataPoint)
    with pytest.raises(TypeError, match="cannot restore NNIterationDataPoint: the pickle lacks 'train_edp'"):
        restored.__setstate__([1e-3, 3, 1, 1])


# The fields each class pickled inside a checkpoint had in NNx 0.2.3, in
# order. Pickles restore by position, so a field may only be appended.
_FIELDS_0_2_3 = {
    NNEvaluationDataPoint: ("f1", "recall", "accuracy", "precision", "loss", "error", "extra"),
    NNIterationDataPoint: ("lr", "iter_idx", "epoch_idx", "batch_idx", "train_edp", "val_edp"),
    NNModelParams: ("net", "device", "loss", "mixed_precision"),
    NNCheckpoint: (
        "net_params",
        "net_state",
        "model_params",
        "idp",
        "transforms",
        "training_state_id",
        "training_state_present",
    ),
    NNCheckpointTransform: ("name", "version", "options"),
    NNParams: _PARAMS_0_2_3,
    NNConvParams: (*_PARAMS_0_2_3, "conv_channels", "in_channels", "kernel_size", "stride", "padding", "pool_size"),
    NNMoEParams: (*_PARAMS_0_2_3, "num_experts", "top_k"),
    NNTransformerParams: (
        *_PARAMS_0_2_3,
        "vocab_size",
        "n_layers",
        "d_model",
        "max_seq_len",
        "ffn_mult",
        "rope_base",
        "tie_embeddings",
        "attn_dropout",
        "resid_dropout",
    ),
}


@pytest.mark.parametrize("cls", list(_FIELDS_0_2_3), ids=lambda cls: cls.__name__)
def test_checkpoint_classes_only_ever_append_fields(cls):
    names = tuple(spec.name for spec in dataclasses.fields(cls))
    assert names[: len(_FIELDS_0_2_3[cls])] == _FIELDS_0_2_3[cls]


def _train_tiny_run(tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=4)
    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    params = NNTrainParams(
        n_epochs=2,
        train_loader=loader,
        val_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
    )
    with _pickled_as_0_2_3():
        return model.train(params=params).id


def _best_target(tmp_path) -> str | None:
    best = tmp_path / "runs" / "best"
    if os.path.islink(best):
        return os.path.basename(os.readlink(best))
    return nn_run._read_best_pointer(str(best)) if os.path.lexists(best) else None


def test_a_run_written_by_0_2_3_stays_eligible_for_runs_best(tmp_path, monkeypatch):
    run_id = _train_tiny_run(tmp_path, monkeypatch)
    runs_root = str(tmp_path / "runs")

    assert nn_run._committed_best_checkpoint(runs_root, run_id, None) is not None
    assert nn_run._elect_best(runs_root, None) == run_id

    NNRun.load(run_id).save()  # re-saving the upgraded best run keeps the pointer
    assert _best_target(tmp_path) is not None
    assert os.path.basename(os.path.normpath(_best_target(tmp_path) or "")) == run_id


def test_a_checkpoint_written_by_0_2_3_exports_again(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    run_id = _train_tiny_run(tmp_path, monkeypatch)

    checkpoint = NNCheckpoint.load(run=run_id, type=Checkpoints.LAST)
    assert checkpoint is not None
    checkpoint.to_file(str(tmp_path / "last.pt"))
    checkpoint.to_file(str(tmp_path / "last.safetensors"), format="safetensors")
    for path in ("last.pt", "last.safetensors"):
        reloaded = NNCheckpoint.from_file(str(tmp_path / path))
        assert reloaded is not None and reloaded.idp == checkpoint.idp

    from nnx.bundles import export_bundle, validate_bundle

    export_bundle(run_id, tmp_path / "bundle")
    assert validate_bundle(tmp_path / "bundle").verified
