"""Pass-2 catalog: O- and U-series tests (API + UX additions).

- O1: predict() accepts numpy / Tensor / DataLoader.
- O2: predict() returns a NamedTuple that still unpacks as (logits, classes).
- O3: NNTrainParams.extra_metrics injects custom metric callables that
  populate NNEvaluationDataPoint.extra; the field round-trips through
  state() / from_state() and is omitted when empty.
- O8: Devices.torch_device() and Devices.get_torch_device() are convenience
  helpers that return torch.device directly.
- O10: Utils.print_tree / print_table accept a file= keyword.
- U2: NNTrainParams.save_phase_checkpoints=False skips FIRST/Q1/Q2/Q3.
- U4: nnx.__version__ exists.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import nnx
from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.nn_model import NNModel, PredictResult
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams
from nnx.utils import Utils


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=3, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )


# --- O1 / O2: predict input flexibility + structured result --------------

def test_o2_predict_result_unpacks_as_tuple():
    """Back-compat: `log, hat = model.predict(X)` keeps working."""
    model = _model()
    X = np.random.RandomState(0).randn(4, 4).astype(np.float32)
    log, hat = model.predict(X)
    assert log.shape == (4, 3)
    assert hat.shape == (4,)


def test_o2_predict_result_has_named_fields():
    model = _model()
    X = np.random.RandomState(0).randn(4, 4).astype(np.float32)
    result = model.predict(X)
    assert isinstance(result, PredictResult)
    assert result.logits.shape == (4, 3)
    assert result.classes.shape == (4,)
    # Field access matches positional unpack.
    log_p, hat_p = result
    assert (log_p == result.logits).all()
    assert (hat_p == result.classes).all()


def test_o1_predict_accepts_tensor_input():
    model = _model()
    X = torch.randn(4, 4)
    result = model.predict(X)
    assert result.logits.shape == (4, 3)


def test_o1_predict_accepts_dataloader():
    model = _model()
    X = torch.randn(10, 4)
    y = torch.randint(0, 3, (10,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
    result = model.predict(loader)
    # DataLoader path should produce a prediction per sample, concatenated.
    assert result.logits.shape == (10, 3)
    assert result.classes.shape == (10,)


# --- O3: custom metrics injection ---------------------------------------

def test_o3_evaluation_data_point_extra_populated_via_of():
    Y = np.array([0, 1, 2, 0, 1, 2])
    Y_hat = np.array([0, 1, 1, 0, 1, 2])
    edp = NNEvaluationDataPoint.of(
        Y, Y_hat,
        extra_metrics={"hamming_loss": lambda y, y_hat: float((y != y_hat).mean())},
    )
    assert "hamming_loss" in edp.extra
    assert edp.extra["hamming_loss"] == 1 / 6  # one mismatch out of six


def test_o3_extra_state_round_trip():
    edp = NNEvaluationDataPoint(
        f1=0.8, recall=0.8, accuracy=0.8, precision=0.8,
        loss=0.1, error=0.2, extra={"my_metric": 0.99},
    )
    rt = NNEvaluationDataPoint.from_state(edp.state())
    assert rt.extra == {"my_metric": 0.99}


def test_o3_extra_omitted_from_state_when_empty():
    """Back-compat: an EDP with no extras must hash identically to pre-extra
    EDPs — so state() omits the key when extra is empty."""
    edp = NNEvaluationDataPoint(f1=0.8, recall=0.8, accuracy=0.8, precision=0.8)
    assert "extra" not in edp.state()


def test_o3_extra_metrics_threaded_through_train(tmp_path, monkeypatch):
    """NNTrainParams.extra_metrics produces edps with the extra dict filled."""
    monkeypatch.chdir(tmp_path)

    model = _model()
    X = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    counter = {"calls": 0}
    def _mean_pred(y, y_hat):
        counter["calls"] += 1
        return float(y_hat.mean())

    run = model.train(params=NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        extra_metrics={"mean_pred": _mean_pred},
    ))

    # extra_metrics ran once per train batch (2 batches).
    assert counter["calls"] >= 2
    for idp in run.idps:
        assert "mean_pred" in idp.train_edp.extra


# --- O8: Devices.torch_device() ----------------------------------------

def test_o8_devices_torch_device_returns_torch_device():
    d = Devices.CPU.torch_device()
    assert isinstance(d, torch.device)
    assert d.type == "cpu"


def test_o8_devices_get_torch_device_one_shot():
    d = Devices.get_torch_device()
    assert isinstance(d, torch.device)
    # The detected device must match one of the supported types.
    assert d.type in {"cpu", "mps", "cuda"}


# --- O10: Utils file= param ---------------------------------------------

def test_o10_print_tree_respects_file_param():
    buf = io.StringIO()
    Utils.print_tree({"a": 1, "b": {"c": 2}}, file=buf)
    out = buf.getvalue()
    assert "[+] a" in out
    assert "[-] b" in out
    assert "[+] c" in out


def test_o10_print_table_respects_file_param():
    buf = io.StringIO()
    Utils.print_table({"k1": "v1", "k2": "v2"}, file=buf)
    out = buf.getvalue()
    assert "k1" in out
    assert "v1" in out


# --- U2: configurable checkpoint cadence --------------------------------

def test_u2_save_phase_checkpoints_false_skips_phase_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _model()
    X = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)

    run = model.train(params=NNTrainParams(
        n_epochs=4,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        save_phase_checkpoints=False,
    ))

    ckpt_dir = tmp_path / "runs" / run.id / "checkpoints"
    # LAST and BEST always present.
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    # FIRST and Q* skipped.
    assert not (ckpt_dir / "first.pt").exists()
    assert not (ckpt_dir / "q1.pt").exists()


def test_u2_default_still_saves_first(tmp_path, monkeypatch):
    """Default save_phase_checkpoints=True still produces FIRST."""
    monkeypatch.chdir(tmp_path)
    model = _model()
    X = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)

    run = model.train(params=NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    ))
    assert (tmp_path / "runs" / run.id / "checkpoints" / "first.pt").exists()


def test_u2_save_phase_checkpoints_state_back_compat():
    """Default True must NOT appear in state() (preserves run.id)."""
    p = NNTrainParams(
        n_epochs=5,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
    )
    assert "save_phase_checkpoints" not in p.state()


def test_u2_save_phase_checkpoints_false_appears_in_state():
    p = NNTrainParams(
        n_epochs=5, save_phase_checkpoints=False,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
    )
    assert p.state()["save_phase_checkpoints"] is False
    rt = NNTrainParams.from_state(p.state())
    assert rt.save_phase_checkpoints is False


# --- U4: __version__ exposed -------------------------------------------

def test_u4_version_string_present():
    assert isinstance(nnx.__version__, str)
    assert len(nnx.__version__) > 0
