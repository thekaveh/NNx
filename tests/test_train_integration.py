"""End-to-end integration test for NNModel.train().

Exercises the full path: build model → train for a few epochs on a tiny
in-memory dataset → assert checkpoints + run files land on disk → reload
the run and reconstruct a model from the BEST checkpoint.

Uses a small random dataset so it stays fast (<10s on CPU) and avoids
network downloads. Uses tmp_path + chdir so the runs/ directory lands in
a pytest temp dir and doesn't pollute the repo."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx.nn.enum.activations import Activations
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.nn_model import NNModel
from nnx.nn.params.nn_checkpoint import NNCheckpoint
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_run import NNRun
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams


def _make_tiny_loaders(n_train: int = 32, n_val: int = 16, input_dim: int = 8, n_classes: int = 3):
    """Random classification data — just enough to drive a forward/backward."""
    torch.manual_seed(0)
    np.random.seed(0)

    X_train = torch.randn(n_train, input_dim)
    y_train = torch.randint(0, n_classes, (n_train,))
    X_val = torch.randn(n_val, input_dim)
    y_val = torch.randint(0, n_classes, (n_val,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8, shuffle=False)
    return train_loader, val_loader


def _make_params(input_dim: int = 8, output_dim: int = 3):
    net_params = NNParams(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[16],
        dropout_prob=0.0,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    return net_params, model_params


def _train_params(train_loader, val_loader, n_epochs: int = 2):
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3,
        ),
    )


def test_train_end_to_end_produces_run_and_checkpoints(tmp_path, monkeypatch):
    """train() saves a run with one idp per batch and at least BEST+LAST
    checkpoints. Reloading the run reconstructs every idp."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, val_loader = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader, n_epochs=2)

    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=train_params)

    # Every batch produces an idp; with batch_size=8 and 32 train samples
    # that's 4 batches × 2 epochs = 8 idps.
    assert len(run.idps) == 8
    # The final idps in each epoch should have val_edp populated.
    last_in_first_epoch = run.idps[3]
    last_in_second_epoch = run.idps[7]
    assert last_in_first_epoch.val_edp is not None
    assert last_in_second_epoch.val_edp is not None

    # On-disk artifacts.
    run_dir = tmp_path / "runs" / run.id
    assert (run_dir / "run.yaml").exists()
    assert (run_dir / "idps.csv").exists()
    assert (run_dir / "checkpoints" / "first.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    # runs/best symlink points at this run (no prior runs in tmp_path).
    assert os.path.islink(tmp_path / "runs" / "best")


def test_run_save_load_round_trip(tmp_path, monkeypatch):
    """NNRun.load(run.id) returns an NNRun whose idps/state match the saved one."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, val_loader = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader, n_epochs=1)

    model = NNModel(net_params=net_params, params=model_params)
    original = model.train(params=train_params)

    reloaded = NNRun.load(id=original.id)

    assert reloaded.id == original.id
    assert reloaded.net == original.net
    assert reloaded.model == original.model
    # train_loader / val_loader are runtime-only (repr=False, not serialized);
    # compare the serializable parts.
    assert reloaded.train.n_epochs == original.train.n_epochs
    assert reloaded.train.optim == original.train.optim
    assert reloaded.train.scheduler == original.train.scheduler
    assert len(reloaded.idps) == len(original.idps)
    # Per-iteration metrics survive CSV → DataFrame → dict round-trip.
    for ridp, oidp in zip(reloaded.idps, original.idps, strict=True):
        assert ridp.iter_idx == oidp.iter_idx
        assert ridp.epoch_idx == oidp.epoch_idx
        assert ridp.batch_idx == oidp.batch_idx
        # Floats may differ by float64 → string → float64 noise; tolerate epsilon.
        assert abs(ridp.train_edp.loss - oidp.train_edp.loss) < 1e-9


def test_checkpoint_reconstruct_predicts(tmp_path, monkeypatch):
    """The BEST checkpoint can be loaded and used to build a working NNModel
    that produces predictions of the right shape."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, val_loader = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader, n_epochs=2)

    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=train_params)

    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert ckpt is not None

    reloaded = NNModel.from_checkpoint(checkpoint=ckpt)
    X = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    log, hat = reloaded.predict(X=X)
    assert log.shape == (4, 3)
    assert hat.shape == (4,)


def test_train_skips_val_loop_when_no_val_loader(tmp_path, monkeypatch):
    """Without a val_loader, idps[*].val_edp is None and the run still saves
    cleanly (regression for the no-val NNRun.save crash)."""
    monkeypatch.chdir(tmp_path)

    net_params, model_params = _make_params()
    train_loader, _ = _make_tiny_loaders()
    train_params = _train_params(train_loader, val_loader=None, n_epochs=1)

    model = NNModel(net_params=net_params, params=model_params)
    run = model.train(params=train_params)

    assert all(idp.val_edp is None for idp in run.idps)
    assert (tmp_path / "runs" / run.id / "run.yaml").exists()


def test_train_rejects_none_or_invalid_params():
    """The first guard in NNModel.train() — params=None or an invalid
    optim config — must raise ValueError loudly rather than letting the
    loop start and produce a garbage run. Pre-audit, this branch had
    zero test coverage."""
    import pytest

    from nnx.nn.params.nn_train_params import NNTrainParams

    net_params, model_params = _make_params()
    model = NNModel(net_params=net_params, params=model_params)

    # 1. None params — surfaces a distinct error from the invalid-optim case.
    with pytest.raises(ValueError, match="^train params must be non-None$"):
        model.train(params=None)

    # 2. invalid optim: Adam with a scalar momentum (Adam wants a tuple).
    train_loader, _ = _make_tiny_loaders()
    bad_optim = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3, momentum=0.9, weight_decay=0.0,
    )
    bad_params = NNTrainParams(
        n_epochs=1,
        train_loader=train_loader,
        optim=bad_optim,
        scheduler=NNSchedulerParams(
            min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3,
        ),
    )
    with pytest.raises(ValueError, match=r"^train params has an invalid optim config:"):
        model.train(params=bad_params)
