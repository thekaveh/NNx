"""Shared pytest fixtures for the nnx test suite.

Eliminates the per-test boilerplate of constructing the same toy
classification dataset / NNModel over and over. Tests that need
variants can still build their own (the fixtures are starting points,
not mandates).
"""
from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.nn_model import NNModel
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams


@pytest.fixture(autouse=True, scope="session")
def _disable_tqdm_in_tests():
    """Set NNX_TQDM_DISABLE=1 for the entire test session so progress
    bars don't pollute pytest output. Autouse so individual tests don't
    have to remember."""
    os.environ["NNX_TQDM_DISABLE"] = "1"
    yield
    os.environ.pop("NNX_TQDM_DISABLE", None)


@pytest.fixture
def tiny_classification_data():
    """16 samples, 4 features, 2 classes — minimal for any test that
    needs *some* data without caring about specifics. Returns
    (X_train, y_train, X_val, y_val)."""
    torch.manual_seed(0)
    X_train = torch.randn(16, 4)
    y_train = torch.randint(0, 2, (16,))
    X_val = torch.randn(8, 4)
    y_val = torch.randint(0, 2, (8,))
    return X_train, y_train, X_val, y_val


@pytest.fixture
def tiny_classification_loaders(tiny_classification_data):
    """DataLoaders wrapping ``tiny_classification_data`` at batch_size=8.
    Returns (train_loader, val_loader)."""
    X_train, y_train, X_val, y_val = tiny_classification_data
    return (
        DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True),
        DataLoader(TensorDataset(X_val, y_val), batch_size=8),
    )


@pytest.fixture
def tiny_net_params():
    return NNParams(
        input_dim=4, output_dim=2, hidden_dims=[8],
        dropout_prob=0.0, activation=Activations.RELU,
    )


@pytest.fixture
def tiny_model_params():
    return NNModelParams(
        net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
    )


@pytest.fixture
def tiny_model(tiny_net_params, tiny_model_params):
    """Fresh NNModel matching the tiny_classification_* fixtures."""
    return NNModel(net_params=tiny_net_params, params=tiny_model_params)


@pytest.fixture
def tiny_train_params(tiny_classification_loaders):
    """Default NNTrainParams wiring train+val from tiny_classification_loaders."""
    train_loader, val_loader = tiny_classification_loaders
    return NNTrainParams(
        n_epochs=1,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    )


@pytest.fixture
def tmp_runs_root(tmp_path, monkeypatch):
    """Chdir to a fresh tmp dir so runs/ ends up there instead of in the
    user's working tree. Returns the tmp_path so tests can build paths
    relative to it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
