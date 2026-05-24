"""Pass-2 catalog: N-series regression tests.

Covers correctness gaps surfaced in the pass-2 audit:
- N1: NNOptimParams.is_valid() must return a bool (not None) for any input.
- N7: NNModel.evaluate() aggregates predictions across all batches so an
  uneven final batch doesn't over-weight metrics.
- N8: evaluate() raises rather than silently returning NaN on empty loaders.
"""
from __future__ import annotations

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


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8],
            dropout_prob=0.0, activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        ),
    )


def test_n1_optim_is_valid_always_returns_bool():
    """is_valid() previously returned None for unknown enum variants,
    which let invalid configs slip through `not params.optim.is_valid()`."""
    p_sgd = NNOptimParams(name=Optims.SGD, max_lr=1e-2, momentum=0.9, weight_decay=0.0)
    assert p_sgd.is_valid() is True

    # SGD with a tuple momentum is invalid (Adam-shaped).
    p_bad = NNOptimParams(name=Optims.SGD, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)
    assert p_bad.is_valid() is False
    assert isinstance(p_bad.is_valid(), bool)


def test_n7_evaluate_aggregates_across_batches():
    """Last-batch over-weighting bug: with 10 samples split into batches of
    [8, 2], per-batch averaging weighted the 2-sample batch 50% in the mean.
    Aggregating before computing should weight by sample count."""
    torch.manual_seed(0)
    model = _model()

    X = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    # batch_size=8 → batches of [8, 2]
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)

    edp = model.evaluate(loader=loader)
    # accuracy is sample-weighted; should match a single-batch computation.
    big_loader = DataLoader(TensorDataset(X, y), batch_size=10, shuffle=False)
    edp_full = model.evaluate(loader=big_loader)
    assert abs(edp.accuracy - edp_full.accuracy) < 1e-9
    assert abs(edp.error - edp_full.error) < 1e-9


def test_n8_evaluate_raises_on_empty_loader():
    """Empty loaders previously yielded NaN metrics from np.mean over [].
    Should raise instead."""
    model = _model()
    X = torch.empty(0, 4)
    y = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=8)
    with pytest.raises(ValueError, match="zero samples"):
        model.evaluate(loader=loader)


def test_n4_train_works_on_iterable_dataset(tmp_path, monkeypatch):
    """train() must tolerate DataLoaders where len() raises (IterableDataset)."""
    monkeypatch.chdir(tmp_path)

    from torch.utils.data import IterableDataset

    class _IterableSet(IterableDataset):
        def __init__(self, n: int):
            super().__init__()
            self.n = n

        def __iter__(self):
            for _ in range(self.n):
                yield torch.randn(4), torch.randint(0, 2, (1,)).squeeze()

    loader = DataLoader(_IterableSet(n=8), batch_size=4)
    # Sanity: len() on this loader raises.
    with pytest.raises(TypeError):
        len(loader)

    from nnx.nn.params.nn_optim_params import NNOptimParams
    from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
    from nnx.nn.params.nn_train_params import NNTrainParams

    model = _model()
    run = model.train(params=NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    ))
    # Successfully completed at least the iterable's worth of batches.
    assert len(run.idps) >= 1


def test_n6_best_symlink_falls_back_to_pointer_file_when_symlink_fails(tmp_path, monkeypatch):
    """On platforms where os.symlink raises (e.g., Windows without dev mode),
    NNRun.save still records the best run via a POINTER.txt file."""
    monkeypatch.chdir(tmp_path)

    from nnx.nn.params import nn_run as nn_run_mod

    def _raise(*a, **kw):
        raise OSError("symlink not supported (simulated Windows)")
    monkeypatch.setattr(nn_run_mod.os, "symlink", _raise)

    # Drive a tiny run end-to-end so NNRun.save() is exercised through
    # the symlink path.
    from torch.utils.data import DataLoader, TensorDataset

    from nnx.nn.params.nn_optim_params import NNOptimParams
    from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
    from nnx.nn.params.nn_train_params import NNTrainParams

    X = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    model = _model()
    run = model.train(params=NNTrainParams(
        n_epochs=1,
        train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
    ))

    pointer = tmp_path / "runs" / "best" / "POINTER.txt"
    assert pointer.exists()
    assert run.id in pointer.read_text()
