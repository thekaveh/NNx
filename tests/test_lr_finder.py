"""Tests for ``nnx.lr_finder`` — fastai-style exponential LR sweep."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import lr_finder
from nnx.lr_finder import LRFinderResult


def _tiny_model_and_loader():
    """Tiny 3-class FFN with synthetic data, suitable for a 50-iter sweep."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    X = torch.randn(64, 4)
    y = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    return model, loader


def test_lr_finder_returns_LRFinderResult():
    """Return type contract."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert isinstance(result, LRFinderResult)


def test_lr_finder_result_has_required_fields():
    """LRFinderResult exposes lrs, losses, suggested_lr, figure."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert isinstance(result.lrs, list)
    assert isinstance(result.losses, list)
    assert isinstance(result.suggested_lr, float)
    assert isinstance(result.figure, go.Figure)


def test_lr_finder_lrs_and_losses_same_length():
    """One loss recorded per LR; lengths must match."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert len(result.lrs) == len(result.losses)
    assert len(result.lrs) >= 1  # may early-exit on divergence


def test_lr_finder_suggested_lr_in_range():
    """suggested_lr falls within the swept range."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert 1e-6 <= result.suggested_lr <= 1.0


def test_lr_finder_restores_model_weights():
    """Model state is non-destructively restored after the sweep."""
    model, loader = _tiny_model_and_loader()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}

    lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )

    for k, v in model.state_dict().items():
        assert torch.equal(v, initial[k]), f"weight {k} was not restored after lr_finder"


def test_lr_finder_rejects_invalid_num_iter():
    """num_iter < 2 raises ValueError."""
    model, loader = _tiny_model_and_loader()
    with pytest.raises(ValueError, match="num_iter"):
        lr_finder(
            model,
            loader,
            loss_fn=nn.functional.cross_entropy,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=1,
        )


def test_lr_finder_rejects_inverted_lr_range():
    """end_lr <= start_lr raises ValueError."""
    model, loader = _tiny_model_and_loader()
    with pytest.raises(ValueError, match="start_lr"):
        lr_finder(
            model,
            loader,
            loss_fn=nn.functional.cross_entropy,
            start_lr=1.0,
            end_lr=1e-6,
            num_iter=30,
        )


def test_lr_finder_rejects_nonpositive_start_lr():
    """start_lr <= 0 raises ValueError."""
    model, loader = _tiny_model_and_loader()
    with pytest.raises(ValueError, match="start_lr"):
        lr_finder(
            model,
            loader,
            loss_fn=nn.functional.cross_entropy,
            start_lr=0.0,
            end_lr=1.0,
            num_iter=30,
        )


def test_lr_finder_figure_has_log_x_axis():
    """The plotted x-axis is log-scaled (the standard fastai LR-finder view)."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert result.figure.layout.xaxis.type == "log"
