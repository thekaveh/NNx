"""Tests for ``NNRun._repr_html_`` — Jupyter rich-display."""

from __future__ import annotations

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


def _tiny_run(tmp_path):
    """Train a tiny FFN for 2 epochs; return the resulting NNRun."""
    X = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8)

    model = NNModel(
        net_params=NNParams(
            input_dim=4,
            output_dim=3,
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
    return model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
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
    )


def test_repr_html_returns_html_string(tmp_path, monkeypatch):
    """_repr_html_ returns a non-empty str containing well-formed HTML markers."""
    monkeypatch.chdir(tmp_path)
    run = _tiny_run(tmp_path)
    html = run._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0
    # Outer container.
    assert html.startswith("<div")


def test_repr_html_contains_config_table(tmp_path, monkeypatch):
    """The HTML includes a <table> showing config values."""
    monkeypatch.chdir(tmp_path)
    run = _tiny_run(tmp_path)
    html = run._repr_html_()
    assert "<table" in html
    # The run.id is the canonical primary identifier; it must appear.
    assert run.id in html


def test_repr_html_contains_metric_chart(tmp_path, monkeypatch):
    """The HTML includes a Plotly chart container."""
    monkeypatch.chdir(tmp_path)
    run = _tiny_run(tmp_path)
    html = run._repr_html_()
    # Plotly's to_html emits a div with class="plotly-graph-div" or
    # an embedded <script>. Either marker is sufficient evidence.
    assert "plotly" in html.lower() or "<script" in html


def test_repr_html_handles_run_without_idps(tmp_path):
    """When idps is None, _repr_html_ still returns a valid string — just
    the config table, no metric chart."""
    from nnx.nn.params.nn_optim_params import NNOptimParams as _NNOptimParams
    from nnx.nn.params.nn_run import NNRun
    from nnx.nn.params.nn_scheduler_params import NNSchedulerParams as _NNSchedulerParams

    run = NNRun(
        net=NNParams(
            input_dim=4,
            output_dim=3,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        model=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
        train=NNTrainParams(
            n_epochs=1,
            train_loader=DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 3, (8,)))),
            optim=_NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=_NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=1, cooldown=1, threshold=1e-3),
        ),
    )
    html = run._repr_html_()
    assert isinstance(html, str)
    assert "<table" in html
