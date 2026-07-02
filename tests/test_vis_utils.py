"""Tests for nnx.vis_utils.

Cover the module import, the public callable surface, AND a real
invocation of `multi_line_plot` so a regression that turns the
function into a no-op or returns `None` is caught.
"""

from __future__ import annotations

import types

import numpy as np
import torch


def test_vis_utils_importable():
    import nnx.vis_utils as vis_utils

    assert vis_utils is not None


def test_vis_utils_module_has_callables():
    """vis_utils should expose at least one callable for figure generation."""
    import inspect

    import nnx.vis_utils as vis_utils

    callables = [name for name, obj in inspect.getmembers(vis_utils) if callable(obj) and not name.startswith("_")]
    assert len(callables) > 0, "vis_utils exposes no public callables"


def test_multi_line_plot_returns_figure_with_traces():
    """A real invocation of `multi_line_plot` — verifies the function
    actually builds a Plotly Figure and adds the expected number of
    traces. Catches regressions that would silently return None or skip
    trace creation."""
    import plotly.graph_objects as go

    from nnx.vis_utils import VisUtils

    x = list(range(10))
    yss = [
        [[1.0 * i for i in range(10)], [2.0 * i for i in range(10)]],  # group 1: 2 lines
        [[0.5 * i for i in range(10)]],  # group 2: 1 line
    ]
    fig = VisUtils.multi_line_plot(
        x=x,
        yss=yss,
        title="test plot",
        yss_legend=(["group A", "group B"], ["line 1", "line 2"]),
        x_axis_label="x",
        y_axis_label="y",
        renderer=None,  # headless
    )
    assert isinstance(fig, go.Figure)
    # Expect at least one trace per actual data line — the function may
    # also add no-trace markers for the legend rows; the floor is the
    # data-line count.
    actual_lines = sum(len(group) for group in yss)
    assert len(fig.data) >= actual_lines, f"expected ≥ {actual_lines} traces for the data lines, got {len(fig.data)}"


def test_multi_line_plot_raises_on_empty_yss():
    """The function should raise loudly rather than silently producing an empty plot."""
    import pytest

    from nnx.vis_utils import VisUtils

    with pytest.raises(ValueError, match="at least one series"):
        VisUtils.multi_line_plot(
            x=[],
            yss=[],
            title="t",
            yss_legend=([], []),
            x_axis_label="x",
            y_axis_label="y",
            renderer=None,
        )


def test_generate_colors_are_pairwise_distinct():
    """Hue is circular, so the old closed linspace gave the first and
    last class identical colors in every scatter/t-SNE plot."""
    from nnx.vis_utils import VisUtils

    for n in (2, 3, 5, 8):
        colors = VisUtils.generate_colors(n)
        assert len(set(colors)) == n, f"duplicate colors at n={n}: {colors}"


def test_two_dim_tsne_checkpoint_logits_collects_across_batches(monkeypatch):
    from nnx import vis_utils
    from nnx.vis_utils import VisUtils

    captured = {}

    class _FakeTSNE:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit_transform(self, X):
            captured["rows"] = len(X)
            return np.zeros((len(X), 2))

    class _FakeNet:
        @staticmethod
        def unpack_batch(batch):
            return batch

    class _FakeModel:
        net = _FakeNet()

        @staticmethod
        def predict(X):
            n = len(X)
            return types.SimpleNamespace(logits=np.arange(n * 2, dtype=float).reshape(n, 2))

    class _FakeNNModel:
        @staticmethod
        def from_checkpoint(checkpoint):
            return _FakeModel()

    ds = types.SimpleNamespace(
        output_dim=2,
        test_loader=[
            (torch.randn(3, 4), torch.tensor([0, 1, 0])),
            (torch.randn(4, 4), torch.tensor([1, 0, 1, 0])),
        ],
    )
    checkpoint = types.SimpleNamespace(idp=types.SimpleNamespace(epoch_idx=3))

    monkeypatch.setattr(vis_utils, "NNModel", _FakeNNModel)
    monkeypatch.setattr(vis_utils, "TSNE", _FakeTSNE)

    VisUtils.two_dim_tsne_checkpoint_logits(checkpoint=checkpoint, ds=ds, n_samples=5, renderer=None)

    assert captured["rows"] == 5
    assert captured["kwargs"]["perplexity"] == 4
    assert captured["kwargs"]["random_state"] == 0
