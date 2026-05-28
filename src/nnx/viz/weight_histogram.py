"""Per-parameter weight histogram grid as a Plotly Figure.

Walks `model.named_parameters()` and emits one histogram trace per
tensor, laid out in a grid subplot. Useful for spotting dead layers,
NaN / Inf weights, or saturation patterns at a glance. Returns a Plotly
`Figure` to stay consistent with the `nnx.vis_utils` idiom (the
existing run-output viz module also returns Plotly figures so callers
can compose them into dashboards or notebook layouts).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import nn

if TYPE_CHECKING:
    from ..nn.nn_model import NNModel


def weight_histogram(
    model: Union[nn.Module, NNModel],
    *,
    bins: int = 64,
    cols: int = 3,
    fig_width: int = 1000,
    row_height: int = 200,
) -> go.Figure:
    """Return a Plotly grid of per-parameter weight histograms.

    Args:
        model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
        bins: Number of histogram bins per parameter tensor.
        cols: Number of columns in the subplot grid. Rows are computed from the
            parameter count.
        fig_width: Figure width in pixels.
        row_height: Per-row height in pixels; total height = `row_height * rows`.

    Returns:
        A Plotly `Figure` with one `Histogram` trace per named parameter tensor.
        Each subplot title is the dotted parameter name (e.g. `layers.0.weight`).
        Empty parameter tensors are skipped from the grid.

    Raises:
        ValueError: If `model` has no named parameters (nothing to plot).
    """
    # Local import to avoid a circular import at package init time.
    from ..nn.nn_model import NNModel

    if isinstance(model, NNModel):
        model = model.net
    params = [(n, p.detach().cpu().flatten().numpy()) for n, p in model.named_parameters()]
    n = len(params)
    if n == 0:
        raise ValueError("weight_histogram: model has no named parameters to plot.")
    rows = math.ceil(n / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[name for name, _ in params],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    for idx, (_name, vals) in enumerate(params):
        r, c = idx // cols + 1, idx % cols + 1
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=bins, showlegend=False),
            row=r,
            col=c,
        )
    fig.update_layout(
        width=fig_width,
        height=row_height * rows,
        title="Weight histograms (per parameter tensor)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
