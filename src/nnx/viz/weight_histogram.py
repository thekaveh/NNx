"""Per-parameter weight histogram grid as a Plotly Figure.

Walks `model.named_parameters()` and emits one histogram trace per
tensor, laid out in a grid subplot. Useful for spotting dead layers,
NaN / Inf weights (counted in each subplot title, since Plotly drops them
when binning), or saturation patterns at a glance. Returns a Plotly
`Figure` to stay consistent with the `nnx.vis_utils` idiom (the
existing run-output viz module also returns Plotly figures so callers
can compose them into dashboards or notebook layouts).
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Union

import numpy as np
from torch import nn

from .._optional import require
from .._validation import shown

if TYPE_CHECKING:
    import plotly.graph_objects as go


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
        ValueError: If `model` has no named parameters (nothing to plot),
            or if `bins` or `cols` is not a positive integer.
    """
    # Any integer, NumPy integers included, passed on as a plain int: plotly's
    # make_subplots(cols=) and nbinsx refuse a numpy.int64.
    for name, value in (("bins", bins), ("cols", cols)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"weight_histogram: {name} must be a positive integer, got {shown(value)}")
    bins, cols = int(bins), int(cols)
    go = require("plotly.graph_objects", "nnx.viz.weight_histogram")
    make_subplots = require("plotly.subplots", "nnx.viz.weight_histogram").make_subplots
    # Local import to avoid a circular import at package init time.
    from ..nn.nn_model import NNModel

    if isinstance(model, NNModel):
        model = model.net
    # numel() > 0: the docstring promises empty tensors are skipped
    # (a zero-size histogram subplot renders as a confusing blank).
    # .float(): NumPy has no bfloat16.
    params = [(n, p.detach().cpu().float().flatten().numpy()) for n, p in model.named_parameters() if p.numel() > 0]
    n = len(params)
    if n == 0:
        raise ValueError("weight_histogram: model has no named parameters to plot.")
    rows = math.ceil(n / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        # Plotly drops NaN / Inf when binning: the title counts them.
        subplot_titles=[
            f"{name} ({count} non-finite)" if (count := int((~np.isfinite(vals)).sum())) else name
            for name, vals in params
        ],
        # Plotly needs spacing <= 1 / (n - 1): shrink it for large grids.
        vertical_spacing=min(0.08, 0.5 / max(rows - 1, 1)),
        horizontal_spacing=min(0.05, 0.5 / max(cols - 1, 1)),
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
