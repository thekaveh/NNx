"""Tests for nnx.vis_utils.

Cover the module import, the public callable surface, AND a real
invocation of `multi_line_plot` so a regression that turns the
function into a no-op or returns `None` is caught.
"""

from __future__ import annotations


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
