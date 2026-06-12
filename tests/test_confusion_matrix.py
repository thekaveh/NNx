"""Tests for VisUtils.confusion_matrix and classification_report.

confusion_matrix now returns a plotly.graph_objects.Figure so callers can
display, save, or compose it. Headless test environments rely on the
default RENDERER=None to skip the .show() call."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nnx.vis_utils import VisUtils


def test_classification_report_returns_dataframe():
    Y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    Y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])  # 2 misclassifications
    df = VisUtils.classification_report(Y_true, Y_pred)
    assert isinstance(df, pd.DataFrame)
    # sklearn report rows include each class plus 'accuracy', 'macro avg', 'weighted avg'
    assert "accuracy" in df.index
    assert "0" in df.index or 0 in df.index


def test_classification_report_with_class_names():
    Y_true = np.array([0, 1, 2])
    Y_pred = np.array([0, 1, 1])
    df = VisUtils.classification_report(
        Y_true,
        Y_pred,
        class_names=["cat", "dog", "fish"],
    )
    assert "cat" in df.index
    assert "dog" in df.index
    assert "fish" in df.index


def test_confusion_matrix_returns_figure():
    """confusion_matrix returns a plotly Figure; the heatmap data matches the
    input cm shape."""
    Y_true = np.array([0, 1, 2, 0, 1, 2])
    Y_pred = np.array([0, 1, 2, 0, 1, 2])
    fig = VisUtils.confusion_matrix(Y_true, Y_pred, class_names=["a", "b", "c"])
    assert isinstance(fig, go.Figure)
    # one heatmap trace whose z is a 3x3 matrix
    assert len(fig.data) == 1
    z = np.asarray(fig.data[0].z)
    assert z.shape == (3, 3)


def test_confusion_matrix_normalize_returns_figure():
    Y_true = np.array([0, 1, 0, 1, 0])
    Y_pred = np.array([0, 1, 1, 1, 0])
    fig = VisUtils.confusion_matrix(Y_true, Y_pred, normalize=True)
    assert isinstance(fig, go.Figure)


def test_confusion_matrix_aligns_labels_when_class_absent():
    """With class_names given and a class absent from the data, sklearn
    orders rows by the classes PRESENT unless labels= is pinned — every
    class after the gap was shifted onto the wrong row/column name."""
    import numpy as np

    from nnx.vis_utils import VisUtils

    # Class 1 never appears; 3 names must still map onto a 3x3 grid.
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 2, 2, 0])
    fig = VisUtils.confusion_matrix(y_true, y_pred, class_names=["a", "b", "c"])
    z = fig.data[0].z
    assert z.shape == (3, 3)
    # Row 1 / column 1 (class "b") must be all zeros, not absorbed by "c".
    assert z[1].sum() == 0
    assert z[:, 1].sum() == 0
    # The real counts sit where the names say they do.
    assert z[0][0] == 1 and z[0][2] == 1 and z[2][2] == 1 and z[2][0] == 1
