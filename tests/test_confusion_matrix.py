"""Tests for VisUtils.classification_report (the renderable confusion_matrix
is tested by import + smoke; full Plotly Figure assertion would be flaky)."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
        Y_true, Y_pred, class_names=["cat", "dog", "fish"],
    )
    assert "cat" in df.index
    assert "dog" in df.index
    assert "fish" in df.index


def test_confusion_matrix_smoke():
    """confusion_matrix renders a Plotly figure; we just check it doesn't raise."""
    Y_true = np.array([0, 1, 2, 0, 1, 2])
    Y_pred = np.array([0, 1, 2, 0, 1, 2])
    # Force the non-interactive renderer for test environments.
    VisUtils.RENDERER = "json"
    try:
        VisUtils.confusion_matrix(Y_true, Y_pred, class_names=["a", "b", "c"])
    finally:
        VisUtils.RENDERER = None
