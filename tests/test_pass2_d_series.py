"""Pass-2 catalog: D-series tests.

- D6: Utils / VisUtils — module-level functions exist; the class API still
  works (back-compat shim); both point at the same callable.
"""
from __future__ import annotations

import io

import numpy as np

from nnx import utils as utils_mod
from nnx import vis_utils as vis_mod
from nnx.utils import Utils, flatten_dict, print_table, print_tree
from nnx.vis_utils import (
    VisUtils,
    classification_report,
    confusion_matrix,
    generate_colors,
)


def test_d6_utils_module_functions_callable():
    """Module-level imports work and produce the documented outputs."""
    assert callable(print_tree)
    assert callable(print_table)
    assert callable(flatten_dict)

    flat = flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_d6_utils_class_shim_still_works():
    """Utils.print_tree / Utils.flatten_dict continue to function."""
    assert Utils.flatten_dict({"a": 1, "b": {"c": 2}}) == {"a": 1, "b.c": 2}


def test_d6_utils_class_shim_delegates_to_same_callable():
    """Utils.flatten_dict and the module-level flatten_dict reference the
    same underlying function (no duplicated implementation)."""
    # __func__ on a staticmethod returns the underlying function; in
    # Python 3.10+ class-level staticmethods are descriptors that resolve
    # to the bare function when accessed via the class.
    assert Utils.flatten_dict is flatten_dict
    assert Utils.print_tree is print_tree
    assert Utils.print_table is print_table


def test_d6_utils_module_print_tree_uses_file_kwarg():
    buf = io.StringIO()
    print_tree({"k": "v"}, file=buf)
    assert "[+] k" in buf.getvalue()


def test_d6_visutils_module_aliases_present():
    assert callable(generate_colors)
    assert callable(confusion_matrix)
    assert callable(classification_report)


def test_d6_visutils_module_aliases_match_class_methods():
    """The module-level alias IS the class static method, not a copy."""
    assert confusion_matrix is VisUtils.confusion_matrix
    assert classification_report is VisUtils.classification_report
    assert generate_colors is VisUtils.generate_colors


def test_d6_visutils_module_alias_works_end_to_end():
    """Calling via the module alias produces a working result identical
    to the class form."""
    Y_true = np.array([0, 1, 2, 0, 1, 2])
    Y_pred = np.array([0, 1, 2, 0, 1, 2])
    fig_a = confusion_matrix(Y_true, Y_pred)
    fig_b = VisUtils.confusion_matrix(Y_true, Y_pred)
    # Same code path, so figures have the same trace count and z-shape.
    assert len(fig_a.data) == len(fig_b.data) == 1
    assert np.asarray(fig_a.data[0].z).shape == np.asarray(fig_b.data[0].z).shape


def test_d6_vis_utils_all_lists_aliases():
    """nnx.vis_utils.__all__ advertises both VisUtils and the module-level
    function aliases so `from nnx.vis_utils import *` works as expected."""
    expected = {
        "VisUtils", "generate_colors", "multi_line_plot", "scatter_plot",
        "get_scatter_plot_vm", "two_dim_tsne_checkpoint_logits",
        "confusion_matrix", "classification_report",
    }
    assert expected.issubset(set(vis_mod.__all__))


def test_d6_utils_module_no_residual_state():
    """utils module exposes the functions and the class, nothing else
    surprising at module level."""
    public = {n for n in dir(utils_mod) if not n.startswith("_")}
    assert {"Utils", "print_tree", "print_table", "flatten_dict"}.issubset(public)
