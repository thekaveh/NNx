"""Smoke tests for nnx.utils helpers.

The Utils class exposes static helpers used by other modules. We
verify the class is importable and at least the obvious methods exist.
"""


def test_utils_class_importable():
    from nnx.utils import Utils

    assert Utils is not None


def test_utils_has_print_methods():
    from nnx.utils import Utils

    assert hasattr(Utils, "print_tree")
    assert hasattr(Utils, "print_table")


def test_utils_print_tree_executes(capsys):
    """Calling print_tree on a small dict shouldn't crash and should produce output."""
    from nnx.utils import Utils

    Utils.print_tree({"a": 1, "b": {"c": 2}})
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_utils_print_tree_tolerates_empty_dicts(capsys):
    """Empty dicts — top-level or as a nested value — are a no-op, not a
    ValueError from max() over zero keys (the pre-fix behavior)."""
    from nnx.utils import Utils

    Utils.print_tree({})
    Utils.print_tree({"a": {}})
    captured = capsys.readouterr()
    assert "[-] a" in captured.out
