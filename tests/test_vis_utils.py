"""Smoke tests for nnx.vis_utils.

We verify the module is importable and that it exposes public callables.
"""


def test_vis_utils_importable():
    import nnx.vis_utils as vis_utils
    assert vis_utils is not None


def test_vis_utils_module_has_callables():
    """vis_utils should expose at least one callable for figure generation."""
    import inspect

    import nnx.vis_utils as vis_utils

    callables = [
        name for name, obj in inspect.getmembers(vis_utils)
        if callable(obj) and not name.startswith("_")
    ]
    assert len(callables) > 0, "vis_utils exposes no public callables"
