"""Optional domain dependencies (FEAT-031).

The graph (``torch_geometric``), vision (``torchvision``) and plotting
(``plotly``) stacks are extras, not core dependencies: ``import nnx`` loads
none of them, and a feature that needs one imports it on use through
:func:`require`, which names the extra to install when the package itself is
missing. An ``ImportError`` raised from *inside* an installed package (one
of its own dependencies missing or broken) propagates unchanged, with its
cause. :func:`require_onnx` does the same for the ``onnx`` extra before a
legacy ONNX export.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

# top-level import name -> (extra, distribution name)
DOMAIN_EXTRAS: dict[str, tuple[str, str]] = {
    "torch_geometric": ("graph", "torch_geometric"),
    "torchvision": ("vision", "torchvision"),
    "plotly": ("plots", "plotly"),
}


def install_hint(package: str) -> str:
    """The install command for a domain package, leading with the NNx extra."""
    extra, distribution = DOMAIN_EXTRAS[package]
    return f'pip install "thekaveh-nnx[{extra}]" (or pip install {distribution})'


def require(module: str, feature: str) -> ModuleType:
    """Import ``module`` for ``feature``.

    Raises:
        ImportError: the domain package ``module`` belongs to is not
            installed; the message names the extra (``thekaveh-nnx[graph]``,
            ``[vision]`` or ``[plots]``) and the original error is the cause.
            Any other import failure propagates unchanged.
    """
    package = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        missing = (error.name or "").split(".")[0]
        if package not in DOMAIN_EXTRAS or missing != package or available(package):
            # a dependency (or a submodule) of an installed package is
            # broken: keep its own error rather than claim it is absent
            raise
        raise ImportError(f"{feature} needs {package}: {install_hint(package)}") from error


def require_onnx(feature: str) -> None:
    """Refuse a legacy (TorchScript) ONNX export without the ``onnx`` package.

    Without it torch's exporter raises a ``RuntimeError``
    (``OnnxExporterError``: "Module onnx is not installed!") that names no
    install command.

    Raises:
        ImportError: ``onnx`` is not installed; the message names
            ``thekaveh-nnx[onnx]``.
    """
    if not available("onnx"):
        raise ImportError(
            f"{feature} requires the `onnx` package. Install via `pip install thekaveh-nnx[onnx]` (or `pip install onnx`)."
        )


def available(package: str) -> bool:
    """Whether a domain package is installed, without importing it."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False
