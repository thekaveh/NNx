"""Optional domain dependencies (FEAT-031).

The graph (``torch_geometric``), vision (``torchvision``) and plotting
(``plotly``) stacks are extras, not core dependencies: ``import nnx`` loads
none of them, and a feature that needs one imports it on use through
:func:`require`, which names the extra to install when the package itself is
missing. An ``ImportError`` raised from *inside* an installed package (one
of its own dependencies missing or broken) propagates unchanged, with its
cause.
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


def available(package: str) -> bool:
    """Whether a domain package is installed, without importing it."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False
