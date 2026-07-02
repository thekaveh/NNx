"""drop_layer — replace a named layer with :class:`nn.Identity`.

This isn't a function-preserving primitive like :func:`widen` or
:func:`deepen` — dropping a layer changes the function the network
computes. What we *do* preserve is the *chain*: the replaced layer
becomes :class:`nn.Identity`, so dotted-name access, downstream
shapes, and the forward pass all still work. The caller is then
expected to refine the dropped network via :meth:`NNModel.train` to
recover quality.

Two call styles:

  - ``drop_layer(model, layer_name="layers.1")`` — drop the named layer
    unconditionally.
  - ``drop_layer(model, layer_name=["layers.0", "layers.1", "layers.2"],
    importance=fn)`` — score each candidate via ``fn(module)`` and drop
    the *minimum*-scoring one (cheapest to lose). Useful for "find the
    least informative layer in this stack" pipelines.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Union

from torch import nn

from ._utils import get_module, set_module


def drop_layer(
    model: nn.Module,
    *,
    layer_name: Union[str, list[str]],
    importance: Callable[[nn.Module], float] | None = None,
) -> nn.Module:
    """Replace a named layer with :class:`nn.Identity`.

    Args:
        model: any :class:`nn.Module`. Deep-copied so the caller's
            reference survives.
        layer_name: either a dotted submodule name, or a list of
            dotted names to choose from. When a list is given,
            ``importance`` must be provided as well.
        importance: optional callable ``fn(submodule) -> float``. When
            ``layer_name`` is a list, the candidate with the *minimum*
            importance score is dropped (lowest = least informative =
            safest to remove). Rejected when ``layer_name`` is a single
            string because there is no candidate-selection step to score.

    Returns:
        A fresh :class:`nn.Module` with the chosen layer replaced by
        :class:`nn.Identity`. Forward shape contract is preserved iff
        the dropped layer was shape-preserving (e.g. an activation or
        a square Linear); otherwise calling forward on the surged
        module will raise — by design, since silently corrupting the
        shape would be worse than a loud failure.

    Raises:
        KeyError: if any candidate name is missing.
        ValueError: if ``layer_name`` is an empty list, or a list
            without ``importance``, or a single string with
            ``importance``.
    """
    chosen = _resolve_target(model, layer_name, importance)

    new_model = copy.deepcopy(model)
    # Verify the chosen name still resolves in the copy — it should,
    # since deepcopy preserves the module tree structure.
    get_module(new_model, chosen)
    set_module(new_model, chosen, nn.Identity())
    return new_model


def _resolve_target(
    model: nn.Module,
    layer_name: Union[str, list[str]],
    importance: Callable[[nn.Module], float] | None,
) -> str:
    """Resolve ``layer_name`` (string or list) + optional ``importance``
    to a single dotted name. Raises on empty / missing / no-importance."""
    if isinstance(layer_name, str):
        if importance is not None:
            raise ValueError(
                "drop_layer: importance= is only valid when layer_name is a candidate list, not a single layer_name"
            )
        # Single name: confirm it resolves.
        get_module(model, layer_name)
        return layer_name

    if not layer_name:
        raise ValueError("drop_layer: empty candidate list")
    if importance is None:
        raise ValueError("drop_layer: when layer_name is a list, importance= must be provided to choose")

    # Score every candidate and pick the minimum.
    scores: list[tuple[float, str]] = []
    for name in layer_name:
        mod = get_module(model, name)
        scores.append((float(importance(mod)), name))
    scores.sort(key=lambda t: t[0])
    return scores[0][1]
