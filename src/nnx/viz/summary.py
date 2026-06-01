"""Keras-style summary table for any NNModel or torch.nn.Module.

Thin wrapper around `torchinfo.summary`. Returns the `ModelStatistics`
object directly — print it for the formatted table; query
`.total_params` / `.trainable_params` / `.total_mult_adds` for
programmatic access. Accepting `NNModel` here (and unwrapping to its
`.net`) keeps the call site short for the common case while still
allowing a raw `nn.Module` for the post-construction-swap idiom that
the multi-optimizer `Trainer` and the diffusion / PEFT specializations
already rely on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch
from torch import nn

if TYPE_CHECKING:
    from torchinfo import ModelStatistics

    from ..nn.nn_model import NNModel


def summary(
    model: Union[nn.Module, NNModel],
    *,
    input_size: tuple[int, ...] | None = None,
    input_data: Union[torch.Tensor, tuple, list, None] = None,
    depth: int = 4,
    col_names: tuple[str, ...] = ("output_size", "num_params", "mult_adds"),
) -> ModelStatistics:
    """Return a `torchinfo.ModelStatistics` summary for `model`.

    Args:
        model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
        input_size: Shape tuple for a synthetic dummy input, e.g. `(1, 3, 224, 224)`.
            Mutually exclusive with `input_data`.
        input_data: An actual tensor / tuple / list to forward through the model.
            Useful when the model takes multiple positional arguments or a non-tensor
            input (graphs, dicts) that `input_size` can't describe.
        depth: Maximum module-nesting depth to expand in the table.
        col_names: Which torchinfo columns to include. Defaults to the three most
            useful ones for spotting parameter / FLOP regressions across runs.

    Returns:
        The `torchinfo.ModelStatistics` instance — print it for the Keras-style
        table, or access `.total_params` / `.trainable_params` / `.total_mult_adds`
        for programmatic regression assertions.

    Raises:
        ImportError: If `torchinfo` isn't installed. Install with `pip install nnx-pytorch[viz]`.
    """
    try:
        from torchinfo import summary as _ti_summary
    except ImportError as e:
        raise ImportError(
            "nnx.viz.summary requires torchinfo — install via `pip install nnx-pytorch[viz]` or `pip install torchinfo>=1.8.0`."
        ) from e
    # Accept NNModel directly — unwrap to .net so callers don't have to.
    # Local import to avoid a circular import at package init time.
    from ..nn.nn_model import NNModel

    if isinstance(model, NNModel):
        model = model.net
    return _ti_summary(
        model,
        input_size=input_size,
        input_data=input_data,
        depth=depth,
        col_names=col_names,
        verbose=0,
    )
