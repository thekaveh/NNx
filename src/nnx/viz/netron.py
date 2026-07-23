"""Netron browser-viewer export.

Wraps `torch.onnx.export` so callers get a single one-liner that
produces an `.onnx` file ready to open with Netron — the de-facto
graph viewer for ONNX / TorchScript / SavedModel artifacts. With
``launch=True`` we additionally call `netron.start(path)` to pop the
viewer open in the user's browser.

The ONNX file itself is plain `pip install thekaveh-nnx[onnx]` territory
(`onnx` ships in the `onnx` extra). The optional Netron *viewer*
process lives in a separate extra (`thekaveh-nnx[viz-interactive]`) so
notebook users who only want the file format don't pull a Flask
server they'll never run.

The interactive launch path is intentionally guarded behind
``launch=False`` by default so tests / CI runs can exercise the
export without spawning a long-lived viewer process.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from torch import nn

from ..utils import _capture_training_modes, _restore_training_modes

if TYPE_CHECKING:
    from ..nn.nn_model import NNModel


def netron_export(
    model: Union[nn.Module, NNModel],
    path: str,
    example_input: Union[torch.Tensor, tuple, np.ndarray],
    *,
    launch: bool = False,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """Export `model` to an ONNX file at `path` (optionally open Netron).

    Args:
        model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
        path: Output filename, e.g. ``"model.onnx"``.
        example_input: A tensor (or tuple of tensors) with realistic
            shape / dtype used to trace the network.
        launch: When True, call `netron.start(path)` to open the model
            in Netron's browser viewer. Requires `pip install thekaveh-nnx[viz-interactive]`
            (or `pip install netron`). Defaults to False so CI / tests
            can exercise export without spawning a long-lived process.
        opset_version: ONNX opset to target. 17 is broadly supported
            by current runtimes.
        dynamic_batch: When True (default), marks dim 0 as dynamic so
            the exported graph accepts any batch size at inference.

    Returns:
        The path written (matches `path` — handy when chaining).

    Raises:
        ImportError: When `launch=True` and the `netron` package isn't
            installed. The ONNX export itself uses `torch.onnx`, which
            is part of core PyTorch.
    """
    # Local import to avoid a circular import at package init time.
    from ..nn.nn_model import NNModel

    if isinstance(model, NNModel):
        net = model.net
        device = model.device
    else:
        net = model
        # Best-effort device probe — fall back to CPU for stateless
        # modules with no parameters.
        try:
            device = next(net.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if isinstance(example_input, torch.Tensor):
        example_input = (example_input.to(device),)
    elif isinstance(example_input, np.ndarray):
        example_input = (torch.from_numpy(example_input).to(device),)
    else:
        example_input = tuple(
            (e.to(device) if isinstance(e, torch.Tensor) else torch.from_numpy(np.asarray(e)).to(device))
            for e in example_input
        )

    in_names = [f"input_{i}" for i in range(len(example_input))]
    out_names = ["output"]
    dynamic_axes = {n: {0: "batch"} for n in in_names + out_names} if dynamic_batch else None

    training_modes = _capture_training_modes(net)
    net.eval()
    try:
        # Mirror NNModel.to_onnx — pin to the legacy TorchScript path so
        # plain `pip install onnx` is enough; the dynamo path needs
        # `onnxscript` which we keep off the viz dep set.
        export_accepts_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
        if export_accepts_dynamo:
            torch.onnx.export(
                net,
                example_input,
                path,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                dynamo=False,
            )
        else:
            torch.onnx.export(
                net,
                example_input,
                path,
                input_names=in_names,
                output_names=out_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
            )
    finally:
        _restore_training_modes(training_modes)

    if launch:
        # Lazy import — keeps `netron` out of required deps. Only users
        # who actually want the interactive viewer pay the install cost.
        try:
            import netron  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "netron_export(launch=True) requires the `netron` package. "
                "Install via `pip install thekaveh-nnx[viz-interactive]` (or `pip install netron`)."
            ) from e
        netron.start(path)

    return path
