"""Activation-map visualization via PyTorch forward hooks.

Given a model and an input tensor, runs a forward pass with a
`register_forward_hook` on the requested layer, captures the
intermediate activation, and emits a Plotly `Figure`:

- **4D activations** ``(N, C, H, W)`` — typical conv feature maps —
  render as a grid of per-channel heatmaps from the first sample in
  the batch. This is the canonical "what is the conv layer looking
  at" view.
- **2D activations** ``(N, F)`` — typical dense feature vectors —
  render as a single heatmap of shape ``(N, F)`` so callers can
  eyeball per-feature activations across the batch.

Other rank tensors fall through to a flattened single-row heatmap
rather than raising — activation maps are a debugging aid, and the
caller would rather get *something* visual than an exception they
have to wrap.

Returning a Plotly `Figure` keeps the module consistent with
`weight_histogram` / `nnx.vis_utils`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch import nn

if TYPE_CHECKING:
    from ..nn.nn_model import NNModel


def activation_map(
    model: Union[nn.Module, NNModel],
    x: torch.Tensor,
    layer_name: str,
    *,
    max_channels: int = 16,
    cols: int = 4,
    fig_width: int = 900,
    cell_size: int = 180,
) -> go.Figure:
    """Capture the activation of `layer_name` for input `x` and render it.

    Registers a forward hook on the named submodule, runs `model(x)` under
    `torch.no_grad()`, then removes the hook and turns the captured tensor
    into a Plotly heatmap layout:

    - 4D ``(N, C, H, W)`` activations: grid of up to `max_channels` per-channel
      heatmaps from the first sample (``N=0``).
    - 2D ``(N, F)`` activations: single ``(N, F)`` heatmap.
    - Other ranks: flattened single-row heatmap (best-effort fallback).

    Args:
        model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
        x: Input tensor (or any object) accepted by `model.__call__`. Moved to
            the same device as the model's first parameter when possible.
        layer_name: Dotted name from `model.named_modules()` — e.g. `"layers.2"`
            for a Sequential, `"conv1"` for a class attribute. Pass an empty
            string `""` to hook the top-level module itself.
        max_channels: Cap on conv-channel subplots (4D case). Defaults to 16 —
            enough to spot patterns without crushing the layout for 256-channel
            feature maps.
        cols: Subplot columns in the 4D grid.
        fig_width: Total figure width in pixels.
        cell_size: Per-subplot square cell size (px). Total height scales
            with the row count.

    Returns:
        A Plotly `Figure` containing one or more `Heatmap` traces.

    Raises:
        ValueError: If `layer_name` doesn't resolve to a submodule of `model`.
        RuntimeError: If the forward hook on `layer_name` never fires
            (the layer is not reached by this input's forward path).
    """
    # Local import to avoid a circular import at package init time.
    from ..nn.nn_model import NNModel

    if isinstance(model, NNModel):
        model = model.net

    modules = dict(model.named_modules())
    if layer_name not in modules:
        # Surface the closest candidates so the caller can fix the typo
        # without re-grep'ing the model. Limit to a short list to keep
        # the error message readable on deep models.
        available = list(modules.keys())
        sample = ", ".join(repr(n) for n in available[:10])
        raise ValueError(
            f"activation_map: layer_name={layer_name!r} not found in model.named_modules(). "
            f"First {min(10, len(available))} available names: [{sample}]"
        )

    target = modules[layer_name]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        # Some layers return tuples / dicts (multi-head attention, etc.).
        # We grab the first tensor we can find so the helper degrades
        # gracefully on non-conv layers rather than raising.
        if isinstance(out, torch.Tensor):
            captured["v"] = out.detach().cpu()
        elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            captured["v"] = out[0].detach().cpu()
        else:
            # Last-ditch: try to coerce.
            captured["v"] = torch.as_tensor(out).detach().cpu()

    handle = target.register_forward_hook(_hook)
    try:
        # Best-effort device alignment — works for the common case where
        # the model has at least one parameter. Pure stateless modules
        # (e.g. nn.ReLU) keep `x` on whatever device the caller passed.
        try:
            device = next(model.parameters()).device
            if isinstance(x, torch.Tensor):
                x = x.to(device)
        except StopIteration:
            pass
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(x)
        finally:
            if was_training:
                model.train()
    finally:
        handle.remove()

    if "v" not in captured:
        # The hook *should* always fire (PyTorch invokes it as part of
        # the forward pass). If it didn't, the forward call must have
        # short-circuited before reaching the layer — surface that
        # rather than KeyError'ing on the next line.
        raise RuntimeError(
            f"activation_map: forward hook on {layer_name!r} did not fire. "
            "The layer may not be reached by this input — check the model's forward path."
        )
    return _activation_to_figure(
        captured["v"],
        layer_name=layer_name,
        max_channels=max_channels,
        cols=cols,
        fig_width=fig_width,
        cell_size=cell_size,
    )


def _activation_to_figure(
    act: torch.Tensor,
    *,
    layer_name: str,
    max_channels: int,
    cols: int,
    fig_width: int,
    cell_size: int,
) -> go.Figure:
    """Render a captured activation tensor as a Plotly Figure.

    Split out so callers (and tests) can render a captured tensor without
    re-running the forward pass. Heatmap rendering is colorscale-only; we
    don't impose a colorbar per subplot since 16 colorbars in one figure
    is more visual noise than signal.
    """
    if act.ndim == 4:
        # (N, C, H, W) — show the first sample, up to max_channels.
        sample = act[0]
        n_channels = min(sample.shape[0], max_channels)
        rows = math.ceil(n_channels / cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"ch{i}" for i in range(n_channels)],
            vertical_spacing=0.06,
            horizontal_spacing=0.04,
        )
        for i in range(n_channels):
            r, c = i // cols + 1, i % cols + 1
            fig.add_trace(
                go.Heatmap(
                    z=sample[i].numpy(),
                    showscale=False,
                    colorscale="Viridis",
                ),
                row=r,
                col=c,
            )
        # Match aspect ratio per cell so conv feature maps don't get
        # stretched into rectangles that hide spatial structure. Plotly
        # accepts the short axis form ("x", "x2", ...) for `scaleanchor`,
        # not the long form ("xaxis2"), so we translate accordingly.
        for axis_key in list(fig.layout):
            if axis_key.startswith("yaxis"):
                short_x = axis_key.replace("yaxis", "x")
                fig.layout[axis_key].scaleanchor = short_x
                fig.layout[axis_key].autorange = "reversed"
        fig.update_layout(
            width=fig_width,
            height=cell_size * rows + 80,
            title=f"Activation map: {layer_name} (sample 0, first {n_channels} channels)",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig

    if act.ndim == 2:
        # (N, F) — single heatmap, rows=batch, cols=feature.
        fig = go.Figure(
            data=go.Heatmap(
                z=act.numpy(),
                colorscale="Viridis",
                colorbar=dict(title="value"),
            )
        )
        fig.update_layout(
            width=fig_width,
            height=max(cell_size, 40 * act.shape[0]) + 80,
            title=f"Activation map: {layer_name} (batch x features)",
            xaxis_title="feature",
            yaxis_title="sample",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    # Fallback: flatten so the caller still gets something useful. This
    # path handles 1D (single sample of features) and >4D (rare, e.g.
    # video conv with extra time axis) without raising.
    flat = act.reshape(1, -1).numpy()
    fig = go.Figure(
        data=go.Heatmap(
            z=flat,
            colorscale="Viridis",
            colorbar=dict(title="value"),
        )
    )
    fig.update_layout(
        width=fig_width,
        height=cell_size + 80,
        title=f"Activation map: {layer_name} (flattened, original shape {tuple(act.shape)})",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
