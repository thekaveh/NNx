"""Captum-backed input-attribution methods with a Plotly heatmap companion.

Single-call wrapper around the six most common Captum attribution
methods. The string-keyed dispatch keeps call sites short and shields
notebook users from Captum's per-method class hierarchy (`IntegratedGradients`,
`GradientShap`, `DeepLift`, `Saliency`, `InputXGradient`, `Occlusion`).
Returns both the raw attribution tensor (for downstream metrics /
sanity-checking) and a Plotly `Figure` for at-a-glance inspection.

Sibling of `nnx.viz.summary` / `nnx.viz.weight_histogram`. Opt-in via
`pip install thekaveh-nnx[viz]` (which now pulls `captum>=0.7.0` alongside
`torchinfo`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import plotly.graph_objects as go
import torch
from torch import nn

if TYPE_CHECKING:
    from ..nn.nn_model import NNModel

# Keep the supported-method list defined once so `attribute()`'s
# error message and the parametrize-all-methods test can both reference it.
SUPPORTED_METHODS: tuple[str, ...] = (
    "integrated_gradients",
    "gradient_shap",
    "deep_lift",
    "saliency",
    "input_x_gradient",
    "occlusion",
)


def _build_attributor(method: str, model: nn.Module) -> Any:
    """Instantiate the Captum attributor for `method` against `model`.

    Lazy-imports `captum.attr` so the rest of `nnx.viz` keeps importing
    without the optional Captum dep installed.
    """
    try:
        from captum.attr import (
            DeepLift,
            GradientShap,
            InputXGradient,
            IntegratedGradients,
            Occlusion,
            Saliency,
        )
    except ImportError as e:
        raise ImportError("nnx.viz.attribute requires captum: pip install captum") from e

    factories = {
        "integrated_gradients": IntegratedGradients,
        "gradient_shap": GradientShap,
        "deep_lift": DeepLift,
        "saliency": Saliency,
        "input_x_gradient": InputXGradient,
        "occlusion": Occlusion,
    }
    return factories[method](model)


def _default_kwargs(method: str, x: torch.Tensor) -> dict[str, Any]:
    """Per-method default kwargs that the user can override via `**method_kwargs`.

    GradientShap requires `baselines`; Occlusion requires `sliding_window_shapes`.
    Supplying sensible defaults keeps the one-call ergonomics intact for
    notebook users while still allowing power-user overrides.
    """
    if method == "gradient_shap":
        return {"baselines": torch.zeros_like(x)}
    if method == "occlusion":
        # Single-element sliding window along the last input dim — works
        # for both tabular (B, D) and image-shaped (B, C, H, W) inputs as
        # a sane "I just want to see something" default. Users tuning
        # occlusion-window size for real image inputs should override.
        feature_shape = tuple(1 for _ in x.shape[1:])
        return {"sliding_window_shapes": feature_shape or (1,)}
    return {}


def _attribution_to_figure(attr: torch.Tensor, x: torch.Tensor) -> go.Figure:
    """Render an attribution tensor as a Plotly heatmap.

    For 3-D / 4-D (image-shaped) tensors we collapse channels via mean
    before plotting so the result is always a 2-D heatmap.
    """
    a = attr.detach().cpu()
    # Image-shaped inputs: (B, C, H, W) or (C, H, W) → mean over channels.
    if a.dim() == 4:
        a = a.mean(dim=1)[0]  # first sample, mean over channels
    elif a.dim() == 3:
        a = a.mean(dim=0)
    elif a.dim() == 1:
        # 1-D → reshape to (1, N) so heatmap still renders.
        a = a.unsqueeze(0)
    # 2-D stays as (B, D) — one row per batch element.
    arr = a.numpy()
    fig = go.Figure(data=go.Heatmap(z=arr, colorscale="RdBu", zmid=0))
    fig.update_layout(
        title=f"Attribution heatmap ({arr.shape[0]}×{arr.shape[1]})",
        xaxis_title="feature index",
        yaxis_title="batch / spatial index",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def attribute(
    model: Union[nn.Module, NNModel],
    x: torch.Tensor,
    *,
    method: str = "integrated_gradients",
    target: Any = None,
    **method_kwargs: Any,
) -> tuple[torch.Tensor, go.Figure]:
    """Compute input attributions via Captum and render a Plotly heatmap.

    Args:
        model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
            The model is set to `eval()` for the duration of the attribution
            call; the original mode is restored on return.
        x: Input tensor to attribute. Shape `(B, ...)`. Gradient-based
            methods will set `requires_grad_(True)` internally as needed.
        method: One of `"integrated_gradients"`, `"gradient_shap"`,
            `"deep_lift"`, `"saliency"`, `"input_x_gradient"`, `"occlusion"`.
        target: Target class index (or per-batch indices) for classification
            attributors. Forwarded verbatim to Captum's `.attribute(target=)`.
        **method_kwargs: Extra kwargs forwarded to the per-method
            `.attribute(...)` call. Overrides any defaults supplied for
            `gradient_shap` (`baselines`) or `occlusion` (`sliding_window_shapes`).

    Returns:
        A tuple `(attribution_tensor, figure)` where `attribution_tensor` is a
            `torch.Tensor` with the same shape as `x` (per Captum's standard
            return contract for these six methods) and `figure` is a Plotly
            `Heatmap` visualizing the attribution. Image-shaped inputs (3-D /
            4-D) are mean-pooled over channels before rendering.

    Raises:
        ImportError: If `captum` is not installed. Install with
            `pip install captum`.
        ValueError: If `method` is not one of the supported keys.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unknown method '{method}'; choose from {list(SUPPORTED_METHODS)}")

    # Local import to avoid a circular import at package init time
    # (NNModel pulls in nnx.viz indirectly through some training paths).
    from ..nn.nn_model import NNModel

    if isinstance(model, NNModel):
        model = model.net

    was_training = model.training
    model.eval()
    try:
        attributor = _build_attributor(method, model)
        kwargs = {**_default_kwargs(method, x), **method_kwargs}
        attr_tensor = attributor.attribute(x, target=target, **kwargs)
    finally:
        if was_training:
            model.train()

    figure = _attribution_to_figure(attr_tensor, x)
    return attr_tensor, figure
