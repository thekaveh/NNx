"""Per-layer gradient-norm bar chart for training-loop diagnostics.

Standard tool for diagnosing vanishing / exploding gradients. Call
after ``loss.backward()`` and before ``optimizer.zero_grad()`` so the
gradients are still attached to the parameters.
"""

from __future__ import annotations

import plotly.graph_objects as go
from torch import nn


def gradient_flow(model: nn.Module) -> go.Figure:
    """Return a Plotly bar chart of per-parameter L2 gradient norms.

    Call AFTER ``loss.backward()`` and BEFORE ``optimizer.zero_grad()``.
    Each bar is one trainable ``nn.Parameter`` of the model whose
    ``.grad`` has been populated by the backward pass; bar height is
    the L2 norm of that gradient.

    Frozen parameters (``requires_grad=False``) are skipped. Parameters
    whose gradient is ``None`` (typically because they weren't reached
    during the forward pass) are also skipped.

    Args:
        model: an ``nn.Module`` whose gradients have just been populated
            by ``loss.backward()``.

    Returns:
        A Plotly ``Figure`` with one bar per trainable parameter,
        labeled by ``named_parameters()`` dotted name.

    Raises:
        ValueError: if no parameter has a populated gradient — most
            often because ``loss.backward()`` wasn't called before
            this function.
    """
    names: list[str] = []
    norms: list[float] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue
        names.append(name)
        norms.append(param.grad.detach().norm().item())

    if not names:
        raise ValueError(
            "gradient_flow: no parameters have populated gradients. "
            "Did you forget to call loss.backward() before this call?"
        )

    fig = go.Figure(data=[go.Bar(x=names, y=norms)])
    fig.update_layout(
        title="Per-layer gradient norms",
        xaxis_title="Parameter",
        yaxis_title="L2 norm of gradient",
        xaxis_tickangle=-45,
        margin=dict(b=120),
    )
    return fig
