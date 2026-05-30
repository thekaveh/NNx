"""fastai-style learning-rate finder.

Runs an exponential LR sweep from ``start_lr`` to ``end_lr`` over
``num_iter`` training iterations, recording loss at each step. The
recommended ``max_lr`` is the LR at the steepest descent point of the
smoothed loss curve — the classic Smith (2017) heuristic.

The sweep is **non-destructive**: the model's initial weights are
snapshotted before the sweep starts and restored on exit, so the
caller can use this as a pre-flight check before the real training
run without disturbing any subsequent reproducibility.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import plotly.graph_objects as go
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class LRFinderResult:
    """Result of an :func:`lr_finder` sweep.

    Attributes:
        lrs: list of learning rates actually exercised. Length matches
            ``losses``. May be shorter than ``num_iter`` if the sweep
            early-exited due to loss divergence.
        losses: list of loss values, one per LR.
        suggested_lr: the recommended ``max_lr`` for a subsequent
            real training run — the LR at the steepest-descent
            point of the smoothed loss curve.
        figure: Plotly ``Figure`` plotting loss vs log(LR) with the
            suggested LR marked.
    """

    lrs: list[float]
    losses: list[float]
    suggested_lr: float
    figure: go.Figure


def lr_finder(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    diverge_threshold: float = 4.0,
    device: Optional[torch.device] = None,
    ema_alpha: float = 0.5,
) -> LRFinderResult:
    """Sweep LRs exponentially from ``start_lr`` to ``end_lr`` and suggest
    a one-cycle ``max_lr``.

    Args:
        model: the network to sweep against. ``model.train()`` is
            called internally; both the original training-mode
            state AND the weights are restored on exit.
        train_loader: a DataLoader yielding ``(X, Y)`` batches the
            model can forward and ``loss_fn`` can score against. The
            loader is iterated, and if the sweep exceeds one epoch
            the loader is re-iterated from the start.
        loss_fn: callable ``(y_hat, Y) -> scalar Tensor`` for the
            per-batch loss. Same shape contract as torch loss
            functions.
        optimizer_cls: optimizer class. Adam by default; SGD also
            works for the sweep.
        start_lr: low end of the sweep range. Must be > 0.
        end_lr: high end of the sweep range. Must be > start_lr.
        num_iter: number of training iterations to run. Must be >= 2.
        diverge_threshold: stop the sweep early if loss exceeds
            ``diverge_threshold * min(observed losses)``. Default 4.
        device: device to move batches to. If None, inferred from
            the first model parameter.
        ema_alpha: smoothing coefficient for the loss curve before
            the steepest-descent search. Default 0.5.

    Returns:
        :class:`LRFinderResult` with the raw sweep data, the suggested
        max_lr, and a Plotly figure of loss vs log(LR).

    Raises:
        ValueError: on invalid arguments (``num_iter < 2``,
            ``start_lr <= 0``, ``end_lr <= start_lr``).
    """
    if num_iter < 2:
        raise ValueError(f"lr_finder num_iter must be >= 2, got {num_iter}")
    if start_lr <= 0:
        raise ValueError(f"lr_finder start_lr must be > 0, got {start_lr}")
    if end_lr <= start_lr:
        raise ValueError(f"lr_finder requires end_lr > start_lr, got start={start_lr} end={end_lr}")

    if device is None:
        device = next(model.parameters()).device

    # Snapshot for non-destructive restore on exit (mode + weights).
    was_training = model.training
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    optimizer = optimizer_cls(model.parameters(), lr=start_lr)
    lrs: list[float] = []
    losses: list[float] = []

    lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
    current_lr = start_lr

    model.train()
    iter_loader = iter(train_loader)
    for _ in range(num_iter):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        X, Y = batch[0].to(device), batch[1].to(device)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, Y)
        loss_val = float(loss.item())

        # Early-exit on divergence (loss balloons past diverge_threshold
        # × the best loss we've seen). Doing the check BEFORE the
        # backward+step keeps the diverging gradients out of the
        # parameter trajectory (which we restore anyway, but cheaper
        # to skip when we can).
        if losses and loss_val > diverge_threshold * min(losses):
            break

        loss.backward()
        optimizer.step()

        lrs.append(current_lr)
        losses.append(loss_val)
        current_lr *= lr_mult

    # Restore weights + training mode — sweep is fully non-destructive.
    model.load_state_dict(initial_state)
    model.train(was_training)

    suggested_lr = _suggest_lr(lrs, losses, ema_alpha) if lrs else start_lr

    fig = _build_figure(lrs, losses, suggested_lr)

    return LRFinderResult(
        lrs=lrs,
        losses=losses,
        suggested_lr=suggested_lr,
        figure=fig,
    )


def _suggest_lr(lrs: list[float], losses: list[float], ema_alpha: float) -> float:
    """Pick the LR at the steepest descent point of EMA-smoothed loss.

    Returns ``lrs[0]`` if the sweep was too short to identify a slope.
    """
    if len(losses) < 5:
        return lrs[0]

    # EMA smoothing reduces the noise impact on the slope estimate.
    smoothed: list[float] = []
    prev = losses[0]
    for loss in losses:
        prev = ema_alpha * loss + (1 - ema_alpha) * prev
        smoothed.append(prev)

    log_lrs = [math.log10(lr) for lr in lrs]
    slopes = [(smoothed[i + 1] - smoothed[i]) / (log_lrs[i + 1] - log_lrs[i]) for i in range(len(smoothed) - 1)]
    # Steepest negative slope = most descent per log-LR step. This is
    # the Smith (2017) heuristic — a value just before the loss
    # bottoms out is typically a safe one-cycle max_lr.
    idx = min(range(len(slopes)), key=lambda i: slopes[i])
    return lrs[idx]


def _build_figure(lrs: list[float], losses: list[float], suggested_lr: float) -> go.Figure:
    """Plotly loss-vs-log(LR) figure with the suggested LR marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lrs, y=losses, mode="lines+markers", name="loss"))
    fig.update_layout(
        title="LR finder — loss vs learning rate",
        xaxis_title="Learning rate (log)",
        yaxis_title="Loss",
        xaxis_type="log",
    )
    fig.add_vline(
        x=suggested_lr,
        line_dash="dash",
        annotation_text=f"suggested ≈ {suggested_lr:.2e}",
    )
    return fig
