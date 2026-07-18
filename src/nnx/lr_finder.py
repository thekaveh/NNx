"""fastai-style learning-rate finder.

Runs an exponential LR sweep from ``start_lr`` to ``end_lr`` over
``num_iter`` training iterations, recording loss at each step. The
recommended ``max_lr`` is the LR at the steepest descent point of the
smoothed loss curve — the classic Smith (2017) heuristic.

The sweep is **non-destructive**: the model's initial weights AND every
RNG stream the sweep consumes (global CPU, the active device's, and any
loader/sampler-attached torch ``generator=``) are snapshotted before
the sweep starts and restored on exit, so the caller can use this as a
pre-flight check before the real training run without disturbing any
subsequent reproducibility. (A non-torch stream a custom sampler may
carry — e.g. a numpy ``Generator`` — is skipped: it stays caller-owned.)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, cast

import plotly.graph_objects as go
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True, kw_only=True, slots=True)
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
            called internally; the original training-mode state, the
            weights, AND the RNG state are all restored on exit.
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
        diverge_threshold: stop the sweep early when the EMA-smoothed
            loss exceeds ``diverge_threshold * smoothed_min`` (the
            minimum EMA-smoothed loss observed so far). Default 4. The
            smoothed check matches fastai's lr_find heuristic — using
            the raw ``min(losses)`` would let a single anomalous low
            first-batch loss pull the threshold too tight and abort
            the sweep prematurely.
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

    # Snapshot for non-destructive restore on exit (mode + weights + RNG).
    # The sweep consumes RNG (dropout under .train(), the DataLoader's
    # base-seed draw) — without the RNG restore, a seeded pipeline that
    # ran lr_finder as a pre-flight diverged from the same pipeline
    # without it, breaking the module docstring's reproducibility claim.
    was_training = model.training
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cpu_rng_state = torch.get_rng_state()
    device_rng_state: Optional[torch.Tensor] = None
    if device.type == "cuda":
        device_rng_state = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        device_rng_state = torch.mps.get_rng_state()
    # A loader built per the PyTorch reproducibility recipe draws its
    # shuffle permutation and worker base-seed from its OWN generator
    # (DataLoader(generator=...) or an explicit sampler's), not global
    # RNG — a third stream the global restore can't cover.
    loader_generators = (
        getattr(train_loader, "generator", None),
        getattr(getattr(train_loader, "sampler", None), "generator", None),
        # An explicit batch_sampler= hides its stream one level deeper —
        # torch fills loader.sampler with a dummy SequentialSampler then.
        getattr(getattr(getattr(train_loader, "batch_sampler", None), "sampler", None), "generator", None),
        # A custom batch sampler may own its generator directly.
        getattr(getattr(train_loader, "batch_sampler", None), "generator", None),
    )
    # isinstance filter: an exotic sampler may carry e.g. a numpy
    # Generator under the same attribute name — no get_state(), and not
    # a stream this helper can restore; it stays caller-owned.
    loader_gen_states = [
        (g, g.get_state()) for g in dict.fromkeys(g for g in loader_generators if isinstance(g, torch.Generator))
    ]
    # persistent_workers caches the first iterator ON the loader: if the
    # sweep creates it, the caller's first epoch would _reset() that
    # cache instead of drawing a fresh worker base seed, shifting their
    # batch stream even though every RNG snapshot here is restored. The
    # finally discards only an iterator the sweep itself created.
    had_cached_iterator = getattr(train_loader, "_iterator", None) is not None

    optimizer = cast(Any, optimizer_cls)(model.parameters(), lr=start_lr)
    lrs: list[float] = []
    losses: list[float] = []

    # num_iter - 1: with num_iter points the LAST one must land on
    # end_lr exactly (1/num_iter stopped one multiplicative step short
    # of the documented sweep ceiling). num_iter >= 2 is enforced above.
    lr_mult = (end_lr / start_lr) ** (1.0 / (num_iter - 1))
    current_lr = start_lr

    # Running EMA-smoothed loss and its minimum. The divergence check
    # compares the current smoothed loss against the smoothed minimum
    # rather than the raw min(losses) — an anomalously low or noisy
    # first-batch loss would otherwise pull the divergence threshold
    # too tight and abort the sweep prematurely. fastai's lr_find uses
    # the same smoothed-min heuristic.
    smoothed_loss: Optional[float] = None
    smoothed_min: Optional[float] = None

    # try/finally guarantees the snapshot is restored even if a user-
    # supplied loss_fn raises, model(X) crashes, backward fails on a
    # NaN, etc. Without it the docstring's "non-destructive" contract
    # held only on the happy path — the caller's model would silently
    # remain modified on any mid-sweep exception.
    try:
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

            smoothed_loss = (
                loss_val if smoothed_loss is None else ema_alpha * loss_val + (1 - ema_alpha) * smoothed_loss
            )
            smoothed_min = smoothed_loss if smoothed_min is None else min(smoothed_min, smoothed_loss)

            # Early-exit on divergence (smoothed loss balloons past
            # diverge_threshold × smoothed minimum). Doing the check BEFORE
            # the backward+step keeps the diverging gradients out of the
            # parameter trajectory (which we restore anyway, but cheaper to
            # skip when we can).
            if smoothed_min is not None and smoothed_loss > diverge_threshold * smoothed_min:
                break

            loss.backward()
            optimizer.step()

            lrs.append(current_lr)
            losses.append(loss_val)
            current_lr *= lr_mult
    finally:
        # Restore weights + training mode + RNG — sweep is fully
        # non-destructive regardless of whether the loop completed,
        # early-exited on divergence, or raised mid-iter.
        model.load_state_dict(initial_state)
        model.train(was_training)
        torch.set_rng_state(cpu_rng_state)
        if device.type == "cuda":
            assert device_rng_state is not None
            torch.cuda.set_rng_state(device_rng_state, device)
        elif device.type == "mps":
            assert device_rng_state is not None
            torch.mps.set_rng_state(device_rng_state)
        for g, s in loader_gen_states:
            g.set_state(s)
        if not had_cached_iterator and getattr(train_loader, "_iterator", None) is not None:
            # Dropping the reference shuts the persistent workers down
            # via the iterator's finalizer.
            train_loader._iterator = None

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

    Fallbacks (in order):

    1. Short sweep (``len(losses) < 5``): no slope estimate is
       reliable; return the LR at the minimum observed loss. This is
       a defensible "best LR we actually saw" answer, much safer than
       returning ``lrs[0]`` (= ``start_lr`` = the lowest swept LR =
       guaranteed to underfit).
    2. Steepest slope is non-negative (loss only went up): there's no
       descent region. Same fallback — LR at minimum observed loss.
    3. Otherwise: the Smith (2017) heuristic — LR at the steepest
       negative slope of the EMA-smoothed loss curve on a log-LR
       axis. A value just before the loss bottoms out is typically a
       safe one-cycle ``max_lr``.
    """
    if len(losses) < 5:
        return _lr_at_min_loss(lrs, losses)

    # EMA smoothing reduces the noise impact on the slope estimate.
    smoothed: list[float] = []
    prev = losses[0]
    for loss in losses:
        prev = ema_alpha * loss + (1 - ema_alpha) * prev
        smoothed.append(prev)

    log_lrs = [math.log10(lr) for lr in lrs]
    slopes = [(smoothed[i + 1] - smoothed[i]) / (log_lrs[i + 1] - log_lrs[i]) for i in range(len(smoothed) - 1)]
    # Steepest negative slope = most descent per log-LR step.
    idx = min(range(len(slopes)), key=lambda i: slopes[i])
    if slopes[idx] >= 0:
        # No descent anywhere — loss only rose. Fall back to LR at the
        # minimum observed (smoothed) loss as the safest suggestion.
        return _lr_at_min_loss(lrs, losses)
    return lrs[idx]


def _lr_at_min_loss(lrs: list[float], losses: list[float]) -> float:
    """LR at the minimum observed loss. Used as the fallback when the
    slope-based heuristic can't return a confident answer."""
    idx = min(range(len(losses)), key=lambda i: losses[i])
    return lrs[idx]


def _build_figure(lrs: list[float], losses: list[float], suggested_lr: float) -> go.Figure:
    """Plotly loss-vs-log(LR) figure with the suggested LR marked.

    Omits the suggested-LR vline if the sweep collected no data
    points (every iteration diverged immediately) — adding a vline
    against an empty trace produces a degenerate figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lrs, y=losses, mode="lines+markers", name="loss"))
    fig.update_layout(
        title="LR finder — loss vs learning rate",
        xaxis_title="Learning rate",
        yaxis_title="Loss",
        xaxis_type="log",
    )
    if lrs:
        fig.add_vline(
            x=suggested_lr,
            line_dash="dash",
            annotation_text=f"suggested ≈ {suggested_lr:.2e}",
        )
    return fig
