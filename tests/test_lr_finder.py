"""Tests for ``nnx.lr_finder`` — fastai-style exponential LR sweep."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import lr_finder
from nnx.lr_finder import LRFinderResult


def _tiny_model_and_loader():
    """Tiny 3-class FFN with synthetic data, suitable for a 50-iter sweep."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    X = torch.randn(64, 4)
    y = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
    return model, loader


def test_lr_finder_returns_LRFinderResult():
    """Return type contract."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert isinstance(result, LRFinderResult)


def test_lr_finder_result_has_required_fields():
    """LRFinderResult exposes lrs, losses, suggested_lr, figure."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert isinstance(result.lrs, list)
    assert isinstance(result.losses, list)
    assert isinstance(result.suggested_lr, float)
    assert isinstance(result.figure, go.Figure)


def test_lr_finder_lrs_and_losses_same_length():
    """One loss recorded per LR; lengths must match."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert len(result.lrs) == len(result.losses)
    assert len(result.lrs) >= 1  # may early-exit on divergence


def test_lr_finder_suggested_lr_in_range():
    """suggested_lr falls within the swept range."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert 1e-6 <= result.suggested_lr <= 1.0


def test_lr_finder_restores_model_weights():
    """Model state is non-destructively restored after the sweep."""
    model, loader = _tiny_model_and_loader()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}

    lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )

    for k, v in model.state_dict().items():
        assert torch.equal(v, initial[k]), f"weight {k} was not restored after lr_finder"


def test_lr_finder_rejects_invalid_num_iter():
    """num_iter < 2 raises ValueError."""
    model, loader = _tiny_model_and_loader()
    with pytest.raises(ValueError, match="num_iter"):
        lr_finder(
            model,
            loader,
            loss_fn=nn.functional.cross_entropy,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=1,
        )


def test_lr_finder_rejects_inverted_lr_range():
    """end_lr <= start_lr raises ValueError."""
    model, loader = _tiny_model_and_loader()
    with pytest.raises(ValueError, match="start_lr"):
        lr_finder(
            model,
            loader,
            loss_fn=nn.functional.cross_entropy,
            start_lr=1.0,
            end_lr=1e-6,
            num_iter=30,
        )


def test_lr_finder_rejects_nonpositive_start_lr():
    """start_lr <= 0 raises ValueError."""
    model, loader = _tiny_model_and_loader()
    with pytest.raises(ValueError, match="start_lr"):
        lr_finder(
            model,
            loader,
            loss_fn=nn.functional.cross_entropy,
            start_lr=0.0,
            end_lr=1.0,
            num_iter=30,
        )


def test_lr_finder_figure_has_log_x_axis():
    """The plotted x-axis is log-scaled (the standard fastai LR-finder view)."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )
    assert result.figure.layout.xaxis.type == "log"


def test_lr_finder_restores_training_mode():
    """Caller's `model.training` is snapshotted and restored on exit.

    The docstring guarantees both weights AND the training-mode flag
    come back exactly as the caller passed them. A previous review
    caught the gap where the function unconditionally `model.train()`'d
    without restoring an `eval()` caller's state.
    """
    model, loader = _tiny_model_and_loader()

    # Caller starts in eval() mode.
    model.eval()
    assert model.training is False

    lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )

    assert model.training is False, "lr_finder left model in train() after eval() caller"

    # And the symmetric case: caller in train() should stay in train().
    model.train()
    assert model.training is True

    lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=30,
    )

    assert model.training is True


def _run_lr_finder_exception_path_restore_check(initial_mode: str) -> None:
    """Drive the lr_finder exception-path test for either initial mode.

    Shared body of the two `test_lr_finder_restores_state_after_exception_*`
    cases; parameterised by the caller's starting training-mode so we
    cover both `train() → exception → restore-to-train()` and
    `eval()  → exception → restore-to-eval()` round-trips with the same
    flaky-loss-fn scaffolding.
    """
    model, loader = _tiny_model_and_loader()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if initial_mode == "eval":
        model.eval()
        assert model.training is False
    elif initial_mode == "train":
        model.train()
        assert model.training is True
    else:  # pragma: no cover — caller-side typo guard.
        raise AssertionError(f"bad initial_mode {initial_mode!r}")

    call_count = {"n": 0}

    def flaky_loss_fn(y_hat, y):
        call_count["n"] += 1
        if call_count["n"] == 4:
            raise RuntimeError("simulated loss_fn failure on iteration 4")
        return nn.functional.cross_entropy(y_hat, y)

    with pytest.raises(RuntimeError, match="simulated loss_fn failure"):
        lr_finder(
            model,
            loader,
            loss_fn=flaky_loss_fn,
            start_lr=1e-6,
            end_lr=1.0,
            num_iter=30,
        )

    # Weights must be restored despite the mid-sweep crash.
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial[k]), (
            f"weight {k} was not restored after lr_finder raised mid-sweep — "
            "the non-destructive contract was broken on the exception path"
        )

    # Training-mode flag must round-trip to the caller's value.
    expected = initial_mode == "train"
    assert model.training is expected, (
        f"lr_finder left model.training={model.training} after exception with "
        f"initial_mode={initial_mode!r} — training-mode restore was skipped on the raise path"
    )


def test_lr_finder_restores_state_after_exception_in_loss_fn_from_eval():
    """Non-destructive restore on the exception path, eval()-mode caller.

    The original implementation only restored after the loop completed,
    so a `loss_fn` that crashed on iteration 3 of 30 would leave the
    caller's model weights silently mutated (the optimizer steps from
    iters 0-2 stayed live) and `model.training` left True even if the
    caller passed in an `eval()` model.

    The fix is a `try/finally` around the loop with restore in the
    `finally` clause. This test pins the eval()-mode round-trip; the
    sibling `_from_train` test covers the symmetric path.
    """
    _run_lr_finder_exception_path_restore_check(initial_mode="eval")


def test_lr_finder_restores_state_after_exception_in_loss_fn_from_train():
    """Non-destructive restore on the exception path, train()-mode caller.

    Symmetric to the `_from_eval` case: a caller in `model.train()` must
    end the call still in `model.train()` after a mid-sweep exception —
    the `try/finally` restore preserves whatever the caller's pre-call
    mode was, not just `eval`.
    """
    _run_lr_finder_exception_path_restore_check(initial_mode="train")


def test_suggest_lr_short_sweep_returns_lr_at_min_loss():
    """When the sweep is too short for a slope estimate (<5 points),
    `_suggest_lr` falls back to the LR at the minimum observed loss
    rather than `lrs[0]` (the lowest swept LR, which is the worst
    possible suggestion). Direct unit test of the fallback path.
    """
    from nnx.lr_finder import _suggest_lr

    # Four-point sweep — below the 5-point threshold.
    lrs = [1e-5, 1e-4, 1e-3, 1e-2]
    losses = [2.0, 1.5, 0.8, 1.2]  # minimum at index 2 (lr=1e-3)
    assert _suggest_lr(lrs, losses, ema_alpha=0.5) == 1e-3


def test_suggest_lr_monotonically_increasing_loss():
    """When the loss only ever rises across the sweep (no descent
    region at all), the slope-based heuristic would return the
    "least bad" point. The fallback instead returns the LR at the
    minimum observed loss — the first iteration when nothing improves.
    """
    from nnx.lr_finder import _suggest_lr

    lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    # Monotonically increasing — no descent anywhere.
    losses = [0.5, 0.6, 0.7, 0.9, 1.3, 2.0]
    assert _suggest_lr(lrs, losses, ema_alpha=0.5) == 1e-7


def test_lr_finder_early_exits_on_divergence():
    """When `loss_fn` diverges (returns escalating values), the sweep
    stops before exhausting `num_iter`. Verifies the EMA-smoothed
    divergence guard (`smoothed_loss > diverge_threshold * smoothed_min`)
    at the top of the loop actually fires.
    """
    model, loader = _tiny_model_and_loader()

    # A divergent loss: scales with iteration count by storing state on
    # a closure-local counter that grows each call.
    counter = {"n": 0}

    def diverging_loss(y_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        counter["n"] += 1
        # First call returns a small loss (sets the min); subsequent calls
        # explode geometrically, blowing past `diverge_threshold * min`
        # within a couple of iterations.
        scale = 1.0 if counter["n"] == 1 else 100.0 ** counter["n"]
        return (y_hat * 0 + scale).sum()

    result = lr_finder(
        model,
        loader,
        loss_fn=diverging_loss,
        start_lr=1e-6,
        end_lr=1.0,
        num_iter=100,
        diverge_threshold=4.0,
    )

    # Sweep should have early-exited well before num_iter=100.
    assert len(result.lrs) < 100
    assert len(result.lrs) >= 1


def test_lr_finder_sweep_reaches_end_lr():
    """The docstring promises a sweep "from start_lr to end_lr"; the old
    1/num_iter exponent stopped one multiplicative step short of
    end_lr."""
    model, loader = _tiny_model_and_loader()
    result = lr_finder(
        model,
        loader,
        loss_fn=nn.functional.cross_entropy,
        start_lr=1e-5,
        end_lr=1.0,
        num_iter=10,
    )
    # Divergence can end the sweep early; when it runs to completion the
    # last LR must hit end_lr (within float noise).
    if len(result.lrs) == 10:
        assert abs(result.lrs[-1] - 1.0) / 1.0 < 1e-6


def test_lr_finder_restores_global_rng_state():
    """The non-destructive contract covers reproducibility: a seeded
    pipeline must produce identical draws whether or not an lr_finder
    pre-flight ran. Pre-fix, the sweep leaked its dropout draws (the
    helper itself calls .train()) and the DataLoader base-seed draw
    into the caller's RNG stream, so seed → lr_finder → train diverged
    from seed → train."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 3))
    X = torch.randn(64, 4)
    y = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)

    torch.manual_seed(7)
    expected = torch.randn(4)
    torch.manual_seed(7)
    lr_finder(model, loader, loss_fn=nn.functional.cross_entropy, num_iter=5)
    assert torch.equal(torch.randn(4), expected)


def test_lr_finder_restores_loader_attached_generator_state():
    """The PyTorch reproducibility recipe attaches a seeded generator to
    the DataLoader (or an explicit sampler); the sweep's iteration draws
    its shuffle permutations from THAT stream, not global RNG. Pre-fix,
    the global-RNG restore left loader-attached generators advanced, so
    pipelines using generator= still diverged after a pre-flight."""
    from torch.utils.data import RandomSampler

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    ds = TensorDataset(torch.randn(64, 4), torch.randint(0, 3, (64,)))

    gen = torch.Generator().manual_seed(42)
    loader = DataLoader(ds, batch_size=8, shuffle=True, generator=gen)
    state = gen.get_state()
    lr_finder(model, loader, loss_fn=nn.functional.cross_entropy, num_iter=5)
    assert torch.equal(gen.get_state(), state)

    # Same contract when the generator rides on an explicit sampler
    # (loader.generator stays None on this construction).
    gen2 = torch.Generator().manual_seed(43)
    loader2 = DataLoader(ds, batch_size=8, sampler=RandomSampler(ds, generator=gen2))
    state2 = gen2.get_state()
    lr_finder(model, loader2, loss_fn=nn.functional.cross_entropy, num_iter=5)
    assert torch.equal(gen2.get_state(), state2)

    # And when it hides inside an explicit batch_sampler= — torch fills
    # loader.sampler with a dummy SequentialSampler on this form, so the
    # live stream sits at loader.batch_sampler.sampler.generator.
    from torch.utils.data import BatchSampler

    gen3 = torch.Generator().manual_seed(44)
    loader3 = DataLoader(ds, batch_sampler=BatchSampler(RandomSampler(ds, generator=gen3), 8, False))
    state3 = gen3.get_state()
    lr_finder(model, loader3, loss_fn=nn.functional.cross_entropy, num_iter=5)
    assert torch.equal(gen3.get_state(), state3)
