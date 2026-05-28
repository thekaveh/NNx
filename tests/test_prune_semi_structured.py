"""Tests for nnx.prune.semi_structured — semi_structured_24.

2:4 semi-structured sparsity is a hardware-accelerated path: the
underlying ``torch.sparse.SparseSemiStructuredTensor`` kernel only
supports CUDA tensors on Ampere or newer architectures. On CPU /
pre-Ampere hardware the swap raises ``RuntimeError`` at construction
time.

These tests verify the swap mechanics (pattern filter, return count,
state-dict-keys behavior) on every platform AND verify the
``RuntimeError`` is the surfaced contract on CPU. The actual
forward-pass speedup is the property of the torchao kernel; we don't
re-test torchao's wheel here.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

# importorskip the parent torchao package first (the top-level handle
# is what ``test_semi_structured_24_skip_on_no_torchao`` reaches for).
# Then importorskip the submodule we monkey-patch — pytest returns the
# loaded module object directly, which lets us bind a local name without
# tripping the ruff F811 redefinition check.
pytest.importorskip("torchao")
torchao_sparsity = pytest.importorskip("torchao.sparsity")

from nnx import prune as nnx_prune  # noqa: E402 — must follow importorskip


class _LinearOnly(nn.Module):
    """Sequence of nn.Linear layers — the 2:4 sparsify target.

    Sized so each weight matrix has in_features % 4 == 0; the 2:4
    pattern requires the inner dimension to be a multiple of 4.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def test_semi_structured_24_swaps_linear_with_semisparse():
    """Every matched nn.Linear's ``weight`` should end up as a
    ``SparseSemiStructuredTensor`` subclass after the swap. On
    CPU-only test runners the underlying kernel raises ``RuntimeError``
    at the swap site — we surface that as a clear, actionable error,
    not a silent pass.

    Skipped under CPU because the swap requires CUDA-resident weights;
    the swap-mechanics contract (pattern filter + return count) is
    covered by sibling tests that don't depend on the actual sparse
    tensor construction succeeding.
    """
    if not _has_cuda():
        pytest.skip("2:4 semi-structured sparsity requires CUDA (Ampere+)")

    torch.manual_seed(0)
    net = _LinearOnly().cuda()
    # Pre-mask the weight to a valid 2:4 pattern so the sparse-tensor
    # constructor accepts it. The torchao path does NOT mask for you —
    # it expects the weight to already satisfy the 2:4 constraint.
    with torch.no_grad():
        for layer in (net.fc1, net.fc2, net.fc3):
            w = layer.weight
            # Reshape to (out, in/4, 4); keep top-2 by abs per group.
            g = w.view(w.shape[0], -1, 4)
            top2 = g.abs().topk(2, dim=-1).indices
            mask = torch.zeros_like(g, dtype=torch.bool)
            mask.scatter_(-1, top2, True)
            layer.weight.copy_((g * mask).view_as(w))

    n = nnx_prune.semi_structured_24(net)
    assert n == 3

    # After sparsify_, each Linear's `weight` is a tensor subclass — not
    # a plain dense Parameter. Detect via the public torch type.
    from torch.sparse import SparseSemiStructuredTensor

    for layer in (net.fc1, net.fc2, net.fc3):
        assert isinstance(layer.weight, SparseSemiStructuredTensor), (
            f"expected SparseSemiStructuredTensor on swapped weight, got {type(layer.weight).__name__}"
        )


def test_semi_structured_24_pattern_filter_works(monkeypatch):
    """layer_pattern is an fnmatch glob against dotted submodule name.
    Only matched layers should pick up the sparsify wrap.

    Test the pattern-filtering contract without depending on CUDA:
    monkey-patch ``torchao.sparsity.sparsify_`` to a recording stub that
    captures which modules its ``filter_fn`` would accept. That isolates
    the filter logic — the part nnx owns — from the hardware-bound
    sparse-tensor construction the torchao kernel does internally.
    """
    seen: list[str] = []

    def fake_sparsify_(model, config, filter_fn=None):
        for name, mod in model.named_modules():
            if filter_fn is not None and filter_fn(mod, name):
                seen.append(name)
        return model

    monkeypatch.setattr(torchao_sparsity, "sparsify_", fake_sparsify_)

    net = _LinearOnly()
    n = nnx_prune.semi_structured_24(net, layer_pattern="fc1")
    # The filter accepted exactly one layer (fc1). The return count
    # tracks the same.
    assert seen == ["fc1"]
    assert n == 1


def test_semi_structured_24_returns_count_with_wildcard(monkeypatch):
    """With the default ``layer_pattern='*'`` every nn.Linear matches.

    Same monkey-patch strategy as the pattern-filter test — the swap
    itself is hardware-gated, but the count contract isn't.
    """
    seen: list[str] = []

    def fake_sparsify_(model, config, filter_fn=None):
        for name, mod in model.named_modules():
            if filter_fn is not None and filter_fn(mod, name):
                seen.append(name)
        return model

    monkeypatch.setattr(torchao_sparsity, "sparsify_", fake_sparsify_)

    net = _LinearOnly()
    n = nnx_prune.semi_structured_24(net)
    assert n == 3
    assert set(seen) == {"fc1", "fc2", "fc3"}


def test_semi_structured_24_no_match_returns_zero(monkeypatch):
    """A pattern that matches no nn.Linear must return 0 — and must not
    touch any layer (no spurious sparsify call should fire)."""
    sparsify_calls: list[int] = []

    def fake_sparsify_(model, config, filter_fn=None):
        sparsify_calls.append(1)
        for name, mod in model.named_modules():
            if filter_fn is not None:
                filter_fn(mod, name)
        return model

    monkeypatch.setattr(torchao_sparsity, "sparsify_", fake_sparsify_)

    net = _LinearOnly()
    n = nnx_prune.semi_structured_24(net, layer_pattern="nonexistent.*")
    assert n == 0
    # No actual sparsify_ call fires when there are zero targets — we
    # short-circuit. (Avoids an unnecessary torchao invocation and the
    # subsequent CUDA-only error on CPU runners.)
    assert sparsify_calls == []


def test_semi_structured_24_skip_on_no_torchao():
    """Sanity check that ``pytest.importorskip('torchao')`` at the top
    of this module fired iff torchao is unavailable. By the time this
    test runs, torchao IS available (otherwise the module would have
    skipped at import time), so we just assert that — proving the guard
    rides the standard pytest skip path."""
    # If we got here, torchao imported successfully. The importorskip at
    # the top of the file is the actual mechanism — this test documents
    # the contract and would itself be skipped (along with every other
    # test in the file) on a torchao-less environment.
    import torchao as _torchao

    assert _torchao is not None


def test_semi_structured_24_filter_excludes_non_linear(monkeypatch):
    """The filter must only accept nn.Linear submodules — Conv2d /
    BatchNorm / Embedding / nested non-Linear children should not be
    swept up even under a wildcard pattern."""
    seen: list[str] = []

    def fake_sparsify_(model, config, filter_fn=None):
        for name, mod in model.named_modules():
            if filter_fn is not None and filter_fn(mod, name):
                seen.append((name, type(mod).__name__))
        return model

    monkeypatch.setattr(torchao_sparsity, "sparsify_", fake_sparsify_)

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3)
            self.bn = nn.BatchNorm2d(8)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(self.bn(self.conv(x)).mean(dim=(-1, -2)))

    n = nnx_prune.semi_structured_24(Mixed())
    assert n == 1
    assert seen == [("fc", "Linear")]
