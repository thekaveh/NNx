"""2:4 semi-structured sparsity via :mod:`torchao.sparsity`.

The 2:4 pattern keeps exactly 2 of every 4 consecutive weight entries
and zeros the other 2. NVIDIA's Sparse Tensor Cores (Ampere+) compute
the dense × 2:4-sparse matmul at roughly 2× the FLOP rate of the
dense × dense path — real wall-clock speedup, unlike unstructured
pruning whose zeros don't accelerate dense GEMM.

The implementation delegates to :func:`torchao.sparsity.sparsify_`,
passing :func:`torchao.sparsity.semi_sparse_weight` as the
per-layer transform plus an fnmatch-glob filter for layer selection.
After the swap, each matched :class:`nn.Linear`'s ``weight`` is a
:class:`torch.sparse.SparseSemiStructuredTensor` subclass that the
PyTorch sparse Tensor-Core kernel dispatches to under the hood.

**Hardware requirement.** The underlying kernel only supports CUDA
tensors on Ampere (sm_80) or newer architectures. CPU / pre-Ampere
calls raise :class:`RuntimeError` at the sparse-tensor construction
site (inside torchao / torch.sparse). Users running on unsupported
hardware should either skip pruning altogether or use
:func:`nnx.prune.magnitude_prune` for the pure-compression path.
"""

from __future__ import annotations

import fnmatch

from torch import nn


def semi_structured_24(net: nn.Module, *, layer_pattern: str = "*") -> int:
    """Swap each matched :class:`nn.Linear`'s weight with a 2:4
    semi-structured sparse tensor via :func:`torchao.sparsity.sparsify_`.

    Args:
        net: root module to walk. The function mutates ``net`` in place.
        layer_pattern: fnmatch glob against dotted submodule name.
            ``"*"`` (the default) matches every :class:`nn.Linear`.

    Returns:
        The number of :class:`nn.Linear` submodules that were swapped.
        ``0`` if ``layer_pattern`` matched nothing (in which case the
        underlying ``torchao.sparsity.sparsify_`` is NOT invoked — this
        avoids an unnecessary torchao dispatch and the CUDA-only kernel
        error on CPU runners with no Linear targets to swap).

    Raises:
        ImportError: if ``torchao`` isn't installed. The import happens
            inside the function body so :mod:`nnx.prune` doesn't pull
            torchao at package-import time; users on the magnitude-only
            path pay no dep cost.
        RuntimeError: surfaced from the underlying
            ``torch.sparse.SparseSemiStructuredTensor`` constructor on
            unsupported hardware (CPU / pre-Ampere GPU) or on weights
            whose inner dimension isn't a multiple of 4. The error
            originates in torch / torchao; we don't intercept it.

    **Pattern semantics:** same fnmatch convention as
    :func:`nnx.peft.apply_lora_to` and
    :func:`nnx.prune.magnitude_prune` — dotted submodule names against
    shell wildcards. Only :class:`nn.Linear` submodules are eligible
    (Conv2d / BatchNorm / Embedding / etc. are skipped even under a
    wildcard pattern).

    **Note on weights:** ``torchao.sparsity.sparsify_`` does NOT enforce
    the 2:4 mask before the swap. Callers are expected to either
    (a) magnitude-prune the weight to a valid 2:4 pattern beforehand
    (via :func:`magnitude_prune` or a custom mask), or
    (b) accept whatever 2:4 approximation
    :func:`torch.sparse.to_sparse_semi_structured` picks (which keeps
    the top-2-by-absolute-value entries per 4-group). For training
    workflows, the standard recipe is to pre-mask, then train the
    surviving entries.
    """
    # Defer the torchao import so consumers of `nnx.prune.magnitude_prune`
    # don't pay the torchao dep cost just by importing the package.
    import torchao.sparsity as ao_sparsity

    # Two-phase: enumerate the matched-Linear set FIRST so we can
    # return the count and short-circuit when there's nothing to do.
    # The same name list also drives the filter_fn we pass to torchao.
    targets: set[str] = set()
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Linear) and fnmatch.fnmatch(name, layer_pattern):
            targets.add(name)

    if not targets:
        # No-op fast path: avoid invoking torchao at all when the
        # pattern matched nothing. Keeps the CPU-only error path off
        # the call graph for users who want to dry-run their pattern.
        return 0

    def _filter(mod: nn.Module, fqn: str) -> bool:
        # torchao calls this for every submodule. Accept only the
        # nn.Linear instances whose dotted name made the targets set
        # — re-checking the isinstance here is defensive (a future
        # torchao change could pass non-Linear modules through the
        # filter for non-Linear sparsity workflows).
        return isinstance(mod, nn.Linear) and fqn in targets

    ao_sparsity.sparsify_(net, ao_sparsity.semi_sparse_weight(), _filter)
    return len(targets)
