"""Low-rank SVD factorization of ``nn.Linear``.

A Linear with weight ``W`` of shape ``(out, in)`` can be factored as
``W ≈ U_k @ S_k @ V_k.T`` where ``U_k`` keeps the top-k left singular
vectors, ``V_k`` the top-k right singular vectors, and ``S_k`` the
top-k singular values. The factored layer is returned as

    Sequential(
        nn.Linear(in, k, bias=False),   # weight = (S_k @ V_k.T)
        nn.Linear(k, out, bias=...),    # weight = U_k, bias preserved
    )

so parameter count drops from ``out * in + bias`` to
``k * (in + out) + bias``. When ``k >= min(out, in)`` the
factorization is exact (within FP rounding) — that's the
function-preservation contract the first test verifies. For
``k < min(out, in)`` the surged forward is an approximation; the
approximation error is bounded by the energy of the discarded singular
values.

Naive truncation; activation-SVD / Fisher-weighted variants are
unshipped follow-ups.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import skip_init

_SUPPORTED_METHODS = ("svd",)


def low_rank_factorize(
    linear: nn.Linear,
    *,
    rank: int,
    method: str = "svd",
) -> nn.Sequential:
    """Factor a Linear into two smaller Linears via rank-``k`` SVD
    truncation.

    Args:
        linear: an :class:`nn.Linear` to factorize. Its weights are
            read but not mutated — the returned Sequential is a fresh
            pair of Linears.
        rank: the truncation rank ``k``. Must be in ``[1, min(out, in)]``.
            When ``k == min(out, in)`` the factorization is exact.
        method: ``"svd"`` (the only option in v1). Reserved for the
            future ``"activation_svd"`` / ``"fisher"`` variants.

    Returns:
        :class:`nn.Sequential` of two Linears whose composition
        approximates the input layer. The first Linear has
        ``bias=False`` (Sx@V.T has no native bias term); the second
        Linear carries the original bias verbatim.

    Raises:
        TypeError: if ``linear`` is not :class:`nn.Linear`.
        ValueError: if ``rank`` is out of range, or ``method`` unknown.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"low_rank_factorize requires an nn.Linear, got {type(linear).__name__}")
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"unsupported method {method!r}; supported: {_SUPPORTED_METHODS}")

    out_features = linear.out_features
    in_features = linear.in_features
    max_rank = min(out_features, in_features)
    if not (1 <= rank <= max_rank):
        raise ValueError(f"rank must be in [1, min(out_features, in_features) = {max_rank}]; got {rank}")

    # Run SVD on the float64-promoted weight to keep the reconstruction
    # numerically clean for the "exact at max rank" test. The factored
    # Linears are downcast back to the original dtype before return.
    W = linear.weight.data
    orig_dtype = W.dtype
    U, S, Vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    U_k = U[:, :rank]  # (out, k)
    S_k = S[:rank]  # (k,)
    Vh_k = Vh[:rank, :]  # (k, in)

    # First Linear: in → k. Its weight (k, in) is (S_k * Vh_k); no bias
    # since the original bias is added after the second matmul.
    down_weight = (S_k.unsqueeze(1) * Vh_k).to(orig_dtype)
    # device= threads the original layer's placement — without it a
    # CUDA-resident model gets CPU layers spliced in and the next
    # forward crashes with a device mismatch (widen already does this).
    # skip_init: every param is fully overwritten, so meta-device
    # construction keeps the surgery off the global RNG stream.
    down = skip_init(nn.Linear, in_features, rank, bias=False, dtype=orig_dtype, device=W.device)
    down.weight.data.copy_(down_weight)

    # Second Linear: k → out. Weight is U_k. Bias is the original.
    up_weight = U_k.to(orig_dtype)
    up = skip_init(nn.Linear, rank, out_features, bias=linear.bias is not None, dtype=orig_dtype, device=W.device)
    up.weight.data.copy_(up_weight)
    if linear.bias is not None:
        up.bias.data.copy_(linear.bias.data)

    return nn.Sequential(down, up)
