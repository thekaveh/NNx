"""Sparse Mixture-of-Experts — :class:`MoELinear` drop-in for ``nn.Linear``.

A MoE layer routes each input token through a *subset* of expert
sub-networks instead of evaluating the full layer end-to-end. The
router (a tiny linear projection) emits ``num_experts`` logits per
token; the top-``k`` experts are selected and their outputs are
combined with softmax-weighted sums. Total parameter count grows
with the number of experts, but per-token FLOPs stay roughly constant
(only ``k`` of ``N`` experts run per token).

Tutorial-grade implementation: a single :class:`MoELinear` layer that
wraps a router + a :class:`torch.nn.ModuleList` of experts. The
forward pass is written for clarity over throughput — it iterates
``top_k × num_experts`` times and dispatches each token group through
the matching expert. Production-scale MoE (MegaBlocks block-sparse
kernels, expert parallelism across GPUs, token-dropping for capacity
factor) is **out of scope**: it would be hollow wrapping over highly
specialized libraries and obscure the mechanism this module is meant
to teach.

The layer exposes ``.last_aux_loss`` after each forward — the
Switch-Transformer load-balancing penalty
``α · N · Σ f_i · P_i`` (`fedus:switch`), where ``f_i`` is the
fraction of dispatched tokens routed to expert ``i`` and ``P_i`` is
the mean router probability for ``i``. The penalty is minimized
(equal to 1) when routing is perfectly uniform across experts. A
companion paradigm factory in :mod:`nnx.paradigms.moe` reads this
attribute across every :class:`MoELinear` in the net and adds it to
the main supervised loss with a configurable weight.
"""

from __future__ import annotations

import torch
from torch import nn


class MoELinear(nn.Module):
    """Sparse top-k Mixture-of-Experts drop-in for :class:`nn.Linear`.

    Forward pass:

      1. Router (a bias-less :class:`nn.Linear`) projects input
         ``(B, in_features) → (B, num_experts)`` logits.
      2. ``top_k`` largest logits per row are kept; a softmax over
         those ``k`` values produces the per-expert gating weight.
      3. Each token is dispatched to its top-``k`` experts; expert
         outputs are weighted by the gating weights and summed into
         the output tensor.
      4. ``self.last_aux_loss`` is populated with the Switch-style
         load-balancing penalty
         ``num_experts · Σ_i f_i · P_i``. This is a scalar tensor with
         gradients wired to the router so optimization of the main
         loss + this term pushes routing toward uniform expert usage.

    Args:
        in_features: input feature dimension (matches ``nn.Linear``).
        out_features: output feature dimension (matches ``nn.Linear``).
        num_experts: number of expert sub-networks. Must be ≥ 2.
            (``num_experts=1`` collapses to a plain linear with extra
            book-keeping; the layer rejects it to surface the misuse.)
        top_k: number of experts each input is routed through. Must
            be ≥ 1 and ≤ ``num_experts``. Defaults to 2 — the
            Switch-Transformer paper uses ``k=1``, but ``k=2`` is the
            broader MoE convention and tolerates a single misrouted
            expert without losing the entire token.

    Attributes:
        router: bias-less :class:`nn.Linear` of shape
            ``(in_features, num_experts)``.
        experts: :class:`nn.ModuleList` of ``num_experts``
            :class:`nn.Linear` layers, each ``(in_features, out_features)``.
        top_k: how many experts run per token.
        num_experts: total expert count.
        last_aux_loss: scalar ``torch.Tensor`` set after each
            :meth:`forward`. ``None`` before the first forward.

    Raises:
        ValueError: if ``num_experts <= 1``, ``top_k <= 0``, or
            ``top_k > num_experts``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_experts: int,
        top_k: int = 2,
    ):
        super().__init__()
        if num_experts <= 1:
            raise ValueError(
                f"num_experts must be ≥ 2 (a single-expert MoELinear is just "
                f"an nn.Linear with extra book-keeping), got {num_experts}"
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed num_experts ({num_experts}) — "
                "a token cannot be dispatched to more experts than exist."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k

        # Router has no bias by convention — the additive offset would
        # bias the softmax toward fixed experts regardless of input,
        # the opposite of what we want for load balancing.
        self.router = nn.Linear(in_features, num_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_experts)])

        # Populated by the first forward. Typed as Optional so static
        # checkers see the pre-forward state correctly.
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear contract: accept (..., in_features). The routing /
        # dispatch logic below is written for 2-D input, so flatten the
        # leading dims here and restore them on return — a (B, T, C)
        # sequence batch routes each token independently, which is the
        # standard MoE-in-transformer semantics.
        lead_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, self.in_features)

        # Router logits: (B, num_experts). The full softmax across
        # all experts is used in the load-balancing penalty (Switch's
        # P_i term needs ALL experts' probability mass, not just the
        # top-k subset).
        logits = self.router(x)
        probs_full = logits.softmax(dim=-1)  # (B, num_experts)

        # Top-k routing: select the k largest logits per row, then
        # renormalize across the k via softmax so the gating weights
        # sum to 1 per token. (Re-softmaxing the k logits — rather
        # than re-normalizing the full probs — is the original Switch
        # formulation; it gives sharper weights when the routed
        # experts have similar pre-top-k logits.)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # both (B, top_k)
        gate_weights = topk_vals.softmax(dim=-1)  # (B, top_k)

        B = x.size(0)
        out = torch.zeros(B, self.out_features, device=x.device, dtype=x.dtype)

        # Dispatch loop: for each of the top_k slots and each expert,
        # pull the rows whose slot routes to that expert, run them
        # through the expert, and scatter back into `out` weighted by
        # the gate. This is O(top_k · num_experts) Python iterations
        # — fine at tutorial scale, replaced by block-sparse kernels
        # at production scale.
        for k in range(self.top_k):
            for e in range(self.num_experts):
                mask = topk_idx[:, k] == e
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    out[mask] = out[mask] + gate_weights[mask, k : k + 1] * expert_out

        # Switch-style load-balancing aux loss
        # ``α · N · Σ_i f_i · P_i`` where:
        #   f_i = fraction of dispatched (token, slot) pairs routed
        #         to expert i, averaged over batch and top_k slots.
        #   P_i = mean router probability for expert i over the batch.
        # Both terms are in [0, 1] and sum to 1 across i. The product
        # is minimized at uniform f and P; at uniform routing each
        # equals 1/N, the sum equals N · 1/N² = 1/N, and the whole
        # penalty equals 1. So the minimum value of this loss term
        # is 1, not 0 — that's the property tested by
        # ``test_moe_linear_aux_loss_zero_at_uniform``.
        #
        # Broadcasting: topk_idx is (B, top_k); the [..., None]
        # expansion yields (B, top_k, 1), comparing against
        # arange[None, None, :] of shape (1, 1, num_experts) gives a
        # boolean tensor of shape (B, top_k, num_experts). Mean over
        # the first two dims collapses to (num_experts,).
        expert_idx = torch.arange(self.num_experts, device=x.device)
        dispatch_mask = topk_idx[..., None] == expert_idx[None, None, :]
        dispatch_frac = dispatch_mask.float().mean(dim=(0, 1))  # (num_experts,)
        mean_prob = probs_full.mean(dim=0)  # (num_experts,)
        self.last_aux_loss = self.num_experts * (dispatch_frac * mean_prob).sum()

        if len(lead_shape) > 1:
            out = out.reshape(*lead_shape, self.out_features)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_experts={self.num_experts}, top_k={self.top_k}"
        )
