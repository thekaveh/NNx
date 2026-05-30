"""Token sampling primitives.

Single-token sampling is split out from the LogitsProcessor chain
because seeding lives here (the chain is otherwise deterministic given
its inputs). ``sample_next_token`` accepts an optional
``torch.Generator`` so the caller can reproduce a generation by
constructing one with the same seed.
"""

from __future__ import annotations

from typing import Optional

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Draw one token id from ``softmax(logits)``.

    Args:
        logits: shape (1, vocab) — single-sequence sample (the LM
            path's batch-1 generate scope).
        generator: optional torch.Generator for reproducible seeded
            sampling. When None, sampling uses the default RNG (still
            affected by torch.manual_seed at the call site).

    Returns:
        An int token id.
    """
    if logits.dim() != 2 or logits.size(0) != 1:
        raise ValueError(f"sample_next_token expects shape (1, vocab); got {tuple(logits.shape)}")
    # Greedy short-circuit: if any +inf appears, the temperature=0 path
    # set it as the argmax marker — pick the first such position. (We
    # don't try to softmax this: softmax(+inf) is NaN when multiple
    # +infs are present, but TemperatureScaling guarantees exactly one
    # +inf per row.)
    pos_inf_mask = torch.isposinf(logits)
    if pos_inf_mask.any():
        return int(pos_inf_mask[0].nonzero(as_tuple=False)[0].item())
    # All-finite-but-degenerate (all -inf after top-k/top-p collapse):
    # fall back to argmax so we don't crash inside multinomial.
    if torch.isinf(logits).all():
        return int(logits.argmax(dim=-1).item())

    probs = torch.softmax(logits, dim=-1)
    # multinomial requires a finite, positive-sum probability vector.
    # Three degenerate cases collapse to argmax on the original logits:
    #   * NaN-in-logits → softmax produces NaN → sum is NaN (not 0.0,
    #     so the original `== 0.0` check missed this — multinomial then
    #     crashed with "probability tensor contains either inf, nan").
    #   * sum is 0.0 (e.g., the entire row underflowed to zero in
    #     reduced-precision contexts).
    #   * sum is +/-inf (shouldn't happen given the +inf guard above,
    #     but cheap belt-and-braces).
    total = probs.sum()
    if not torch.isfinite(total) or total.item() <= 0.0:
        return int(logits.argmax(dim=-1).item())
    next_id = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_id.item())
