"""Logits processors for autoregressive sampling.

Each processor takes the current logits tensor (shape ``(batch, vocab)``)
plus the token history so far and returns an adjusted logits tensor of
the same shape. Composing them into a chain — apply_chain — gives the
classic temperature → top-k → top-p → repetition-penalty decoding setup
without coupling sampling logic into the model.
"""

from __future__ import annotations

from typing import Protocol

import torch


class LogitsProcessor(Protocol):
    """Callable protocol: ``logits, token_history -> adjusted_logits``.

    ``token_history`` is a flat list of int token ids generated so far
    (across batch dim 0 — we assume a single-sequence batch in
    generate(), which is the SP-4 scope). Processors that don't care
    about history (temperature, top-k, top-p) simply ignore the arg.
    """

    def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor: ...


class TemperatureScaling:
    """Divide logits by ``temperature`` before sampling.

    ``temperature == 0`` is a special case: the chain reduces to greedy
    decoding (argmax). We map argmax positions to +inf and others to
    -inf so the downstream sampler picks deterministically without
    branching on the temperature value.
    """

    def __init__(self, temperature: float):
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
        if self.temperature == 0.0:
            out = torch.full_like(logits, float("-inf"))
            argmax = logits.argmax(dim=-1, keepdim=True)
            out.scatter_(dim=-1, index=argmax, value=float("inf"))
            return out
        return logits / self.temperature


class TopKFilter:
    """Keep only the top-k logits per row; set the rest to -inf.

    -inf survives the temperature divide (still -inf) and gets mapped
    to 0 probability mass by softmax, so the order top-k → temperature
    or temperature → top-k both work; we don't enforce an ordering.
    """

    def __init__(self, top_k: int):
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        self.top_k = top_k

    def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
        k = min(self.top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k=k, dim=-1)
        # threshold[i] = smallest of the top-k values in row i.
        threshold = topk_vals[..., -1:].expand_as(logits)
        return torch.where(logits >= threshold, logits, torch.full_like(logits, float("-inf")))


class TopPFilter:
    """Nucleus (top-p) sampling: keep the smallest set of tokens whose
    cumulative probability exceeds ``top_p``.

    Edge case: if a single token already has probability >= top_p, only
    that token is retained.
    """

    def __init__(self, top_p: float):
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        self.top_p = top_p

    def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Tokens to remove: cum_probs strictly greater than top_p,
        # shifted by 1 so the token that pushes cum_probs over the
        # threshold is itself kept.
        sorted_remove = cum_probs > self.top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False  # always keep the top token
        # Scatter back to the original position order.
        remove = torch.zeros_like(sorted_remove)
        remove.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)
        return logits.masked_fill(remove, float("-inf"))


class RepetitionPenalty:
    """Penalize previously-seen tokens (HF-style).

    For each token id ``i`` in ``token_history``:
      * if ``logits[..., i] > 0``: divide by penalty (decreases mass).
      * if ``logits[..., i] < 0``: multiply by penalty (increases
        magnitude → further decreases relative mass after softmax).

    A penalty of 1.0 is a no-op (the back-compat default).
    """

    def __init__(self, penalty: float):
        if penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {penalty}")
        self.penalty = penalty

    def __call__(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
        if not token_history or self.penalty == 1.0:
            return logits
        out = logits.clone()
        # We assume batch size 1 throughout SP-4's generate(); for
        # multi-batch generation a per-row history would be needed.
        idx = torch.tensor(sorted(set(token_history)), dtype=torch.long, device=logits.device)
        # Apply penalty per-token: divide positive, multiply negative.
        selected = out[..., idx]
        positive = selected > 0
        adjusted = torch.where(positive, selected / self.penalty, selected * self.penalty)
        out[..., idx] = adjusted
        return out


def apply_chain(
    logits: torch.Tensor,
    *,
    token_history: list[int],
    processors: list[LogitsProcessor],
) -> torch.Tensor:
    """Apply every processor in order. No-op when ``processors`` is empty."""
    for proc in processors:
        logits = proc(logits, token_history)
    return logits
