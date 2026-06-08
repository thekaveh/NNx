"""Prompt tuning — learnable soft-prompt embeddings prepended to inputs.

Prompt tuning (`lester:prompt-tuning`) is the leanest of the prefix
family: freeze the entire pretrained model and learn a single
``(n_prompt_tokens, d_model)`` tensor of "soft prompt" embeddings.
Those embeddings are concatenated to the token embeddings at the input
of the first transformer block — every subsequent block sees them as
ordinary tokens (the model adapts to them through attention, not via
any per-layer parameter).

Compared to :class:`PrefixTuner`, the trainable budget is much smaller
(no per-layer K/V tensors), and the inductive bias is weaker — the
prompt only affects the model through the input embedding layer.
Prompt tuning is the right default when:

  - the base model is large enough that the per-layer prefix budget
    would still be small in absolute terms but the storage is annoying;
  - you only need to adapt to a single new task and want the fastest
    fine-tune;
  - you want to compose multiple soft prompts at inference time without
    swapping per-layer state.

Mechanism in this file:

  - :class:`PromptTuner` wraps a :class:`TransformerNN`, freezes every
    base parameter, and allocates one ``(n_prompt_tokens, d_model)``
    embedding tensor.
  - The wrapper's ``forward`` runs the token embedding manually,
    prepends the prompt embeddings (broadcast across batch), then
    drives the rest of the transformer stack — RMSNorm + LM head —
    by hand. The returned logits cover only the REAL tokens, not the
    soft-prompt positions (those are scaffolding, not output positions).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Union

import torch
from torch import nn

from ..nn.net.transformer_nn import TransformerNN
from ._source import _resolve_source_to_state_dict


class PromptTuner(nn.Module):
    """Wrap a :class:`TransformerNN` with a learnable soft prompt.

    Freezes every base parameter and allocates an
    ``(n_prompt_tokens, d_model)`` embedding tensor. The wrapper's
    forward prepends the prompt to the token embeddings, runs the
    stack, then trims the prompt positions off the logits before
    returning.

    Args:
        model: a :class:`TransformerNN` instance. Its parameters are
            mutated in place (set to ``requires_grad=False``).
        n_prompt_tokens: number of soft-prompt slots. Must be > 0.

    The soft prompt is initialized with ``nn.init.normal_(std=0.02)``
    — the same scale Lester et al. use as their "random init" baseline.
    """

    def __init__(
        self,
        model: TransformerNN,
        *,
        n_prompt_tokens: int = 20,
    ):
        super().__init__()
        if not isinstance(model, TransformerNN):
            raise TypeError(f"PromptTuner requires a TransformerNN, got {type(model).__name__}")
        if n_prompt_tokens <= 0:
            raise ValueError(f"n_prompt_tokens must be positive, got {n_prompt_tokens}")

        self.model = model
        self.n_prompt_tokens = n_prompt_tokens

        # Freeze every parameter of the wrapped TransformerNN — the
        # soft prompt is the only trainable bit going forward.
        for p in self.model.parameters():
            p.requires_grad = False

        d_model = model.params.d_model
        # (n_prompt_tokens, d_model). Normal(0, 0.02) init.
        self.soft_prompt = nn.Parameter(torch.empty(n_prompt_tokens, d_model))
        nn.init.normal_(self.soft_prompt, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the wrapped model with the soft prompt prepended.

        Args:
            tokens: (batch, seq) long tensor of token ids.

        Returns:
            (batch, seq, vocab_size) logits over the REAL token
            positions only. The soft-prompt positions are scaffolding
            and their logits are discarded.

        Raises:
            ValueError: if ``seq + n_prompt_tokens`` exceeds the
                wrapped model's ``max_seq_len``. The soft prompt
                consumes positions in the RoPE table just like real
                tokens do.
        """
        b, t = tokens.shape
        total = t + self.n_prompt_tokens
        if total > self.model.params.max_seq_len:
            raise ValueError(
                f"input length ({t}) + soft prompt ({self.n_prompt_tokens}) = {total} "
                f"exceeds wrapped model's max_seq_len={self.model.params.max_seq_len}"
            )

        # Manual replication of TransformerNN.forward, but with the
        # soft prompt spliced in front of the token embeddings.
        x = self.model.tok_embed(tokens)  # (B, T, d_model)
        prompt = self.soft_prompt.unsqueeze(0).expand(b, -1, -1)  # (B, n_prompt, d_model)
        x = torch.cat([prompt, x], dim=1)  # (B, n_prompt + T, d_model)

        for block in self.model.blocks:
            x, _ = block(x, use_cache=False)
        x = self.model.norm_out(x)
        logits = self.model.lm_head(x)  # (B, n_prompt + T, vocab)
        # Trim the soft-prompt positions off the returned logits — the
        # caller only ever wants predictions for real input tokens.
        return logits[:, self.n_prompt_tokens :, :]

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the soft-prompt tensor.

        The wrapped model's parameters are frozen on construction; this
        is the iterator you hand to an optimizer.
        """
        yield self.soft_prompt

    def prompt_state_dict(self) -> dict:
        """Return a state-dict containing only the soft-prompt tensor,
        keyed for round-trip via :meth:`load_prompt_weights`."""
        return {k: v for k, v in self.state_dict().items() if "soft_prompt" in k}


def save_prompt_weights(tuner: PromptTuner, path: Union[str, Path]) -> str:
    """Save ONLY the soft-prompt tensor of ``tuner`` to ``path``.

    Args:
        tuner: a :class:`PromptTuner` instance.
        path: destination file path.

    Returns:
        The path written, so calls can be chained.
    """
    sd = tuner.prompt_state_dict()
    torch.save(sd, str(path))
    return str(path)


def load_prompt_weights(tuner: PromptTuner, source: Union[str, Path, dict]) -> int:
    """Load the soft-prompt tensor into ``tuner`` from ``source``.

    Args:
        tuner: must already have the same prompt shape as the source
            (same n_prompt_tokens, d_model). A shape mismatch is
            surfaced by ``load_state_dict``.
        source: a path to a file produced by :func:`save_prompt_weights`,
            or a state-dict dict directly.

    Returns:
        The number of parameter tensors loaded.
    """
    sd = _resolve_source_to_state_dict(source, "load_prompt_weights")
    sd = {k: v for k, v in sd.items() if "soft_prompt" in k}
    tuner.load_state_dict(sd, strict=False)
    return len(sd)
