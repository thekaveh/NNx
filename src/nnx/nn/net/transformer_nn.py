"""TransformerNN — decoder-only language model.

Composes ``TransformerBlock``s with a token embedding at the front and
a tied LM head at the back. RMSNorm output normalization sits between
the last block and the LM head, matching the LLaMA family.

Scope explicit: TinyStories-class single-GPU LM. The forward pass takes
``(B, T)`` token ids and returns ``(B, T, vocab)`` logits — autoregressive
sampling lives in ``GenerativeNNModel.generate()`` (PR 4).
"""

from __future__ import annotations

import torch
from torch import nn

from ..params.nn_transformer_params import NNTransformerParams
from .transformer_layers import RMSNorm, TransformerBlock


class TransformerNN(nn.Module):
    def __init__(self, params: NNTransformerParams):
        super().__init__()
        self.params = params

        self.tok_embed = nn.Embedding(num_embeddings=params.vocab_size, embedding_dim=params.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=params.d_model,
                    n_heads=params.n_heads,
                    ffn_mult=params.ffn_mult,
                    max_seq_len=params.max_seq_len,
                    rope_base=params.rope_base,
                    attn_dropout=params.attn_dropout,
                    resid_dropout=params.resid_dropout,
                )
                for _ in range(params.n_layers)
            ]
        )
        self.norm_out = RMSNorm(params.d_model)
        # LM head: a no-bias Linear from d_model → vocab_size. When
        # tie_embeddings=True, we share the weight tensor with the
        # token embedding (parameter-shared, not just initialized to
        # the same values) — the standard LLaMA trick that cuts
        # ~vocab*d_model params for free.
        self.lm_head = nn.Linear(params.d_model, params.vocab_size, bias=False)
        if params.tie_embeddings:
            # Identity-assign the parameter — not just copy the data —
            # so .parameters() doesn't double-count and gradients flow
            # through a single shared tensor.
            self.lm_head.weight = self.tok_embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: (batch, seq) long tensor of token ids.

        Returns:
            (batch, seq, vocab_size) logits — pre-softmax.
        """
        b, t = tokens.shape
        if t > self.params.max_seq_len:
            raise ValueError(f"input sequence length ({t}) exceeds max_seq_len={self.params.max_seq_len}")
        x = self.tok_embed(tokens)  # (B, T, d_model)
        for block in self.blocks:
            x, _ = block(x, use_cache=False)
        x = self.norm_out(x)
        logits = self.lm_head(x)  # (B, T, vocab)
        return logits

    def unpack_batch(self, batch):
        """Make TransformerNN compatible with the standard supervised
        NNModel training loop.

        For an LM the canonical batch is ``(tokens, targets)`` where
        ``targets = tokens[:, 1:]`` shifted by one. We don't shift here —
        the caller assembles the tuple — but we accept either a 2-tuple
        ``(X, Y)`` or a plain tensor of tokens (next-token loss is then
        computed in the train step).
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, Y = batch
            return (X,), Y
        if isinstance(batch, torch.Tensor):
            # Self-supervised next-token: targets are inputs shifted by 1.
            return (batch[:, :-1],), batch[:, 1:]
        raise TypeError(
            f"TransformerNN.unpack_batch expects a (tokens, targets) tuple "
            f"or a tensor of token ids; got {type(batch).__name__}"
        )

    def __str__(self) -> str:
        return f"TransformerNN={self.params}"

    def to_file(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @staticmethod
    def from_file(path: str, params: NNTransformerParams) -> TransformerNN:
        net = TransformerNN(params)
        net.load_state_dict(torch.load(path, weights_only=True))
        return net

    @staticmethod
    def from_state(state_dict: dict, params: NNTransformerParams) -> TransformerNN:
        net = TransformerNN(params)
        net.load_state_dict(state_dict)
        return net
