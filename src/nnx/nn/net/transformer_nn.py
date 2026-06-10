"""TransformerNN — decoder-only language model.

Composes ``TransformerBlock``s with a token embedding at the front and
a tied LM head at the back. RMSNorm output normalization sits between
the last block and the LM head, matching the LLaMA family.

Scope explicit: TinyStories-class single-GPU LM. The forward pass takes
``(B, T)`` token ids and returns ``(B, T, vocab)`` logits — autoregressive
sampling lives in ``GenerativeNNModel.generate()`` (PR 4).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..params.nn_transformer_params import NNTransformerParams
from .transformer_layers import RMSNorm, TransformerBlock

# Type alias: per-layer KV cache entry; None means "no cache yet for
# this layer". The full cache is a list with one entry per transformer
# block, threaded through ``forward_with_cache`` between decode steps.
LayerKV = Optional[tuple[torch.Tensor, torch.Tensor]]


class TransformerNN(nn.Module):
    def __init__(self, params: NNTransformerParams):
        super().__init__()
        self.params = params

        self.tok_embed = nn.Embedding(num_embeddings=params.vocab_size, embedding_dim=params.d_model)
        # GPT-2/LLaMA-style small-std init. nn.Embedding's default is
        # N(0, 1) — with tied embeddings, the input token's own logit
        # then includes e·e ≈ d_model, so an untrained model starts at
        # CE ≈ d_model instead of ln(vocab) and greedy/sampled decoding
        # degenerates into repeating the last prompt token. The shared
        # tensor also covers lm_head when tie_embeddings=True.
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
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

    def forward_with_cache(
        self,
        tokens: torch.Tensor,
        past_kvs: Optional[list[LayerKV]] = None,
    ) -> tuple[torch.Tensor, list[LayerKV]]:
        """Cache-threading forward used by ``GenerativeNNModel.generate``.

        Behaves like ``forward`` but additionally accepts a per-layer
        list of (k, v) caches (or ``None`` entries on the first call)
        and returns the updated per-layer caches alongside the logits.

        The total attended-to length per layer is
        ``past_kv_len + tokens.shape[1]`` — the caller is responsible
        for ensuring that this stays within ``max_seq_len`` (the
        generate loop slides a window when it would otherwise overflow).

        Args:
            tokens: (batch, seq) long tensor of token ids. During
                incremental decode, ``seq == 1``; on the prefill step
                the prompt's full length is fed in one shot.
            past_kvs: list of length ``n_layers`` with each entry a
                ``(k, v)`` tuple or ``None``. ``None`` means "no
                history for this layer" (i.e., first call).

        Returns:
            A tuple ``(logits, new_kvs)`` where ``logits`` is
                ``(batch, seq, vocab)`` — the *new* tokens' logits (with
                ``past_kvs != None`` and ``seq=1`` the returned
                ``logits[:, -1, :]`` is the next-token distribution
                conditioned on the full cached prefix) — and ``new_kvs``
                is a list of length ``n_layers`` of updated ``(k, v)``
                tuples; pass this back in for the next step.
        """
        b, t = tokens.shape
        # Length already cached, taken from layer 0 (all layers share length).
        cached_len = 0
        if past_kvs is not None and past_kvs[0] is not None:
            cached_len = past_kvs[0][0].size(-2)
        total_len = cached_len + t
        if total_len > self.params.max_seq_len:
            raise ValueError(
                f"cached_len ({cached_len}) + new tokens ({t}) = {total_len} "
                f"exceeds max_seq_len={self.params.max_seq_len}"
            )

        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        if len(past_kvs) != len(self.blocks):
            raise ValueError(f"past_kvs has {len(past_kvs)} entries but model has {len(self.blocks)} layers")

        x = self.tok_embed(tokens)  # (B, T, d_model)
        new_kvs: list[LayerKV] = []
        for block, layer_past in zip(self.blocks, past_kvs, strict=True):
            x, new_kv = block(x, past_kv=layer_past, use_cache=True)
            new_kvs.append(new_kv)
        x = self.norm_out(x)
        logits = self.lm_head(x)  # (B, T, vocab)
        return logits, new_kvs

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
