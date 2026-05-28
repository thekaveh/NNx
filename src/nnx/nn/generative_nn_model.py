"""GenerativeNNModel — NNModel subclass with autoregressive ``generate()``.

Wires a TransformerNN + NNTokenizerParams together so the standard
NNModel.train() machinery still works for next-token-prediction
training, while exposing the LM-specific ``generate(prompt, ...)`` API
on top.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..generation.logits_processors import (
    LogitsProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
)
from ..generation.sampling import sample_next_token
from .nn_model import NNModel
from .params.nn_model_params import NNModelParams
from .params.nn_params import NNParams
from .params.nn_tokenizer_params import NNTokenizerParams


class GenerativeNNModel(NNModel):
    """Language model with an autoregressive ``generate()`` method.

    ``tokenizer`` is held as a regular instance attribute (not a
    constructor-arg of NNModel) so existing NNModel callers don't
    have to know about it. It's required for ``generate()`` but
    optional at construction — train-time you can build the model
    first and attach the tokenizer later.
    """

    def __init__(
        self,
        net_params: NNParams,
        params: NNModelParams,
        tokenizer: Optional[NNTokenizerParams] = None,
    ):
        super().__init__(net_params=net_params, params=params)
        self.tokenizer = tokenizer

    # ---------- generate ----------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ) -> str:
        """Autoregressive decode from ``prompt``.

        Args:
            prompt: input text. Encoded via ``self.tokenizer``.
            max_new_tokens: hard cap on new tokens emitted. Generation
                also stops if the context window (max_seq_len) would be
                exceeded and the model can't shrink the window further,
                or if a ``stop`` string is decoded.
            temperature: 0 means greedy (argmax). Higher values produce
                more diverse output. Routes through TemperatureScaling.
            top_k: keep only the top-k logits. None disables.
            top_p: nucleus (top-p) cutoff. None disables.
            repetition_penalty: divide previously-seen tokens' positive
                logits by this. 1.0 is no-op (default).
            stop: list of stop strings — generation halts when any of
                them appears in the decoded prefix.
            seed: when set, sampling is reproducible — two calls with
                the same seed + prompt + model produce identical output.
            use_cache: when True (default), uses an incremental KV
                cache — each new token only re-runs attention on the
                last position, not the whole prefix. When False, falls
                back to the SP-4 full-recompute path (kept for
                regression testing). Both paths produce the same
                tokens for greedy decoding (sampling paths agree given
                the same seed).

        Returns:
            The full decoded string (prompt + generated continuation).
        """
        if self.tokenizer is None:
            raise ValueError(
                "GenerativeNNModel.generate requires a tokenizer. "
                "Construct with `GenerativeNNModel(..., tokenizer=NNTokenizerParams.of(tk, path))`."
            )
        self.net.eval()

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            # Use a single space as a non-empty seed when the prompt
            # encodes to nothing (e.g., an empty string with no BOS).
            prompt_ids = self.tokenizer.encode(" ") or [0]

        max_seq_len = getattr(self.net_params, "max_seq_len", None)
        if max_seq_len is None:
            raise ValueError("net_params must expose `max_seq_len` (got a non-Transformer net?)")

        # Build the processor chain from the kwargs. Order matters:
        # repetition penalty first (operates on raw logits), then
        # top-k / top-p (filtering), then temperature (scaling). This
        # is the conventional HF transformers ordering.
        processors: list[LogitsProcessor] = []
        if repetition_penalty != 1.0:
            processors.append(RepetitionPenalty(penalty=repetition_penalty))
        if top_k is not None:
            processors.append(TopKFilter(top_k=top_k))
        if top_p is not None:
            processors.append(TopPFilter(top_p=top_p))
        # Temperature is always applied — temperature=0 is the greedy
        # short-circuit. We still inject the processor so the sampler
        # path can be uniform.
        processors.append(TemperatureScaling(temperature=temperature))

        # Optional seeded torch.Generator for reproducible sampling.
        # We pin it to the model's device so multinomial doesn't fall
        # back to CPU silently on a CUDA model.
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))

        # The KV-cache path needs ``forward_with_cache`` on the net.
        # Plain LSTM/MLP nets (or older Transformer forks) don't have
        # it — fall back transparently rather than crashing.
        if use_cache and not hasattr(self.net, "forward_with_cache"):
            use_cache = False

        generated: list[int] = list(prompt_ids)
        with torch.no_grad():
            if use_cache:
                self._generate_with_cache(
                    generated=generated,
                    max_new_tokens=max_new_tokens,
                    max_seq_len=max_seq_len,
                    processors=processors,
                    gen=gen,
                    stop=stop,
                )
            else:
                self._generate_no_cache(
                    generated=generated,
                    max_new_tokens=max_new_tokens,
                    max_seq_len=max_seq_len,
                    processors=processors,
                    gen=gen,
                    stop=stop,
                )

        return self.tokenizer.decode(generated)

    # ---------- generate helpers ----------

    def _generate_no_cache(
        self,
        *,
        generated: list[int],
        max_new_tokens: int,
        max_seq_len: int,
        processors: list[LogitsProcessor],
        gen: Optional[torch.Generator],
        stop: Optional[list[str]],
    ) -> None:
        """SP-4 full-recompute path: every step re-runs the model on the
        last ``max_seq_len`` tokens. O(T^2) attention cost; kept for
        regression-testing parity against the cached path."""
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len from the right so the
            # most recent tokens stay in the window. Sliding window
            # — the simple production-ready approach for SP-4 scope.
            context_ids = generated[-max_seq_len:]
            ctx = torch.tensor([context_ids], dtype=torch.long, device=self.device)
            logits = self.net(ctx)  # (1, T, vocab)
            next_logits = logits[:, -1, :]  # (1, vocab) — last token's logits
            adjusted = apply_chain(next_logits, token_history=generated, processors=processors)
            next_id = sample_next_token(adjusted, generator=gen)
            generated.append(next_id)

            # Optional stop-string check. We only decode once per
            # iteration when stops are configured — keeps the hot
            # loop cheap when they aren't.
            if stop:
                text_so_far = self.tokenizer.decode(generated)
                if any(s in text_so_far for s in stop):
                    break

    def _generate_with_cache(
        self,
        *,
        generated: list[int],
        max_new_tokens: int,
        max_seq_len: int,
        processors: list[LogitsProcessor],
        gen: Optional[torch.Generator],
        stop: Optional[list[str]],
    ) -> None:
        """KV-cache path (SP-10c). Runs one prefill pass on the
        truncated prompt, then per new token re-runs only the last
        position's attention against the cached prefix. O(T) per
        step instead of O(T^2).

        Sliding-window safety: when the prompt + max_new_tokens would
        overflow ``max_seq_len``, we drop the oldest entry from each
        layer's cached k/v before appending the next step. This keeps
        long-context generation correct without rebuilding the cache
        from scratch every overflow.
        """
        # ----- Prefill pass on the prompt (sliding window). -----
        context_ids = generated[-max_seq_len:]
        ctx = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        logits, past_kvs = self.net.forward_with_cache(ctx, past_kvs=None)
        next_logits = logits[:, -1, :]
        adjusted = apply_chain(next_logits, token_history=generated, processors=processors)
        next_id = sample_next_token(adjusted, generator=gen)
        generated.append(next_id)
        if stop:
            text_so_far = self.tokenizer.decode(generated)
            if any(s in text_so_far for s in stop):
                return

        # ----- Incremental decode loop. -----
        for _ in range(max_new_tokens - 1):
            # If appending another token would overflow max_seq_len,
            # drop the oldest cached k/v position on every layer
            # (sliding window — matches the no-cache path's behaviour).
            cached_len = past_kvs[0][0].size(-2) if past_kvs and past_kvs[0] is not None else 0
            if cached_len + 1 > max_seq_len:
                past_kvs = [(k[..., 1:, :], v[..., 1:, :]) for (k, v) in past_kvs]

            last_id = generated[-1]
            ctx = torch.tensor([[last_id]], dtype=torch.long, device=self.device)
            logits, past_kvs = self.net.forward_with_cache(ctx, past_kvs=past_kvs)
            next_logits = logits[:, -1, :]
            adjusted = apply_chain(next_logits, token_history=generated, processors=processors)
            next_id = sample_next_token(adjusted, generator=gen)
            generated.append(next_id)

            if stop:
                text_so_far = self.tokenizer.decode(generated)
                if any(s in text_so_far for s in stop):
                    break
