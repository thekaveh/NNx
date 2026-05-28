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

        generated: list[int] = list(prompt_ids)
        with torch.no_grad():
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

        return self.tokenizer.decode(generated)
