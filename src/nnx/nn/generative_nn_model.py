"""GenerativeNNModel — NNModel subclass with autoregressive ``generate()``.

Wires a TransformerNN + NNTokenizerParams together so the standard
NNModel.train() machinery still works for next-token-prediction
training, while exposing the LM-specific ``generate(prompt, ...)`` API
on top.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, cast

import torch

from ..generation.logits_chain import LogitsChain
from ..generation.logits_processors import (
    LogitsProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
)
from ..generation.sampling import sample_next_token
from ..utils import _capture_training_modes, _restore_training_modes
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

    def _hub_reconstruction_config(self, save_directory) -> dict[str, Any]:
        if self.tokenizer is None:
            return {}
        filename = "tokenizer.json"
        self.tokenizer.tokenizer.save(str(Path(save_directory) / filename))  # type: ignore[union-attr]
        return {"tokenizer": {"filename": filename}}

    @classmethod
    def _hub_reconstruction_kwargs(cls, config: Mapping[str, Any], config_directory) -> dict[str, Any]:
        tokenizer_config = config.get("tokenizer")
        if not isinstance(tokenizer_config, Mapping) or not isinstance(tokenizer_config.get("filename"), str):
            return {}
        filename = cast(str, tokenizer_config["filename"])
        return {"tokenizer": NNTokenizerParams.from_state({"path": str(Path(config_directory) / filename)})}

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
        logits_chain: Optional[LogitsChain] = None,
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
            stop: list of stop strings — generation halts once any of
                them appears in the decoded CONTINUATION (the prompt
                itself is not searched, so a prompt containing a stop
                string doesn't halt generation immediately; a stop
                string straddling the prompt/continuation boundary is
                likewise not detected — matching the generated-text-only
                convention HF uses).
            seed: when set, sampling is reproducible — two calls with
                the same seed + prompt + model produce identical output.
            use_cache: when True (default), uses an incremental KV
                cache — each new token only re-runs attention on the
                last position, not the whole prefix. When False, falls
                back to the full-recompute path (kept for regression
                testing). Both paths produce the same tokens for greedy
                decoding (sampling paths agree given the same seed).
            logits_chain: optional pre-built ``LogitsChain`` (see
                ``nnx.LogitsChain.builder()``). When provided, the
                inline chain construction from ``temperature`` /
                ``top_k`` / ``top_p`` / ``repetition_penalty`` kwargs
                is skipped — the supplied chain is used as-is.
                Power-user path for custom logit processors (e.g.,
                logit-bias for forbidden tokens). When ``None`` (the
                default), behavior is unchanged.

        Returns:
            The full decoded string (prompt + generated continuation).

        Non-destructive: ``self.net.training`` is snapshotted before
        switching to ``eval()`` and restored on exit (including the
        exception path via ``try/finally``). Matches the convention
        used by ``NNModel.predict`` / ``NNModel.evaluate``,
        ``nnx.diffusion.sample``, ``nnx.embeddings.embed_texts``,
        ``nnx.viz.activation_map``, and ``nnx.lr_finder``.
        """
        if self.tokenizer is None:
            raise ValueError(
                "GenerativeNNModel.generate requires a tokenizer. "
                "Construct with `GenerativeNNModel(..., tokenizer=NNTokenizerParams.of(tk, path))`."
            )
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            # Use a single space as a non-empty seed when the prompt
            # encodes to nothing (e.g., an empty string with no BOS).
            prompt_ids = self.tokenizer.encode(" ") or [0]

        max_seq_len = getattr(self.net_params, "max_seq_len", None)
        if max_seq_len is None:
            raise ValueError("net_params must expose `max_seq_len` (got a non-Transformer net?)")
        # Wrappers that consume window slots (PromptTuner's soft prompt)
        # advertise a smaller effective window — without this, the
        # sliding window overflows the wrapped model mid-generation.
        max_seq_len = getattr(self.net, "effective_max_seq_len", max_seq_len)

        # Build the processor chain. Two paths:
        # (a) If the caller supplied a pre-built ``logits_chain``, use
        #     its processors as-is — they've already been ordered.
        # (b) Otherwise build the standard chain from kwargs in NNx's
        #     canonical order: repetition penalty first (raw logits),
        #     then top-k / top-p (filtering), then temperature
        #     (scaling — deliberately last: temperature=0's ±inf greedy
        #     markers must not be re-filtered). Temperature is always applied;
        #     temperature=0 is the greedy short-circuit (still routed
        #     through TemperatureScaling so the sampler path is
        #     uniform).
        processors: list[LogitsProcessor]
        if logits_chain is not None:
            processors = list(logits_chain.processors)
        else:
            processors = []
            if repetition_penalty != 1.0:
                processors.append(RepetitionPenalty(penalty=repetition_penalty))
            if top_k is not None:
                processors.append(TopKFilter(top_k=top_k))
            if top_p is not None:
                processors.append(TopPFilter(top_p=top_p))
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
        # Snapshot training-mode for non-destructive restore on exit
        # (matches NNModel.predict / evaluate / nnx.viz.activation_map /
        # nnx.lr_finder). Deliberately the LAST thing before the
        # try/finally: an exception in the validation / chain-building
        # code above would otherwise strand the net in eval() with no
        # finally to restore it.
        training_modes = _capture_training_modes(self.net)
        self.net.eval()
        try:
            with torch.no_grad():
                if use_cache:
                    self._generate_with_cache(
                        generated=generated,
                        n_prompt=len(prompt_ids),
                        max_new_tokens=max_new_tokens,
                        max_seq_len=max_seq_len,
                        processors=processors,
                        gen=gen,
                        stop=stop,
                    )
                else:
                    self._generate_no_cache(
                        generated=generated,
                        n_prompt=len(prompt_ids),
                        max_new_tokens=max_new_tokens,
                        max_seq_len=max_seq_len,
                        processors=processors,
                        gen=gen,
                        stop=stop,
                    )
        finally:
            _restore_training_modes(training_modes)

        return self.tokenizer.decode(generated)

    # ---------- generate helpers ----------

    def _generate_no_cache(
        self,
        *,
        generated: list[int],
        n_prompt: int,
        max_new_tokens: int,
        max_seq_len: int,
        processors: list[LogitsProcessor],
        gen: Optional[torch.Generator],
        stop: Optional[list[str]],
    ) -> None:
        """Full-recompute path: every step re-runs the model on the
        last ``max_seq_len`` tokens. O(T^2) attention cost; kept for
        regression-testing parity against the cached path."""
        assert self.tokenizer is not None
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len from the right so the
            # most recent tokens stay in the window. Sliding window —
            # the simple production-ready approach.
            context_ids = generated[-max_seq_len:]
            ctx = torch.tensor([context_ids], dtype=torch.long, device=self.device)
            logits = self.net(ctx)  # (1, T, vocab)
            next_logits = logits[:, -1, :]  # (1, vocab) — last token's logits
            adjusted = apply_chain(next_logits, token_history=generated, processors=processors)
            next_id = sample_next_token(adjusted, generator=gen)
            generated.append(next_id)

            # Optional stop-string check, scoped to the CONTINUATION —
            # a stop string already present in the prompt must not halt
            # generation after one token. Only decodes when stops are
            # configured, keeping the hot loop cheap otherwise.
            if stop:
                continuation = self.tokenizer.decode(generated[n_prompt:])
                if any(s in continuation for s in stop):
                    break

    def _generate_with_cache(
        self,
        *,
        generated: list[int],
        n_prompt: int,
        max_new_tokens: int,
        max_seq_len: int,
        processors: list[LogitsProcessor],
        gen: Optional[torch.Generator],
        stop: Optional[list[str]],
    ) -> None:
        """KV-cache path. Runs one prefill pass on the
        truncated prompt, then per new token re-runs only the last
        position's attention against the cached prefix. O(T) per
        step instead of O(T^2).

        Sliding-window safety: once appending another token would
        overflow ``max_seq_len``, the cache is rebuilt from the current
        window. Cached k/v are RoPE-stamped at the absolute position
        they were written at, so merely dropping the oldest entry would
        pin every later token's offset at ``max_seq_len - 1`` and
        corrupt the relative position geometry (logits drift vs the
        no-cache path). Rebuilding re-rotates the window to positions
        ``0..max_seq_len-1`` — exactly what the no-cache path computes,
        so greedy/seeded parity holds across overflow. Post-overflow
        steps therefore cost one full window forward, same as the
        no-cache path; the O(T) win applies within the window.
        """
        assert self.tokenizer is not None
        if max_new_tokens <= 0:
            # Hard cap honored on this path too: the prefill below
            # always samples one token, which would emit 1 instead of 0.
            return

        # ----- Prefill pass on the prompt (sliding window). -----
        context_ids = generated[-max_seq_len:]
        ctx = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        cached_net = cast(Any, self.net)
        logits, past_kvs = cached_net.forward_with_cache(ctx, past_kvs=None)
        next_logits = logits[:, -1, :]
        adjusted = apply_chain(next_logits, token_history=generated, processors=processors)
        next_id = sample_next_token(adjusted, generator=gen)
        generated.append(next_id)
        if stop:
            # Continuation-scoped, like the loop check below.
            continuation = self.tokenizer.decode(generated[n_prompt:])
            if any(s in continuation for s in stop):
                return

        # ----- Incremental decode loop. -----
        for _ in range(max_new_tokens - 1):
            cached_len = past_kvs[0][0].size(-2) if past_kvs and past_kvs[0] is not None else 0
            if cached_len + 1 > max_seq_len:
                # Window full: rebuild the cache from the current
                # window (see the docstring — dropping the oldest k/v
                # would corrupt RoPE relative positions). The window
                # includes generated[-1], so this forward both refills
                # the cache and yields the next token's logits.
                ctx = torch.tensor([generated[-max_seq_len:]], dtype=torch.long, device=self.device)
                logits, past_kvs = cached_net.forward_with_cache(ctx, past_kvs=None)
            else:
                ctx = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
                logits, past_kvs = cached_net.forward_with_cache(ctx, past_kvs=past_kvs)
            next_logits = logits[:, -1, :]
            adjusted = apply_chain(next_logits, token_history=generated, processors=processors)
            next_id = sample_next_token(adjusted, generator=gen)
            generated.append(next_id)

            if stop:
                # Continuation-scoped — see _generate_no_cache.
                continuation = self.tokenizer.decode(generated[n_prompt:])
                if any(s in continuation for s in stop):
                    break
