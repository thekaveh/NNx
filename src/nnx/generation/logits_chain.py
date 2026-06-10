"""LogitsChain — a typed, ordered chain of LogitsProcessors.

Plan 5 of the Builder-pattern rollout (see
`docs/superpowers/specs/2026-05-31-builder-pattern-investigation.md`
§3.5). The Builder is the opt-in alternative to
`GenerativeNNModel.generate(temperature=..., top_k=..., top_p=...,
repetition_penalty=...)`: it lets the user compose the chain
declaratively and add custom processors (e.g., logit-bias for
forbidden tokens) without having to drop down to
`apply_chain(processors=[...])` and reconstruct the canonical order
themselves.

Builder ordering: regardless of the method-call order, `.build()`
sorts the processors into NNx's canonical order (the same order
`GenerativeNNModel.generate` builds from its inline kwargs):
    RepetitionPenalty → TopKFilter → TopPFilter → TemperatureScaling
Temperature is deliberately LAST — temperature=0 greedy is implemented
via ±inf argmax markers that must not be re-filtered, so this differs
from HF's transformers, which applies temperature before top-k/top-p
(for temperature≠1 the nucleus token set can differ).
Custom processors (added via `.custom(processor)`) are appended after
TemperatureScaling — the "post-default" slot for user extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .logits_processors import (
    LogitsProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopKFilter,
    TopPFilter,
    apply_chain,
)

# Canonical processor order. Used by LogitsChainBuilder.build() to
# sort whatever the user chained into the conventional HF sequence.
# Custom processors (added via .custom()) go AFTER this group, in
# the order they were added.
_CANONICAL_ORDER: tuple[type[LogitsProcessor], ...] = (
    RepetitionPenalty,
    TopKFilter,
    TopPFilter,
    TemperatureScaling,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class LogitsChain:
    """A typed, ordered sequence of LogitsProcessors.

    Build via `LogitsChain.builder()` for the safe / discoverable
    path; or construct directly from a list for advanced cases. The
    `.apply()` method runs the processors against a logits tensor in
    order, returning the adjusted tensor.
    """

    processors: list[LogitsProcessor] = field(default_factory=list)

    def apply(self, logits: torch.Tensor, token_history: list[int]) -> torch.Tensor:
        """Run every processor in `self.processors` in order. Thin
        wrapper around `apply_chain`."""
        return apply_chain(logits, token_history=token_history, processors=self.processors)

    @classmethod
    def builder(cls) -> LogitsChainBuilder:
        """Return a fluent builder. See `LogitsChainBuilder`."""
        return LogitsChainBuilder()


class LogitsChainBuilder:
    """Fluent builder for a `LogitsChain`.

    Method order at the call site doesn't matter — `.build()` sorts
    the standard processors into NNx's canonical order (matching
    `generate()`'s inline-kwargs chain; see the module docstring for
    why temperature is deliberately last):
    `RepetitionPenalty → TopKFilter → TopPFilter → TemperatureScaling`.
    Custom processors (added via `.custom(processor)`) are appended in
    the order they were added, after the canonical group.
    """

    def __init__(self) -> None:
        self._standard: dict[type[LogitsProcessor], LogitsProcessor] = {}
        self._custom: list[LogitsProcessor] = []

    def repetition_penalty(self, penalty: float) -> LogitsChainBuilder:
        """Add a RepetitionPenalty processor with the given penalty."""
        self._standard[RepetitionPenalty] = RepetitionPenalty(penalty=penalty)
        return self

    def top_k(self, k: int) -> LogitsChainBuilder:
        """Add a TopKFilter with the given k."""
        self._standard[TopKFilter] = TopKFilter(top_k=k)
        return self

    def top_p(self, p: float) -> LogitsChainBuilder:
        """Add a TopPFilter (nucleus sampling) with the given p."""
        self._standard[TopPFilter] = TopPFilter(top_p=p)
        return self

    def temperature(self, t: float) -> LogitsChainBuilder:
        """Add a TemperatureScaling processor with the given temperature."""
        self._standard[TemperatureScaling] = TemperatureScaling(temperature=t)
        return self

    def custom(self, processor: LogitsProcessor) -> LogitsChainBuilder:
        """Append a user-supplied LogitsProcessor after the canonical
        group. Useful for logit-bias / forbidden-token / domain-specific
        adjustments. Multiple `.custom(...)` calls append in order."""
        self._custom.append(processor)
        return self

    def build(self) -> LogitsChain:
        """Construct the LogitsChain with processors in canonical order.

        Standard processors that were chained are emitted in the
        fixed `_CANONICAL_ORDER`; custom processors come after, in
        the order they were added.
        """
        ordered: list[LogitsProcessor] = []
        for proc_type in _CANONICAL_ORDER:
            if proc_type in self._standard:
                ordered.append(self._standard[proc_type])
        ordered.extend(self._custom)
        return LogitsChain(processors=ordered)
