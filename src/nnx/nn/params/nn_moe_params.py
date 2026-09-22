"""Mixture-of-Experts feed-forward params (#88).

``NNMoEParams`` subclasses ``NNParams`` — the same lift-via-subclassing pattern
``NNTransformerParams`` uses. It adds the two MoE routing knobs consumed by
:class:`~nnx.nn.net.feed_fwd_moe_nn.FeedFwdMoENN`'s ``MoELinear`` hidden layers:

- ``hidden_dims`` — inherited from ``NNParams``, but specialized here to require
  at least one entry so every ``FeedFwdMoENN`` contains an expert layer.

- ``num_experts`` — experts per hidden layer (required, and must be at least 2;
  a single-expert network is not a mixture and is rejected by ``MoELinear``).
- ``top_k`` — experts each token routes to (default 2, the Switch/Mixtral
  convention). Omitted from ``state()`` at its default — the omit-when-default
  invariant that keeps a "vanilla" MoE config hashing to a stable run.id as
  knobs accrue.

``num_experts`` is ALWAYS emitted by ``state()``: it is the discriminator
``NNParams.resolve_from_state`` dispatches on (mirroring how ``vocab_size``
identifies a transformer state), and hashing it is exactly what keeps an MoE
run's id distinct from its plain-FeedFwd twin.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..._validation import require_count
from .nn_params import NNParams


@dataclass(frozen=True, kw_only=True, slots=True)
class NNMoEParams(NNParams):
    """Serializable parameters for an expert-bearing feed-forward MoE.

    Unlike base :class:`NNParams`, ``hidden_dims`` must contain at least one
    layer because only hidden layers are replaced by ``MoELinear`` modules.
    """

    # Required: experts per hidden MoELinear layer.
    num_experts: int
    # Routed experts per token; 2 is the Switch/Mixtral convention.
    top_k: int = 2

    def __post_init__(self):
        # Explicit unbound call — same slotted-dataclass reasoning as
        # NNTransformerParams.__post_init__.
        NNParams.__post_init__(self)
        if not self.hidden_dims:
            raise ValueError("NNMoEParams requires at least one hidden layer in hidden_dims")
        # Expert and top-k counts are validated as integers and normalized
        # to plain `int` before the bound check between them (FIX-021).
        object.__setattr__(
            self, "num_experts", require_count(self.num_experts, "num_experts", owner="NNMoEParams", minimum=2)
        )
        object.__setattr__(
            self, "top_k", require_count(self.top_k, "top_k", owner="NNMoEParams", minimum=0, exclusive_min=True)
        )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"NNMoEParams requires top_k <= num_experts, got top_k={self.top_k} > num_experts={self.num_experts}"
            )

    def state(self) -> dict:
        d = NNParams.state(self)
        # Always emitted: the resolve_from_state discriminator AND the hash
        # distinctness guard vs a plain-FeedFwd config.
        d["num_experts"] = self.num_experts
        # Omit-when-default (module docstring).
        if self.top_k != 2:
            d["top_k"] = self.top_k
        return d

    @staticmethod
    def from_state(state: dict) -> NNMoEParams:
        base = NNParams.from_state(state)
        return NNMoEParams(
            input_dim=base.input_dim,
            output_dim=base.output_dim,
            hidden_dims=base.hidden_dims,
            dropout_prob=base.dropout_prob,
            activation=base.activation,
            activations=base.activations,
            dropout_probs=base.dropout_probs,
            n_heads=base.n_heads,
            num_experts=state["num_experts"],
            top_k=state.get("top_k", 2),
        )
