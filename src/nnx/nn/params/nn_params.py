from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from ..enum.activations import Activations


@dataclass(frozen=True, kw_only=True, slots=True)
class NNParams:
    dropout_prob: float
    n_heads: Optional[int] = field(default=None)
    activation: Optional[Activations] = field(default=Activations.LEAKY_RELU)

    input_dim: int = field(repr=False)
    output_dim: int = field(repr=False)
    hidden_dims: Optional[list[int]] = field(repr=False, default=None)
    _dims: Optional[list[int]] = field(repr=False, init=False, default=None)

    @property
    def dims(self) -> list[int]:
        # `_dims` is set unconditionally in __post_init__, so the
        # Optional type on the field is purely a dataclass artifact
        # (no init=True default for slotted frozen subclasses). The
        # assert documents that contract for both readers and the
        # type-checker (pyright can't model __post_init__-set fields
        # via `object.__setattr__`).
        assert self._dims is not None
        return self._dims

    def __post_init__(self):
        dims = [self.input_dim]
        dims += self.hidden_dims if self.hidden_dims is not None else []
        dims += [self.output_dim]

        object.__setattr__(self, "_dims", dims)

    def state(self) -> dict:
        ret = dict(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            dropout_prob=self.dropout_prob,
            hidden_dims=str(self.hidden_dims),
            activation=str(self.activation),
        )

        if self.n_heads is not None:
            ret["n_heads"] = self.n_heads

        return ret

    @staticmethod
    def from_state(state: dict) -> NNParams:
        return NNParams(
            input_dim=state["input_dim"],
            output_dim=state["output_dim"],
            dropout_prob=state["dropout_prob"],
            activation=Activations(state["activation"]),
            hidden_dims=ast.literal_eval(state["hidden_dims"]),
            n_heads=state["n_heads"] if "n_heads" in state else None,
        )
