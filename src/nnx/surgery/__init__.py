"""Model surgery — function-preserving network edits.

This subpackage ships primitives that take a trained :class:`nn.Module`
and return a fresh module with a structural change applied:

  - :func:`widen` — Net2WiderNet: grow a Linear's ``out_features`` by
    duplicating randomly chosen output units, dividing the next layer's
    corresponding incoming weights by each unit's replication count so
    the forward output is preserved exactly.
  - :func:`deepen` — Net2DeeperNet: insert an identity-initialized
    Linear (+ ReLU) after the named layer. ReLU-only; other activations
    break function-preservation.
  - :func:`drop_layer` — replace a named layer with :class:`nn.Identity`,
    optionally guided by an importance metric.
  - :func:`low_rank_factorize` — replace an :class:`nn.Linear` with the
    rank-k SVD truncation, returned as ``Sequential(Linear, Linear)``.
  - :func:`expand_embedding` — resize an :class:`nn.Embedding`, preserving
    original rows exactly and returning a frozen-mask for the original
    rows so the train loop can keep them locked while new rows learn.

All primitives return fresh modules (deep-copied + edited) so the
caller's reference to the original survives. The "refine after surgery"
loop composes with :meth:`nnx.NNModel.train` — the unique compositional
payoff of surgery + training-loop in one toolkit.
"""

from __future__ import annotations

from .deepen import deepen
from .drop_layer import drop_layer
from .embedding import expand_embedding
from .low_rank import low_rank_factorize
from .widen import widen

__all__ = [
    "widen",
    "deepen",
    "drop_layer",
    "low_rank_factorize",
    "expand_embedding",
]
