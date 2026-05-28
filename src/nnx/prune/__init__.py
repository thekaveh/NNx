"""Network pruning — magnitude-unstructured and 2:4 semi-structured.

Two complementary pruning strategies layered on top of plain :class:`nn.Linear`
submodules without touching the surrounding training loop:

  - **Magnitude unstructured** — :func:`magnitude_prune`. Zeros the
    s-fraction smallest-magnitude entries of each matched layer's
    ``weight`` matrix. Pure compression; no hardware speedup on dense
    matmul, but the resulting weight matrices store more zeros and
    quantize / compress on disk dramatically better.
  - **2:4 semi-structured** — :func:`semi_structured_24`. Swaps each
    matched :class:`nn.Linear`'s ``weight`` with a 2:4 structured-sparse
    tensor via ``torchao.sparsity``. **Real wall-clock speedup** on
    Ampere+ GPUs (~1.1× inference, ~1.3× training per torchao's ViT/SAM
    benchmarks); CPU and pre-Ampere hardware fall back to the dense path.

The two functions share the same pattern-filter convention as
:func:`nnx.peft.apply_lora_to` and :func:`nnx.finetune.freeze` —
fnmatch globs against dotted submodule names from
``module.named_modules()``.

Structured pruning that REMOVES channels / heads (and so changes the
weight matrix shape) is deferred — it breaks ``state_dict``
shape-compat invariants the same way changing :class:`nn.Linear`
in_features would, and needs a per-architecture surgery API that
the existing checkpoint format doesn't yet support.
"""

from __future__ import annotations

from .magnitude import magnitude_prune
from .semi_structured import semi_structured_24

__all__ = [
    "magnitude_prune",
    "semi_structured_24",
]
