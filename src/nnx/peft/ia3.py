"""IA3 — Infused Adapter by Inhibiting and Amplifying Inner Activations (Liu et al., NeurIPS 2022).

IA3 is the smallest adapter in the PEFT family: a single learned vector
of per-output-dim scaling factors applied multiplicatively to a frozen
:class:`nn.Linear`'s output. Trainable parameter count per wrapped
layer is exactly ``out_features`` — roughly two orders of magnitude
smaller than LoRA at the same effective adaptation budget. Empirically
competitive on smaller tasks; useful when LoRA + AdapterLayer are
still too much trainable surface for the data on hand.

This module ships:

  - :class:`IA3Linear` — wraps an :class:`nn.Linear`, freezes its
    base parameters on construction, exposes a trainable
    ``scaling`` parameter (shape: ``out_features``) initialized to
    all-ones so the layer's output at step 0 equals the base layer's
    output exactly.
  - :func:`apply_ia3_to(module, *patterns)` — fnmatch-glob in-place
    wrap mirroring :func:`nnx.peft.apply_lora_to`.
  - :func:`save_ia3_weights(module, path)` /
    :func:`load_ia3_weights(module, source)` — persist ONLY the
    ``scaling`` parameters, symmetric to LoRA's save/load idiom.
    The resulting checkpoint is tiny — a single vector per wrapped
    layer.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Union

import torch
from torch import nn

from ._source import _resolve_source_to_state_dict


class IA3Linear(nn.Module):
    """Linear layer wrapped with an IA3 per-output-dim scaling vector.

    The original :class:`nn.Linear` lives at ``self.base`` with its
    parameters frozen (``requires_grad=False``) on construction.
    ``scaling`` is the only trainable parameter: a length-``out_features``
    vector initialized to all-ones so the layer's output at step 0
    equals the base layer's output exactly.

    Forward: ``y = base(x) * scaling`` (broadcast over the trailing dim).

    Args:
        base: the :class:`nn.Linear` to wrap.
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"IA3Linear requires an nn.Linear base, got {type(base).__name__}")

        self.base = base
        # Freeze every parameter of the wrapped layer — the scaling
        # vector is the only trainable bit going forward.
        for p in self.base.parameters():
            p.requires_grad = False

        # Per-output-dim scaling vector, initialized to all-ones so
        # the layer's output at step 0 equals base(x) exactly.
        self.scaling = nn.Parameter(torch.ones(base.out_features))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `self.scaling` broadcasts over leading batch / sequence dims:
        # base(x) is (..., out_features) and scaling is (out_features,).
        return self.base(x) * self.scaling

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features} (IA3)"


def apply_ia3_to(module: nn.Module, *name_patterns: str) -> int:
    """Wrap every :class:`nn.Linear` submodule whose dotted name matches
    any of ``name_patterns`` with an :class:`IA3Linear`. Returns the
    number of layers wrapped.

    Mirrors :func:`nnx.peft.apply_lora_to` — same fnmatch glob conventions,
    same two-phase (collect-then-mutate) traversal, same idempotency
    contract (existing IA3 wrappers are skipped via the parent-is-IA3Linear
    check).

    Args:
        module: root module to walk. Mutated in place.
        name_patterns: at least one fnmatch glob.

    Returns:
        The count of layers wrapped (may be 0 if no patterns match
        or every match is already wrapped).

    Raises:
        ValueError: if ``name_patterns`` is empty.
    """
    if not name_patterns:
        raise ValueError("apply_ia3_to requires at least one name pattern")

    targets: list[str] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        # Skip the inner .base of an existing IA3Linear — re-applying
        # must be idempotent.
        parent_path, _, _ = name.rpartition(".")
        parent = module if not parent_path else module.get_submodule(parent_path)
        if isinstance(parent, IA3Linear):
            continue
        if any(fnmatch.fnmatch(name, p) for p in name_patterns):
            targets.append(name)

    for name in targets:
        parent_path, _, attr = name.rpartition(".")
        parent = module if not parent_path else module.get_submodule(parent_path)
        old = getattr(parent, attr)
        setattr(parent, attr, IA3Linear(old))

    return len(targets)


def _ia3_keys_only(state_dict: dict) -> dict:
    """Filter a state_dict to entries belonging to IA3 ``scaling`` keys.
    Used by both save and load.

    Note: the substring ``scaling`` is also the name of an inherited
    attribute on :class:`LoRALinear` (``self.scaling = alpha / r`` — a
    plain float, not a Parameter, so it does NOT appear in
    ``state_dict()``). The filter is therefore unambiguous in practice:
    only IA3's ``nn.Parameter`` named ``scaling`` lands in a state_dict
    key matching this substring.
    """
    return {k: v for k, v in state_dict.items() if "scaling" in k}


def save_ia3_weights(module: nn.Module, path: Union[str, Path]) -> str:
    """Save ONLY the IA3 ``scaling`` parameters of ``module`` to ``path``.

    The output is a ``torch.save`` of a dict-subset of the full
    state_dict, containing only keys whose name includes ``scaling``.
    Loadable via :func:`load_ia3_weights`.

    Args:
        module: any module that has been processed by
            :func:`apply_ia3_to`. If no IA3 params exist, an empty
            dict is saved.
        path: destination file path.

    Returns:
        The path written (so calls can be chained).
    """
    sd = _ia3_keys_only(module.state_dict())
    torch.save(sd, str(path))
    return str(path)


def load_ia3_weights(module: nn.Module, source: Union[str, Path, dict]) -> int:
    """Load IA3 ``scaling`` parameters into ``module`` from ``source``.

    Args:
        module: must already have :class:`IA3Linear` wrappers in the
            same positions as the source — apply_ia3_to FIRST, then
            call this. Otherwise the keys won't match and 0 params load.
        source: either a path to a file produced by
            :func:`save_ia3_weights`, or a state-dict dict directly.

    Returns:
        The number of parameter tensors loaded.

    Loads via ``module.load_state_dict(..., strict=False)`` so the
    base layer's frozen weights — which are NOT in the IA3-only
    checkpoint — don't trigger a missing-keys error.
    """
    sd = _resolve_source_to_state_dict(source, "load_ia3_weights")
    sd = _ia3_keys_only(sd)
    module.load_state_dict(sd, strict=False)
    return len(sd)
