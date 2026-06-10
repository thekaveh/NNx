"""LoRA — Low-Rank Adaptation of pretrained linear layers.

LoRA (`hu:lora`) freezes a Linear's full-rank weight matrix and adds
two trainable low-rank projections ``A`` (r × in) and ``B`` (out × r)
whose product approximates a low-rank residual:

    y = W·x  ⟶  y = W·x + (α/r) · B(A(x))

Only ``A`` and ``B`` (a tiny fraction of the original parameter count)
update during fine-tuning. The frozen ``W`` stays in place, so the
adapted model can be served by composing W with B·A at deploy time
(or by keeping the LoRA wrapper, which adds one matrix-multiply per
adapted layer).

This module ships:

  - :class:`LoRALinear` — wraps an :class:`nn.Linear`, exposes the
    original weight via ``self.base`` (frozen on construction), adds
    trainable ``lora_A`` and ``lora_B`` parameters.
  - :func:`apply_lora_to(module, *patterns, r, alpha, dropout)` —
    walks a module, finds :class:`nn.Linear` children whose dotted
    names match any glob in ``patterns``, and replaces them in-place
    with :class:`LoRALinear` wrappers.
  - :func:`save_lora_weights(module, path)` /
    :func:`load_lora_weights(module, source)` — persist ONLY the
    LoRA parameters (``lora_A`` / ``lora_B``) so adapter checkpoints
    are tiny compared to a full ``state_dict``.

The fnmatch glob conventions match those of :func:`nnx.finetune.freeze`
— dotted parameter / module names against shell wildcards. The two
modules are designed to compose: ``apply_lora_to`` for the structural
swap, ``freeze`` for any additional non-LoRA layers the user wants to
hold fixed.
"""

from __future__ import annotations

import fnmatch
import math
from pathlib import Path
from typing import Union

import torch
from torch import nn

from ._source import _resolve_source_to_state_dict


class LoRALinear(nn.Module):
    """Linear layer wrapped with a LoRA low-rank residual.

    The original :class:`nn.Linear` lives at ``self.base`` with its
    parameters frozen (``requires_grad=False``) on construction.
    ``lora_A`` and ``lora_B`` are trainable; ``lora_A`` uses
    Kaiming-uniform init and ``lora_B`` is zero-initialized so the
    layer's output at step 0 equals the base layer's output exactly
    — fine-tuning starts from the pretrained behavior and diverges
    only as B picks up gradient.

    The wrapper preserves the base layer's ``in_features`` /
    ``out_features``, so consumers that read ``base.weight.shape`` or
    pass tensors through the layer don't change.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires an nn.Linear base, got {type(base).__name__}")
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

        self.base = base
        # Freeze every parameter of the wrapped layer — the LoRA residual
        # is the only trainable bit going forward.
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        # `scaling = alpha / r` keeps the effective LR independent of r
        # — common LoRA convention.
        self.scaling = alpha / r

        in_features = base.in_features
        out_features = base.out_features

        # A: (r, in). Init with Kaiming-uniform (sqrt(5) gain), matching
        # the original LoRA implementation. B: (out, r), zero-init so
        # the residual contributes 0 at step 0.
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward stays unchanged — frozen W·x + b. The LoRA path
        # adds (α/r) · B(A(dropout(x))).
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"
        )


def apply_lora_to(
    module: nn.Module,
    *name_patterns: str,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Wrap every :class:`nn.Linear` submodule whose dotted name matches
    any of ``name_patterns`` with a :class:`LoRALinear`. Returns the
    number of layers wrapped.

    Patterns use shell-style globs (``fnmatch``) against the dotted
    submodule name as it appears in ``module.named_modules()`` — e.g.,
    ``"layers.0"``, ``"encoder.*"``, ``"*"`` for every Linear.

    The wrap is in-place: each matched layer is removed from its parent
    and replaced with a :class:`LoRALinear` wrapping it. The base
    layer's parameters end up frozen as a side effect of LoRALinear's
    construction; the LoRA parameters (``lora_A`` / ``lora_B``) are
    trainable by default.

    Args:
        module: root module to walk. The function mutates ``module``
            in place.
        name_patterns: at least one fnmatch glob. Empty raises.
        r: LoRA rank — passed through to :class:`LoRALinear`.
        alpha: LoRA scaling numerator — passed through.
        dropout: dropout on the LoRA path — passed through.

    Returns:
        The count of layers wrapped (may be 0 if no patterns match).

    Raises:
        ValueError: if ``name_patterns`` is empty.

    **Idempotency note:** if a layer is already a :class:`LoRALinear`,
    its inner ``.base`` is skipped — re-applying ``apply_lora_to``
    against the same patterns is a no-op for layers that already
    carry a LoRA wrapper. The function returns the count of NEW wraps.
    """
    if not name_patterns:
        raise ValueError("apply_lora_to requires at least one name pattern")

    # Two-phase: collect targets first, then mutate. Iterating
    # named_modules() while reassigning child attributes would
    # invalidate the traversal in ways that depend on dict iteration
    # order. Collecting first keeps the loop predictable.
    targets: list[str] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        # Skip the inner .base of an existing LoRALinear — its parent
        # is a LoRALinear, which we surface via get_submodule.
        parent_path, _, _ = name.rpartition(".")
        parent = module if not parent_path else module.get_submodule(parent_path)
        if isinstance(parent, LoRALinear):
            continue
        if any(fnmatch.fnmatch(name, p) for p in name_patterns):
            targets.append(name)

    for name in targets:
        parent_path, _, attr = name.rpartition(".")
        parent = module if not parent_path else module.get_submodule(parent_path)
        old = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(old, r=r, alpha=alpha, dropout=dropout))

    return len(targets)


def _lora_keys_only(state_dict: dict) -> dict:
    """Filter a state_dict to entries belonging to LoRA parameters
    (``lora_A`` / ``lora_B``). Used by both save and load."""
    return {k: v for k, v in state_dict.items() if "lora_A" in k or "lora_B" in k}


def save_lora_weights(module: nn.Module, path: Union[str, Path]) -> str:
    """Save ONLY the LoRA parameters of ``module`` to ``path``.

    The output is a plain ``torch.save`` of a dict-subset of the full
    state_dict, containing only keys with ``lora_A`` or ``lora_B`` in
    them. Loadable via :func:`load_lora_weights`.

    Args:
        module: any module that has been processed by
            :func:`apply_lora_to`. If no LoRA params exist, an empty
            dict is saved (the caller decides whether that's an error).
        path: destination file path.

    Returns:
        The path written (so calls can be chained).
    """
    sd = _lora_keys_only(module.state_dict())
    torch.save(sd, str(path))
    return str(path)


def load_lora_weights(module: nn.Module, source: Union[str, Path, dict]) -> int:
    """Load LoRA parameters into ``module`` from ``source``.

    Args:
        module: must already have :class:`LoRALinear` wrappers in the
            same positions as the source — apply_lora_to FIRST, then
            call this. Otherwise the keys won't match and 0 params load.
        source: either a path to a file produced by
            :func:`save_lora_weights`, or a state-dict dict directly.

    Returns:
        The number of parameter tensors loaded.

    Loads via ``module.load_state_dict(..., strict=False)`` so the
    base layer's frozen weights — which are NOT in the LoRA-only
    checkpoint — don't trigger a missing-keys error.
    """
    sd = _resolve_source_to_state_dict(source, "load_lora_weights")
    sd = _lora_keys_only(sd)
    result = module.load_state_dict(sd, strict=False)
    # strict=False silently drops keys that don't exist on the module
    # (e.g. loading into an un-adapted model) — subtract them so the
    # return value is the number of tensors that actually landed.
    return len(sd) - len(result.unexpected_keys)
