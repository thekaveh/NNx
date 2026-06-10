"""DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al., NVIDIA, ICML 2024 Oral).

DoRA refines LoRA by decomposing each weight matrix into a magnitude
vector ``m`` (per output row) and a direction matrix obtained by L2-
normalizing the low-rank-augmented weight column-wise. Concretely:

    V = W_0 + (α/r) · B · A      (same residual as LoRA)
    W = m * V / ||V||_c          (DoRA decomposition; ||·||_c is per-row L2)
    y = W · x + b

where ``m`` (shape: ``out_features``) is a trainable per-output-row
magnitude. Empirically, this often outperforms LoRA at the same rank
while adding only ``out_features`` extra parameters — negligible vs
LoRA's ``r · (in_features + out_features)`` baseline.

This module ships:

  - :class:`DoRALinear` — subclass of :class:`LoRALinear`, inherits the
    LoRA residual matrices ``lora_A`` / ``lora_B``, adds a trainable
    ``magnitude`` parameter initialized from ``||W_0||_c`` so the
    forward output at step 0 equals the base layer's output exactly.
  - :func:`apply_dora_to(module, *patterns, ...)` — fnmatch-based
    in-place wrap, mirroring :func:`nnx.peft.apply_lora_to`.

Because ``DoRALinear`` is a subclass of ``LoRALinear``, the existing
:func:`nnx.peft.save_lora_weights` and :func:`nnx.peft.load_lora_weights`
helpers still pick up the ``lora_A`` / ``lora_B`` matrices unchanged.
The ``magnitude`` parameter is not LoRA-shaped; callers wanting to
persist a full DoRA adapter should use the standard
``module.state_dict()`` round-trip (the magnitude key is small —
single vector of length ``out_features`` per wrapped layer).
"""

from __future__ import annotations

import fnmatch

import torch
from torch import nn

from .lora import LoRALinear


class DoRALinear(LoRALinear):
    """Linear layer wrapped with a DoRA weight decomposition.

    Subclasses :class:`LoRALinear` to inherit the frozen-base + trainable
    low-rank residual machinery (``lora_A``, ``lora_B``, alpha/r scaling,
    optional dropout, base-freeze-on-construction). Adds a trainable
    per-output-row ``magnitude`` parameter (shape: ``out_features``)
    initialized from the column-wise L2 norm of the base weight.

    The forward composes the LoRA residual into a combined weight
    ``V = W_0 + (α/r) · BA``, normalizes ``V`` row-wise, then re-scales
    by the trainable magnitude:

        ``W = magnitude.unsqueeze(1) * V / ||V||_c``
        ``y = W · x + b``

    At step 0, ``B`` is zero-initialized (inherited from LoRALinear)
    so ``V = W_0`` and ``||V||_c == magnitude``, giving ``W == W_0``
    exactly — fine-tuning starts from the pretrained behavior.

    Args:
        base: the :class:`nn.Linear` to wrap. Its parameters are frozen
            on construction (inherited from LoRALinear).
        r: low-rank dim for the LoRA residual. Must be positive.
        alpha: scaling numerator. Effective LoRA scale is ``alpha / r``.
        dropout: dropout on the LoRA update path. Range ``[0, 1)``.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__(base, r=r, alpha=alpha, dropout=dropout)
        # Magnitude initialized from the per-output-row L2 norm of the
        # base weight. With ``B = 0`` (LoRA zero-init), the combined
        # weight ``V = W_0``, so ``magnitude * V / ||V||_c`` reduces to
        # ``||W_0||_c * (W_0 / ||W_0||_c) == W_0`` — output equals
        # base(x) exactly at step 0.
        with torch.no_grad():
            init_mag = self.base.weight.norm(p=2, dim=1)  # (out_features,)
        self.magnitude = nn.Parameter(init_mag.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compose V = W_0 + (α/r) · BA. Note: LoRALinear stores
        # ``self.scaling = alpha / r``, used for the inherited LoRA
        # forward; we reuse it here for the DoRA recomposition.
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        # Optional dropout on the LoRA matrix (not the input). NOTE:
        # this is DropConnect-style — it zeroes/rescales entries of the
        # composed BA *weight matrix* batch-wide, unlike LoRALinear,
        # which drops entries of the *input activations* per sample.
        # Deliberate: dropping x here would also perturb the row norms
        # the magnitude renormalization below depends on.
        # ``self.lora_dropout`` is Identity when dropout=0, so this is
        # a no-op in the common case.
        lora_update = self.lora_dropout(lora_update)
        V = self.base.weight + lora_update
        # Per-output-row L2 norm. clamp_min avoids divide-by-zero in
        # the pathological case where a row of V is all zeros (which
        # shouldn't happen for a sensibly-init'd base, but be safe).
        norm = V.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        V_normalized = V / norm
        W = self.magnitude.unsqueeze(1) * V_normalized
        return torch.nn.functional.linear(x, W, self.base.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling} (DoRA)"
        )


def apply_dora_to(
    module: nn.Module,
    *name_patterns: str,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Wrap every :class:`nn.Linear` submodule whose dotted name matches
    any of ``name_patterns`` with a :class:`DoRALinear`. Returns the
    number of layers wrapped.

    Mirrors :func:`nnx.peft.apply_lora_to` — same fnmatch glob conventions,
    same two-phase (collect-then-mutate) traversal, same idempotency
    contract (existing DoRA/LoRA wrappers are not re-wrapped — the
    parent-is-LoRALinear check covers DoRALinear by inheritance).

    Args:
        module: root module to walk. Mutated in place.
        name_patterns: at least one fnmatch glob.
        r: LoRA rank — passed through.
        alpha: LoRA scaling numerator — passed through.
        dropout: dropout on the LoRA update path — passed through.

    Returns:
        The count of layers wrapped (may be 0 if no patterns match
        or every match is already wrapped).

    Raises:
        ValueError: if ``name_patterns`` is empty.
    """
    if not name_patterns:
        raise ValueError("apply_dora_to requires at least one name pattern")

    # Two-phase traversal — same rationale as apply_lora_to: avoid
    # invalidating the iterator while reassigning child attributes.
    targets: list[str] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        # Skip the inner .base of an existing LoRALinear (which covers
        # DoRALinear via inheritance) — re-applying must be idempotent.
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
        setattr(parent, attr, DoRALinear(old, r=r, alpha=alpha, dropout=dropout))

    return len(targets)
