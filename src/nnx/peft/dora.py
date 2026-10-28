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

import torch
from torch import nn

from ._targets import wrap_selected_linears
from .lora import LoRALinear


def _stored_weights(base: nn.Linear) -> list[torch.Tensor]:
    """The tensors that store ``base``'s weight, read without running a
    parametrization: its ``original`` tensors when ``weight`` is parametrized,
    else the registered ``weight``."""
    from torch.nn.utils import parametrize

    if parametrize.is_parametrized(base, "weight"):
        plist: nn.Module = base.parametrizations["weight"]  # type: ignore[index,assignment]
        stored = {**plist._parameters, **plist._buffers}
        return [t for name, t in stored.items() if name.startswith("original") and t is not None]
    weight = base._parameters.get("weight")
    return [weight] if weight is not None else []


def _is_torchao_tensor(t: torch.Tensor) -> bool:
    """Whether ``t`` is a torchao (quantized) tensor subclass, checked by module
    so torchao need not be installed."""
    if type(t) in (torch.Tensor, nn.Parameter):  # the common case, and all dynamo tracing needs to see
        return False
    return any(cls.__module__.split(".")[0] == "torchao" for cls in type(t).__mro__)


def _bypassed_feature(base: nn.Linear) -> tuple[str, str] | None:
    """The first part of ``base``'s own forward path that DoRA would skip,
    with why and how to resolve it; ``None`` when there is none (FIX-056).

    DoRA reads ``base.weight`` / ``base.bias`` and runs ``F.linear`` itself,
    so ``base(x)`` never runs: an instance ``forward``, hooks, an unbaked
    prune mask or a legacy ``weight_norm`` / ``spectral_norm`` (the last two
    recompute ``base.weight`` in a forward pre-hook, so DoRA would read a
    stale tensor) never apply, and a quantized weight cannot be normed. The
    ``torch.nn.utils.parametrizations`` norms recompute ``base.weight`` on
    every access, so DoRA reads them correctly. Hooks are recognized by
    type, never by parameter names (as in ``surgery._utils.refuse_parametrized``).

    The weight is inspected through its stored tensors, never ``base.weight``:
    on a parametrized base that read would run the parametrization again (and
    step ``parametrizations.spectral_norm``'s power iteration in training).
    A quantized weight is recognized positively as a ``torchao`` tensor, so
    the FakeTensors of ``torch.export`` and fake-mode tracing still pass.
    """
    from torch.nn.utils.prune import BasePruningMethod
    from torch.nn.utils.spectral_norm import SpectralNorm
    from torch.nn.utils.weight_norm import WeightNorm

    bypassed = "DoRA reads the layer's weight and never calls the layer, so it would be bypassed."
    hooked = f"{bypassed} Remove the hook, or use LoRA, which calls the base layer and its hooks."
    stale = "It applies only when the layer is called, so DoRA would read a stale weight."
    if "forward" in base.__dict__:  # not vars(): torch 2.4 dynamo cannot trace it
        return "an instance forward override", f"{bypassed} Use LoRA, which calls the base layer."
    pre_hooks = list(base._forward_pre_hooks.values())
    masks = sorted(getattr(h, "_tensor_name", "?") for h in pre_hooks if isinstance(h, BasePruningMethod))
    if masks:
        return (
            f"an unbaked prune mask on {masks}",
            f"{stale} Bake the mask first (torch.nn.utils.prune.remove, or magnitude_prune(bake=True)), or use "
            "LoRA, which calls the base layer.",
        )
    norms = sorted(getattr(h, "name", "?") for h in pre_hooks if isinstance(h, (WeightNorm, SpectralNorm)))
    if norms:
        return (
            f"a legacy weight_norm / spectral_norm on {norms}",
            f"{stale} Remove the norm (torch.nn.utils.remove_weight_norm / remove_spectral_norm) or use its "
            "torch.nn.utils.parametrizations form, which DoRA reads on every forward, or use LoRA.",
        )
    if pre_hooks:
        return "a forward pre-hook", hooked
    if base._forward_hooks:
        return "a forward hook", hooked
    if base._backward_hooks or base._backward_pre_hooks:
        return "a backward hook", hooked
    for w in _stored_weights(base):  # a loop, not next(..., None): torch 2.4 dynamo cannot trace that
        if _is_torchao_tensor(w):
            return (
                f"a weight of type {type(w).__name__} (an int8 quantized weight, say)",
                "DoRA norms and adds to the weight itself, which a quantized weight does not support. Use LoRA, "
                "which composes with an int8 base.",
            )
    return None


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

    The row normalization is carried out in FP32 for FP16/BF16 layers
    (FP64 stays FP64), so a zero or tiny row of ``V`` — from pruning,
    explicit zero initialization or a zero row in a loaded base — yields
    a finite, zero output equal to the base instead of NaN, and learned
    nonzero updates stay differentiable. The effective weight is cast
    back to the layer dtype before the matmul, so the output dtype is
    unchanged. Persist a DoRA adapter with the full ``state_dict()``:
    the LoRA-only helpers omit ``magnitude``. Like :class:`LoRALinear`,
    the wrapper and its update dropout inherit the base layer's
    train/eval mode at construction.

    DoRA reads the base's ``weight`` and ``bias`` and never calls the base
    layer, so it refuses a base whose own forward path would be skipped:
    a subclass or instance ``forward``, forward / pre / backward hooks, an
    unbaked prune mask or a legacy ``torch.nn.utils.weight_norm`` /
    ``spectral_norm`` (they update ``weight`` only when the layer is called)
    and a quantized weight (torchao int8), which DoRA cannot norm. The
    refusal comes before the base is frozen, at construction, and at
    forward for a base changed after wrapping. Use :class:`LoRALinear` for
    such a layer. A ``torch.nn.utils.parametrizations`` norm recomputes
    ``weight`` on every access and is supported.

    Args:
        base: the :class:`nn.Linear` to wrap. Its parameters are frozen
            on construction (inherited from LoRALinear).
        r: low-rank dim for the LoRA residual. Must be positive.
        alpha: scaling numerator. Effective LoRA scale is ``alpha / r``.
        dropout: dropout on the LoRA update path. Range ``[0, 1)``.

    Raises:
        TypeError: if ``base`` is not an :class:`nn.Linear`, or carries one
            of the features above (at forward too, when added after
            wrapping). Nothing is frozen by a refused construction.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        if isinstance(base, nn.Linear) and type(base).forward is not nn.Linear.forward:  # DoRA recomputes the matmul
            raise TypeError(
                f"DoRA cannot wrap a {type(base).__name__}: it has its own forward (fake quantization, say), "
                "which DoRA's recomposed matmul would bypass. Use LoRA, which calls the base layer."
            )
        # Checked before LoRALinear freezes the base, so a refusal changes nothing.
        bypassed = _bypassed_feature(base) if isinstance(base, nn.Linear) else None
        if bypassed is not None:
            feature, remedy = bypassed
            raise TypeError(f"DoRA cannot wrap a {type(base).__name__} with {feature}: {remedy} Nothing was modified.")
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
        if type(self.base).forward is not nn.Linear.forward:  # replaced later (QAT prepare, say)
            raise TypeError(
                f"DoRA's base is now a {type(self.base).__name__} with its own forward (QAT prepare, say), "
                "which DoRA's recomposed matmul would bypass silently; use LoRA for a layer that QAT prepares"
            )
        bypassed = _bypassed_feature(self.base)  # added after wrapping
        if bypassed is not None:
            feature, remedy = bypassed
            raise TypeError(f"DoRA's base now has {feature}, added after wrapping: {remedy}")
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
        # Row normalization runs in a promoted dtype (FIX-015): FP16 cannot
        # represent the 1e-8 guard (it rounds to 0), so an all-zero row —
        # after pruning, explicit zero init, or a zero row in a loaded
        # base — used to divide 0/0 and poison the whole output with NaN.
        # FP16/BF16 promote to FP32; FP32 and FP64 keep their own dtype so
        # doubles are never truncated. The norm, clamp, division and
        # magnitude rescale are computed coherently in that dtype, then
        # the effective weight is cast back so the caller-facing
        # dtype/device contract (and the next half layer) is unchanged.
        # Gradients flow through the casts to A, B and magnitude; a zero
        # row stays exactly zero (0 / 1e-8 == 0), never an invented
        # direction.
        acc_dtype = torch.float32 if V.dtype in (torch.float16, torch.bfloat16) else V.dtype
        V_acc = V.to(acc_dtype)
        norm = V_acc.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        W = (self.magnitude.to(acc_dtype).unsqueeze(1) * (V_acc / norm)).to(V.dtype)
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
        ValueError: if ``name_patterns`` is empty, if a selected Linear
            is read directly by its parent (``nn.MultiheadAttention.out_proj``,
            a torch Transformer layer's ``linear1`` / ``linear2``), or if it
            is registered under more than one name — the whole matched set
            is validated before any wrapper is built, so nothing is
            modified. A Linear inside a shared container is one slot and is
            wrapped once; tied tensors between distinct modules are outside
            this check (see :func:`nnx.peft.apply_lora_to`).
        TypeError: if a selected Linear is a subclass with its own
            ``forward`` (a QAT fake-quantized layer), or carries an instance
            ``forward``, a hook, an unbaked prune mask, a legacy
            ``weight_norm`` / ``spectral_norm`` or a quantized weight, which
            DoRA's recomposed matmul would bypass (see :class:`DoRALinear`).
            Every wrapper is built before any slot is replaced, so a refused
            call modifies nothing.
    """
    # Same two-phase select/validate-then-mutate as apply_lora_to (FIX-014);
    # the LoRALinear skip covers DoRALinear by inheritance.
    return wrap_selected_linears(
        module,
        name_patterns,
        skip_inside=LoRALinear,
        helper="apply_dora_to",
        wrap=lambda base: DoRALinear(base, r=r, alpha=alpha, dropout=dropout),
    )
