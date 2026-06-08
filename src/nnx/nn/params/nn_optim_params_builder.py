"""Builder for NNOptimParams — variant-gated optimizer config.

Adam variants take `betas: tuple[float, float]` (PyTorch's spelling);
SGD variants take `momentum: float`. Both map onto the underlying
`NNOptimParams.momentum` field, which holds whichever shape is correct
for the chosen optimizer kind (see `NNOptimParams.is_valid()`).

The rename is purely Builder-side. `from_state` and the direct-kwarg
ctor still take `momentum`, so on-disk YAML round-trips unchanged. The
Builder is the spot where we present the PyTorch-native spelling.

See `docs/superpowers/specs/2026-05-31-builder-pattern-investigation.md`
§3.3 for the rubric scoring and design rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ..enum.optims import Optims
from .nn_optim_params import NNOptimParams

if TYPE_CHECKING:
    from ...finetune.param_groups import NNParamGroupSpec


class NNOptimParamsBuilder:
    """Variant-aware builder for `NNOptimParams`.

    Reach via `NNOptimParams.builder()`. Pick exactly one variant
    method (`adam`, `adam_amsgrad`, `sgd`, `sgd_nesterov`), then chain
    optional methods (`grad_clip`, `accumulate_grad`, `param_groups`),
    then `.build()`. Method-call order is independent — a modifier
    called before a variant survives the variant call, and the last
    variant always wins.
    """

    # Fields that a variant method owns. `_set_variant` drops these
    # from `self._fields` before applying the new variant so a second
    # variant call cleanly replaces the first AND any modifier-set
    # keys (grad_clip_norm / accumulate_grad_batches / param_groups)
    # survive.
    _VARIANT_KEYS: ClassVar[tuple[str, ...]] = ("name", "max_lr", "momentum", "weight_decay")

    def __init__(self) -> None:
        self._fields: dict[str, Any] = {}

    def _set_variant(self, **fields: Any) -> None:
        for k in self._VARIANT_KEYS:
            self._fields.pop(k, None)
        self._fields.update(fields)

    # ---------- variant methods ----------

    def adam(
        self,
        *,
        max_lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ) -> NNOptimParamsBuilder:
        """torch.optim.Adam. `betas` is PyTorch's name for the
        (beta1, beta2) tuple; the Builder maps it onto the underlying
        `NNOptimParams.momentum` field (which holds the tuple for Adam
        variants).
        """
        self._set_variant(
            name=Optims.ADAM,
            max_lr=max_lr,
            momentum=betas,
            weight_decay=weight_decay,
        )
        return self

    def adam_amsgrad(
        self,
        *,
        max_lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
    ) -> NNOptimParamsBuilder:
        """torch.optim.Adam with `amsgrad=True`. Same `betas` mapping
        as `adam()`.
        """
        self._set_variant(
            name=Optims.ADAM_AMSGRAD,
            max_lr=max_lr,
            momentum=betas,
            weight_decay=weight_decay,
        )
        return self

    def sgd(
        self,
        *,
        max_lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> NNOptimParamsBuilder:
        """torch.optim.SGD. The float momentum stays as `momentum`
        (no rename) — `betas` is an Adam-family term.
        """
        self._set_variant(
            name=Optims.SGD,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        return self

    def sgd_nesterov(
        self,
        *,
        max_lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> NNOptimParamsBuilder:
        """torch.optim.SGD with `nesterov=True`. Same momentum shape
        as `sgd()`.
        """
        self._set_variant(
            name=Optims.SGD_NESTEROV,
            max_lr=max_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        return self

    # ---------- optional modifiers (chain after variant) ----------

    def grad_clip(self, norm: float) -> NNOptimParamsBuilder:
        """Global-L2 gradient-norm clipping. None = no clipping (the
        dataclass default; this method is the opt-in path).
        """
        self._fields["grad_clip_norm"] = norm
        return self

    def accumulate_grad(self, batches: int) -> NNOptimParamsBuilder:
        """Accumulate gradients over `batches` mini-batches before
        stepping. Default (no call) leaves the dataclass at 1.
        """
        self._fields["accumulate_grad_batches"] = batches
        return self

    def param_groups(self, groups: list[NNParamGroupSpec]) -> NNOptimParamsBuilder:
        """Per-layer-group LR / weight_decay overrides (the fine-tuning
        idiom). Default (no call) leaves the dataclass at None
        (single-group behavior).
        """
        self._fields["param_groups"] = groups
        return self

    # ---------- terminator ----------

    def build(self) -> NNOptimParams:
        """Construct the dataclass from the fields the user touched.

        Pre-empts the dataclass's missing-required-argument TypeError
        with an actionable Builder-level ValueError naming the variant
        methods — matches the [[builder-pattern-shape]] §11b convention
        that PR #52 established on NNTrainerParamsBuilder.

        Forwards only the keys present in `self._fields` so the
        dataclass defaults govern every untouched optional field —
        that's what preserves the omit-when-default state() invariant.
        """
        if "name" not in self._fields:
            raise ValueError(
                "NNOptimParamsBuilder: call one of .adam(...), "
                ".adam_amsgrad(...), .sgd(...), or .sgd_nesterov(...) "
                "before .build() — a variant selects the optimizer kind "
                "and sets the required name/max_lr/momentum/weight_decay fields."
            )
        return NNOptimParams(**self._fields)
