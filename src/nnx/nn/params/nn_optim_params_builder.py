"""Builder for NNOptimParams — variant-gated optimizer config.

Adam variants (`adam`, `adam_amsgrad`, `adamw`) take
`betas: tuple[float, float]` (PyTorch's spelling) and an optional `eps`;
SGD variants take `momentum: float`. Both map onto the underlying
`NNOptimParams.momentum` field, which holds whichever shape is correct
for the chosen optimizer kind (see `NNOptimParams.is_valid()`).

The rename is purely Builder-side. `from_state` and the direct-kwarg
ctor still take `momentum`, so on-disk YAML round-trips unchanged. The
Builder is the spot where we present the PyTorch-native spelling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from ..._builders import copy_containers, params_init_values
from ..enum.optims import Optims
from .nn_optim_params import NNOptimParams

if TYPE_CHECKING:
    from ...finetune.param_groups import NNParamGroupSpec


class NNOptimParamsBuilder:
    """Variant-aware builder for `NNOptimParams`.

    Reach via `NNOptimParams.builder()`. Pick exactly one variant
    method (`adam`, `adam_amsgrad`, `adamw`, `sgd`, `sgd_nesterov`), then chain
    optional methods (`grad_clip`, `accumulate_grad`, `param_groups`),
    then `.build()`. Method-call order is independent — a modifier
    called before a variant survives the variant call, and the last
    variant always wins.

    `copy()` branches a (possibly partial) builder and `from_params()`
    rebuilds one from an existing `NNOptimParams`, so shared setup is
    written once and varied per branch.
    """

    # Fields that a variant method owns. `_set_variant` drops these
    # from `self._fields` before applying the new variant so a second
    # variant call cleanly replaces the first AND any modifier-set
    # keys (grad_clip_norm / accumulate_grad_batches / param_groups)
    # survive.
    _VARIANT_KEYS: ClassVar[tuple[str, ...]] = ("name", "max_lr", "momentum", "weight_decay", "eps")
    # Every `NNOptimParams` init field `from_params` can carry.
    _PARAMS_FIELDS: ClassVar[tuple[str, ...]] = (
        "name",
        "max_lr",
        "weight_decay",
        "momentum",
        "grad_clip_norm",
        "accumulate_grad_batches",
        "param_groups",
        "eps",
    )
    # Configuration containers a branch owns (its items stay shared).
    _CONTAINER_FIELDS: ClassVar[tuple[str, ...]] = ("param_groups",)

    def __init__(self) -> None:
        self._fields: dict[str, Any] = {}

    # ---------- branching ----------

    def copy(self) -> NNOptimParamsBuilder:
        """Return an independent branch of this builder, complete or partial.

        The branch starts with the same fields; afterwards setters on
        either builder never affect the other, and neither touches values
        already built. Configuration containers (the `param_groups` list)
        are copied; the immutable `NNParamGroupSpec` rows are shared.
        Nothing is built or validated here — `build()` validates each
        branch on its own.
        """
        branch = type(self)()
        branch._fields = copy_containers(self._fields, self._CONTAINER_FIELDS)
        return branch

    @classmethod
    def from_params(cls, params: NNOptimParams) -> NNOptimParamsBuilder:
        """Return a builder pre-loaded with every field of `params`.

        `from_params(params).build()` equals `params`, with the same
        `state()` (key order and omitted defaults included). The builder
        is ordinary afterwards: a new variant call replaces the variant
        fields (`name` / `max_lr` / `momentum` / `weight_decay` / `eps`)
        and keeps the modifiers, exactly as in a hand-written chain.

        Raises:
            TypeError: if `params` is not exactly an `NNOptimParams` — a
                registered-factory `NNOptimFactoryParams` has no builder
                (pass it straight to `NNTrainerParamsBuilder.optimizer`),
                and subclasses are rejected rather than downgraded.
            ValueError: if `params` carries a field this builder cannot
                reproduce.
        """
        # Lazy import: nnx.optimizers imports this params module.
        from ...optimizers import NNOptimFactoryParams

        builder = cls()
        builder._fields = params_init_values(
            "NNOptimParamsBuilder",
            params,
            NNOptimParams,
            cls._PARAMS_FIELDS,
            containers=cls._CONTAINER_FIELDS,
            hint=(
                "registered-factory optimizers have no builder — pass the NNOptimFactoryParams "
                "value straight to NNTrainerParamsBuilder.optimizer(name, params)"
                if isinstance(params, NNOptimFactoryParams)
                else None
            ),
        )
        return builder

    def _set_variant(self, **fields: Any) -> None:
        for k in self._VARIANT_KEYS:
            self._fields.pop(k, None)
        # `eps=None` means "not set": the dataclass default (1e-8) governs,
        # which keeps eps out of state() and the run id unchanged.
        if fields.get("eps", 0.0) is None:
            fields.pop("eps")
        self._fields.update(fields)

    # ---------- variant methods ----------

    def adam(
        self,
        *,
        max_lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: Optional[float] = None,
    ) -> NNOptimParamsBuilder:
        """torch.optim.Adam. `betas` is PyTorch's name for the
        (beta1, beta2) tuple; the Builder maps it onto the underlying
        `NNOptimParams.momentum` field (which holds the tuple for Adam
        variants). `weight_decay` is Adam's coupled L2 term (added to the
        gradient); use `adamw()` for decoupled decay. `eps` defaults to
        torch's `1e-8` and is then omitted from `state()`.
        """
        self._set_variant(
            name=Optims.ADAM,
            max_lr=max_lr,
            momentum=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        return self

    def adam_amsgrad(
        self,
        *,
        max_lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: Optional[float] = None,
    ) -> NNOptimParamsBuilder:
        """torch.optim.Adam with `amsgrad=True`. Same `betas` / `eps`
        mapping as `adam()`.
        """
        self._set_variant(
            name=Optims.ADAM_AMSGRAD,
            max_lr=max_lr,
            momentum=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        return self

    def adamw(
        self,
        *,
        max_lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        eps: Optional[float] = None,
    ) -> NNOptimParamsBuilder:
        """torch.optim.AdamW — Adam with decoupled weight decay: each step
        first scales the weights by ``1 - lr * weight_decay`` and then
        applies the Adam update, instead of adding ``weight_decay * p`` to
        the gradient. `weight_decay` defaults to torch's `1e-2`; `betas` /
        `eps` map as in `adam()`.
        """
        self._set_variant(
            name=Optims.ADAMW,
            max_lr=max_lr,
            momentum=betas,
            weight_decay=weight_decay,
            eps=eps,
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

        Raises:
            ValueError: if no variant method (`.adam`, `.adam_amsgrad`,
                `.adamw`, `.sgd`, `.sgd_nesterov`) was called before
                `.build()`. The message names the five methods so the user
                can fix the chain without consulting the dataclass schema.
        """
        if "name" not in self._fields:
            raise ValueError(
                "NNOptimParamsBuilder: call one of .adam(...), "
                ".adam_amsgrad(...), .adamw(...), .sgd(...), or .sgd_nesterov(...) "
                "before .build() — a variant selects the optimizer kind "
                "and sets the required name/max_lr/momentum/weight_decay fields."
            )
        return NNOptimParams(**self._fields)
