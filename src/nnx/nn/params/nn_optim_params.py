from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

from ..enum.optims import Optims

if TYPE_CHECKING:
    from ...finetune.param_groups import NNParamGroupSpec
    from .nn_optim_params_builder import NNOptimParamsBuilder


@dataclass(frozen=True, kw_only=True, slots=True)
class NNOptimParams:
    """Optimizer config.

    `momentum` is overloaded by optimizer kind:
      - For SGD / SGD_NESTEROV: a single float, the SGD momentum coefficient.
      - For ADAM / ADAM_AMSGRAD: a (beta1, beta2) tuple, passed as the
        Adam `betas=` argument. The name is retained for backwards
        compatibility — `is_valid()` enforces the per-optim shape.

    `grad_clip_norm` clips gradients by global L2 norm before optimizer.step().
    None = no clipping (back-compat default). Typical values: 1.0 for
    transformers, 5.0 for RNNs.

    `accumulate_grad_batches` enables gradient accumulation — the effective
    batch size becomes batch_size * accumulate_grad_batches. The loss is
    scaled by 1/N so the accumulated gradient is the mean across N batches.
    Default 1 (back-compat: step every batch).

    `param_groups` enables per-layer-group LR / weight_decay overrides — the
    fine-tuning idiom of "small LR on the backbone, large LR on the head."
    None = single-group behavior (every parameter at `max_lr` / `weight_decay`).
    When set, the optimizer factory dispatches via
    :func:`nnx.finetune.param_groups.build_param_groups` to construct
    per-group dicts.
    """

    name: Optims
    max_lr: float
    weight_decay: float
    momentum: Union[float, tuple[float, float]]

    grad_clip_norm: Optional[float] = None
    accumulate_grad_batches: int = 1
    param_groups: Optional[list[NNParamGroupSpec]] = field(default=None)

    def __post_init__(self):
        # Fail fast on plain dicts: they construct fine but crash much
        # later inside state() during NNRun hashing with an opaque
        # AttributeError. Same construction-time convention as the
        # dataset classes.
        if self.param_groups is not None:
            if not isinstance(self.param_groups, (list, tuple)):
                # A generator would be silently EXHAUSTED by the
                # validation loop below — state() would then emit an
                # empty param_groups and training would run single-group
                # with a shifted run.id.
                raise TypeError(
                    f"param_groups must be a list/tuple of NNParamGroupSpec, got {type(self.param_groups).__name__}"
                )
            # Lazy import — keeps this low-level dataclass importable
            # without eagerly loading the finetune subpackage (no cycle
            # today; same deferral style as from_state below).
            from ...finetune.param_groups import NNParamGroupSpec

            for i, g in enumerate(self.param_groups):
                if not isinstance(g, NNParamGroupSpec):
                    raise TypeError(
                        f"param_groups[{i}] must be an NNParamGroupSpec, got {type(g).__name__} — "
                        "wrap it: NNParamGroupSpec(name_pattern=..., lr=...)."
                    )

    def __str__(self):
        return f"[name={self.name}, max_lr={self.max_lr:1.0e}, weight_decay={self.weight_decay:1.0e}, momentum={self.momentum}, grad_clip={self.grad_clip_norm}, accum={self.accumulate_grad_batches}]"

    def state(self):
        d = dict(max_lr=self.max_lr, momentum=str(self.momentum), name=str(self.name), weight_decay=self.weight_decay)
        # grad_clip_norm / accumulate_grad_batches / param_groups: only emit
        # when set to a non-default value, so a NNOptimParams with none of
        # them set hashes to the same run.id as before these fields existed.
        # Existing on-disk YAML without these keys is loadable via the .get()
        # defaults below. (This invariant was broken once before for
        # grad_clip_norm — every existing run.id shifted. Same omit-when-
        # default pattern is now enforced on every params dataclass; see
        # test_params_round_trip.py for the regression tests.)
        if self.grad_clip_norm is not None:
            d["grad_clip_norm"] = self.grad_clip_norm
        if self.accumulate_grad_batches != 1:
            d["accumulate_grad_batches"] = self.accumulate_grad_batches
        if self.param_groups is not None:
            d["param_groups"] = [g.state() for g in self.param_groups]
        return d

    @staticmethod
    def from_state(state: dict) -> NNOptimParams:
        # Lazy import — defers the finetune subpackage so this
        # low-level dataclass stays light at import time (no actual
        # cycle today: param_groups.py imports only stdlib + torch).
        from ...finetune.param_groups import NNParamGroupSpec

        raw_pg = state.get("param_groups")
        param_groups = [NNParamGroupSpec.from_state(g) for g in raw_pg] if raw_pg is not None else None
        return NNOptimParams(
            max_lr=state["max_lr"],
            name=Optims(state["name"]),
            weight_decay=state["weight_decay"],
            momentum=ast.literal_eval(state["momentum"]),
            # .get() preserves back-compat with older YAML that predates
            # grad_clip_norm / accumulate_grad_batches / param_groups.
            grad_clip_norm=state.get("grad_clip_norm"),
            accumulate_grad_batches=state.get("accumulate_grad_batches", 1),
            param_groups=param_groups,
        )

    def is_valid(self) -> bool:
        if self.name == Optims.SGD or self.name == Optims.SGD_NESTEROV:
            return isinstance(self.momentum, float)
        if self.name == Optims.ADAM or self.name == Optims.ADAM_AMSGRAD:
            return (
                isinstance(self.momentum, tuple)
                and len(self.momentum) == 2
                and all(isinstance(x, float) for x in self.momentum)
            )
        # Unknown enum variant — refuse rather than implicitly returning None
        # (which would short-circuit `not params.optim.is_valid()` in train()).
        return False

    @classmethod
    def builder(cls) -> NNOptimParamsBuilder:
        """Return a variant-aware builder. See `NNOptimParamsBuilder`."""
        from .nn_optim_params_builder import NNOptimParamsBuilder

        return NNOptimParamsBuilder()
