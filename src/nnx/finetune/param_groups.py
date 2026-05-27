"""Per-layer-group learning rate config for fine-tuning.

The standard fine-tuning recipe is "small LR on the pretrained
backbone, large LR on the freshly-initialized head." PyTorch's
optimizers support this natively via the param-groups feature:

    Adam([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params,     "lr": 1e-3},
    ], lr=1e-3)

This module lets users describe such groups declaratively in
``NNOptimParams.param_groups`` (a list of :class:`NNParamGroupSpec`)
so the optimizer factory builds the per-group dicts automatically.
Parameters that don't match any pattern fall into the default group
with ``lr=NNOptimParams.max_lr``.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Optional

from torch import nn


@dataclass(frozen=True, kw_only=True, slots=True)
class NNParamGroupSpec:
    """One row in :attr:`NNOptimParams.param_groups`.

    Matches parameters whose dotted name matches ``name_pattern``
    (fnmatch glob) and applies the specified ``lr`` (absolute) or
    ``lr_multiplier`` (multiplied by ``NNOptimParams.max_lr``) and
    optional ``weight_decay`` override.

    Exactly one of ``lr`` and ``lr_multiplier`` may be set. If both are
    None the matched parameters use the optimizer's default LR — handy
    when you only want to override ``weight_decay`` for a group.

    Example:
        # Freeze nothing, but train the backbone at 1/100th the head's LR
        # and disable weight_decay on every bias term.
        NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-3,
            momentum=(0.9, 0.999),
            weight_decay=5e-4,
            param_groups=[
                NNParamGroupSpec(name_pattern="encoder.*", lr_multiplier=0.01),
                NNParamGroupSpec(name_pattern="*.bias", weight_decay=0.0),
            ],
        )
    """

    name_pattern: str
    lr: Optional[float] = None
    lr_multiplier: Optional[float] = None
    weight_decay: Optional[float] = None

    def __post_init__(self) -> None:
        if self.lr is not None and self.lr_multiplier is not None:
            raise ValueError(
                f"NNParamGroupSpec(name_pattern={self.name_pattern!r}): "
                "specify at most one of `lr` and `lr_multiplier`, not both"
            )

    def state(self) -> dict:
        d: dict = dict(name_pattern=self.name_pattern)
        if self.lr is not None:
            d["lr"] = self.lr
        if self.lr_multiplier is not None:
            d["lr_multiplier"] = self.lr_multiplier
        if self.weight_decay is not None:
            d["weight_decay"] = self.weight_decay
        return d

    @staticmethod
    def from_state(state: dict) -> NNParamGroupSpec:
        return NNParamGroupSpec(
            name_pattern=state["name_pattern"],
            lr=state.get("lr"),
            lr_multiplier=state.get("lr_multiplier"),
            weight_decay=state.get("weight_decay"),
        )


def build_param_groups(
    module: nn.Module,
    specs: list[NNParamGroupSpec],
    *,
    default_lr: float,
    default_weight_decay: float,
    strict: bool = False,
) -> list[dict]:
    """Walk ``module``'s parameters, bucket them by the first matching
    spec (or into a fallback default group), and return the list of
    param-group dicts the optimizer expects.

    Parameters with ``requires_grad=False`` are dropped — they're
    frozen, the optimizer doesn't need to see them. (Without this, the
    optimizer would still hold them in its state but they'd never
    update; harmless but wasteful and confusing in `optimizer.param_groups`.)

    Args:
        module: source of parameters to bucket.
        specs: list of :class:`NNParamGroupSpec` in priority order.
            The first spec whose ``name_pattern`` matches a parameter's
            dotted name wins.
        default_lr: LR for parameters that don't match any spec, or
            for specs that omit both ``lr`` and ``lr_multiplier``.
        default_weight_decay: WD for parameters that don't match any
            spec's ``weight_decay`` override.
        strict: when False (default, fine-tuning semantics), parameters
            that match no spec go into a default group at ``default_lr``
            so every trainable parameter ends up in the optimizer. When
            True (multi-optimizer Trainer semantics), unmatched parameters
            are DROPPED from the optimizer entirely — the contract is
            "this optimizer owns only what the specs explicitly select",
            which is what allows disjoint optimizers in
            :class:`nnx.trainer.Trainer`.

    Returns:
        A list of dicts suitable for ``torch.optim.Optimizer(
        params, ...)`` — each entry has ``"params"`` plus any overrides.
    """
    # One bucket per spec, in priority order. Each parameter goes into
    # the FIRST matching spec's bucket (break on hit, below); anything
    # unmatched falls through to default_bucket.
    buckets: list[tuple[NNParamGroupSpec, list[nn.Parameter]]] = [(spec, []) for spec in specs]
    default_bucket: list[nn.Parameter] = []

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        for spec, bucket in buckets:
            if fnmatch.fnmatch(name, spec.name_pattern):
                bucket.append(param)
                break
        else:
            default_bucket.append(param)

    out: list[dict] = []
    for spec, bucket in buckets:
        if not bucket:
            continue
        group: dict = {"params": bucket}
        if spec.lr is not None:
            group["lr"] = spec.lr
        elif spec.lr_multiplier is not None:
            group["lr"] = default_lr * spec.lr_multiplier
        else:
            group["lr"] = default_lr
        group["weight_decay"] = spec.weight_decay if spec.weight_decay is not None else default_weight_decay
        out.append(group)

    # Default bucket: included under fine-tuning semantics (every trainable
    # param ends up in the optimizer), suppressed under strict mode (the
    # caller — typically nnx.trainer.Trainer — wants this optimizer to own
    # only what the specs explicitly select).
    if default_bucket and not strict:
        out.append(
            {
                "params": default_bucket,
                "lr": default_lr,
                "weight_decay": default_weight_decay,
            }
        )

    if not out:
        raise ValueError(
            "build_param_groups produced no parameter groups — every parameter "
            "is either frozen, unmatched (under strict mode), or the module "
            "has none. Check the specs' name_pattern against module.named_parameters()."
        )

    return out
