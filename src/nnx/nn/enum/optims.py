from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from torch import nn, optim


def resolve_param_groups(
    net: nn.Module,
    param_groups: Optional[list],
    *,
    lr_start: float,
    weight_decay: float,
    strict_param_groups: bool = False,
) -> list[dict]:
    """The parameter-ownership rule every optimizer is built with.

    ``param_groups=None``: one group holding every parameter of ``net``
    (the historical single-group behaviour — frozen parameters included,
    they simply never receive gradients). Otherwise the
    :func:`nnx.finetune.param_groups.build_param_groups` buckets: first
    matching spec wins, frozen parameters dropped, and unmatched
    parameters either joined into a default group (``strict=False``) or
    left out entirely (``strict=True``, the multi-optimizer Trainer).
    Built-in :class:`Optims` and registered optimizer factories
    (:mod:`nnx.optimizers`) share this one rule.
    """
    if net is None:
        raise ValueError("net must not be None")
    if param_groups is None:
        return [{"params": list(net.parameters())}]
    # Lazy import — defers the finetune subpackage until a param-grouped
    # optimizer is actually built (no cycle today; matches NNOptimParams'
    # deferral style).
    from ...finetune.param_groups import build_param_groups

    return build_param_groups(
        net,
        param_groups,
        default_lr=lr_start,
        default_weight_decay=weight_decay,
        strict=strict_param_groups,
    )


class Optims(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAM_AMSGRAD = "adam_amsgrad"
    SGD_NESTEROV = "sgd_nesterov"
    # Decoupled weight decay (Loshchilov & Hutter): torch.optim.AdamW decays
    # the weights directly instead of adding an L2 term to the gradient as
    # ADAM / ADAM_AMSGRAD do.
    ADAMW = "adamw"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(
        self,
        net: nn.Module,
        lr_start: float,
        weight_decay: float,
        momentum: Union[float, tuple[float, float]],
        param_groups: Optional[list] = None,
        strict_param_groups: bool = False,
        eps: float = 1e-8,
    ) -> optim.Optimizer:
        """Build the underlying torch optimizer.

        When ``param_groups`` is None (back-compat default), constructs
        the optimizer with a single group: every trainable parameter of
        ``net`` at ``lr=lr_start``, ``weight_decay=weight_decay``.

        When ``param_groups`` is set to a list of
        :class:`nnx.finetune.NNParamGroupSpec`, dispatches to
        :func:`nnx.finetune.param_groups.build_param_groups` to bucket
        parameters by fnmatch pattern and apply per-group LR /
        weight_decay overrides. Frozen parameters (``requires_grad=False``)
        are dropped — the optimizer doesn't need to know about them.

        ``strict_param_groups`` toggles between fine-tuning and
        multi-optimizer-Trainer semantics: when False (default),
        unmatched parameters go into a default group at ``lr_start``;
        when True, unmatched parameters are dropped from the optimizer
        entirely. The Trainer passes True so disjoint optimizers don't
        end up co-owning the same params via implicit default buckets.

        ``eps`` is forwarded to the Adam family (ADAM, ADAM_AMSGRAD,
        ADAMW) and ignored by the SGD variants; ``1e-8`` is torch's
        default, so omitting it builds exactly what earlier versions did.
        ADAM / ADAM_AMSGRAD apply ``weight_decay`` as a coupled L2 term on
        the gradient; ADAMW applies it as decoupled weight decay.
        """
        params_or_groups = resolve_param_groups(
            net,
            param_groups,
            lr_start=lr_start,
            weight_decay=weight_decay,
            strict_param_groups=strict_param_groups,
        )

        match self:
            case Optims.SGD:
                assert isinstance(momentum, float)
                return optim.SGD(
                    params_or_groups,
                    lr=lr_start,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
            case Optims.ADAM:
                assert isinstance(momentum, tuple)
                return optim.Adam(
                    params_or_groups,
                    lr=lr_start,
                    betas=momentum,
                    weight_decay=weight_decay,
                    eps=eps,
                )
            case Optims.ADAMW:
                assert isinstance(momentum, tuple)
                return optim.AdamW(
                    params_or_groups,
                    lr=lr_start,
                    betas=momentum,
                    weight_decay=weight_decay,
                    eps=eps,
                )
            case Optims.ADAM_AMSGRAD:
                assert isinstance(momentum, tuple)
                return optim.Adam(
                    params_or_groups,
                    amsgrad=True,
                    lr=lr_start,
                    betas=momentum,
                    weight_decay=weight_decay,
                    eps=eps,
                )
            case Optims.SGD_NESTEROV:
                assert isinstance(momentum, float)
                return optim.SGD(
                    params_or_groups,
                    nesterov=True,
                    lr=lr_start,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
