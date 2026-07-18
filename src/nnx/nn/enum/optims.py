from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from torch import nn, optim


class Optims(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAM_AMSGRAD = "adam_amsgrad"
    SGD_NESTEROV = "sgd_nesterov"

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
        """
        if net is None:
            raise ValueError("net must not be None")

        if param_groups is not None:
            # Lazy import — defers the finetune subpackage until a
            # param-grouped optimizer is actually built (no cycle today;
            # matches NNOptimParams' deferral style).
            from ...finetune.param_groups import build_param_groups

            params_or_groups = build_param_groups(
                net,
                param_groups,
                default_lr=lr_start,
                default_weight_decay=weight_decay,
                strict=strict_param_groups,
            )
        else:
            params_or_groups = net.parameters()

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
                )
            case Optims.ADAM_AMSGRAD:
                assert isinstance(momentum, tuple)
                return optim.Adam(
                    params_or_groups,
                    amsgrad=True,
                    lr=lr_start,
                    betas=momentum,
                    weight_decay=weight_decay,
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
