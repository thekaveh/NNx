from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from torch import nn, optim


class Optims(Enum):
    SGD             = "sgd"
    ADAM            = "adam"
    ADAM_AMSGRAD    = "adam_amsgrad"
    SGD_NESTEROV    = "sgd_nesterov"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def __call__(
        self
        , net           : nn.Module
        , lr_start      : float
        , weight_decay  : float
        , momentum      : Union[float, tuple[float, float]]
        , param_groups  : Optional[list] = None
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
        """
        if net is None:
            raise ValueError("net must not be None")

        if param_groups is not None:
            # Lazy import to avoid a cycle: build_param_groups lives in
            # nnx.finetune, which depends on this module via NNOptimParams.
            from ...finetune.param_groups import build_param_groups
            params_or_groups = build_param_groups(
                net, param_groups,
                default_lr=lr_start,
                default_weight_decay=weight_decay,
            )
        else:
            params_or_groups = net.parameters()

        match self:
            case Optims.SGD:
                return optim.SGD(
                    params_or_groups,
                    lr=lr_start,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
            case Optims.ADAM:
                return optim.Adam(
                    params_or_groups,
                    lr=lr_start,
                    betas=momentum,
                    weight_decay=weight_decay,
                )
            case Optims.ADAM_AMSGRAD:
                return optim.Adam(
                    params_or_groups,
                    amsgrad=True,
                    lr=lr_start,
                    betas=momentum,
                    weight_decay=weight_decay,
                )
            case Optims.SGD_NESTEROV:
                return optim.SGD(
                    params_or_groups,
                    nesterov=True,
                    lr=lr_start,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
