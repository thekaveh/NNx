"""Helpers for the paradigm-step TaskSpec tests (#513).

Each test deep-copies one base model, trains it for one epoch on one batch
with plain SGD and compares the weights and records with the default step's.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Optional

import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    TaskSpec,
)
from nnx.nn.params.nn_conv_params import NNConvParams
from nnx.nn.params.nn_moe_params import NNMoEParams

X = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
IMAGES = torch.randn(6, 1, 8, 8, generator=torch.Generator().manual_seed(1))
MIXED = torch.tensor([0, 1, 2, 0, 1, 2])  # rows 0 and 3 are ignored with ignore_index=0
ALL_IGNORED = torch.zeros(6, dtype=torch.long)
_RUNS = itertools.count()  # a distinct run id per fit (a freed step function's repr can recur)


def model(
    task: Optional[TaskSpec],
    *,
    net: Nets = Nets.FEED_FWD,
    loss: Losses = Losses.CROSS_ENTROPY,
    output_dim: int = 3,
    seed: int = 0,
) -> NNModel:
    """A small 4-feature model (an 8x8 one-channel image model for
    ``Nets.CONV``) with ``task`` declared."""
    torch.manual_seed(seed)
    shared: dict[str, Any] = dict(output_dim=output_dim, hidden_dims=[8], dropout_prob=0.0)
    if net is Nets.CONV:
        net_params: NNParams = NNConvParams(
            input_dim=64, conv_channels=[2], kernel_size=3, activation=Activations.RELU, **shared
        )
    elif net is Nets.FEED_FWD_MOE:
        net_params = NNMoEParams(input_dim=4, num_experts=2, top_k=1, **shared)
    else:
        net_params = NNParams(input_dim=4, **shared)
    return NNModel(net_params=net_params, params=NNModelParams(net=net, device=Devices.CPU, loss=loss, task=task))


def categorical(ignore_index: int = 0, **kw: Any) -> NNModel:
    return model(TaskSpec.categorical(3, ignore_index=ignore_index), **kw)


def fit(base: NNModel, Y: torch.Tensor, *, inputs: torch.Tensor = X, lr: float = 0.5, **kw: Any) -> tuple[NNModel, Any]:
    """Train a deep copy of ``base`` for one epoch on ``(inputs, Y)``; ``kw``
    are the ``NNModel.train`` hooks (``train_step_fn`` / ``objective``)."""
    trained = copy.deepcopy(base)
    run = trained.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=[(inputs, Y)],
            optim=NNOptimParams.builder().sgd(max_lr=lr, momentum=0.0).build(),
        ),
        salt=f"task-steps-{next(_RUNS)}",
        **kw,
    )
    return trained, run


def weights(m: NNModel) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in m.net.parameters()])


def moved(m: NNModel, base: NNModel) -> float:
    """The largest change of any weight from ``base``."""
    return float((weights(m) - weights(base)).abs().max())


def record(run: Any) -> Any:
    """The run's last training record."""
    return run.idps[-1].train_edp


def no_run_left(root: Path) -> bool:
    """No run directory (or ``runs/best``) and no run lease under ``root``."""
    runs = root / "runs"
    if not runs.exists():
        return True
    leases = list((runs / ".leases").glob("*.lock")) if (runs / ".leases").exists() else []
    return not leases and not [p for p in runs.iterdir() if not p.name.startswith(".")]


def multilabel_targets() -> torch.Tensor:
    """0/1 labels with NaN masking a third of them."""
    Y = (torch.randn(6, 3, generator=torch.Generator().manual_seed(2)) > 0).float()
    Y[0, 0] = Y[2, 1] = Y[3, 2] = Y[5, 0] = Y[4, 1] = Y[1, 2] = float("nan")
    return Y
