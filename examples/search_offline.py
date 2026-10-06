"""Budgeted experiment search with Optuna, offline (FEAT-033).

A three-trial study over the learning rate of a small classifier:

  1. **A base plan, unchanged.** ``ExperimentPlan`` holds the network, the
     training parameters (validation-loss monitor) and the data as
     factories; ``with_lr`` returns a *new* plan per trial.
  2. **Search.** ``nnx.search.search(plan, space, apply=with_lr,
     monitor=MonitorSpec("loss"), budget=SearchBudget(trials=3), ...)``
     asks a grid of three learning rates. A ``ThresholdPruner`` prunes the
     trial whose first validation loss is above 1.0 — the tiny learning
     rate barely moves — so one outcome is ``pruned`` and the best of the
     completed ones names its run and BEST checkpoint.
  3. **Reopen under budget.** Searching the stored study again with the
     same budget asks no new trial (every trial — completed, pruned, failed
     or orphaned — counts against it).

Requires the ``optuna`` extra (``pip install "thekaveh-nnx[optuna]"``).
Optuna's study lives in ``<output>/study.db`` (SQLite); NNx's runs in
``<output>/runs/``.

Run:
    python examples/search_offline.py [--output search-out]

The bounded ``search_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx.monitors import MonitorSpec
from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_train_params import NNTrainParams
from nnx.plans import ExperimentPlan
from nnx.search import CategoricalParam, SearchBudget, SearchSpace, search

LEARNING_RATES = (0.1, 1e-4, 0.05)
DATA = torch.Generator().manual_seed(0)
X = torch.randn(48, 4, generator=DATA)
Y = (X[:, 0] > 0).long() + (X[:, 1] > 0).long()  # three classes from two features
X_VAL = torch.randn(24, 4, generator=DATA)
Y_VAL = (X_VAL[:, 0] > 0).long() + (X_VAL[:, 1] > 0).long()


def base_plan() -> ExperimentPlan:
    return (
        ExperimentPlan()
        .with_net(NNParams(input_dim=4, output_dim=3, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU))
        .with_model(NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY))
        .with_train(
            NNTrainParams(
                n_epochs=3, optim=NNOptimParams.builder().sgd(max_lr=0.05).build(), monitor=MonitorSpec("loss")
            )
        )
        .with_data(
            lambda: DataLoader(TensorDataset(X, Y), batch_size=8),
            val=lambda: DataLoader(TensorDataset(X_VAL, Y_VAL), batch_size=12),
            identity="toy-three-class",
        )
        .with_seed(0)
    )


def with_lr(plan: ExperimentPlan, params) -> ExperimentPlan:
    return plan.with_optim(replace(plan.train.optim, max_lr=params["lr"]))


def run_search(output: str):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{os.path.join(output, 'study.db')}"
    return search(
        base_plan(),
        SearchSpace(CategoricalParam(name="lr", choices=LEARNING_RATES)),
        apply=with_lr,
        monitor=MonitorSpec("loss"),
        budget=SearchBudget(trials=3),
        study_name="lr-search",
        storage=storage,
        sampler=optuna.samplers.GridSampler({"lr": list(LEARNING_RATES)}, seed=0),
        pruner=optuna.pruners.ThresholdPruner(upper=1.0),
    )


def search_workflow(output: str = ".") -> dict:
    """Search, check the outcomes, reopen under budget."""
    previous = os.getcwd()
    os.makedirs(output, exist_ok=True)
    os.chdir(output)  # runs/ and study.db live in the output directory
    try:
        first = run_search(".")
        states = sorted(outcome.state for outcome in first.outcomes)
        if states != ["completed", "completed", "pruned"]:
            raise RuntimeError(f"expected two completed trials and one pruned, got {states}")
        best = first.best
        if best is None or best.checkpoint is None or not os.path.exists(best.checkpoint):
            raise RuntimeError("the best trial must name a completed run and its BEST checkpoint")
        again = run_search(".")  # the same budget: nothing left to ask
        if again.asked != 0 or len(again.outcomes) != 3:
            raise RuntimeError(f"reopening under budget asked {again.asked} trial(s)")
        return {
            "outcomes": [
                {
                    "lr": o.params["lr"],
                    "state": o.state,
                    "reason": o.reason,
                    "val_loss": o.monitor_value,
                    "epochs": o.epochs,
                }
                for o in first.outcomes
            ],
            "best": {"lr": best.params["lr"], "run": best.run_id, "checkpoint": os.path.relpath(best.checkpoint)},
            "reopened_asked": again.asked,
            "optuna_storage": os.path.join(output, "study.db"),
            "nnx_runs": os.path.join(output, "runs"),
        }
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description="Budgeted search with Optuna, offline")
    parser.add_argument("--output", default=None, help="where study.db and runs/ go (default: a temporary directory)")
    args = parser.parse_args()
    if args.output is None:
        with tempfile.TemporaryDirectory() as output:
            print(json.dumps(search_workflow(output), indent=2))
    else:
        print(json.dumps(search_workflow(args.output), indent=2))


if __name__ == "__main__":
    main()
