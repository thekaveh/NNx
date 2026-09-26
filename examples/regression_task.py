"""Regression and multilabel through task adapters (FEAT-002).

``NNModel`` assumes categorical classification unless told otherwise. A
``TaskSpec`` on ``NNModelParams(task=...)`` declares the task instead, and
its adapter drives the *default* training step, ``evaluate()``,
``predict()`` and ``predict_proba()`` — no custom step functions:

  * **regression** — continuous targets shaped like the output (NaN marks a
    masked target), loss and metrics averaged over the valid targets only,
    records carrying ``mse`` / ``mae`` and *no* classification fields,
    ``predict()`` returning the values themselves (no argmax), and
    ``predict_proba()`` returning a result with ``probabilities=None``.
  * **multilabel** — independent labels decoded at a declared probability
    threshold, with subset and element accuracy reported side by side.

This script fits a two-target regression with one masked target, validates
it every epoch, saves and reloads the run and its checkpoint (the task and
every record survive), and checks continuous rich prediction; then it
scores a tiny multilabel batch where subset and element accuracy differ.

Fully offline, CPU only.

Run:
    python examples/regression_task.py

The bounded ``regression_task_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory, and the
script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    TaskSpec,
    set_seed,
    task_adapter,
)


def regression_task_workflow() -> None:
    set_seed(0)
    task = TaskSpec.regression(2, labels=["sum", "difference"])
    model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=2, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=task),
    )
    x = torch.randn(48, 3)
    y = torch.stack([x.sum(dim=1), x[:, 0] - x[:, 1]], dim=1)
    y[5, 1] = math.nan  # one missing target: masked out of loss and metrics alike
    train_loader = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=10)
    val_loader = DataLoader(TensorDataset(x[32:], y[32:]), batch_size=10)

    params = (
        NNTrainParams(n_epochs=3, optim=NNOptimParams.builder().adam(max_lr=1e-2).build())
        .with_train_loader(train_loader)
        .with_val_loader(val_loader)
    )
    run = model.train(params=params)

    # Every record is a regression record: counts of valid targets, MSE / MAE,
    # and no fabricated accuracy / f1 / recall / precision.
    first, last = run.idps[0].train_edp, run.idps[-1]
    assert first.kind == "regression" and first.count == 19 and first.f1 is None  # 10 rows x 2, one masked
    assert last.val_edp is not None and last.val_edp.count == 32 and set(last.val_edp.metrics) == {"mse", "mae"}
    assert last.val_edp.accuracy is None and last.val_edp.error is None

    # Save / load: the task rides on the run and the checkpoint.
    reloaded = NNRun.load(run.id)
    assert reloaded.model.task == task and reloaded.idps[-1].val_edp.kind == "regression"  # type: ignore[union-attr]
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.model_params.task == task
    restored = NNModel.from_checkpoint(checkpoint)

    # Continuous prediction: values, not argmax; rich prediction has no probabilities.
    probe = x[32:36].numpy()
    values = restored.predict(probe).classes
    assert values.shape == (4, 2) and values.dtype.kind == "f"
    rich = restored.predict_proba(probe)
    assert rich.probabilities is None and rich.spec is None and rich.kind == "continuous"
    np.testing.assert_allclose(rich.decoded, values)
    evaluated = restored.evaluate(val_loader)
    assert math.isclose(evaluated.metrics["mse"], last.val_edp.metrics["mse"], rel_tol=1e-6)

    # Multilabel: independent labels; subset accuracy (every label right)
    # differs from element accuracy (each label counted on its own).
    multilabel = task_adapter(TaskSpec.multilabel(2, threshold=0.5))
    logits = torch.tensor([[4.0, -4.0], [4.0, -4.0]])  # predicts [1, 0] twice
    record = multilabel.record(logits, torch.tensor([[1.0, 0.0], [1.0, 1.0]]), loss=0.0)
    assert record.metrics["subset_accuracy"] == 0.5 and record.metrics["element_accuracy"] == 0.75

    print(
        f"regression run {run.id}: val mse={last.val_edp.metrics['mse']:.4f}, "
        f"mae={last.val_edp.metrics['mae']:.4f}; multilabel subset/element accuracy = "
        f"{record.metrics['subset_accuracy']}/{record.metrics['element_accuracy']}"
    )


def main() -> None:
    regression_task_workflow()


if __name__ == "__main__":
    main()
