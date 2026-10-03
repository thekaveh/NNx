"""Streaming prediction and mergeable metrics (FEAT-020).

``NNModel.predict()`` returns only after the whole loader has run.
``NNModel.iter_predict(loader)`` streams the same predictions one loader
batch at a time and holds nothing once a batch is handed over. This script
shows:

  1. **Ordered categorical and regression streams.** Batches of ``[2, 2, 1]``
     rows arrive in loader order; concatenated they are exactly ``predict()``
     and ``predict_proba()``, sample ids included — class probabilities for a
     categorical task, the continuous values for a regression task.
  2. **An early close.** Leaving the ``with`` block after one batch closes
     the stream, restores the network's training mode and drops the stream's
     hold on the loader; the loader itself can be iterated again.
  3. **Mergeable metrics.** ``StreamingMetrics`` accumulates NLL and Brier
     score per batch; two halves merged (in either order) equal one pass.
  4. **Accumulator-backed validation.** ``train(eval_step_fn=streaming_eval_step)``
     builds each epoch's validation record from counts and sums and matches
     the default (eager) validation record on uneven, partly masked batches.

Fully offline, CPU only; the training runs are written under a temporary
directory.

Run:
    python examples/prediction_stream.py

The bounded ``prediction_stream_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory, and the
script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    MetricSpec,
    MonitorSpec,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    StreamingMetrics,
    TaskSpec,
    set_seed,
    streaming_eval_step,
)
from nnx.streaming import concatenate_predictions

DATA = torch.Generator().manual_seed(0)
X = torch.randn(5, 4, generator=DATA)


def _model(task: TaskSpec, loss: Losses) -> NNModel:
    set_seed(0)
    width = task.num_outputs or 1
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=width, hidden_dims=[8], dropout_prob=0.1, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss, task=task),
    )


def prediction_stream_workflow() -> dict:
    # 1. Ordered streams: [2, 2, 1] rows, concatenating to the eager results.
    classes = TaskSpec.categorical(3, labels=("low", "mid", "high"))
    classifier = _model(classes, Losses.CROSS_ENTROPY)
    labels = torch.tensor([0, 2, 1, 2, 0])
    loader = DataLoader(TensorDataset(X, labels), batch_size=2)
    with classifier.iter_predict(loader, rich=True) as stream:
        streamed = list(stream)
    rows = [len(batch.sample_ids) for batch in streamed]
    whole = concatenate_predictions(streamed)
    eager = classifier.predict_proba(loader)
    assert rows == [2, 2, 1]
    assert np.array_equal(whole.probabilities, eager.probabilities)  # type: ignore[union-attr]
    assert np.array_equal(whole.sample_ids, eager.sample_ids)

    regressor = _model(TaskSpec.regression(2), Losses.MEAN_SQUARED_ERROR)
    targets = torch.zeros(5, 2)
    values_loader = DataLoader(TensorDataset(X, targets), batch_size=2)
    with regressor.iter_predict(values_loader) as stream:
        values = concatenate_predictions(list(stream))
    assert np.array_equal(values.classes, regressor.predict(values_loader).classes)  # type: ignore[union-attr]

    # 2. An early close: one batch, then out of the with block.
    classifier.net.train()
    with classifier.iter_predict(loader) as stream:
        first = next(stream)
    assert stream.closed and classifier.net.training  # closed, and the network is back in train mode
    assert len(list(classifier.iter_predict(loader))) == 3  # the loader is still the caller's

    # 3. Mergeable metrics: two halves merged == one pass, in either order.
    declared = [MetricSpec("nll"), MetricSpec("brier")]
    halves = [StreamingMetrics.for_task(declared, classes), StreamingMetrics.for_task(declared, classes)]
    one_pass = StreamingMetrics.for_task(declared, classes)
    for index, batch in enumerate(streamed):
        truth = labels[batch.sample_ids]
        halves[index % 2].update(truth, probabilities=batch.probabilities)
        one_pass.update(truth, probabilities=batch.probabilities)
    merged = halves[0].merge(halves[1]).finalize()
    swapped = halves[1].merge(halves[0]).finalize()
    assert merged.count == one_pass.count == 5
    assert np.allclose(list(merged.values.values()), list(one_pass.finalize().values.values()))
    assert np.allclose(list(swapped.values.values()), list(merged.values.values()))

    # 4. Accumulator-backed validation on uneven, partly masked batches.
    nan = float("nan")
    y = torch.tensor([[1.0, nan], [0.5, -1.0], [nan, nan], [2.0, 0.0], [0.0, nan]])
    batches = [(X[0:2], y[0:2]), (X[2:4], y[2:4]), (X[4:5], y[4:5])]

    def fit(eval_step_fn=None, data_id=None):
        model = _model(TaskSpec.regression(2), Losses.MEAN_SQUARED_ERROR)
        params = NNTrainParams(
            n_epochs=3,
            data_id=data_id,  # the step is not part of the run id: keep the two runs apart
            train_loader=batches,
            val_loader=batches,
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            metrics=[MetricSpec("mae")],
            monitor=MonitorSpec(metric="mae"),
            seed=0,
        )
        return model.train(params=params, eval_step_fn=eval_step_fn)

    eager_run, streamed_run = fit(data_id="eager"), fit(streaming_eval_step, data_id="streamed")
    eager_val = eager_run.idps[-1].val_edp
    streamed_val = streamed_run.idps[-1].val_edp
    assert eager_val is not None and streamed_val is not None
    assert streamed_val.count == eager_val.count == 6  # masked targets are not counted
    assert np.isclose(streamed_val.loss, eager_val.loss) and np.isclose(
        streamed_val.metrics["mae"], eager_val.metrics["mae"]
    )

    summary = {
        "rows": rows,
        "early_close": {"first_rows": len(first), "closed": stream.closed},
        "metrics": {name: round(value, 4) for name, value in merged.values.items()},
        "validation": {"count": streamed_val.count, "mae": round(streamed_val.metrics["mae"], 4)},
    }
    print(f"streamed batches of {rows} rows == predict_proba(), sample ids included")
    print(f"regression stream == predict(): {values.classes.shape[0]} rows of 2 values")  # type: ignore[union-attr]
    print(f"early close after {len(first)} rows: stream closed, network back in train mode, loader reusable")
    print(f"merged metrics (either order) == one pass: {summary['metrics']}")
    print(
        f"streaming validation == eager validation: count {streamed_val.count}, mae {streamed_val.metrics['mae']:.4f}"
    )
    return summary


def main() -> None:
    home = os.getcwd()
    with tempfile.TemporaryDirectory() as root:
        os.chdir(root)  # runs/ lands in the temporary directory
        try:
            prediction_stream_workflow()
        finally:
            os.chdir(home)


if __name__ == "__main__":
    main()
