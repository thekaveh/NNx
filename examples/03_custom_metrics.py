"""Plug a custom metric callable into NNTrainParams.extra_metrics.

Demonstrates how to record any metric beyond the four hard-coded ones
(f1/recall/precision/accuracy). Each metric is called as
``fn(y_true, y_pred)`` — the truth first, the decoded class predictions
second: per batch by the default training step and once on the aggregate
by ``evaluate()`` (the default validation pass). Custom metrics show up in
``idp.train_edp.extra`` and ``idp.val_edp.extra`` and survive the
NNRun.save → NNRun.load round-trip.

``true_class0_rate`` below is deliberately *asymmetric* — it reads only the
truth — so the example can check the argument order against the validation
labels it knows: swapping the arguments would report the predicted class-0
rate instead.

Named metrics and monitors (FEAT-003) are the declarative counterpart:
``NNTrainParams(metrics=[MetricSpec("nll"), ...], monitor=MonitorSpec(...))``
computes registered metrics over the *full* sample of each epoch — labels,
probabilities or continuous outputs, as each metric declares — and one
monitor drives BEST selection, ``ReduceLROnPlateau`` and any
``EarlyStopping`` given the same spec. ``named_monitor_workflow`` below runs
one beside a decoded callable on uneven CPU batches, reloads the records and
verifies the reported value and the BEST epoch.

Run:
    python examples/03_custom_metrics.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    EarlyStopping,
    Losses,
    MetricSpec,
    MonitorSpec,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    set_seed,
)


def _model() -> NNModel:
    return NNModel(
        net_params=NNParams(input_dim=8, output_dim=3, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def named_monitor_workflow() -> dict:
    """Bounded demonstration of named metrics and a named monitor (FEAT-003).

    Uneven batches on purpose — 50 training rows in batches of 16 (the last
    holds 2) and 21 validation rows in batches of 8 — so a mean of batch
    values would differ from the full-sample value. A decoded callable
    (``hamming_error``) runs beside the declared ``nll`` / ``brier`` /
    ``accuracy`` metrics; the validation NLL is the monitor for BEST
    selection, the plateau scheduler and ``EarlyStopping``. After reloading
    the run, the reported value is recomputed independently from the BEST
    model's probabilities, and the BEST epoch is the last one the monitor
    marked as improved.
    """
    set_seed(2)
    X, y = torch.randn(50, 8), torch.randint(0, 3, (50,))
    X_val, y_val = torch.randn(21, 8), torch.randint(0, 3, (21,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16)  # batches [16, 16, 16, 2]
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8)  # batches [8, 8, 5]

    def hamming_error(y_true, y_pred):
        return float((y_true != y_pred).mean())

    monitor = MonitorSpec(metric="nll", min_delta=1e-3)
    run = _model().train(
        params=NNTrainParams(
            n_epochs=4,
            train_loader=loader,
            val_loader=val_loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            metrics=[MetricSpec("nll"), MetricSpec("brier"), MetricSpec("accuracy")],
            monitor=monitor,
            extra_metrics={"hamming_error": hamming_error},
            data_id="named-monitor-demo",
        ),
        callbacks=[EarlyStopping(monitor=monitor, patience=2)],
    )

    reloaded = NNRun.load(run.id)
    epochs = [idp for idp in reloaded.idps if idp.selection is not None]
    for idp in epochs:
        assert idp.val_edp is not None and idp.train_summary is not None
        # The monitored value is the declared metric of the whole validation set.
        assert abs(idp.selection.value - idp.val_edp.metrics["nll"]) < 1e-12
        # The decoded callable still reports beside it.
        assert "hamming_error" in idp.val_edp.extra

    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.idp.selection is not None
    best_epoch = max(idp.epoch_idx for idp in epochs if idp.selection.improved)
    assert best.idp.epoch_idx == best_epoch, (best.idp.epoch_idx, best_epoch)

    # Recompute the reported value from the BEST model's probabilities.
    best_model = NNModel.from_checkpoint(checkpoint=best)
    probabilities = torch.softmax(torch.as_tensor(best_model.predict(X_val.numpy()).logits), dim=1)
    nll = float(-torch.log(probabilities[torch.arange(len(y_val)), y_val]).mean())
    assert abs(nll - best.idp.selection.value) < 1e-5, (nll, best.idp.selection.value)

    summary = {
        "best_epoch": best_epoch,
        "val_nll": round(best.idp.selection.value, 6),
        "improved": [idp.selection.improved for idp in epochs],
    }
    print(f"named monitor: {summary}")
    return summary


def main():
    set_seed(1)
    X = torch.randn(128, 8)
    y = torch.randint(0, 3, (128,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    X_val = torch.randn(40, 8)
    y_val = torch.randint(0, 3, (40,))
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

    model = NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[16],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )

    # Two custom metrics: 0-1 error (mirrors `error` but computed differently)
    # and the predicted-class entropy as a confidence proxy.
    def hamming_error(y_true, y_pred):
        return float((y_true != y_pred).mean())

    def predicted_class_entropy(_y_true, y_pred):
        # Distribution of predicted classes, then Shannon entropy in nats.
        _, counts = np.unique(y_pred, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    def true_class0_rate(y_true, _y_pred):
        # Asymmetric on purpose: depends on the truth only.
        return float((np.asarray(y_true) == 0).mean())

    train_params = NNTrainParams(
        n_epochs=3,
        train_loader=loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=1,
            cooldown=1,
            threshold=1e-3,
        ),
        extra_metrics={
            "hamming_error": hamming_error,
            "predicted_class_entropy": predicted_class_entropy,
            "true_class0_rate": true_class0_rate,
        },
    )

    run = model.train(params=train_params)
    last = run.idps[-1]
    print("\nCustom metrics on the final batch:")
    for name, value in last.train_edp.extra.items():
        print(f"  {name:30s} = {value:.4f}")

    # Truth-first order, checked against the known validation labels: the
    # aggregate validation value equals the class-0 rate of y_val.
    expected = float((y_val.numpy() == 0).mean())
    assert last.val_edp is not None
    assert abs(last.val_edp.extra["true_class0_rate"] - expected) < 1e-9, dict(last.val_edp.extra)

    reloaded = NNRun.load(run.id)
    assert reloaded is not None
    reloaded_last = reloaded.idps[-1]
    assert reloaded_last.val_edp is not None
    for name, value in last.val_edp.extra.items():
        assert abs(reloaded_last.val_edp.extra[name] - value) < 1e-9, name
    print(f"\nValidation true_class0_rate = {expected:.4f} (matches y_val); values survive NNRun.load().")

    named_monitor_workflow()


if __name__ == "__main__":
    main()
