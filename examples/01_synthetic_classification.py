"""Train a feed-forward classifier on synthetic 3-class data.

Demonstrates the core NNModel flow: build params → train with callbacks →
inspect the resulting NNRun → reload the BEST checkpoint and predict — both
with the classic ``predict()`` (logits + classes) and with the opt-in,
probability-aware ``predict_proba(X, ProbabilitySpec(...))``: equal logits
and classes, rows of probabilities summing to 1, sample ids aligned with the
input rows, and a confusion matrix from the decoded classes plus
``spec.labels`` identical to the legacy one.

``native_nll_workflow`` below is a bounded variant using
``Losses.NEGATIVE_LOG_LIKELIHOOD``: built-in nets emit raw logits, and NNx
applies ``log_softmax`` internally before native ``torch.nn.NLLLoss`` during
training and evaluation, while ``predict().logits`` stays raw. The helper
trains briefly, reloads BEST and checks the reported loss against an
explicit log-softmax/NLL reference.

Run:
    python examples/01_synthetic_classification.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    EarlyStopping,
    Losses,
    LRMonitor,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    ProbabilitySpec,
    VisUtils,
    prediction_from_logits,
    set_seed,
)


def _synthetic_loaders(n_train: int = 256, n_val: int = 64) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    # Labels derive from the inputs through a fixed random projection so
    # the task is LEARNABLE — with random labels the val error would sit
    # at 3-class chance and the BEST checkpoint would be selecting noise.
    proj = torch.randn(8, 3)
    X_train = torch.randn(n_train, 8)
    y_train = (X_train @ proj).argmax(dim=1)
    X_val = torch.randn(n_val, 8)
    y_val = (X_val @ proj).argmax(dim=1)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    return train_loader, val_loader, X_val


def native_nll_workflow() -> dict:
    """Bounded native-NLL variant (writes ``runs/`` under the current working
    directory; the smoke test runs it in a temporary one).

    Two fixed-logit rows show the contract numerically, then one short
    training epoch runs, BEST is reloaded, and its evaluation loss is
    compared against an explicit ``F.nll_loss(F.log_softmax(raw))``
    reference computed from the reloaded network's raw output.
    """
    set_seed(0)
    train_loader, val_loader, _ = _synthetic_loaders(n_train=64, n_val=32)
    nll_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.NEGATIVE_LOG_LIKELIHOOD)

    # Fixed logits [2, 1] for target 0: normalized NLL is -log softmax(2) = 0.3133,
    # while the unnormalized objective would report -2.0.
    probe = NNModel(net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[], dropout_prob=0.0), params=nll_params)
    with torch.no_grad():
        probe.net.layers[-1].weight.zero_()
        probe.net.layers[-1].bias.copy_(torch.tensor([2.0, 1.0]))
    x, y = torch.zeros(2, 2), torch.zeros(2, dtype=torch.long)
    probe_loss = probe.evaluate(DataLoader(TensorDataset(x, y), batch_size=2)).loss
    assert abs(probe_loss - 0.31326166) < 1e-6, probe_loss
    assert probe.predict(x).logits.tolist() == [[2.0, 1.0], [2.0, 1.0]]  # raw, not log-probabilities

    model = NNModel(
        net_params=NNParams(input_dim=8, output_dim=3, hidden_dims=[16], dropout_prob=0.0, activation=Activations.RELU),
        params=nll_params,
    )
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            seed=0,
            train_loader=train_loader,
            val_loader=val_loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=3, cooldown=1, threshold=1e-3),
        )
    )
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert ckpt is not None
    best_model = NNModel.from_checkpoint(checkpoint=ckpt)
    assert isinstance(best_model.loss_fn, torch.nn.NLLLoss)  # descriptor unchanged
    reported = best_model.evaluate(val_loader).loss
    X_val, y_val = val_loader.dataset.tensors  # type: ignore[attr-defined]
    with torch.no_grad():
        raw = best_model.net(X_val)
    reference = float(F.nll_loss(F.log_softmax(raw, dim=1), y_val))
    assert abs(reported - reference) < 1e-5, (reported, reference)

    summary = {"probe_loss": round(probe_loss, 6), "best_val_loss": round(reported, 6), "run_id": run.id}
    print(f"native-NLL workflow: {summary}")
    return summary


def main():
    set_seed(0)

    # 1. Build a tiny synthetic dataset. Labels derive from the inputs
    #    through a fixed random projection so the task is LEARNABLE —
    #    with random labels the val error would sit at 3-class chance
    #    and the BEST checkpoint would be selecting noise.
    n_train, n_val = 256, 64
    proj = torch.randn(8, 3)
    X_train = torch.randn(n_train, 8)
    y_train = (X_train @ proj).argmax(dim=1)
    X_val = torch.randn(n_val, 8)
    y_val = (X_val @ proj).argmax(dim=1)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # 2. Model.
    net_params = NNParams(
        input_dim=8,
        output_dim=3,
        hidden_dims=[32, 16],
        dropout_prob=0.1,
        activation=Activations.RELU,
    )
    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    model = NNModel(net_params=net_params, params=model_params)

    # 3. Train.
    train_params = NNTrainParams(
        n_epochs=20,
        seed=0,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
            momentum=(0.9, 0.999),
            weight_decay=5e-5,
            grad_clip_norm=1.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=3,
            cooldown=1,
            threshold=1e-3,
        ),
    )

    lr_monitor = LRMonitor()
    run = model.train(
        params=train_params,
        callbacks=[EarlyStopping(patience=8), lr_monitor],
    )

    # 4. Inspect.
    print(f"\nrun.id = {run.id}")
    print(f"completed iterations: {len(run.idps)}")
    last = run.idps[-1]
    print(f"final train loss: {last.train_edp.loss:.4f}, error: {last.train_edp.error:.4f}")
    if last.val_edp is not None:
        print(f"final val   loss: {last.val_edp.loss:.4f}, error: {last.val_edp.error:.4f}")
    print(f"LR trajectory: {[f'{lr:.4f}' for lr in lr_monitor.history]}")

    # 5. Reload the BEST checkpoint and run prediction.
    ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    best_model = NNModel.from_checkpoint(checkpoint=ckpt)
    result = best_model.predict(X=X_val)
    print(f"predicted classes for {len(result.classes)} val samples; first 8: {result.classes[:8]}")

    # 6. Probability-aware prediction (opt-in): declare the task explicitly.
    spec = ProbabilitySpec(kind="categorical", class_axis=1, labels=("class_0", "class_1", "class_2"))
    rich = best_model.predict_proba(X_val, spec)
    assert np.array_equal(rich.logits, result.logits) and np.array_equal(rich.decoded, result.classes)
    assert np.allclose(rich.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert rich.sample_ids.tolist() == list(range(len(X_val)))  # row i of every array is X_val[i]
    # Batched through a loader, every row keeps its identity.
    batched = best_model.predict_proba(DataLoader(TensorDataset(X_val), batch_size=24), spec)
    assert batched.sample_ids.tolist() == rich.sample_ids.tolist()
    assert np.array_equal(batched.decoded, rich.decoded)
    # Decoded classes + spec.labels give exactly the legacy confusion matrix.
    names = list(spec.labels or ())
    legacy_cm = VisUtils.confusion_matrix(y_val, result.classes, class_names=names)
    rich_cm = VisUtils.confusion_matrix(y_val, rich.class_indices, class_names=names)
    assert np.array_equal(np.asarray(legacy_cm.data[0].z), np.asarray(rich_cm.data[0].z))
    # Independent Bernoulli outputs are per-output indicators, never classes.
    bernoulli = prediction_from_logits(rich.logits, ProbabilitySpec(kind="bernoulli", labels=spec.labels))
    try:
        _ = bernoulli.class_indices
    except TypeError:
        pass
    else:
        raise AssertionError("bernoulli outputs must not be usable as class indices")
    confidence = rich.probabilities.max(axis=1)
    print(
        f"predict_proba: first 4 labels {rich.decoded_labels()[:4].tolist()}, "
        f"mean top-class probability {confidence.mean():.3f}"
    )


if __name__ == "__main__":
    main()
