"""Feature-KD (FitNets-style) — distill a teacher's intermediate activations.

Pipeline:

  1. Train a large teacher classifier.
  2. Build a small student with a matching output width at the
     auxiliary layer (so the MSE term doesn't need a projector).
  3. Train the student via `nnx.paradigms.feature_kd_train_step_factory`
     with one teacher_layer → student_layer auxiliary pair plus the
     usual KL-soft + hard-label terms.
  4. Compare student val accuracy with vs without the feature-KD term.

Feature-KD adds an MSE term between named teacher / student
intermediate activations: L = α * KL_soft * T² + β * MSE(student_act, teacher_act) + (1 - α) * L_hard.
Forward hooks capture the activations per batch.

Key API notes:
  - `feature_kd_train_step_factory` takes a full `NNModel` for `teacher`
    (not just `teacher.net`).
  - `auxiliary_layers` is a `dict[str, str]` mapping teacher layer name
    to student layer name. Both layers must produce activations of the
    same shape (v1 does not ship a projector).
  - FeedFwdNN with hidden_dims=[64, 32] has: layers.0 (16→64),
    layers.1 (64→32), layers.2 (32→3). Activation at layers.1 has shape
    (B, 32). Student hidden_dims=[32, 32]: layers.0 (16→32) also gives
    (B, 32) — the pair {"layers.1": "layers.0"} shape-matches.

Run:
    pip install nnx
    python examples/24_feature_kd.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    feature_kd_train_step_factory,
    set_seed,
)


def _make_data():
    # No torch.manual_seed here — the caller does set_seed(42) before
    # calling us. Re-seeding torch inside this helper would silently
    # override the caller's seed (the same bug that PR #31's review
    # caught in examples 19 / 21 / 23).
    X = torch.randn(1024, 16)
    proj = torch.randn(16, 3)
    y = (X @ proj).argmax(dim=1)
    return (
        DataLoader(TensorDataset(X[:800], y[:800]), batch_size=64, shuffle=True),
        DataLoader(TensorDataset(X[800:], y[800:]), batch_size=64),
    )


def main() -> None:
    set_seed(42)
    train_loader, val_loader = _make_data()

    model_params = NNModelParams(
        net=Nets.FEED_FWD,
        device=Devices.CPU,
        loss=Losses.CROSS_ENTROPY,
    )
    base_train_params = NNTrainParams(
        n_epochs=5,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=1e-2,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-6,
            factor=0.5,
            patience=2,
            cooldown=1,
            threshold=1e-3,
        ),
    )

    # ----- Teacher: wide FFN -----
    # FeedFwdNN hidden_dims=[64, 32]:
    #   layers.0: Linear(16 → 64)
    #   layers.1: Linear(64 → 32)   ← activation shape (B, 32)
    #   layers.2: Linear(32 → 3)
    print("─── Phase 1: train wide teacher ───")
    teacher = NNModel(
        net_params=NNParams(
            input_dim=16,
            output_dim=3,
            hidden_dims=[64, 32],
            dropout_prob=0.1,
            activation=Activations.RELU,
        ),
        params=model_params,
    )
    teacher_run = teacher.train(params=base_train_params)
    teacher_val_idps = [idp for idp in teacher_run.idps if idp.val_edp is not None]
    teacher_val_acc = 1.0 - teacher_val_idps[-1].val_edp.error
    print(f"Teacher val accuracy: {teacher_val_acc:.3f}")

    # ----- Student: smaller FFN with matching aux-layer output width -----
    # Student hidden_dims=[32, 32]:
    #   layers.0: Linear(16 → 32)   ← activation shape (B, 32) — matches teacher.layers.1
    #   layers.1: Linear(32 → 32)
    #   layers.2: Linear(32 → 3)
    print("─── Phase 2: train student with feature-KD ───")
    student = NNModel(
        net_params=NNParams(
            input_dim=16,
            output_dim=3,
            hidden_dims=[32, 32],
            dropout_prob=0.1,
            activation=Activations.RELU,
        ),
        params=model_params,
    )
    # Pair: teacher.layers.1 (output 32) → student.layers.0 (output 32).
    # auxiliary_layers is dict[teacher_layer_name, student_layer_name].
    feature_kd_step = feature_kd_train_step_factory(
        teacher=teacher,
        auxiliary_layers={"layers.1": "layers.0"},
        alpha=0.5,
        beta=0.5,
        temperature=4.0,
    )
    student_run = student.train(params=base_train_params, train_step_fn=feature_kd_step)
    student_val_idps = [idp for idp in student_run.idps if idp.val_edp is not None]
    student_val_acc = 1.0 - student_val_idps[-1].val_edp.error
    print(f"Student val accuracy (with feature-KD): {student_val_acc:.3f}")
    print(
        f"Teacher params: {sum(p.numel() for p in teacher.net.parameters()):,}  "
        f"Student params: {sum(p.numel() for p in student.net.parameters()):,}"
    )


if __name__ == "__main__":
    main()
