"""Knowledge distillation — train a small student to mimic a large teacher.

Two-phase flow:

  1. Train a "large" teacher classifier on a tabular toy task.
  2. Build a smaller student (a fraction of the teacher's params —
     hidden_dims=[16] vs [64, 64], so the student is roughly a 4-5%
     parameter count) and distill via :func:`kd_train_step_factory`,
     mixing the teacher's softened logits (KL term) with the standard
     hard-label loss (CE term). The exact ratio is printed at runtime.

The example demonstrates the *mechanism*: factory call, teacher
freezing, the train_step_fn hook. It does NOT claim distillation
beats a non-distilled baseline — on toy tabular data with clean
labels, the dark-knowledge effect is small or inconsistent.
Distillation's real benefit shows up on harder real-data tasks with
class confusion, noisy labels, or extreme student capacity gaps.

Run:
    python examples/10_knowledge_distillation.py
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
    kd_train_step_factory,
    set_seed,
)


def _make_classifier(hidden_dims: list[int]) -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=4,
            hidden_dims=hidden_dims,
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    """A 4-class toy task with overlapping Gaussians. Class means live
    close together (separation ~0.8) and the per-sample noise is
    almost-as-large, so the optimal classifier is well under 100%.
    Distillation's benefit needs a problem where the teacher knows
    something the labels alone don't tell you."""
    g = torch.Generator().manual_seed(seed)
    # Means clustered near the origin — small inter-class separation.
    means = torch.randn(4, 8, generator=g) * 0.8

    def make(n: int):
        cls = torch.randint(0, 4, (n,), generator=g)
        X = means[cls] + 0.7 * torch.randn(n, 8, generator=g)
        return X, cls

    # Small training set: distillation tends to help most when the student
    # is data-starved (the teacher gets to see the same data many times
    # and bakes its dark knowledge into its softmax).
    X_train, y_train = make(128)
    X_val, y_val = make(256)
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    return train, val


def _train_params(n_epochs: int, train_loader, val_loader, lr: float = 1e-2):
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=NNOptimParams(
            name=Optims.ADAM,
            max_lr=lr,
            momentum=(0.9, 0.999),
            weight_decay=0.0,
        ),
        scheduler=NNSchedulerParams(
            min_lr=1e-7,
            factor=0.5,
            patience=3,
            cooldown=1,
            threshold=1e-3,
        ),
    )


def main():
    set_seed(0)
    train_loader, val_loader = _loaders(seed=0)

    # ---- Phase 1: train the teacher.
    print("=" * 60)
    print("Phase 1: training teacher (hidden_dims=[64, 64])")
    print("=" * 60)
    teacher = _make_classifier(hidden_dims=[64, 64])
    teacher_run = teacher.train(params=_train_params(8, train_loader, val_loader))
    teacher_err = teacher_run.idps[-1].val_edp.error
    teacher_params = sum(p.numel() for p in teacher.net.parameters())
    print(f"\nteacher: {teacher_params} params, val error {teacher_err:.4f}")

    # Snapshot teacher weights — we'll verify the factory keeps them
    # frozen by re-checking after the student's training run.
    teacher_snapshot = {k: v.clone() for k, v in teacher.net.state_dict().items()}

    # ---- Phase 2: distill into a smaller student.
    print("\n" + "=" * 60)
    print("Phase 2: distilling into student (hidden_dims=[16])")
    print("=" * 60)
    set_seed(1)
    student = _make_classifier(hidden_dims=[16])
    student_params = sum(p.numel() for p in student.net.parameters())
    print(f"student: {student_params} params ({student_params * 100 / teacher_params:.1f}% of teacher)")

    step_fn = kd_train_step_factory(teacher, alpha=0.5, temperature=4.0)
    student_run = student.train(
        params=_train_params(8, train_loader, val_loader),
        train_step_fn=step_fn,
    )
    student_err = student_run.idps[-1].val_edp.error
    print(f"\nstudent (distilled) val error: {student_err:.4f}")

    # Verify the factory kept the teacher frozen across the student's
    # training run. (kd_train_step_factory sets requires_grad=False on
    # every teacher parameter; this is a runtime sanity check.)
    for k, v in teacher.net.state_dict().items():
        if not torch.equal(v, teacher_snapshot[k]):
            raise RuntimeError(f"teacher param {k!r} drifted during distillation")
    print("teacher weights unchanged across student training: confirmed")


if __name__ == "__main__":
    main()
