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

**Objective mode (FEAT-004).** ``kd_objective(teacher, ...)`` describes the
same loss as loss terms with explicit denominators and hands the update to
NNx's shared update engine, so distillation also gets gradient accumulation
(exact for uneven microbatches and short windows), mixed precision and
clipping — the imperative factory refuses the first two.
``objective_mode()`` below runs it with a short accumulation window and
checks the one committed update against a full-batch reference.

Run:
    python examples/10_knowledge_distillation.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Callback,
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
    kd_objective,
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


def objective_mode() -> dict:
    """Bounded demonstration of distillation as an objective (FEAT-004).

    Three samples in microbatches of [2, 1] with ``accumulate_grad(4)``:
    the window is cut short by the end of the epoch, so the engine commits
    exactly one update — and it equals one full-batch step on
    ``alpha · KL_T(teacher ‖ student) · T² + (1 − alpha) · CE`` over all
    three samples. The teacher's tensors never change.
    """
    set_seed(3)
    teacher = _make_classifier(hidden_dims=[64, 64])
    set_seed(4)
    student = _make_classifier(hidden_dims=[16])
    X, y = torch.randn(3, 8), torch.tensor([0, 2, 1])
    alpha, temperature = 0.5, 4.0

    # Reference: one SGD step on the full batch, from a copy of the student.
    reference = _make_classifier(hidden_dims=[16])
    reference.net.load_state_dict(student.net.state_dict())
    with torch.no_grad():
        teacher_logits = teacher.net(X)
    logits = reference.net(X)
    soft = F.kl_div(
        F.log_softmax(logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    loss = alpha * soft + (1 - alpha) * F.cross_entropy(logits, y)
    loss.backward()
    with torch.no_grad():
        for param in reference.net.parameters():
            param -= 0.05 * param.grad

    teacher_snapshot = {k: v.clone() for k, v in teacher.net.state_dict().items()}
    updates = []

    class CountUpdates(Callback):
        def on_optimizer_update(self, ctx, event):
            updates.append((event.update_idx, event.microbatches))

    student.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=[(X[:2], y[:2]), (X[2:], y[2:])],  # uneven microbatches
            optim=NNOptimParams.builder().sgd(max_lr=0.05, momentum=0.0).accumulate_grad(4).build(),
            save_phase_checkpoints=False,
        ),
        objective=kd_objective(teacher, alpha=alpha, temperature=temperature),
        callbacks=[CountUpdates()],
    )

    assert updates == [(1, 2)], updates  # one committed update over both microbatches
    for key, value in reference.net.state_dict().items():
        assert torch.allclose(student.net.state_dict()[key], value, rtol=1e-6, atol=1e-7), key
    for key, value in teacher.net.state_dict().items():
        assert torch.equal(value, teacher_snapshot[key]), key
    summary = {"updates": updates, "matches_reference": True}
    print(f"objective mode: {summary}")
    return summary


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

    objective_mode()


if __name__ == "__main__":
    main()
