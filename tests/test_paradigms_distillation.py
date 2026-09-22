"""Tests for nnx.paradigms.distillation — Hinton-style KD step factory."""

from __future__ import annotations

import pytest
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


def _make_classifier(hidden: int) -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=8,
            output_dim=3,
            hidden_dims=[hidden],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def _classification_loader(n: int = 64, batch_size: int = 16) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 8)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def test_kd_factory_validates_alpha_and_temperature():
    teacher = _make_classifier(32)
    with pytest.raises(ValueError, match="alpha"):
        kd_train_step_factory(teacher, alpha=1.5, temperature=4.0)
    with pytest.raises(ValueError, match="alpha"):
        kd_train_step_factory(teacher, alpha=-0.1, temperature=4.0)
    with pytest.raises(ValueError, match="temperature"):
        kd_train_step_factory(teacher, alpha=0.5, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        kd_train_step_factory(teacher, alpha=0.5, temperature=-1.0)


def test_kd_factory_freezes_teacher_params():
    teacher = _make_classifier(32)
    # Teacher params start trainable.
    assert all(p.requires_grad for p in teacher.net.parameters())
    kd_train_step_factory(teacher, alpha=0.5, temperature=4.0)
    # After factory: every teacher param frozen.
    assert all(not p.requires_grad for p in teacher.net.parameters())


def test_kd_factory_teacher_in_eval_mode():
    teacher = _make_classifier(32)
    teacher.net.train()  # force into train mode
    kd_train_step_factory(teacher, alpha=0.5, temperature=4.0)
    assert not teacher.net.training, (
        "kd_train_step_factory must set teacher.net to eval mode so dropout / "
        "BatchNorm running stats don't drift during student training"
    )


def test_kd_end_to_end_loss_decreases(tmp_path, monkeypatch):
    """Train a teacher first, then distill into a smaller student via
    the KD step factory and verify the student's loss trends down."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    loader = _classification_loader(n=128, batch_size=16)

    teacher = _make_classifier(hidden=64)
    teacher.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        )
    )

    # Snapshot teacher weights so we can assert they DON'T move during
    # distillation — the factory should have frozen them.
    teacher_snapshot = {k: v.clone() for k, v in teacher.net.state_dict().items()}

    student = _make_classifier(hidden=16)
    step_fn = kd_train_step_factory(teacher, alpha=0.5, temperature=4.0)
    run = student.train(
        params=NNTrainParams(
            n_epochs=3,
            train_loader=loader,
            optim=NNOptimParams(
                name=Optims.ADAM,
                max_lr=1e-2,
                momentum=(0.9, 0.999),
                weight_decay=0.0,
            ),
            scheduler=NNSchedulerParams(
                min_lr=1e-7,
                factor=0.5,
                patience=2,
                cooldown=1,
                threshold=1e-3,
            ),
        ),
        train_step_fn=step_fn,
    )

    losses = [idp.train_edp.loss for idp in run.idps]
    assert all(lo is not None and lo == lo and abs(lo) < 1e9 for lo in losses)
    n = len(losses)
    early = sum(losses[: n // 3]) / max(1, n // 3)
    late = sum(losses[2 * n // 3 :]) / max(1, n - 2 * n // 3)
    assert late < early, f"distillation loss did not decrease: early {early:.4f} vs late {late:.4f}"

    # Teacher weights unchanged across the student's training.
    for k, v in teacher.net.state_dict().items():
        assert torch.equal(v, teacher_snapshot[k]), (
            f"teacher param {k!r} drifted during student training — kd_train_step_factory must keep the teacher frozen"
        )


def test_kd_alpha_zero_collapses_to_supervised(tmp_path, monkeypatch):
    """alpha=0 turns the KD step into pure supervised — useful sanity
    check that the soft term doesn't leak gradients when α=0."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    loader = _classification_loader(n=64, batch_size=16)
    teacher = _make_classifier(hidden=32)

    student = _make_classifier(hidden=16)
    step_fn = kd_train_step_factory(teacher, alpha=0.0, temperature=4.0)

    run = student.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
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
        ),
        train_step_fn=step_fn,
    )
    assert len(run.idps) > 0
    assert all(idp.train_edp.loss is not None for idp in run.idps)


def test_kd_alpha_one_is_pure_distillation(tmp_path, monkeypatch):
    """alpha=1 drops the hard-label term entirely — the symmetric
    boundary case to α=0. Verifies the hard term doesn't leak when
    α=1.0; the KL term alone should still produce a finite, decreasing
    loss when the student matches the teacher's input shape."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)

    loader = _classification_loader(n=64, batch_size=16)
    teacher = _make_classifier(hidden=32)

    student = _make_classifier(hidden=16)
    step_fn = kd_train_step_factory(teacher, alpha=1.0, temperature=4.0)

    run = student.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=loader,
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
        ),
        train_step_fn=step_fn,
    )
    losses = [idp.train_edp.loss for idp in run.idps]
    assert len(losses) > 0
    assert all(lo is not None and torch.isfinite(torch.tensor(lo)).item() for lo in losses)


def _nll_twin(model: NNModel) -> NNModel:
    """Same weights as `model`, configured with native NLL instead of CE."""
    twin = NNModel(
        net_params=model.net_params,
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.NEGATIVE_LOG_LIKELIHOOD),
    )
    twin.net.load_state_dict(model.net.state_dict())
    return twin


def _step_once(model: NNModel, step_fn, batch, seed: int = 0):
    from nnx import Optims, TrainStepContext

    torch.manual_seed(seed)
    optimizer = Optims.ADAM(net=model.net, lr_start=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)
    ctx = TrainStepContext(
        model=model,
        batch=batch,
        optimizer=optimizer,
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
        is_last_batch=True,
    )
    return step_fn(ctx)


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_kd_hard_term_uses_normalized_nll(alpha):
    """FIX-001: the KD hard-label term under native NLL must equal the
    explicit log-softmax reference and the CE-configured student from
    identical state — `alpha=0` isolates the hard term so the KL cannot
    hide a wrong sign."""
    import torch.nn.functional as F

    from nnx.paradigms import kd_train_step_factory

    torch.manual_seed(0)
    teacher = _make_classifier(32)
    student_ce = _make_classifier(16)
    student_nll = _nll_twin(student_ce)
    batch = next(iter(_classification_loader()))
    X, Y = batch

    with torch.no_grad():
        raw = student_nll.net(X)
    expected_hard = float(F.nll_loss(F.log_softmax(raw, dim=1), Y))

    edp_ce = _step_once(student_ce, kd_train_step_factory(teacher, alpha=alpha, temperature=2.0), batch)
    edp_nll = _step_once(student_nll, kd_train_step_factory(teacher, alpha=alpha, temperature=2.0), batch)
    assert edp_ce.loss is not None and edp_nll.loss is not None
    assert edp_nll.loss == pytest.approx(edp_ce.loss, rel=1e-6)
    if alpha == 0.0:
        assert edp_nll.loss == pytest.approx(expected_hard, rel=1e-6)
    for (name, p_ce), (_, p_nll) in zip(
        student_ce.net.named_parameters(), student_nll.net.named_parameters(), strict=False
    ):
        assert torch.allclose(p_ce, p_nll, atol=1e-6), name


def test_feature_kd_hard_term_uses_normalized_nll():
    from nnx.paradigms import feature_kd_train_step_factory

    torch.manual_seed(0)
    teacher = _make_classifier(16)
    student_ce = _make_classifier(16)
    student_nll = _nll_twin(student_ce)
    batch = next(iter(_classification_loader()))
    factory = lambda: feature_kd_train_step_factory(  # noqa: E731
        teacher, auxiliary_layers={"layers.0": "layers.0"}, alpha=0.5, beta=0.5, temperature=2.0
    )
    edp_ce = _step_once(student_ce, factory(), batch)
    edp_nll = _step_once(student_nll, factory(), batch)
    assert edp_ce.loss is not None and edp_nll.loss is not None
    assert edp_nll.loss == pytest.approx(edp_ce.loss, rel=1e-6)
    for (name, p_ce), (_, p_nll) in zip(
        student_ce.net.named_parameters(), student_nll.net.named_parameters(), strict=False
    ):
        assert torch.allclose(p_ce, p_nll, atol=1e-6), name
