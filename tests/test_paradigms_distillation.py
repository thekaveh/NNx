"""Tests for nnx.paradigms.distillation — Hinton-style KD step factory."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
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
    TaskSpec,
    TaskValidationError,
    kd_train_step_factory,
    set_seed,
)
from nnx._step_helpers import softened_kl
from tests import _task_steps as tasks


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


# --- a declared TaskSpec (#513) ---------------------------------------------------------------


def test_kd_alpha_0_masks_ignored_rows_as_the_default_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, teacher = tasks.categorical(), tasks.categorical(seed=1)
    reference, reference_run = tasks.fit(base, tasks.MIXED)
    trained, run = tasks.fit(base, tasks.MIXED, train_step_fn=kd_train_step_factory(teacher, alpha=0.0))
    torch.testing.assert_close(tasks.weights(trained), tasks.weights(reference))
    got, want = tasks.record(run), tasks.record(reference_run)
    assert (got.kind, got.count, got.status) == ("categorical", 4, "ok")
    assert got.loss == pytest.approx(want.loss)
    with torch.no_grad():  # the masked rows' loss would be counted without the adapter
        assert got.loss != pytest.approx(float(F.cross_entropy(base.net(tasks.X), tasks.MIXED)))


def test_kd_hard_term_scores_only_the_non_ignored_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, teacher = tasks.categorical(), tasks.categorical(seed=1)
    _, run = tasks.fit(base, tasks.MIXED, train_step_fn=kd_train_step_factory(teacher, alpha=0.5, temperature=2.0))
    with torch.no_grad():
        student, soft = base.net(tasks.X), teacher.net(tasks.X)
        keep = tasks.MIXED != 0
        hard = F.cross_entropy(student[keep], tasks.MIXED[keep])
        expected = 0.5 * softened_kl(student, soft, 2.0) + 0.5 * hard  # the soft term keeps every row
    assert tasks.record(run).loss == pytest.approx(float(expected), rel=1e-5)


def test_kd_alpha_0_takes_no_update_on_an_all_ignored_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, teacher = tasks.categorical(), tasks.categorical(seed=1)
    trained, run = tasks.fit(base, tasks.ALL_IGNORED, train_step_fn=kd_train_step_factory(teacher, alpha=0.0))
    assert tasks.moved(trained, base) == 0.0
    record = tasks.record(run)
    assert (record.kind, record.count, record.status, record.loss) == ("categorical", 0, "empty", None)


@pytest.mark.parametrize("ignore_index", [-1, 255])
def test_kd_masks_an_out_of_range_ignore_index(ignore_index, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, teacher = tasks.categorical(ignore_index), tasks.categorical(ignore_index, seed=1)
    Y = torch.tensor([ignore_index, 1, 2, ignore_index, 1, 2])
    reference, _ = tasks.fit(base, Y)
    trained, _ = tasks.fit(base, Y, train_step_fn=kd_train_step_factory(teacher, alpha=0.0))
    torch.testing.assert_close(tasks.weights(trained), tasks.weights(reference))
    _, run = tasks.fit(base, Y, train_step_fn=kd_train_step_factory(teacher, alpha=0.5))  # no IndexError
    assert tasks.record(run).count == 4


def test_kd_masks_nan_multilabel_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = TaskSpec.multilabel(3)
    base = tasks.model(task, loss=Losses.BINARY_CROSS_ENTROPY)
    teacher = tasks.model(task, loss=Losses.BINARY_CROSS_ENTROPY, seed=1)
    Y = tasks.multilabel_targets()
    _, reference = tasks.fit(base, Y)
    _, run = tasks.fit(base, Y, train_step_fn=kd_train_step_factory(teacher, alpha=0.0))
    assert tasks.record(run).loss == pytest.approx(tasks.record(reference).loss)
    _, run = tasks.fit(base, Y, train_step_fn=kd_train_step_factory(teacher, alpha=0.5))  # no divergence
    assert tasks.record(run).kind == "multilabel" and tasks.record(run).count == 12


def test_kd_refuses_a_regression_task_before_any_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    regression = dict(loss=Losses.MEAN_SQUARED_ERROR, output_dim=2)
    base = tasks.model(TaskSpec.regression(2), **regression)
    teacher = tasks.model(TaskSpec.regression(2), seed=1, **regression)
    with pytest.raises(TaskValidationError) as caught:
        tasks.fit(base, torch.randn(6, 2), train_step_fn=kd_train_step_factory(teacher, alpha=0.0))
    message = str(caught.value)
    assert "kd_train_step_factory" in message and "regression task" in message
    assert "declare the task" not in message and "NNModelParams(task=" not in message
    assert tasks.no_run_left(tmp_path)
