"""Tests for nnx.paradigms.born_again — iterated self-distillation wrapper.

Born-again training is a thin composition over the KD step factory:
generation 0 trains plain, generation k > 0 uses a frozen deepcopy of
the model after generation k-1 as the teacher. The tests focus on the
structural invariants of the wrapper itself (right number of runs, KD
factory used on later generations, teacher frozen + eval-mode) rather
than the metric trajectory — convergence is exercised in depth by
test_paradigms_distillation.py.
"""

from __future__ import annotations

import copy

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
    NNRun,
    NNSchedulerParams,
    NNTrainParams,
    Optims,
    born_again_train,
    set_seed,
)
from nnx.paradigms import born_again as born_again_module


def _make_classifier(hidden: int = 16) -> NNModel:
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


def _classification_loader(n: int = 32, batch_size: int = 16) -> DataLoader:
    torch.manual_seed(0)
    X = torch.randn(n, 8)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def _train_params(n_epochs: int = 1) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        train_loader=_classification_loader(),
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


def test_born_again_validates_generations():
    """generations < 1 is meaningless (no run produced); raise early."""
    model = _make_classifier()
    with pytest.raises(ValueError, match="generations"):
        born_again_train(model, generations=0, train_params=_train_params())
    with pytest.raises(ValueError, match="generations"):
        born_again_train(model, generations=-1, train_params=_train_params())


def test_born_again_runs_n_generations(tmp_path, monkeypatch):
    """generations=3 → exactly 3 NNRun objects in the returned list."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_classifier()
    runs = born_again_train(model, generations=3, train_params=_train_params())
    assert isinstance(runs, list)
    assert len(runs) == 3
    assert all(isinstance(r, NNRun) for r in runs)


def test_born_again_first_generation_uses_plain_training(tmp_path, monkeypatch):
    """Generation 0 has no teacher → must call NNModel.train with
    train_step_fn=None (i.e., the default supervised path), not via the
    KD factory. Spy on both call sites to assert this."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_classifier()

    kd_calls: list[NNModel] = []
    original_kd_factory = born_again_module.kd_train_step_factory

    def spy_kd(teacher: NNModel, **kwargs):
        kd_calls.append(teacher)
        return original_kd_factory(teacher, **kwargs)

    monkeypatch.setattr(born_again_module, "kd_train_step_factory", spy_kd)

    train_step_fn_observed: list[object] = []
    original_train = NNModel.train

    def spy_train(self, params, callbacks=None, train_step_fn=None):
        train_step_fn_observed.append(train_step_fn)
        return original_train(self, params=params, callbacks=callbacks, train_step_fn=train_step_fn)

    monkeypatch.setattr(NNModel, "train", spy_train)

    runs = born_again_train(model, generations=1, train_params=_train_params())
    assert len(runs) == 1
    # Generation 0: kd_train_step_factory NOT invoked.
    assert len(kd_calls) == 0, "generation 0 must not construct a KD step"
    # Generation 0: train() called with train_step_fn=None (plain supervised).
    assert len(train_step_fn_observed) == 1
    assert train_step_fn_observed[0] is None


def test_born_again_each_generation_uses_previous_as_teacher(tmp_path, monkeypatch):
    """Generations 1+ must invoke kd_train_step_factory. With G=3 we
    expect exactly 2 KD invocations (generations 1 and 2). Each teacher
    passed in must be a *different object* than the live `model` (the
    wrapper snapshots via deepcopy)."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_classifier()

    teachers_seen: list[NNModel] = []
    original_kd_factory = born_again_module.kd_train_step_factory

    def spy_kd(teacher: NNModel, **kwargs):
        teachers_seen.append(teacher)
        return original_kd_factory(teacher, **kwargs)

    monkeypatch.setattr(born_again_module, "kd_train_step_factory", spy_kd)

    runs = born_again_train(model, generations=3, train_params=_train_params())
    assert len(runs) == 3
    # G=3 → KD invoked at generations 1 and 2 → 2 calls total.
    assert len(teachers_seen) == 2, f"expected 2 KD invocations for G=3, got {len(teachers_seen)}"
    # Each teacher is a snapshot, never the live model itself.
    for t in teachers_seen:
        assert t is not model, (
            "born_again_train must pass a deepcopy snapshot as teacher, "
            "not the live model — otherwise the teacher's params drift "
            "during the student's training"
        )


def test_born_again_teacher_frozen_and_eval_mode(tmp_path, monkeypatch):
    """At every generation k > 0, the teacher passed into the KD factory
    must already have requires_grad=False on every parameter AND be in
    eval mode. Verified at the moment of the KD-factory call so we catch
    state established by born_again_train itself (not the side effect of
    kd_train_step_factory, which also freezes belt-and-braces)."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_classifier()

    captured_states: list[tuple[bool, bool]] = []
    # (any_requires_grad, training_mode_flag) per KD invocation.
    original_kd_factory = born_again_module.kd_train_step_factory

    def spy_kd(teacher: NNModel, **kwargs):
        captured_states.append(
            (
                any(p.requires_grad for p in teacher.net.parameters()),
                teacher.net.training,
            )
        )
        return original_kd_factory(teacher, **kwargs)

    monkeypatch.setattr(born_again_module, "kd_train_step_factory", spy_kd)

    born_again_train(model, generations=3, train_params=_train_params())
    # 2 KD invocations across G=3.
    assert len(captured_states) == 2
    for any_grad, is_training in captured_states:
        assert not any_grad, (
            "teacher must have requires_grad=False on every parameter before being handed to kd_train_step_factory"
        )
        assert not is_training, "teacher.net must be in eval mode"


def test_born_again_kd_kwargs_forwarded(tmp_path, monkeypatch):
    """alpha / temperature kwargs must reach kd_train_step_factory
    unchanged. Demonstrates the **kd_kwargs forwarding contract."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_classifier()

    seen_kwargs: list[dict] = []
    original_kd_factory = born_again_module.kd_train_step_factory

    def spy_kd(teacher: NNModel, **kwargs):
        seen_kwargs.append(kwargs)
        return original_kd_factory(teacher, **kwargs)

    monkeypatch.setattr(born_again_module, "kd_train_step_factory", spy_kd)

    born_again_train(
        model,
        generations=2,
        train_params=_train_params(),
        alpha=0.7,
        temperature=2.5,
    )
    assert len(seen_kwargs) == 1
    assert seen_kwargs[0] == {"alpha": 0.7, "temperature": 2.5}


def test_born_again_teacher_isolated_from_subsequent_training(tmp_path, monkeypatch):
    """The teacher snapshot at generation k must not drift when the
    live model continues training in generation k+1. Snapshot the
    teacher's state_dict at the moment it's handed in; compare to its
    state_dict after the next generation completes."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_classifier()

    captured_teachers: list[NNModel] = []
    captured_snapshots: list[dict] = []
    original_kd_factory = born_again_module.kd_train_step_factory

    def spy_kd(teacher: NNModel, **kwargs):
        captured_teachers.append(teacher)
        captured_snapshots.append({k: v.clone() for k, v in teacher.net.state_dict().items()})
        return original_kd_factory(teacher, **kwargs)

    monkeypatch.setattr(born_again_module, "kd_train_step_factory", spy_kd)

    born_again_train(model, generations=3, train_params=_train_params(n_epochs=2))

    # Two teachers captured. Both must be unchanged at the end of training.
    for teacher, snapshot in zip(captured_teachers, captured_snapshots, strict=True):
        for k, v in teacher.net.state_dict().items():
            assert torch.equal(v, snapshot[k]), (
                f"teacher param {k!r} drifted across the next generation's "
                "training — born_again_train must freeze the teacher"
            )


def test_born_again_exported_from_top_level():
    """born_again_train must be reachable from `nnx` (top level) and
    from `nnx.paradigms`."""
    import nnx
    import nnx.paradigms

    assert hasattr(nnx, "born_again_train")
    assert hasattr(nnx.paradigms, "born_again_train")
    assert nnx.born_again_train is nnx.paradigms.born_again_train


def test_born_again_returns_runs_and_mutates_model_in_place(tmp_path, monkeypatch):
    """Sanity: the live model is mutated (weights change across the
    course of born-again training). The returned runs list holds the
    per-generation NNRun objects intact."""
    monkeypatch.chdir(tmp_path)
    set_seed(0)
    model = _make_classifier()
    initial = copy.deepcopy({k: v.clone() for k, v in model.net.state_dict().items()})
    runs = born_again_train(model, generations=2, train_params=_train_params())
    assert len(runs) == 2
    assert all(isinstance(r, NNRun) for r in runs)
    # At least one parameter has moved — the model was trained.
    final = model.net.state_dict()
    assert any(not torch.equal(final[k], v) for k, v in initial.items())
