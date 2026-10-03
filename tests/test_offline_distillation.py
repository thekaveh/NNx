"""FEAT-022: offline teacher-distribution datasets and their objective
(``nnx.paradigms.offline_distillation``)."""

from __future__ import annotations

import copy
import math
import socket

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
    NNTrainParams,
)
from nnx.components import ComponentRestoreError
from nnx.nn.enum.optims import Optims
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.objectives import ObjectiveContext, kd_objective
from nnx.paradigms import born_again_train, feature_kd_train_step_factory, kd_train_step_factory
from nnx.paradigms.offline_distillation import (
    RECORD_FORMAT,
    DistillationReport,
    OfflineDistillationObjective,
    TeacherBatch,
    TeacherDataset,
    TeacherRecord,
    TeacherRecordError,
    evaluate_offline,
    read_teacher_records,
    write_teacher_records,
)

CANDIDATES = ("cat", "dog")
SCHEMA = "sha256:pets-v1"
PROVENANCE = {"source": "pets-teacher-export-2026-09", "training_rights": "declared: internal use, licence ref L-17"}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _record(sample_id: str, probabilities, *, candidates=CANDIDATES, label=None, schema=SCHEMA, **overrides):
    fields = dict(
        sample_id=sample_id,
        question_id="pet-kind",
        candidates=tuple(candidates),
        probabilities=tuple(probabilities),
        teacher="pets-teacher",
        revision="r42",
        schema_digest=schema,
        semantics="predictive",
        provenance=PROVENANCE,
        label=label,
    )
    fields.update(overrides)
    return TeacherRecord(**fields)


def _model(input_dim: int = 2, *, zero: bool = False, seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    model = NNModel(
        net_params=NNParams(
            input_dim=input_dim,
            output_dim=len(CANDIDATES),
            hidden_dims=[],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    if zero:
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.zero_()
    return model


def _inputs(n: int, input_dim: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {f"s{i}": torch.randn(input_dim, generator=generator) for i in range(n)}


def _dataset(n: int = 3, *, labelled: bool = True, schema: str = SCHEMA) -> TeacherDataset:
    teacher = [(0.75, 0.25), (0.2, 0.8), (0.6, 0.4), (0.1, 0.9)]
    labels = ["cat", "dog", "dog", "dog"]
    records = [_record(f"s{i}", teacher[i], label=labels[i] if labelled else None, schema=schema) for i in range(n)]
    return TeacherDataset(records, _inputs(n), candidates=CANDIDATES, schema_digest=schema)


def _sgd(accumulate: int = 1) -> NNOptimParams:
    return NNOptimParams(
        name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0, accumulate_grad_batches=accumulate
    )


def _params(loader, *, epochs: int = 1, accumulate: int = 1, **overrides) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=epochs, train_loader=loader, optim=_sgd(accumulate), save_phase_checkpoints=False, **overrides
    )


def _batch(dataset: TeacherDataset) -> TeacherBatch:
    (batch,) = list(dataset.loader(batch_size=len(dataset)))
    return batch


# ------------------------------------------------------------------ records


def test_record_schema(tmp_path):
    record = _record("s0", (0.75, 0.25), label="cat")
    assert record.state()["format"] == RECORD_FORMAT
    assert TeacherRecord.from_state(record.state()) == record
    assert record.distribution() == {"cat": 0.75, "dog": 0.25}
    path = tmp_path / "teacher.jsonl"
    write_teacher_records(path, [record, _record("s1", (0.5, 0.5))])
    assert read_teacher_records(path) == [record, _record("s1", (0.5, 0.5))]

    for missing in ("sample_id", "question_id", "candidates", "teacher", "revision", "schema_digest", "semantics"):
        state = record.state()
        del state[missing]
        with pytest.raises(TeacherRecordError, match=missing):
            TeacherRecord.from_state(state)
    for empty in ("sample_id", "question_id", "teacher", "revision", "schema_digest"):
        with pytest.raises(TeacherRecordError, match=empty):
            TeacherRecord.from_state({**record.state(), empty: ""})
    for provenance in ({}, {"source": "x"}, {"training_rights": "declared"}, {"source": "", "training_rights": "x"}):
        with pytest.raises(TeacherRecordError, match="provenance"):
            _record("s0", (0.75, 0.25), provenance=provenance)
    with pytest.raises(TeacherRecordError, match="semantics"):
        _record("s0", (0.75, 0.25), semantics="logits")
    with pytest.raises(TeacherRecordError, match="unknown keys"):
        TeacherRecord.from_state({**record.state(), "logits": [1.0, 0.0]})
    with pytest.raises(TeacherRecordError, match="label"):
        _record("s0", (0.75, 0.25), label="fox")
    path.write_text(path.read_text().replace('"r42"', '""', 1))
    with pytest.raises(TeacherRecordError, match=r"teacher\.jsonl:1"):
        read_teacher_records(path)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ((0.7, 0.2), "sum to"),
        ((1.2, -0.2), "nonnegative"),
        ((math.nan, 1.0), "finite"),
        ((math.inf, 0.0), "finite"),
        ((0.5,), "one probability per candidate"),
        ((0.5, 0.5 + 2e-6), "sum to"),
    ],
)
def test_probabilities_are_a_distribution(probabilities, message):
    with pytest.raises(TeacherRecordError, match=message):
        _record("s0", probabilities)
    _record("s0", (0.5, 0.5 + 5e-7))  # within 1e-6


def test_candidate_alignment():
    canonical = _dataset(3)
    reordered = TeacherDataset(
        [
            _record("s0", (0.25, 0.75), candidates=("dog", "cat"), label="cat"),
            _record("s1", (0.8, 0.2), candidates=("dog", "cat"), label="dog"),
            _record("s2", (0.6, 0.4), label="dog"),
        ],
        _inputs(3),
        candidates=CANDIDATES,
        schema_digest=SCHEMA,
    )
    torch.testing.assert_close(_batch(reordered).teacher, _batch(canonical).teacher, rtol=0, atol=0)
    assert torch.equal(_batch(reordered).target, _batch(canonical).target)
    a, b = _model(), _model()
    a.train(_params(canonical.loader(batch_size=2), data_id="canonical"), objective=canonical.objective(alpha=0.5))
    b.train(_params(reordered.loader(batch_size=2), data_id="reordered"), objective=reordered.objective(alpha=0.5))
    for key, value in a.net.state_dict().items():
        torch.testing.assert_close(b.net.state_dict()[key], value, rtol=0, atol=0)


def test_missing_candidate(monkeypatch):
    def no_optimizer(*args, **kwargs):
        raise AssertionError("a misaligned record must fail before any optimizer is built")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", no_optimizer)
    cases = [
        (_record("s0", (0.5, 0.5)), ("cat", "dog", "fox"), "lacks candidates \\['fox'\\]"),
        (
            _record("s0", (0.5, 0.25, 0.25), candidates=("cat", "dog", "fox")),
            CANDIDATES,
            "unknown candidates \\['fox'\\]",
        ),
    ]
    for record, candidates, message in cases:
        with pytest.raises(TeacherRecordError, match=message):  # the dataset fails first: no optimizer is built
            data = TeacherDataset([record], _inputs(1), candidates=candidates, schema_digest=SCHEMA)
            _model().train(_params(data.loader(batch_size=1)), objective=data.objective(alpha=1.0))
    with pytest.raises(TeacherRecordError, match="duplicate candidate"):
        _record("s0", (0.5, 0.5), candidates=("cat", "cat"))
    with pytest.raises(TeacherRecordError, match="duplicate sample"):
        TeacherDataset(
            [_record("s0", (0.5, 0.5)), _record("s0", (0.5, 0.5))],
            _inputs(1),
            candidates=CANDIDATES,
            schema_digest=SCHEMA,
        )
    with pytest.raises(TeacherRecordError, match="no input"):
        TeacherDataset([_record("s9", (0.5, 0.5))], _inputs(1), candidates=CANDIDATES, schema_digest=SCHEMA)
    with pytest.raises(TeacherRecordError, match="schema digest"):
        TeacherDataset(
            [_record("s0", (0.5, 0.5), schema="sha256:other")], _inputs(1), candidates=CANDIDATES, schema_digest=SCHEMA
        )
    with pytest.raises(TeacherRecordError, match="duplicate candidate"):
        TeacherDataset([_record("s0", (0.5, 0.5))], _inputs(1), candidates=("cat", "cat"), schema_digest=SCHEMA)


# ------------------------------------------------------------------ objective


def test_kl_oracle():
    dataset = TeacherDataset(
        [_record("s0", (0.75, 0.25), label="cat")], _inputs(1), candidates=CANDIDATES, schema_digest=SCHEMA
    )
    model = _model(zero=True)  # zero student logits: p = [0.5, 0.5]
    batch = _batch(dataset)
    result = dataset.objective(alpha=1.0)(ObjectiveContext(model=model, batch=batch, epoch_idx=0, batch_idx=0))
    (soft,) = result.terms
    assert soft.name == "teacher_kl" and soft.denominator == 1 and soft.weight == 1.0
    assert soft.value == pytest.approx(0.130812, abs=1e-6)  # 0.75 ln 1.5 + 0.25 ln 0.5, no T²
    (soft.numerator / soft.denominator).backward()
    output_bias = model.net.layers[0].bias.grad
    torch.testing.assert_close(output_bias, torch.tensor([-0.25, 0.25]))  # p - q
    model.net.zero_grad()
    result = dataset.objective(alpha=0.0)(ObjectiveContext(model=model, batch=batch, epoch_idx=0, batch_idx=0))
    (hard,) = result.terms
    assert hard.name == "hard_ce" and hard.value == pytest.approx(math.log(2)) and hard.denominator == 1
    for alpha in (-0.1, 1.5, math.nan, math.inf, True, "0.5"):
        with pytest.raises((TeacherRecordError, TypeError), match="alpha"):
            dataset.objective(alpha=alpha)


def test_shared_denominator():
    dataset = _dataset(3)
    model = _model()
    result = dataset.objective(alpha=0.3)(
        ObjectiveContext(model=model, batch=_batch(dataset), epoch_idx=0, batch_idx=0)
    )
    soft, hard = result.terms
    assert (soft.name, hard.name) == ("teacher_kl", "hard_ce")
    assert soft.denominator == hard.denominator == 3 and (soft.weight, hard.weight) == (0.3, pytest.approx(0.7))
    assert result.record is not None and result.record.loss == pytest.approx(0.3 * soft.value + 0.7 * hard.value)

    partly = TeacherDataset(
        [_record("s0", (0.75, 0.25), label="cat"), _record("s1", (0.5, 0.5))],
        _inputs(2),
        candidates=CANDIDATES,
        schema_digest=SCHEMA,
    )
    assert partly.labelled == "partial"
    for alpha in (1.0, 0.5):  # refused before any optimizer exists, whatever the batch size
        with pytest.raises(TeacherRecordError, match="partly labelled"):
            partly.objective(alpha=alpha)
    direct = OfflineDistillationObjective(schema_digest=SCHEMA, candidates=CANDIDATES, alpha=1.0)
    with pytest.raises(TeacherRecordError, match="partly labelled batch"):  # a hand-built loader's batch
        direct(ObjectiveContext(model=model, batch=_batch(partly), epoch_idx=0, batch_idx=0))
    unlabelled = _dataset(2, labelled=False)
    (only,) = unlabelled.objective(alpha=1.0)(
        ObjectiveContext(model=model, batch=_batch(unlabelled), epoch_idx=0, batch_idx=0)
    ).terms
    assert only.name == "teacher_kl" and only.denominator == 2
    with pytest.raises(TeacherRecordError, match="every record needs a label"):
        unlabelled.objective(alpha=0.5)


def test_objective_never_steps(monkeypatch):
    dataset = _dataset(3)
    model = _model()
    batch = _batch(dataset)
    batch.teacher.requires_grad_(True)  # a caller's tensor that still tracks gradients

    def forbidden(*args, **kwargs):
        raise AssertionError("an objective never zeroes, back-propagates, scales, steps or schedules")

    with monkeypatch.context() as patch:
        for owner, name in (
            (torch.optim.Optimizer, "zero_grad"),
            (torch.optim.SGD, "step"),
            (nn.Module, "zero_grad"),
            (torch.Tensor, "backward"),
            (torch.amp.GradScaler, "scale"),
            (torch.amp.GradScaler, "step"),
            (torch.amp.GradScaler, "update"),
            (torch.optim.lr_scheduler.LRScheduler, "step"),
        ):
            patch.setattr(owner, name, forbidden)
        result = dataset.objective(alpha=0.5)(ObjectiveContext(model=model, batch=batch, epoch_idx=0, batch_idx=0))
    assert all(parameter.grad is None for parameter in model.net.parameters())
    sum(term.weight * term.numerator / term.denominator for term in result.terms).backward()
    assert batch.teacher.grad is None  # the teacher's probabilities are detached
    assert all(parameter.grad is not None for parameter in model.net.parameters())


def test_accumulation_matches_full_batch():
    dataset = _dataset(3)
    accumulated, full = _model(seed=1), _model(seed=1)
    accumulated.train(_params(dataset.loader(batch_size=2), accumulate=2), objective=dataset.objective(alpha=0.4))
    full.train(_params(dataset.loader(batch_size=3)), objective=dataset.objective(alpha=0.4))

    reference = _model(seed=1)
    optimizer = torch.optim.SGD(reference.net.parameters(), lr=0.1)
    batch = _batch(dataset)
    log_p = F.log_softmax(reference.net(batch.inputs), dim=-1)
    q = batch.teacher
    kl = (q * (q.log() - log_p)).sum(-1).mean()
    ce = F.nll_loss(log_p, batch.target)
    (0.4 * kl + 0.6 * ce).backward()
    optimizer.step()
    for key, value in reference.net.state_dict().items():
        torch.testing.assert_close(full.net.state_dict()[key], value, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(accumulated.net.state_dict()[key], value, rtol=1e-6, atol=1e-7)


def test_no_network_in_hot_path(monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError("offline distillation reached the network")

    for name in ("create_connection", "getaddrinfo", "gethostbyname"):
        monkeypatch.setattr(socket, name, offline)
    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setattr(socket.socket, "connect_ex", offline)
    dataset = _dataset(3)
    assert sum(len(batch.sample_ids) for batch in dataset.loader(batch_size=2, shuffle=True, seed=0)) == 3
    model = _model()
    run = model.train(
        _params(dataset.loader(batch_size=2), val_loader=dataset.loader(batch_size=2), epochs=2),
        objective=dataset.objective(alpha=0.5),
        eval_step_fn=dataset.eval_step(),
    )
    assert run.idps[-1].val_edp is not None and "teacher_agreement" in run.idps[-1].val_edp.metrics
    assert isinstance(evaluate_offline(model, dataset), DistillationReport)


def test_resume_rejects_schema_change():
    dataset = _dataset(3)
    model = _model()
    objective = dataset.objective(alpha=0.5)
    identity = objective.identity()
    assert identity == {
        "id": "offline_distillation",
        "version": 1,
        "schema_digest": SCHEMA,
        "alpha": 0.5,
        "adapter": {"kind": "candidate_ids", "candidates": list(CANDIDATES)},
    }
    first = model.train(_params(dataset.loader(batch_size=2)), objective=objective)

    reordered = TeacherDataset(dataset.records, _inputs(3), candidates=("dog", "cat"), schema_digest=SCHEMA)
    other_schema = _dataset(3, schema="sha256:pets-v2")
    for data, changed, message in (
        (other_schema, other_schema.objective(alpha=0.5), "schema digest"),
        (dataset, dataset.objective(alpha=0.25), "alpha"),
        (reordered, reordered.objective(alpha=0.5), "candidate alignment"),
    ):
        fresh = _model(seed=3)
        before = copy.deepcopy(fresh.net.state_dict())
        with pytest.raises(ComponentRestoreError, match=message):
            fresh.train(_params(data.loader(batch_size=2), epochs=2, resume_from_run_id=first.id), objective=changed)
        assert all(torch.equal(fresh.net.state_dict()[k], v) for k, v in before.items())  # no resumed update
    resumed = _model(seed=3).train(
        _params(dataset.loader(batch_size=2), epochs=2, resume_from_run_id=first.id),
        objective=dataset.objective(alpha=0.5),
    )
    assert resumed.resume_status is not None and resumed.resume_status.mode == "stateful"


def test_agreement_vs_quality():
    records = [
        _record("s0", (0.9, 0.1), label="cat"),
        _record("s1", (0.8, 0.2), label="dog"),  # the teacher is wrong here
        _record("s2", (0.7, 0.3)),
        _record("s3", (0.6, 0.4)),
    ]
    dataset = TeacherDataset(records, _inputs(4), candidates=CANDIDATES, schema_digest=SCHEMA)
    model = _model(zero=True)
    with torch.no_grad():
        model.net.layers[0].bias.copy_(torch.tensor([1.0, 0.0]))  # always "cat", as the teacher
    report = evaluate_offline(model, dataset, batch_size=3)
    assert (report.agreement.value, report.agreement.denominator) == (1.0, 4)
    assert (report.label_accuracy.value, report.label_accuracy.denominator) == (0.5, 2)
    log_p = F.log_softmax(torch.tensor([1.0, 0.0]), dim=-1)
    expected = (
        sum(sum(q * (math.log(q) - lp) for q, lp in zip(r.probabilities, log_p.tolist(), strict=True)) for r in records)
        / 4
    )
    assert report.imitation_loss.denominator == 4 and report.imitation_loss.value == pytest.approx(expected)
    assert report.label_nll.denominator == 2
    assert report.label_nll.value == pytest.approx(-(log_p[0].item() + log_p[1].item()) / 2)
    text = report.text()
    assert "teacher agreement 1.0000 (4 records)" in text and "label accuracy 0.5000 (2 labelled records)" in text
    unlabelled = evaluate_offline(model, _dataset(2, labelled=False))
    assert unlabelled.label_accuracy.value is None and unlabelled.label_accuracy.denominator == 0
    assert "no labelled records" in unlabelled.label_accuracy.reason
    assert report.state()["agreement"] == {"value": 1.0, "denominator": 4, "reason": None}


# ------------------------------------------------------------------ live KD keeps its contract


def test_live_distillation_rejects_probability_only_records():
    dataset = _dataset(3)
    records = [_record("s0", (0.75, 0.25))]
    for teacher in (dataset, records, records[0], dataset.loader(batch_size=2)):
        with pytest.raises(TypeError, match="live NNModel teacher"):
            kd_train_step_factory(teacher)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="live NNModel teacher"):
            kd_objective(teacher)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="live NNModel teacher"):
            feature_kd_train_step_factory(teacher, auxiliary_layers={"layers.0": "layers.0"})  # type: ignore[arg-type]
    student = _model()
    with pytest.raises(TypeError, match="TeacherDataset.objective"):
        student.train(_params(dataset.loader(batch_size=2)), train_step_fn=kd_train_step_factory(_model(seed=4)))
    with pytest.raises(TypeError, match="TeacherDataset.objective"):
        student.train(_params(dataset.loader(batch_size=2)))  # the default supervised step
    with pytest.raises(TypeError, match="born_again_train distills live generations"):
        born_again_train(student, generations=2, train_params=_params(dataset.loader(batch_size=2)))


def test_a_batch_in_another_candidate_order_is_refused():
    """Alignment is checked per batch, not only when a dataset is built: an
    objective or evaluation never reads another dataset's columns as its own."""
    canonical = _dataset(3)
    reordered = TeacherDataset(canonical.records, _inputs(3), candidates=("dog", "cat"), schema_digest=SCHEMA)
    model = _model()
    batch = _batch(reordered)
    assert batch.candidates == ("dog", "cat")
    with pytest.raises(TeacherRecordError, match="probability columns"):
        canonical.objective(alpha=0.5)(ObjectiveContext(model=model, batch=batch, epoch_idx=0, batch_idx=0))
    with pytest.raises(TeacherRecordError, match="probability columns"):
        canonical.eval_step()(type("Ctx", (), {"model": model, "val_loader": reordered.loader(batch_size=2)})())
    assert evaluate_offline(model, reordered.loader(batch_size=2)).agreement.denominator == 3  # its own order


def test_zero_teacher_probabilities_add_nothing_and_stay_finite():
    one_hot = TeacherDataset(
        [_record("s0", (1.0, 0.0), label="cat")], _inputs(1), candidates=CANDIDATES, schema_digest=SCHEMA
    )
    model = _model(zero=True)
    ctx = ObjectiveContext(model=model, batch=_batch(one_hot), epoch_idx=0, batch_idx=0)
    (soft,) = one_hot.objective(alpha=1.0)(ctx).terms
    assert soft.value == pytest.approx(math.log(2))
    (soft.numerator / soft.denominator).backward()
    torch.testing.assert_close(model.net.layers[0].bias.grad, torch.tensor([-0.5, 0.5]))
    model.net.zero_grad()
    with torch.no_grad():
        model.net.layers[0].bias.copy_(
            torch.tensor([0.0, -1e4])
        )  # the zero-probability candidate is all but impossible
    (soft,) = one_hot.objective(alpha=1.0)(ctx).terms
    (soft.numerator / soft.denominator).backward()
    assert math.isfinite(soft.value) and soft.value == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(model.net.layers[0].bias.grad).all()


def test_inputs_batch_in_one_shape_and_the_default_dtype():
    import numpy as np

    records = [_record("s0", (0.5, 0.5), label="cat"), _record("s1", (0.5, 0.5), label="dog")]
    as_float64 = {"s0": np.zeros(2), "s1": np.ones(2)}  # NumPy's default float64
    data = TeacherDataset(records, as_float64, candidates=CANDIDATES, schema_digest=SCHEMA)
    assert _batch(data).inputs.dtype == torch.get_default_dtype()
    ragged = {"s0": torch.zeros(2), "s1": torch.zeros(3)}
    with pytest.raises(TeacherRecordError, match="one shape"):
        TeacherDataset(records, ragged, candidates=CANDIDATES, schema_digest=SCHEMA)
    mixed = [records[0], _record("s1", (0.5, 0.5), question_id="pet-age")]
    with pytest.raises(TeacherRecordError, match="answers one question"):
        TeacherDataset(mixed, _inputs(2), candidates=CANDIDATES, schema_digest=SCHEMA)


def test_only_the_marker_itself_counts_as_teacher_records():
    from unittest import mock

    from nnx.paradigms.offline_distillation import _is_teacher_records

    assert not _is_teacher_records(mock.MagicMock())  # a mock answers every attribute, truthily
    assert _is_teacher_records(_dataset(1)) and _is_teacher_records([_record("s0", (0.5, 0.5))])
