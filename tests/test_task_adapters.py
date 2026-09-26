"""FEAT-002: opt-in task adapters for categorical, multilabel and regression.

Covers the TaskSpec contract, model preflight, per-batch validation before
any optimizer update, masking, loss units and whole-dataset metrics
(including the regression and multilabel oracles), legacy categorical
parity on uneven batches and accumulated windows, and continuous rich
prediction.
"""

from __future__ import annotations

import math

import numpy as np
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
    NNTrainParams,
    ProbabilitySpec,
    TaskSpec,
    TaskValidationError,
    set_seed,
    task_adapter,
)

# ------------------------------------------------------------------ helpers


def _model(task: TaskSpec | None, *, loss: Losses, output_dim: int, input_dim: int = 3, seed: int = 0) -> NNModel:
    set_seed(seed)
    return NNModel(
        net_params=NNParams(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=[6], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss, task=task),
    )


def _sgd(lr: float = 0.1, accumulate: int = 1) -> NNOptimParams:
    return NNOptimParams.builder().sgd(max_lr=lr, momentum=0.0).accumulate_grad(accumulate).build()


def _train(model: NNModel, loader, *, accumulate: int = 1, val_loader=None, n_epochs: int = 1):
    params = NNTrainParams(n_epochs=n_epochs, optim=_sgd(accumulate=accumulate)).with_train_loader(loader)
    if val_loader is not None:
        params = params.with_val_loader(val_loader)
    return model.train(params=params)


def _params_snapshot(model: NNModel) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.net.named_parameters()}


def _fixed(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# ------------------------------------------------------------- TaskSpec


def test_task_spec_constructors_and_validation():
    assert TaskSpec.categorical(3, ignore_index=-100).state() == {
        "version": 1,
        "kind": "categorical",
        "num_outputs": 3,
        "ignore_index": -100,
    }
    assert TaskSpec.multilabel(labels=["a", "b"]).num_outputs == 2
    assert TaskSpec.regression().state() == {"version": 1, "kind": "regression"}
    assert TaskSpec.multilabel(2, threshold=0.3).state()["threshold"] == 0.3
    with pytest.raises(TaskValidationError, match="kind"):
        TaskSpec("ranking")
    with pytest.raises(TaskValidationError, match="ignore_index applies to categorical"):
        TaskSpec("regression", ignore_index=0)
    with pytest.raises(TaskValidationError, match="threshold applies to multilabel"):
        TaskSpec("categorical", threshold=0.3)
    with pytest.raises(TaskValidationError, match=r"threshold must be in \(0, 1\)"):
        TaskSpec.multilabel(threshold=1.0)
    with pytest.raises(TaskValidationError, match=">= 2"):
        TaskSpec.categorical(1)
    with pytest.raises(TaskValidationError, match="unique"):
        TaskSpec.categorical(labels=["a", "a"])
    with pytest.raises(TaskValidationError, match="num_outputs"):
        TaskSpec.regression(3, labels=["a", "b"])
    with pytest.raises(TaskValidationError, match="integer"):
        TaskSpec.regression(2.0)  # type: ignore[arg-type]


def test_task_spec_versioned_round_trip_rejects_unknown_versions_and_keys():
    spec = TaskSpec.multilabel(3, threshold=0.25, labels=["x", "y", "z"])
    assert TaskSpec.from_state(spec.state()) == spec
    with pytest.raises(TaskValidationError, match="version"):
        TaskSpec.from_state({**spec.state(), "version": 2})
    with pytest.raises(TaskValidationError, match="version"):
        TaskSpec.from_state({"kind": "regression"})
    with pytest.raises(TaskValidationError, match="unknown"):
        TaskSpec.from_state({**spec.state(), "mask": "nan"})


# ------------------------------------------------------------ preflight


@pytest.mark.parametrize(
    ("task", "loss", "output_dim", "match"),
    [
        (TaskSpec.regression(), Losses.CROSS_ENTROPY, 1, "MEAN_SQUARED_ERROR"),
        (TaskSpec.categorical(), Losses.MEAN_SQUARED_ERROR, 3, "CROSS_ENTROPY"),
        (TaskSpec.multilabel(), Losses.CROSS_ENTROPY, 2, "BINARY_CROSS_ENTROPY"),
        (TaskSpec.regression(2), Losses.MEAN_SQUARED_ERROR, 3, "declares 2 output"),
    ],
)
def test_model_construction_rejects_an_incompatible_task(task, loss, output_dim, match):
    with pytest.raises(TaskValidationError, match=match):
        _model(task, loss=loss, output_dim=output_dim)


def test_language_model_tasks_are_out_of_scope():
    adapter = task_adapter(TaskSpec.categorical())
    with pytest.raises(TaskValidationError, match="TRANSFORMER"):
        adapter.check_model(net=Nets.TRANSFORMER, loss=Losses.CROSS_ENTROPY, output_dim=None)


# -------------------------------------- validation before any optimizer update


@pytest.mark.parametrize(
    ("task", "loss", "output_dim", "x", "y", "match"),
    [
        # categorical: float targets, wrong shape, out-of-range class index
        (TaskSpec.categorical(), Losses.CROSS_ENTROPY, 3, torch.randn(4, 3), torch.rand(4), "integer class indices"),
        (
            TaskSpec.categorical(),
            Losses.CROSS_ENTROPY,
            3,
            torch.randn(4, 3),
            torch.zeros(4, 1, dtype=torch.long),
            "shape",
        ),
        (TaskSpec.categorical(), Losses.CROSS_ENTROPY, 3, torch.randn(4, 3), torch.tensor([0, 1, 2, 3]), r"\[0, 3\)"),
        # multilabel: non-binary targets, wrong shape
        (TaskSpec.multilabel(), Losses.BINARY_CROSS_ENTROPY, 2, torch.randn(4, 3), torch.full((4, 2), 0.5), "0 or 1"),
        (TaskSpec.multilabel(), Losses.BINARY_CROSS_ENTROPY, 2, torch.randn(4, 3), torch.ones(4, 3), "shape"),
        # regression: integer targets, broadcasting shape, ±inf
        (
            TaskSpec.regression(),
            Losses.MEAN_SQUARED_ERROR,
            1,
            torch.randn(4, 3),
            torch.ones(4, dtype=torch.long),
            "floating",
        ),
        (TaskSpec.regression(), Losses.MEAN_SQUARED_ERROR, 2, torch.randn(4, 3), torch.ones(4), "no broadcasting"),
        (
            TaskSpec.regression(),
            Losses.MEAN_SQUARED_ERROR,
            1,
            torch.randn(4, 3),
            _fixed([[1.0], [math.inf], [0.0], [0.0]]),
            "finite",
        ),
    ],
    ids=[
        "categorical-float-target",
        "categorical-shape",
        "categorical-range",
        "multilabel-range",
        "multilabel-shape",
        "regression-dtype",
        "regression-broadcast",
        "regression-inf",
    ],
)
def test_incompatible_batches_are_rejected_before_any_update(
    task, loss, output_dim, x, y, match, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = _model(task, loss=loss, output_dim=output_dim)
    before = _params_snapshot(model)
    with pytest.raises(TaskValidationError, match=match):
        _train(model, DataLoader(TensorDataset(x, y), batch_size=4))
    for name, param in model.net.named_parameters():
        assert torch.equal(param.detach(), before[name]), f"{name} changed before validation"


# ------------------------------------------------------------ regression


def test_regression_record_matches_the_mse_mae_oracle_and_its_masked_variant():
    adapter = task_adapter(TaskSpec.regression(2))
    output = _fixed([[1, 3], [5, 7]])
    record = adapter.record(output, _fixed([[0, 1], [4, 5]]), loss=2.5)
    assert record.kind == "regression" and record.status == "ok" and record.count == 4
    assert record.metrics["mse"] == pytest.approx(2.5) and record.metrics["mae"] == pytest.approx(1.5)
    assert (record.f1, record.accuracy, record.recall, record.precision, record.error) == (None,) * 5

    # Masked column: only the first column counts, for loss and metrics alike.
    masked = adapter.record(output, _fixed([[0, math.nan], [4, math.nan]]), loss=None)
    assert masked.count == 2
    assert masked.metrics["mse"] == pytest.approx(1.0) and masked.metrics["mae"] == pytest.approx(1.0)
    prepared = adapter.prepare(output, _fixed([[0, math.nan], [4, math.nan]]))
    display, numerator, weight = adapter.loss_terms(torch.nn.MSELoss(), *prepared)
    assert float(display) == pytest.approx(1.0) and weight == 2 and float(numerator) == pytest.approx(2.0)


def test_regression_trains_evaluates_and_predicts_continuous_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = _model(TaskSpec.regression(2, labels=["height", "width"]), loss=Losses.MEAN_SQUARED_ERROR, output_dim=2)
    x = torch.randn(12, 3)
    y = torch.stack([x.sum(dim=1), x[:, 0] - x[:, 1]], dim=1)
    y[0, 1] = math.nan  # a masked target
    run = _train(
        model, DataLoader(TensorDataset(x, y), batch_size=5), val_loader=DataLoader(TensorDataset(x, y), batch_size=5)
    )
    idp = run.idps[-1]
    assert idp.train_edp.kind == "regression" and idp.train_edp.f1 is None
    assert idp.val_edp is not None and idp.val_edp.count == 23 and set(idp.val_edp.metrics) == {"mse", "mae"}

    evaluated = model.evaluate(DataLoader(TensorDataset(x, y), batch_size=5))
    assert evaluated.count == 23 and evaluated.error is None and evaluated.accuracy is None
    with torch.no_grad():
        outputs = model.net(x)
    diff = (outputs - y)[~torch.isnan(y)].double()
    assert evaluated.metrics["mse"] == pytest.approx(float((diff**2).mean()), rel=1e-6)
    assert evaluated.metrics["mae"] == pytest.approx(float(diff.abs().mean()), rel=1e-6)
    assert evaluated.loss == pytest.approx(evaluated.metrics["mse"], rel=1e-5)

    predicted = model.predict(x.numpy())
    np.testing.assert_allclose(predicted.classes, outputs.numpy(), rtol=1e-6)  # continuous, no argmax
    assert predicted.classes.dtype.kind == "f"

    rich = model.predict_proba(x.numpy())
    assert rich.probabilities is None and rich.spec is None and rich.kind == "continuous"
    np.testing.assert_allclose(rich.decoded, outputs.numpy(), rtol=1e-6)
    np.testing.assert_array_equal(rich.sample_ids, np.arange(12))
    with pytest.raises(TypeError, match="continuous"):
        _ = rich.class_axis


# ------------------------------------------------------------ multilabel


def test_multilabel_subset_accuracy_differs_from_element_accuracy():
    adapter = task_adapter(TaskSpec.multilabel(2))
    logits = _fixed([[4.0, -4.0], [4.0, -4.0]])  # predicts [1, 0] for both rows
    record = adapter.record(logits, _fixed([[1, 0], [1, 1]]), loss=0.0)
    assert record.metrics["subset_accuracy"] == pytest.approx(0.5)
    assert record.metrics["element_accuracy"] == pytest.approx(0.75)
    assert record.accuracy == pytest.approx(0.5) and record.error == pytest.approx(0.5)


def test_multilabel_masked_targets_leave_loss_and_metrics_alike():
    adapter = task_adapter(TaskSpec.multilabel(3))
    target = _fixed([[1, math.nan, 0], [0, 1, math.nan]])
    a = _fixed([[2.0, 5.0, -1.0], [-3.0, 1.0, 9.0]])
    b = _fixed([[2.0, -7.0, -1.0], [-3.0, 1.0, -2.0]])  # differs only at masked positions
    loss_fn = torch.nn.BCEWithLogitsLoss()
    terms_a = adapter.loss_terms(loss_fn, *adapter.prepare(a, target))
    terms_b = adapter.loss_terms(loss_fn, *adapter.prepare(b, target))
    assert float(terms_a[0]) == pytest.approx(float(terms_b[0])) and terms_a[2] == terms_b[2] == 4
    rec_a, rec_b = adapter.record(a, target, loss=float(terms_a[0])), adapter.record(b, target, loss=float(terms_b[0]))
    assert rec_a.state() == rec_b.state()
    assert rec_a.count == 4


def test_multilabel_threshold_is_the_decoding_rule(tmp_path, monkeypatch):
    model = _model(TaskSpec.multilabel(2, threshold=0.3), loss=Losses.BINARY_CROSS_ENTROPY, output_dim=2)
    x = torch.randn(6, 3)
    rich = model.predict_proba(x.numpy())
    assert rich.kind == "bernoulli" and rich.probabilities is not None
    np.testing.assert_array_equal(rich.decoded, (rich.probabilities >= 0.3).astype(np.int64))
    np.testing.assert_array_equal(model.predict(x.numpy()).classes, rich.decoded)


# ---------------------------------------------- categorical legacy parity


def _categorical_data():
    set_seed(3)
    x = torch.randn(5, 3)
    y = torch.tensor([0, 2, -100, 1, 2])  # one ignored target
    return x, y


def test_categorical_evaluate_matches_legacy_on_uneven_batches_and_ignored_targets():
    x, y = _categorical_data()
    legacy = _model(None, loss=Losses.CROSS_ENTROPY, output_dim=3)
    task = _model(TaskSpec.categorical(3, ignore_index=-100), loss=Losses.CROSS_ENTROPY, output_dim=3)
    task.net.load_state_dict(legacy.net.state_dict())

    reference = legacy.evaluate(DataLoader(TensorDataset(x, y), batch_size=5))  # one five-row batch
    uneven = task.evaluate(DataLoader(TensorDataset(x, y), batch_size=2))  # [2, 2, 1]
    for name in ("loss", "error", "accuracy", "f1", "recall", "precision"):
        assert getattr(uneven, name) == pytest.approx(getattr(reference, name)), name
    assert uneven.kind == "categorical" and uneven.count == 4


def test_categorical_accumulated_window_equals_one_full_batch_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    x, y = _categorical_data()
    x, y = x[:3], torch.tensor([0, 2, 1])
    full = _model(None, loss=Losses.CROSS_ENTROPY, output_dim=3)
    window = _model(TaskSpec.categorical(3), loss=Losses.CROSS_ENTROPY, output_dim=3)
    window.net.load_state_dict(full.net.state_dict())

    _train(full, DataLoader(TensorDataset(x, y), batch_size=3))  # one update over 3 rows
    _train(window, DataLoader(TensorDataset(x, y), batch_size=2), accumulate=2)  # microbatches [2, 1]
    for (name, a), (_, b) in zip(full.net.named_parameters(), window.net.named_parameters(), strict=True):
        torch.testing.assert_close(a, b, msg=name)


def test_categorical_predictions_match_legacy_decoding():
    x, _ = _categorical_data()
    legacy = _model(None, loss=Losses.CROSS_ENTROPY, output_dim=3)
    task = _model(TaskSpec.categorical(3, labels=["a", "b", "c"]), loss=Losses.CROSS_ENTROPY, output_dim=3)
    task.net.load_state_dict(legacy.net.state_dict())
    np.testing.assert_array_equal(task.predict(x.numpy()).classes, legacy.predict(x.numpy()).classes)
    rich = task.predict_proba(x.numpy())
    explicit = legacy.predict_proba(x.numpy(), ProbabilitySpec("categorical", class_axis=1, labels=("a", "b", "c")))
    np.testing.assert_allclose(rich.probabilities, explicit.probabilities)
    assert list(rich.decoded_labels()) == list(explicit.decoded_labels())


def test_predict_proba_without_spec_needs_a_task():
    model = _model(None, loss=Losses.CROSS_ENTROPY, output_dim=3)
    with pytest.raises(TypeError, match="ProbabilitySpec"):
        model.predict_proba(np.zeros((2, 3), dtype=np.float32))


# ------------------------------------------------------- all-masked data


def test_an_all_masked_window_takes_no_update_and_records_an_empty_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = _model(TaskSpec.regression(1), loss=Losses.MEAN_SQUARED_ERROR, output_dim=1)
    before = _params_snapshot(model)
    x = torch.randn(4, 3)
    y = torch.full((4, 1), math.nan)
    # Nothing to learn from and no loss to schedule on: the plateau scheduler says so.
    with pytest.warns(RuntimeWarning, match="skipping ReduceLROnPlateau step"):
        run = _train(
            model,
            DataLoader(TensorDataset(x, y), batch_size=4),
            val_loader=DataLoader(TensorDataset(x, y), batch_size=4),
        )
    for name, param in model.net.named_parameters():
        assert torch.equal(param.detach(), before[name]), f"{name} moved on an all-masked window"
    record = run.idps[-1].train_edp
    assert (record.status, record.count, record.loss, dict(record.metrics)) == ("empty", 0, None, {})
    val = run.idps[-1].val_edp
    assert val is not None and (val.status, val.count, val.loss) == ("empty", 0, None)


# ------------------------------------------------------ review regressions


def test_multilabel_masked_loss_keeps_per_label_pos_weight():
    adapter = task_adapter(TaskSpec.multilabel(3))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 2.0, 3.0]))
    output = _fixed([[0.5, -1.0, 2.0], [1.5, 0.2, -0.3], [-2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    target = _fixed([[1, 0, 1], [0, math.nan, 1], [1, 1, 0], [0, 1, math.nan]])
    display, numerator, weight = adapter.loss_terms(loss_fn, *adapter.prepare(output, target))
    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        output, torch.nan_to_num(target), pos_weight=loss_fn.pos_weight, reduction="none"
    )
    valid = ~torch.isnan(target)
    assert weight == 10
    assert float(numerator) == pytest.approx(float(per_element[valid].sum()))
    assert float(display) == pytest.approx(float(per_element[valid].mean()))


def test_regression_targets_keep_their_dtype_under_reduced_precision_outputs():
    adapter = task_adapter(TaskSpec.regression(1))
    output = torch.tensor([[70000.0], [1000.0]], dtype=torch.float16)  # as an autocast Linear would emit
    target = torch.tensor([[70000.0], [1000.1]], dtype=torch.float32)
    prepared_output, prepared_target, valid = adapter.prepare(output, target)
    assert prepared_target.dtype == torch.float32 and bool(valid.all())
    assert prepared_target[1, 0].item() == pytest.approx(1000.1, rel=1e-7)  # not rounded to fp16
    record = adapter.record(output, target, loss=None)
    assert record.metrics["mae"] == pytest.approx((abs(float(output[0, 0]) - 70000.0) + 0.1) / 2, rel=1e-3)


@pytest.mark.parametrize(
    ("task", "output", "target"),
    [
        (TaskSpec.regression(2), _fixed([[1, 3], [5, 7]]), _fixed([[0, math.nan], [4, 5]])),
        (TaskSpec.multilabel(2), _fixed([[4.0, -4.0], [4.0, 4.0]]), _fixed([[1, math.nan], [1, 1]])),
    ],
    ids=["regression", "multilabel"],
)
def test_extra_metrics_see_only_valid_targets(task, output, target):
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(y_true, y_pred):
        seen.append((np.asarray(y_true), np.asarray(y_pred)))
        return 1.0

    record = task_adapter(task).record(output, target, loss=None, extra_metrics={"spy": spy})
    ((y_true, y_pred),) = seen
    assert not np.isnan(y_true).any() and y_true.shape == y_pred.shape == (3,)
    assert record.extra["spy"] == 1.0


def test_direct_step_callers_keep_a_window_whose_last_batch_is_masked():
    from nnx import TrainStepContext, default_train_step

    model = _model(TaskSpec.regression(1), loss=Losses.MEAN_SQUARED_ERROR, output_dim=1)
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
    before = _params_snapshot(model)
    batches = [(torch.randn(3, 3), torch.randn(3, 1)), (torch.randn(2, 3), torch.full((2, 1), math.nan))]
    for idx, batch in enumerate(batches):
        default_train_step(
            TrainStepContext(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scaler=None,
                grad_clip_norm=None,
                extra_metrics=None,
                accumulate_grad_batches=2,
                batch_idx=idx,
                epoch_idx=0,
            )
        )
    # The first batch's gradients were not thrown away: the window stepped.
    assert any(not torch.equal(p.detach(), before[n]) for n, p in model.net.named_parameters())


def test_multilabel_decoding_agrees_across_predict_proba_and_evaluate():
    model = _model(TaskSpec.multilabel(2, threshold=0.3), loss=Losses.BINARY_CROSS_ENTROPY, output_dim=2)
    x = torch.randn(64, 3)
    with torch.no_grad():
        logits = model.net(x)
    edge = math.log(0.3 / 0.7)
    expected = (logits.numpy() >= edge).astype(np.int64)
    np.testing.assert_array_equal(model.predict(x.numpy()).classes, expected)
    np.testing.assert_array_equal(model.predict_proba(x.numpy()).decoded, expected)
    np.testing.assert_array_equal(model.task_adapter.decode(logits).numpy(), expected)  # type: ignore[union-attr]
    with np.errstate(over="raise"):  # the logit-space rule never overflows exp
        model.task_adapter.decode_array(np.array([[-1000.0, 1000.0]]))  # type: ignore[union-attr]


def test_regression_predict_classes_is_not_the_logits_array():
    model = _model(TaskSpec.regression(1), loss=Losses.MEAN_SQUARED_ERROR, output_dim=1)
    result = model.predict(np.random.randn(4, 3).astype(np.float32))
    assert result.classes is not result.logits
    np.testing.assert_array_equal(result.classes, result.logits)


def test_a_non_elementwise_loss_cannot_be_masked_but_scores_complete_batches():
    class RowNorm(torch.nn.Module):
        def forward(self, output, target):
            return (output - target).norm(dim=1).mean()

    adapter = task_adapter(TaskSpec.regression(2))
    output = _fixed([[1, 3], [5, 7]])
    display, _, _ = adapter.loss_terms(RowNorm(), *adapter.prepare(output, _fixed([[0, 1], [4, 5]])))
    assert float(display) == pytest.approx(float(RowNorm()(output, _fixed([[0, 1], [4, 5]]))))
    with pytest.raises(TaskValidationError, match="cannot be masked"):
        adapter.loss_terms(RowNorm(), *adapter.prepare(output, _fixed([[0, math.nan], [4, 5]])))
