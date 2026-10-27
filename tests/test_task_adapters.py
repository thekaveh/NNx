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
    MetricSpec,
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
from tests import _task_steps as task_steps

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


class _ExplodingLoader:
    def __iter__(self):
        raise AssertionError("the loss_fn check must fail before any batch is read")


# (task, construction loss, output width, a runtime loss_fn the task cannot score)
_INCOMPATIBLE_RUNTIME_LOSSES = [
    pytest.param(TaskSpec.regression(2), Losses.MEAN_SQUARED_ERROR, 2, torch.nn.CrossEntropyLoss(), id="regression"),
    pytest.param(TaskSpec.multilabel(2), Losses.BINARY_CROSS_ENTROPY, 2, torch.nn.MSELoss(), id="multilabel"),
    pytest.param(TaskSpec.categorical(3), Losses.CROSS_ENTROPY, 3, torch.nn.MultiMarginLoss(), id="categorical"),
]


@pytest.mark.parametrize(("task", "loss", "output_dim", "runtime_loss"), _INCOMPATIBLE_RUNTIME_LOSSES)
def test_evaluate_rechecks_a_runtime_loss_fn_before_reading_a_batch(
    task, loss, output_dim, runtime_loss, tmp_path, monkeypatch
):
    """#565: evaluate() refuses a runtime loss_fn the declared task cannot
    score with train()'s TaskValidationError, before any batch is read
    (it reported a meaningless loss)."""
    monkeypatch.chdir(tmp_path)
    model = _model(task, loss=loss, output_dim=output_dim)
    model.loss_fn = runtime_loss
    with pytest.raises(TaskValidationError) as from_train:
        _train(model, _ExplodingLoader())
    with pytest.raises(TaskValidationError) as from_evaluate:
        model.evaluate(_ExplodingLoader())  # type: ignore[arg-type]
    assert str(from_evaluate.value) == str(from_train.value)


class _TaggedCrossEntropy(torch.nn.CrossEntropyLoss):
    pass


class _TaggedBCE(torch.nn.BCEWithLogitsLoss):
    pass


@pytest.mark.parametrize(
    ("task", "loss", "output_dim", "runtime_loss", "y"),
    [
        pytest.param(
            TaskSpec.categorical(3),
            Losses.CROSS_ENTROPY,
            3,
            _TaggedCrossEntropy(),
            torch.tensor([0, 1, 2, 1, 0, 2]),
            id="categorical-subclass",
        ),
        pytest.param(
            TaskSpec.multilabel(2),
            Losses.BINARY_CROSS_ENTROPY,
            2,
            _TaggedBCE(),
            _fixed([[0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]]),
            id="multilabel-subclass",
        ),
        pytest.param(
            TaskSpec.regression(2),
            Losses.MEAN_SQUARED_ERROR,
            2,
            torch.nn.MSELoss(),
            torch.linspace(-1.0, 1.0, 12).reshape(6, 2),
            id="regression-fresh-mse",
        ),
    ],
)
def test_evaluate_scores_a_compatible_runtime_loss_fn_as_before(task, loss, output_dim, runtime_loss, y):
    """#565: a runtime loss the task can score (a subclass of its loss) is
    accepted, and scores exactly as the construction loss."""
    set_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(6, 3), y), batch_size=4)
    model = _model(task, loss=loss, output_dim=output_dim)
    reference = model.evaluate(loader)
    model.loss_fn = runtime_loss
    swapped = model.evaluate(loader)
    assert swapped.state() == reference.state()


def test_evaluate_does_not_check_the_loss_fn_of_a_task_free_model_or_a_stand_in():
    """#565: a model without a task (and a legacy stand-in that borrows
    evaluate() without a task_adapter) keeps any runtime loss_fn."""

    class StandIn:
        evaluate = NNModel.evaluate

        def __init__(self, net, loss_fn):
            self.net, self.loss_fn, self.device = net, loss_fn, torch.device("cpu")

        def _fwd_pass(self, batch):
            X, Y = batch
            logits = self.net(X)
            return X, Y, logits, logits.argmax(dim=1)

    loader = DataLoader(TensorDataset(torch.randn(6, 3), torch.tensor([0, 1, 2, 1, 0, 2])), batch_size=4)
    task_free = _model(None, loss=Losses.CROSS_ENTROPY, output_dim=3)
    task_free.loss_fn = torch.nn.MultiMarginLoss()  # a categorical task would refuse it
    record = task_free.evaluate(loader)
    assert record.loss is not None and math.isfinite(record.loss)
    stand_in = StandIn(task_free.net, torch.nn.MultiMarginLoss())
    assert stand_in.evaluate(loader).state() == record.state()


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


def test_a_zero_row_multilabel_batch_records_empty_like_the_other_kinds():
    # A loader can hand a zero-row batch; the multilabel accumulator used to
    # crash reshaping it (#549) where categorical and regression recorded empty.
    for spec, target in (
        (TaskSpec.categorical(3), torch.zeros(0, dtype=torch.long)),
        (TaskSpec.multilabel(3), torch.zeros(0, 3)),
        (TaskSpec.regression(3), torch.zeros(0, 3)),
    ):
        rec = task_adapter(spec).record(torch.zeros(0, 3), target, loss=None)
        assert (rec.status, rec.count) == ("empty", 0)


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


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.int64])
def test_a_bfloat16_multilabel_batch_is_recorded(target_dtype):
    """NumPy has no bfloat16: a bf16 output (and a bf16 or integer target
    cast to it) is recorded instead of failing after the update."""
    from nnx.tasks import task_adapter

    output = torch.tensor([[2.0, -2.0, 1.0], [-1.0, 3.0, -3.0]]).bfloat16()
    target = torch.tensor([[1, 0, 1], [0, 1, 1]]).to(target_dtype)
    record = task_adapter(TaskSpec.multilabel(3)).record(output, target, loss=0.1)
    assert record.accuracy == pytest.approx(0.5)  # exact-match rows: the first row only


def test_a_task_state_without_its_kind_is_a_task_error():
    from nnx.tasks import TaskValidationError

    with pytest.raises(TaskValidationError, match="'kind'"):
        TaskSpec.from_state({"version": 1})


@pytest.mark.parametrize(
    ("spec", "target"),
    [
        (TaskSpec.categorical(3, ignore_index=-100), torch.tensor([-100, -100])),
        (TaskSpec.regression(3), torch.full((2, 3), float("nan"))),
    ],
    ids=["categorical", "regression"],
)
def test_a_fully_masked_batch_with_infinite_outputs_adds_an_exact_zero(spec, target):
    """A -inf class mask in the outputs must not turn the all-masked batch's
    zero numerator into NaN (it would poison the evaluated loss sum)."""
    adapter = task_adapter(spec)
    output = torch.tensor([[0.0, 1.0, -float("inf")], [1.0, 0.0, -float("inf")]], requires_grad=True)
    loss_fn = torch.nn.CrossEntropyLoss() if spec.kind == "categorical" else torch.nn.MSELoss()
    prepared = adapter.prepare(output, target)
    _, numerator, weight = adapter.loss_terms(loss_fn, *prepared)
    assert weight == 0.0 and float(numerator) == 0.0
    numerator.backward()
    assert torch.isfinite(output.grad).all()


def test_the_default_loss_terms_of_an_all_ignored_batch_are_an_exact_zero():
    """Without a task, an all-ignored CE batch with a -inf logit contributes
    an exact, graph-connected zero (logits.sum() * 0.0 was NaN)."""
    from nnx.nn.nn_model import _loss_terms

    logits = torch.tensor([[0.0, 1.0, -float("inf")]], requires_grad=True)
    _, numerator, weight = _loss_terms(torch.nn.CrossEntropyLoss(ignore_index=-100), logits, torch.tensor([-100]))
    assert weight == 0.0 and float(numerator) == 0.0
    numerator.backward()
    assert torch.isfinite(logits.grad).all()


_CATEGORICAL_LOSSES = {
    "ce": lambda: torch.nn.CrossEntropyLoss(),
    "ce-sum": lambda: torch.nn.CrossEntropyLoss(reduction="sum"),
    "ce-class-weight": lambda: torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.0, 3.0])),
    "ce-label-smoothing": lambda: torch.nn.CrossEntropyLoss(label_smoothing=0.1),
    "nll": lambda: torch.nn.NLLLoss(),
    "nll-sum": lambda: torch.nn.NLLLoss(reduction="sum"),
    "nll-class-weight": lambda: torch.nn.NLLLoss(weight=torch.tensor([0.2, 1.0, 3.0])),
}


@pytest.mark.parametrize("shape", [(8, 3), (4, 3, 5)], ids=["rows", "sequence"])
@pytest.mark.parametrize("loss_name", list(_CATEGORICAL_LOSSES))
def test_a_masked_categorical_batch_scores_the_selected_rows_directly(loss_name, shape):
    """A partly ignored categorical batch has the value, numerator, weight
    and gradients of the loss itself over the valid rows alone: class
    weights, label smoothing and both reductions apply unchanged."""
    loss_fn = _CATEGORICAL_LOSSES[loss_name]()
    gen = torch.Generator().manual_seed(0)
    raw = torch.randn(*shape, generator=gen)
    target = torch.randint(0, 3, (shape[0], *shape[2:]), generator=gen)
    target.view(-1)[::3] = -1  # every third position ignored
    adapter = task_adapter(TaskSpec.categorical(3, ignore_index=-1))

    logits = raw.clone().requires_grad_(True)
    display, numerator, weight = adapter.loss_terms(loss_fn, *adapter.prepare(logits, target))
    display_grad, numerator_grad = (torch.autograd.grad(t, logits, retain_graph=True)[0] for t in (display, numerator))

    reference_logits = raw.clone().requires_grad_(True)
    rows, flat = reference_logits.movedim(1, -1).reshape(-1, 3), target.reshape(-1)
    keep = flat != -1
    selected = rows[keep]
    if isinstance(loss_fn, torch.nn.NLLLoss):
        selected = selected.log_softmax(1)  # NLLLoss scores the net's logits as log-probabilities (FIX-001)
    reference = loss_fn(selected, flat[keep])
    if loss_fn.reduction == "sum":
        expected_weight, expected_numerator = None, reference
    else:
        class_weight = loss_fn.weight
        expected_weight = float(class_weight[flat[keep]].sum()) if class_weight is not None else float(keep.sum())
        expected_numerator = reference * expected_weight
    reference_grad, reference_numerator_grad = (
        torch.autograd.grad(t, reference_logits, retain_graph=True)[0] for t in (reference, expected_numerator)
    )

    assert torch.allclose(display, reference) and torch.allclose(numerator, expected_numerator)
    assert weight == pytest.approx(expected_weight) if expected_weight is not None else weight is None
    assert torch.allclose(display_grad, reference_grad) and torch.allclose(numerator_grad, reference_numerator_grad)
    assert (display_grad.movedim(1, -1).reshape(-1, 3)[~keep] == 0).all()  # ignored rows get no gradient


def test_the_categorical_adapter_overrides_only_the_masked_branch():
    """The all-masked and unmasked branches of ``loss_terms`` are shared;
    the categorical adapter supplies only its row selection."""
    categorical = type(task_adapter(TaskSpec.categorical(3)))
    assert "loss_terms" not in vars(categorical)
    assert "_masked_loss_terms" in vars(categorical)


# ------------------------------------------- #504: sum-reduction task windows

_SUM_TASKS = {
    "categorical-ce-sum": (
        lambda: TaskSpec.categorical(3, ignore_index=-100),
        Losses.CROSS_ENTROPY,
        lambda: torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100),
        3,
        torch.tensor([0, 1]),
        torch.tensor([-100, -100]),
    ),
    "regression-mse-sum": (
        lambda: TaskSpec.regression(1),
        Losses.MEAN_SQUARED_ERROR,
        lambda: torch.nn.MSELoss(reduction="sum"),
        1,
        torch.tensor([[0.5], [-1.0]]),
        torch.full((2, 1), math.nan),
    ),
}
_SUM_X = torch.randn(4, 3, generator=torch.Generator().manual_seed(1))


def _fit_sum_task(case: str, loader, *, via: str = "model", objective=None, optim=None, tag: str = "sum"):
    from nnx import NNTrainerParams, Trainer

    task, loss, loss_fn, output_dim, _, _ = _SUM_TASKS[case]
    model = _model(task(), loss=loss, output_dim=output_dim)
    model.loss_fn = loss_fn()
    optim = optim or _sgd(accumulate=2)
    if via == "trainer":
        params = (
            NNTrainerParams.builder()
            .n_epochs(1)
            .train_loader(loader)
            .optimizer("main", optim)
            .save_phase_checkpoints(False)
            .data_id(f"{tag}-{case}-trainer")
            .build()
        )
        run = Trainer(model).train(params=params, objective=objective)
    else:
        params = NNTrainParams(
            n_epochs=1, train_loader=loader, optim=optim, save_phase_checkpoints=False, data_id=f"{tag}-{case}"
        )
        run = model.train(params=params, objective=objective)
    return model, run


@pytest.mark.filterwarnings("ignore:epoch 0. skipping ReduceLROnPlateau step")  # a masked last record: no loss
@pytest.mark.parametrize("via", ["model", "trainer"])
@pytest.mark.parametrize("order", ["valid-first", "masked-first"])
@pytest.mark.parametrize("case", list(_SUM_TASKS))
def test_a_sum_task_window_mixing_valid_and_all_masked_microbatches_trains_the_valid_rows(
    tmp_path, monkeypatch, case, order, via
):
    """#504: the valid microbatch's term is summed, the all-masked one's is
    a no-sample normalized term; the engine refused the mix ("mixes 'sum'
    and 'mean' reductions") where the default step trained the valid rows."""
    from nnx.objectives import supervised_objective

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    valid, masked = _SUM_TASKS[case][4], _SUM_TASKS[case][5]
    reference, _ = _fit_sum_task(case, [(_SUM_X[:2], valid)], tag="reference")  # one step on the valid rows
    window = [(_SUM_X[:2], valid), (_SUM_X[2:], masked)]
    if order == "masked-first":
        window.reverse()
    model, run = _fit_sum_task(case, window, via=via, objective=supervised_objective(), tag=order)
    for got, want in zip(model.net.parameters(), reference.net.parameters(), strict=True):
        torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)
    assert run.idps[-1].update_count == 1


@pytest.mark.parametrize("optim", ["sgd", "sgd-momentum-wd"])
@pytest.mark.parametrize("step", ["objective", "default"])
@pytest.mark.parametrize("case", list(_SUM_TASKS))
def test_a_lone_all_masked_sum_task_window_takes_no_update(tmp_path, monkeypatch, case, step, optim):
    """#504 guard: accepting a no-sample term beside summed ones must not
    make a lone all-masked window commit — momentum and weight decay would
    then move the weights with nothing to learn from."""
    import warnings

    from nnx.objectives import supervised_objective

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    task, loss, _, output_dim, _, masked = _SUM_TASKS[case]
    before = _params_snapshot(_model(task(), loss=loss, output_dim=output_dim))
    momentum, weight_decay = (0.9, 0.1) if optim == "sgd-momentum-wd" else (0.0, 0.0)
    built = NNOptimParams.builder().sgd(max_lr=0.1, momentum=momentum, weight_decay=weight_decay).build()
    objective = supervised_objective() if step == "objective" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nothing to schedule on: the plateau scheduler says so
        model, run = _fit_sum_task(case, [(_SUM_X[2:], masked)], objective=objective, optim=built)
    for name, param in model.net.named_parameters():
        assert torch.equal(param.detach(), before[name]), f"{name} moved on an all-masked window"
    assert run.idps[-1].train_edp.loss is None
    if step == "objective":
        assert run.idps[-1].update_count == 0


@pytest.mark.filterwarnings("ignore:epoch 0. skipping ReduceLROnPlateau step")  # a masked last record: no loss
@pytest.mark.parametrize("via", ["model", "trainer"])
def test_a_recordless_objective_on_a_task_model_keeps_a_masked_microbatch_loss_none(tmp_path, monkeypatch, via):
    """#504: a custom objective on a ``TaskSpec`` model that returns no
    record still records ``None`` for an all-masked microbatch, as the
    default step does for a task model — never the window's running loss."""
    from nnx.objectives import ObjectiveResult, supervised_objective

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    supervised = supervised_objective()

    def recordless(ctx):
        return ObjectiveResult(supervised(ctx).terms)

    case = "categorical-ce-sum"
    valid, masked = _SUM_TASKS[case][4], _SUM_TASKS[case][5]
    window = [(_SUM_X[:2], valid), (_SUM_X[2:], masked)]
    _, run = _fit_sum_task(case, window, via=via, objective=recordless, tag="recordless")
    trained, empty = (idp.train_edp for idp in run.idps)
    assert trained.loss is not None
    assert empty.loss is None
    assert run.idps[-1].update_count == 1


# ------------------------------------------- narrow integer targets (#511)

_WIDE = 200  # class 156 is a real class: -100 wraps to it in uint8


_MASK_CASES = [
    (-100, torch.uint8, [156, 3, 7], [True, True, True]),  # -100 is 156 in uint8
    (256, torch.uint8, [0, 1, 4], [True, True, True]),  # 256 is 0 in uint8
    (255, torch.uint8, [255, 156, 3], [False, True, True]),  # ignore values that fit the dtype still mask
    (-100, torch.int8, [-100, 100, 3], [False, True, True]),
    (-100, torch.int16, [-100, 156, 3], [False, True, True]),
    (-100, torch.int32, [-100, 156, 3], [False, True, True]),
]


def test_ignore_index_is_compared_after_the_int64_cast():
    """#511: the categorical mask compares ``ignore_index`` with int64
    targets; in a narrow dtype the Python int wrapped and masked a real class."""
    logits = torch.randn(3, _WIDE)
    for ignore, dtype, values, expected in _MASK_CASES:
        _, target, valid = task_adapter(TaskSpec.categorical(_WIDE, ignore_index=ignore)).prepare(
            logits, torch.tensor(values, dtype=dtype)
        )
        assert valid.tolist() == expected, (ignore, dtype)
        assert target.dtype == torch.int64 and target.tolist() == values, (ignore, dtype)


def test_an_out_of_range_uint8_class_is_refused_like_an_int64_one():
    """#511: with three classes, a uint8 156 (-100 wrapped) is out of range,
    not ignored."""
    adapter = task_adapter(TaskSpec.categorical(3, ignore_index=-100))
    for dtype in (torch.int64, torch.uint8):
        with pytest.raises(TaskValidationError, match=r"\[0, 3\) or ignore_index=-100, got values in \[1, 156\]"):
            adapter.prepare(torch.randn(3, 3), torch.tensor([156, 1, 2], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_wide_unsigned_targets_are_scored_like_int64_targets(dtype):
    """#511: uint16 / uint32 / uint64 class indices are cast to int64 before
    any check (aminmax raised NotImplementedError on them before)."""
    logits = torch.randn(3, _WIDE)
    for spec in (TaskSpec.categorical(_WIDE, ignore_index=-100), TaskSpec.categorical(_WIDE)):
        adapter = task_adapter(spec)
        _, target, valid = adapter.prepare(logits, torch.tensor([156, 3, 7], dtype=dtype))
        assert target.dtype == torch.int64 and target.tolist() == [156, 3, 7] and valid.all()
        wide = adapter.record(logits, torch.tensor([156, 3, 7], dtype=dtype), loss=None)
        assert wide == adapter.record(logits, torch.tensor([156, 3, 7]), loss=None)
    with pytest.raises(TaskValidationError, match="above 2"):  # beyond int64: refused, never wrapped negative
        task_adapter(TaskSpec.categorical(3, ignore_index=-100)).prepare(
            torch.randn(1, 3), torch.tensor([2**64 - 100], dtype=torch.uint64)
        )


def _wide_model(task: TaskSpec | None, input_dim: int) -> NNModel:
    return _model(task, loss=Losses.CROSS_ENTROPY, output_dim=_WIDE, input_dim=input_dim)


_WIDE_TASKS = [TaskSpec.categorical(_WIDE, ignore_index=-100), None]


@pytest.mark.parametrize("task", _WIDE_TASKS, ids=["task", "no-task"])
def test_evaluate_scores_uint8_labels_like_int64_labels(task):
    """#511: class-156 rows stay in accuracy, declared metrics, the loss and
    the task count, over one batch and over two (the loss weights)."""
    set_seed(0)
    x, y = torch.randn(6, 4), torch.tensor([156, 156, 156, 3, 4, 5])
    model = _wide_model(task, 4)
    with torch.no_grad():
        [m for m in model.net.modules() if isinstance(m, torch.nn.Linear)][-1].bias[156] += 1000.0
    for loader in ([(x, y)], [(x[:3], y[:3]), (x[3:], y[3:])]):
        records = [
            model.evaluate([(xb, yb.to(dtype)) for xb, yb in loader], metrics=[MetricSpec("accuracy")])
            for dtype in (torch.int64, torch.uint8)
        ]
        wide, narrow = records
        assert wide.accuracy == 0.5
        assert (narrow.accuracy, dict(narrow.metrics), narrow.count) == (
            wide.accuracy,
            dict(wide.metrics),
            wide.count,
        )
        assert narrow.loss == pytest.approx(wide.loss, rel=1e-6)


@pytest.mark.parametrize("task", _WIDE_TASKS, ids=["task", "no-task"])
def test_a_loader_of_only_uint8_class_156_rows_is_scored(task):
    """#511: every uint8 156 was taken as ignored, so the loader was empty
    (a ValueError without a task, an empty record with one)."""
    set_seed(0)
    x, y = torch.randn(2, 4), torch.tensor([156, 156])
    model = _wide_model(task, 4)
    wide = model.evaluate([(x, y)])
    narrow = model.evaluate([(x, y.to(torch.uint8))])
    assert narrow.loss == pytest.approx(wide.loss, rel=1e-6)
    assert narrow.count == wide.count == (2 if task is not None else None)


@pytest.mark.parametrize("task", _WIDE_TASKS, ids=["task", "no-task"])
def test_training_on_uint8_labels_ends_where_int64_labels_do(tmp_path, monkeypatch, task):
    """#511: one batch of 8, batches of 4 led by an all-156 batch, and those
    batches accumulated over 2 end with the same parameters for both dtypes
    (an all-156 uint8 batch or window added no gradient before)."""
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    set_seed(0)
    x, y = torch.randn(8, 3), torch.tensor([156] * 4 + [3] * 4)
    for batch_size, accumulate in ((8, 1), (4, 1), (4, 2)):
        trained = []
        for dtype in (torch.int64, torch.uint8):
            run_dir = tmp_path / f"{batch_size}-{accumulate}-{dtype}"  # one run directory per fit
            run_dir.mkdir()
            monkeypatch.chdir(run_dir)
            model = _wide_model(task, 3)
            loader = DataLoader(TensorDataset(x, y.to(dtype)), batch_size=batch_size)
            _train(model, loader, accumulate=accumulate, n_epochs=3)
            trained.append(_params_snapshot(model))
        for name, value in trained[0].items():
            torch.testing.assert_close(trained[1][name], value, msg=f"{batch_size}/{accumulate}: {name}")


# ------------------------------- #512: stock-forward elementwise loss subclasses


class MyMSE(torch.nn.MSELoss):  # the stock forward: only a name
    pass


class MyL1(torch.nn.L1Loss):
    pass


class MySmoothL1(torch.nn.SmoothL1Loss):
    pass


class MyHuber(torch.nn.HuberLoss):
    pass


class Huber05(torch.nn.HuberLoss):  # an __init__-only default, the stock forward
    def __init__(self) -> None:
        super().__init__(delta=0.5)


class MyBCE(torch.nn.BCEWithLogitsLoss):
    pass


def _regression_rows() -> tuple[torch.Tensor, torch.Tensor]:
    """The #512 rows: batch 1 (rows 0-3) unmasked, batch 2 (rows 4-7) with one NaN target."""
    generator = torch.Generator().manual_seed(1)
    x, y = torch.randn(8, 3, generator=generator), torch.randn(8, 3, generator=generator)
    y[5, 1] = math.nan
    return x, y


def _multilabel_rows() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    x, y = torch.randn(8, 3, generator=generator), torch.randint(0, 2, (8, 3), generator=generator).float()
    y[6, 0] = math.nan
    return x, y


_STOCK_FORWARD_TASKS = {
    "mse": (lambda: TaskSpec.regression(3), Losses.MEAN_SQUARED_ERROR, MyMSE, _regression_rows),
    "bce": (lambda: TaskSpec.multilabel(3), Losses.BINARY_CROSS_ENTROPY, MyBCE, _multilabel_rows),
}


def _stock_forward_model(case: str, loss_fn: torch.nn.Module | None = None) -> NNModel:
    task, loss, subclass, _ = _STOCK_FORWARD_TASKS[case]
    model = _model(task(), loss=loss, output_dim=3)
    model.loss_fn = loss_fn if loss_fn is not None else subclass()
    return model


def _masked_mean(case: str, model: NNModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The loss over the valid targets of one full batch, from the net directly."""
    valid = ~torch.isnan(y)
    if case == "mse":
        per_element = (model.net(x) - y.nan_to_num()) ** 2
    else:
        per_element = torch.nn.functional.binary_cross_entropy_with_logits(
            model.net(x), y.nan_to_num(), reduction="none"
        )
    return per_element[valid].mean()


def test_a_stock_forward_mse_subclass_evaluates_to_the_masked_mse(tmp_path, monkeypatch):
    """#512: a mixed loader's loss weighted the unmasked batch by its rows
    (0.832098) where its masked batch counted valid elements, so
    ``evaluate().loss`` and the streaming ``val_edp.loss`` disagreed with
    ``metrics['mse']`` (0.981019)."""
    from nnx.streaming import streaming_eval_step

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    x, y = _regression_rows()
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    for loss_fn in (torch.nn.MSELoss(), MyMSE()):
        model = _stock_forward_model("mse", loss_fn)
        record = model.evaluate(loader)
        assert record.loss == pytest.approx(0.981019, abs=1e-6) == pytest.approx(record.metrics["mse"])
        frozen = NNOptimParams.builder().sgd(max_lr=0.0).build()
        params = NNTrainParams(
            n_epochs=1, optim=frozen, train_loader=loader, val_loader=loader, data_id=type(loss_fn).__name__
        )
        val = model.train(params=params, eval_step_fn=streaming_eval_step).idps[-1].val_edp
        assert val.loss == pytest.approx(0.981019, abs=1e-6) == pytest.approx(val.metrics["mse"])


def test_a_stock_forward_bce_subclass_evaluates_to_the_masked_bce():
    """#512: ``MyBCE`` reported 0.699854 on the mixed multilabel loader."""
    x, y = _multilabel_rows()
    for loss_fn in (torch.nn.BCEWithLogitsLoss(), MyBCE()):
        model = _stock_forward_model("bce", loss_fn)
        record = model.evaluate(DataLoader(TensorDataset(x, y), batch_size=4))
        with torch.no_grad():
            assert (
                record.loss
                == pytest.approx(0.680013, abs=1e-6)
                == pytest.approx(float(_masked_mean("bce", model, x, y)))
            )


@pytest.mark.parametrize(
    "loss_fn",
    [
        torch.nn.MSELoss(),
        MyMSE(),
        MyL1(),
        MySmoothL1(),
        MyHuber(),
        Huber05(),
        MySmoothL1(beta=0.5),
        MyBCE(),
        MyBCE(pos_weight=torch.tensor([1.0, 2.0, 0.5])),
    ],
    ids=lambda fn: type(fn).__name__,
)
def test_stock_forward_elementwise_subclasses_weight_unmasked_batches_per_element(loss_fn):
    """#512: an unmasked ``(4, 3)`` batch weighs its 12 elements (subclasses
    got 4.0, the rows), a batch with one NaN target its 11 valid ones; the
    display loss is the module's own mean."""
    multilabel = isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)
    adapter = task_adapter(TaskSpec.multilabel(3) if multilabel else TaskSpec.regression(3))
    generator = torch.Generator().manual_seed(3)
    output = torch.randn(4, 3, generator=generator)
    target = torch.randint(0, 2, (4, 3), generator=generator).float() if multilabel else torch.randn(4, 3)
    display, numerator, weight = adapter.loss_terms(loss_fn, *adapter.prepare(output, target))
    assert weight == 12.0
    torch.testing.assert_close(display, loss_fn(output, target))
    torch.testing.assert_close(numerator, display * 12.0)
    target[2, 1] = math.nan
    display, numerator, weight = adapter.loss_terms(loss_fn, *adapter.prepare(output, target))
    assert weight == 11.0
    torch.testing.assert_close(display, numerator / 11.0)


def _sgd_step_on(loss: torch.Tensor, model: NNModel, lr: float = 0.1) -> dict[str, torch.Tensor]:
    """The parameters after one plain SGD step on ``loss`` (``model`` is left untouched)."""
    grads = torch.autograd.grad(loss, list(model.net.parameters()))
    return {name: (p - lr * g).detach() for (name, p), g in zip(model.net.named_parameters(), grads, strict=True)}


@pytest.mark.parametrize("case", list(_STOCK_FORWARD_TASKS))
def test_a_mixed_window_of_a_stock_forward_subclass_equals_one_full_batch_update(tmp_path, monkeypatch, case):
    """#512: under the default step, an ``[unmasked, masked]`` window is one
    step on the valid targets of the full batch (``MyMSE`` was 8.160e-03
    off, ``MyBCE`` 2.292e-03; tests/test_objective_engine.py covers
    ``supervised_objective()``)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    x, y = _STOCK_FORWARD_TASKS[case][3]()
    initial = _stock_forward_model(case)
    reference = _sgd_step_on(_masked_mean(case, initial, x, y), initial)
    model = _stock_forward_model(case)
    params = NNTrainParams(
        n_epochs=1, optim=_sgd(accumulate=2), train_loader=DataLoader(TensorDataset(x, y), batch_size=4)
    )
    model.train(params=params)
    for name, value in reference.items():
        torch.testing.assert_close(dict(model.net.named_parameters())[name].detach(), value, msg=name)


def test_task_weights_of_other_losses_and_shapes_are_unchanged():
    """#512 leaves these alone: categorical CE / NLL subclasses count rows
    (4.0, 3.0), sum losses are additive (None), an ``(N, 1)`` regression
    counts its rows (4.0, 3.0), and a ``forward``-overriding subclass keeps
    row weights on a complete batch and cannot be masked."""

    class MyCE(torch.nn.CrossEntropyLoss):
        pass

    class MyNLL(torch.nn.NLLLoss):
        pass

    class OwnForward(torch.nn.MSELoss):
        def forward(self, output, target):
            return ((output - target) ** 2).sum(dim=1).mean()

    def weights(spec, loss_fn, output, complete, masked):
        adapter = task_adapter(spec)
        return tuple(adapter.loss_terms(loss_fn, *adapter.prepare(output, target))[2] for target in (complete, masked))

    logits = torch.randn(4, 3, generator=torch.Generator().manual_seed(4))
    categorical = TaskSpec.categorical(3, ignore_index=-100)
    classes, ignored = torch.tensor([0, 1, 2, 1]), torch.tensor([0, -100, 2, 1])
    for loss_fn in (MyCE(), MyNLL()):
        assert weights(categorical, loss_fn, logits, classes, ignored) == (4.0, 3.0)
    assert weights(categorical, MyCE(reduction="sum"), logits, classes, ignored) == (None, None)
    values = torch.randn(4, 3, generator=torch.Generator().manual_seed(5))
    one_masked = values.clone()
    one_masked[1, 2] = math.nan
    assert weights(TaskSpec.regression(3), MyMSE(reduction="sum"), logits, values, one_masked) == (None, None)
    column, column_masked = values[:, :1].clone(), values[:, :1].clone()
    column_masked[0, 0] = math.nan
    assert weights(TaskSpec.regression(1), MyMSE(), logits[:, :1], column, column_masked) == (4.0, 3.0)
    adapter = task_adapter(TaskSpec.regression(3))
    assert adapter.loss_terms(OwnForward(), *adapter.prepare(logits, values))[2] == 4.0
    with pytest.raises(TaskValidationError, match="cannot be masked"):
        adapter.loss_terms(OwnForward(), *adapter.prepare(logits, one_masked))


# ------------------------------------------------------------------ paradigm steps (#513)


def _paradigm_hooks(teacher: NNModel) -> dict[str, dict]:
    from nnx.objectives import kd_objective
    from nnx.paradigms import feature_kd_train_step_factory, kd_train_step_factory, moe_train_step_factory

    return {
        "kd_train_step_factory": {"train_step_fn": kd_train_step_factory(teacher, alpha=0.0)},
        "feature_kd_train_step_factory": {
            "train_step_fn": feature_kd_train_step_factory(
                teacher, auxiliary_layers={"layers.0": "layers.0"}, alpha=0.0, beta=0.0
            )
        },
        "moe_train_step_factory": {"train_step_fn": moe_train_step_factory(aux_loss_weight=0.0)},
        "kd_objective": {"objective": kd_objective(teacher, alpha=0.0)},
    }


@pytest.mark.parametrize(
    "name", ["kd_train_step_factory", "feature_kd_train_step_factory", "moe_train_step_factory", "kd_objective"]
)
def test_paradigm_records_are_the_default_steps_task_records(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = task_steps.categorical()
    hook = _paradigm_hooks(task_steps.categorical(seed=1))[name]
    for Y in (task_steps.MIXED, task_steps.ALL_IGNORED):
        _, reference = task_steps.fit(base, Y)
        _, run = task_steps.fit(base, Y, **hook)
        got, want = task_steps.record(run), task_steps.record(reference)
        assert (got.kind, got.count, got.status) == (want.kind, want.count, want.status)
        assert got.loss == pytest.approx(want.loss) if want.loss is not None else got.loss is None
        assert (got.accuracy, got.f1, got.error) == pytest.approx((want.accuracy, want.f1, want.error))


def test_a_marked_step_refuses_a_task_kind_it_does_not_train():
    from nnx.paradigms import kd_train_step_factory, mixup_train_step_factory
    from nnx.tasks import _check_step_task

    step = mixup_train_step_factory()
    for task in (None, TaskSpec.categorical(3), TaskSpec.multilabel(3)):
        _check_step_task(step, task)  # trained through the adapter (or no task at all)
    _check_step_task(lambda ctx: None, TaskSpec.regression(2))  # a user's own step is the user's business
    with pytest.raises(TaskValidationError, match=r"mixup_train_step_factory trains categorical and multilabel"):
        _check_step_task(step, TaskSpec.regression(2))
    teacher = _model(None, loss=Losses.CROSS_ENTROPY, output_dim=3)
    with pytest.raises(TaskValidationError, match=r"this model declares a regression task \(TaskSpec\.regression\)"):
        _check_step_task(kd_train_step_factory(teacher), TaskSpec.regression())
