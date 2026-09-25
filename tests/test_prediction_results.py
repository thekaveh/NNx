"""FEAT-001: probability-aware prediction results.

`ProbabilitySpec` declares the task (categorical softmax vs independent
Bernoulli sigmoid, class axis, ordered labels); `NNModel.predict_proba` and
`prediction_from_logits` return a `PredictionResult` with logits,
probabilities, decoded values and sample ids. `predict()` / `PredictResult`
are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Checkpoints,
    Devices,
    Losses,
    Nets,
    NNCheckpoint,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    Optims,
    PredictionResult,
    PredictionValidationError,
    ProbabilitySpec,
    prediction_from_logits,
)

LABELS = ("low", "mid", "high")


def _model(loss: Losses = Losses.CROSS_ENTROPY, output_dim: int = 3) -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=4, output_dim=output_dim, hidden_dims=[8], dropout_prob=0.25, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=loss),
    )


def _x(n: int = 5) -> np.ndarray:
    return np.random.RandomState(0).randn(n, 4).astype(np.float32)


# ---------------------------------------------------------------- oracles


def test_categorical_oracle_softmax_rows_sum_to_one():
    spec = ProbabilitySpec(kind="categorical", class_axis=1, labels=LABELS)
    result = prediction_from_logits(np.log([[0.15, 0.70, 0.15]]), spec)
    np.testing.assert_allclose(result.probabilities, [[0.15, 0.70, 0.15]], atol=1e-12)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.decoded.tolist() == [1] and result.class_indices.tolist() == [1]
    assert result.decoded_labels().tolist() == ["mid"]
    assert result.sample_ids.tolist() == [0] and result.class_axis == 1 and result.labels == LABELS


def test_bernoulli_oracle_is_elementwise_and_not_normalized():
    result = prediction_from_logits(np.zeros((2, 3), dtype=np.float32), ProbabilitySpec(kind="bernoulli"))
    assert result.probabilities.dtype == np.float32
    np.testing.assert_allclose(result.probabilities, 0.5)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), [1.5, 1.5])  # rows are NOT normalized
    assert result.decoded.shape == (2, 3) and result.decoded.tolist() == [[1, 1, 1], [1, 1, 1]]
    with pytest.raises(TypeError, match="only defined for categorical"):
        _ = result.class_indices
    extreme = prediction_from_logits(np.array([[-1000.0, 1000.0]]), ProbabilitySpec(kind="bernoulli"))
    assert extreme.probabilities.tolist() == [[0.0, 1.0]]  # stable, no overflow


def test_class_last_and_class_first_axes():
    logits = np.random.RandomState(1).randn(2, 3, 5)  # (B, T, V) class-last
    last = prediction_from_logits(logits, ProbabilitySpec(kind="categorical", class_axis=-1))
    np.testing.assert_allclose(last.probabilities.sum(axis=-1), 1.0, atol=1e-6)
    assert last.decoded.shape == (2, 3) and last.class_axis == 2
    first = prediction_from_logits(logits, ProbabilitySpec(kind="categorical", class_axis=1))
    np.testing.assert_allclose(first.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert first.decoded.shape == (2, 5)


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "softmax"},
        {"kind": "categorical", "class_axis": True},
        {"kind": "categorical", "class_axis": 1.0},
        {"kind": "categorical", "labels": "abc"},
        {"kind": "categorical", "labels": ("a", "a", "b")},
        {"kind": "categorical", "labels": ("a", "")},
        {"kind": "categorical", "labels": ("only",)},
        {"kind": "bernoulli", "labels": ()},
    ],
)
def test_invalid_spec_raises(kwargs):
    with pytest.raises(PredictionValidationError):
        ProbabilitySpec(**kwargs)


@pytest.mark.parametrize(
    ("logits", "spec", "match"),
    [
        (np.zeros((2, 3)), ProbabilitySpec(kind="categorical", labels=("a", "b")), "2 label"),
        (np.zeros((2, 3)), ProbabilitySpec(kind="categorical", class_axis=2), "out of range"),
        (np.zeros((2, 3)), ProbabilitySpec(kind="categorical", class_axis=-3), "out of range"),
        (np.zeros((2, 3)), ProbabilitySpec(kind="categorical", class_axis=0), "sample axis"),
        (np.zeros(3), ProbabilitySpec(kind="categorical"), "at least 2-D"),
        (np.zeros((2, 1)), ProbabilitySpec(kind="categorical"), "at least 2 classes"),
        (np.array([[0.0, np.nan]]), ProbabilitySpec(kind="categorical"), "non-finite"),
        (np.array([[0.0, np.inf]]), ProbabilitySpec(kind="bernoulli"), "non-finite"),
        (np.array([["a", "b"]]), ProbabilitySpec(kind="bernoulli"), "numeric"),
    ],
)
def test_logits_that_do_not_fit_the_spec_raise(logits, spec, match):
    with pytest.raises(PredictionValidationError, match=match):
        prediction_from_logits(logits, spec)


def test_sample_ids_must_align_with_rows():
    spec = ProbabilitySpec(kind="categorical")
    with pytest.raises(PredictionValidationError, match="one integer id per sample"):
        prediction_from_logits(np.zeros((3, 2)), spec, sample_ids=[0, 1])
    with pytest.raises(PredictionValidationError, match="one integer id per sample"):
        prediction_from_logits(np.zeros((2, 2)), spec, sample_ids=[0.5, 1.5])
    assert prediction_from_logits(np.zeros((2, 2)), spec, sample_ids=torch.tensor([7, 3])).sample_ids.tolist() == [7, 3]


def test_spec_state_round_trips_label_order_and_axis():
    spec = ProbabilitySpec(kind="categorical", class_axis=-1, labels=("zeta", "alpha", "mid"))
    restored = ProbabilitySpec.from_state(yaml.safe_load(yaml.safe_dump(spec.state())))
    assert restored == spec and restored.labels == ("zeta", "alpha", "mid") and restored.class_axis == -1
    assert ProbabilitySpec.from_state(ProbabilitySpec(kind="bernoulli").state()) == ProbabilitySpec(kind="bernoulli")


# ------------------------------------------------------------ NNModel API


def test_array_tensor_tuple_and_loader_inputs_agree():
    model = _model()
    x = _x(5)
    spec = ProbabilitySpec(kind="categorical", class_axis=1, labels=LABELS)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=2)  # batches of 2, 2, 1
    assert [len(batch[0]) for batch in loader] == [2, 2, 1]
    results = [model.predict_proba(inp, spec) for inp in (x, torch.from_numpy(x), (x,), loader)]
    legacy = model.predict(x)
    for result in results:
        assert isinstance(result, PredictionResult)
        np.testing.assert_allclose(result.logits, legacy.logits, rtol=1e-6, atol=1e-6)
        assert np.array_equal(result.decoded, legacy.classes)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert result.sample_ids.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(results[0].probabilities, results[3].probabilities, rtol=1e-6, atol=1e-6)


def test_bernoulli_spec_matches_the_bce_threshold_of_predict():
    model = _model(loss=Losses.BINARY_CROSS_ENTROPY, output_dim=2)
    x = _x(6)
    result = model.predict_proba(x, ProbabilitySpec(kind="bernoulli", labels=("tag_a", "tag_b")))
    legacy = model.predict(x)
    assert np.array_equal(result.decoded, legacy.classes)
    np.testing.assert_allclose(result.probabilities, 1 / (1 + np.exp(-legacy.logits)), rtol=1e-6)


def test_transformer_class_last_matches_predict():
    from nnx.nn.params.nn_transformer_params import NNTransformerParams

    torch.manual_seed(0)
    model = NNModel(
        net_params=NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            max_seq_len=8,
        ),
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    tokens = torch.randint(0, 16, (3, 5))
    legacy = model.predict(tokens)
    result = model.predict_proba(tokens, ProbabilitySpec(kind="categorical", class_axis=-1))
    assert result.probabilities.shape == (3, 5, 16) and np.array_equal(result.decoded, legacy.classes)
    np.testing.assert_allclose(result.probabilities.sum(axis=-1), 1.0, atol=1e-6)


def test_predict_is_unchanged():
    model = _model()
    logits, classes = model.predict(_x(4))
    assert logits.shape == (4, 3) and classes.shape == (4,)


def test_parameters_gradients_and_training_modes_are_untouched():
    model = _model()
    model.net.train()
    model.net.layers[0].eval()  # a mixed-mode tree must come back exactly as it was
    modes = {name: module.training for name, module in model.net.named_modules()}
    params = {name: p.detach().clone() for name, p in model.net.named_parameters()}
    spec = ProbabilitySpec(kind="categorical", labels=LABELS)

    model.predict_proba(_x(), spec)
    assert {name: module.training for name, module in model.net.named_modules()} == modes
    assert all(torch.equal(p, params[name]) and p.grad is None for name, p in model.net.named_parameters())

    with pytest.raises(PredictionValidationError, match="label"):
        model.predict_proba(_x(), ProbabilitySpec(kind="categorical", labels=("a", "b")))
    assert {name: module.training for name, module in model.net.named_modules()} == modes
    with pytest.raises(ValueError, match="predict_proba\\(\\) loader produced zero batches"):
        model.predict_proba(DataLoader(TensorDataset(torch.zeros(0, 4)), batch_size=2), spec)
    assert {name: module.training for name, module in model.net.named_modules()} == modes
    assert all(torch.equal(p, params[name]) and p.grad is None for name, p in model.net.named_parameters())


def test_best_reconstructed_model_round_trip_keeps_parameters_and_run_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = torch.from_numpy(_x(16))
    y = torch.randint(0, 3, (16,), generator=torch.Generator().manual_seed(0))
    run = _model().train(
        NNTrainParams(
            n_epochs=2,
            train_loader=DataLoader(TensorDataset(x, y), batch_size=8),
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        )
    )
    best = NNModel.from_checkpoint(NNCheckpoint.load(run=run.id, type=Checkpoints.BEST))
    before = {name: p.detach().clone() for name, p in best.net.named_parameters()}

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(ProbabilitySpec(kind="categorical", class_axis=1, labels=LABELS).state()))
    spec = ProbabilitySpec.from_state(yaml.safe_load(spec_path.read_text()))
    assert spec.labels == LABELS and spec.class_axis == 1

    result = best.predict_proba(x.numpy(), spec)
    legacy = best.predict(x.numpy())
    assert np.array_equal(result.decoded, legacy.classes) and result.sample_ids.tolist() == list(range(16))
    assert all(torch.equal(p, before[name]) for name, p in best.net.named_parameters())
    assert NNRun.load(run.id).id == run.id


# ------------------------------------------------------- review hardening


def test_from_state_validates_labels_instead_of_splitting_strings():
    with pytest.raises(PredictionValidationError, match="sequence of strings"):
        ProbabilitySpec.from_state({"kind": "categorical", "class_axis": 1, "labels": "abc"})
    with pytest.raises(PredictionValidationError, match="sequence of strings"):
        ProbabilitySpec.from_state({"kind": "categorical", "labels": {"a": 0, "b": 1}})


def test_reduced_precision_logits():
    """bfloat16 tensors (no NumPy dtype) are upcast; float16 is computed in
    float32 and returned as float16 without overflow."""
    bf16 = prediction_from_logits(torch.zeros(2, 3, dtype=torch.bfloat16), ProbabilitySpec(kind="categorical"))
    np.testing.assert_allclose(bf16.probabilities, 1 / 3, rtol=1e-6)
    fp16 = prediction_from_logits(
        np.array([[60000.0, 0.0, -60000.0]], dtype=np.float16), ProbabilitySpec(kind="categorical")
    )
    assert fp16.probabilities.dtype == np.float16 and fp16.probabilities.tolist() == [[1.0, 0.0, 0.0]]
    as_int = prediction_from_logits(np.array([[0, 0]]), ProbabilitySpec(kind="bernoulli"))
    assert as_int.probabilities.dtype == np.float64 and as_int.probabilities.tolist() == [[0.5, 0.5]]


def test_shuffling_loader_warns_that_ids_are_positions():
    model = _model()
    x = torch.from_numpy(_x(6))
    with pytest.warns(UserWarning, match="iteration positions"):
        result = model.predict_proba(
            DataLoader(TensorDataset(x), batch_size=4, shuffle=True), ProbabilitySpec(kind="categorical")
        )
    assert result.sample_ids.tolist() == list(range(6))


def test_spec_is_rejected_on_the_first_loader_batch():
    """A spec that does not fit the logits fails before the rest of the
    loader is run."""
    model = _model()
    seen: list[int] = []

    class _Loader(DataLoader):
        def __init__(self):
            super().__init__(dataset=[0])

        def __iter__(self):
            for i in range(3):
                seen.append(i)
                yield [torch.from_numpy(_x(2))]

    with pytest.raises(PredictionValidationError, match="label"):
        model.predict_proba(_Loader(), ProbabilitySpec(kind="categorical", labels=("a", "b")))
    assert seen == [0]
