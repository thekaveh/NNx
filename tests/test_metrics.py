"""Direct unit tests for the internal `_resolve_metric` helper.

The helper was extracted to dedupe the val→train, error→loss fallback
across four call sites (NNModel._step_scheduler / _update_tqdm_postfix
and Trainer._step_scheduler / _update_tqdm_postfix). Until this file,
its contract was only exercised indirectly through those call sites,
which left the "both edps yield no signal" branch (returns None) and
the val/train ordering both untested in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nnx._metrics import _resolve_metric


def _edp(error=None, loss=None):
    """A minimal EDP-shaped object — _resolve_metric only reads `.error` / `.loss`."""
    return SimpleNamespace(error=error, loss=loss)


def test_resolve_metric_both_none_returns_none():
    """Both edps None → None. Callers treat None as "skip the step"
    (e.g., ReduceLROnPlateau.step(None) would crash inside float())."""
    assert _resolve_metric(None, None) is None


def test_resolve_metric_both_edps_have_no_signal_returns_none():
    """Both edps present but with .error AND .loss unset → still None.
    Reachable via custom train_step_fn factories that report neither."""
    assert _resolve_metric(_edp(), _edp()) is None


def test_resolve_metric_prefers_val_error_over_everything():
    """val_edp.error wins when set, ahead of val.loss and any train field."""
    val = _edp(error=0.5, loss=0.9)
    train = _edp(error=0.1, loss=0.2)
    assert _resolve_metric(val, train) == 0.5


def test_resolve_metric_falls_back_to_val_loss_when_val_error_is_none():
    """Within val_edp, .loss is the fallback when .error is unset."""
    val = _edp(error=None, loss=0.7)
    train = _edp(error=0.1, loss=0.2)
    assert _resolve_metric(val, train) == 0.7


def test_resolve_metric_falls_back_to_train_when_val_edp_is_none():
    """val_edp=None (no validation loader configured) → consult train_edp.
    .error preferred over .loss within train, same as within val."""
    train = _edp(error=0.3, loss=0.4)
    assert _resolve_metric(None, train) == 0.3


def test_resolve_metric_falls_back_to_train_loss_when_only_loss_set_anywhere():
    """val_edp has no signal AND train_edp.error is unset →
    train_edp.loss is the last fallback before None."""
    val = _edp()  # both None
    train = _edp(error=None, loss=0.55)
    assert _resolve_metric(val, train) == 0.55


def test_classification_edp_arithmetic():
    """Direct unit test for the shared classification epilogue: top-1
    error, loss attachment, and extra-metric invocation (previously
    only transitively covered through the step-factory tests)."""
    import torch

    from nnx._metrics import classification_edp

    Y = torch.tensor([0, 1, 1, 0])
    Y_hat = torch.tensor([0, 1, 0, 0])  # 3 of 4 correct
    edp = classification_edp(
        Y=Y,
        Y_hat=Y_hat,
        loss=0.5,
        extra_metrics={"n_samples": lambda y, y_hat: float(len(y))},
    )
    assert edp.loss == 0.5
    assert edp.error == 0.25
    assert edp.accuracy == 0.75
    assert edp.extra["n_samples"] == 4.0


def test_classification_edp_multilabel_error_matches_subset_accuracy():
    import torch

    from nnx._metrics import classification_edp

    y = torch.tensor([[1, 0, 1], [0, 1, 0]])
    prediction = torch.tensor([[1, 0, 1], [1, 1, 0]])
    edp = classification_edp(Y=y, Y_hat=prediction, loss=0.5)

    assert edp.accuracy == 0.5
    assert edp.error == 0.5


def test_evaluation_data_point_extra_is_immutable_and_hashable():
    from nnx import NNEvaluationDataPoint

    edp = NNEvaluationDataPoint(f1=1.0, recall=1.0, accuracy=1.0, precision=1.0, extra={"score": 1.0})
    with pytest.raises(TypeError):
        edp.extra["score"] = 2.0  # type: ignore[index]
    assert isinstance(hash(edp), int)


def test_resolve_metric_skips_nonfinite_in_priority_order():
    """FIX-009: NaN / +inf / -inf are not signals. A non-finite `.error`
    must still let the SAME edp's finite `.loss` win before falling
    through to train, and an entirely non-finite walk yields None —
    never a fabricated zero."""
    train = _edp(error=0.2, loss=0.5)
    for invalid in (float("nan"), float("inf"), float("-inf")):
        val = _edp(error=invalid, loss=1.0)
        assert _resolve_metric(val, train) == 1.0
        val.loss = invalid
        assert _resolve_metric(val, train) == 0.2
    assert _resolve_metric(_edp(error=None, loss=None), None) is None
    every_candidate_invalid = _resolve_metric(
        _edp(error=float("nan"), loss=float("inf")),
        _edp(error=float("-inf"), loss=float("nan")),
    )
    assert every_candidate_invalid is None


def test_resolve_metric_with_provenance_reports_rejected_fields():
    """Callers that warn need to say WHICH field/split was rejected and
    which one was selected; absent (None) fields are not rejections."""
    from nnx._metrics import _resolve_metric_with_provenance

    val = _edp(error=float("nan"), loss=1.0)
    train = _edp(error=0.2, loss=0.5)
    assert _resolve_metric_with_provenance(val, train) == (1.0, "val_edp.loss", ("val_edp.error=nan",))

    value, source, rejected = _resolve_metric_with_provenance(_edp(error=float("inf"), loss=float("-inf")), _edp())
    assert value is None
    assert source is None
    assert rejected == ("val_edp.error=inf", "val_edp.loss=-inf")

    assert _resolve_metric_with_provenance(_edp(), _edp()) == (None, None, ())
    assert _resolve_metric_with_provenance(None, _edp(error=None, loss=0.3)) == (0.3, "train_edp.loss", ())
