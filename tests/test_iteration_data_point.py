"""Tests for NNIterationDataPoint serialization round-trip.

Specifically, that from_state() handles val_edp=None gracefully — the case
that arises when loading a saved run from a no-validation experiment.
"""
from __future__ import annotations

from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint


def _make_edp(loss: float) -> NNEvaluationDataPoint:
    return NNEvaluationDataPoint(
        loss=loss, error=1 - loss, accuracy=0.9, f1=0.9, recall=0.9, precision=0.9,
    )


def test_from_state_round_trip_with_val_edp():
    """Standard case: both train_edp and val_edp present."""
    original = NNIterationDataPoint(
        lr=1e-3, iter_idx=10, epoch_idx=1, batch_idx=5,
        train_edp=_make_edp(0.5), val_edp=_make_edp(0.6),
    )
    state = {
        'lr': original.lr,
        'iter_idx': original.iter_idx,
        'epoch_idx': original.epoch_idx,
        'batch_idx': original.batch_idx,
        'train_edp.loss': original.train_edp.loss,
        'train_edp.error': original.train_edp.error,
        'train_edp.accuracy': original.train_edp.accuracy,
        'train_edp.f1': original.train_edp.f1,
        'train_edp.recall': original.train_edp.recall,
        'train_edp.precision': original.train_edp.precision,
        'val_edp.loss': original.val_edp.loss,
        'val_edp.error': original.val_edp.error,
        'val_edp.accuracy': original.val_edp.accuracy,
        'val_edp.f1': original.val_edp.f1,
        'val_edp.recall': original.val_edp.recall,
        'val_edp.precision': original.val_edp.precision,
    }
    reconstructed = NNIterationDataPoint.from_state(state)
    assert reconstructed.val_edp is not None
    assert reconstructed.val_edp.loss == original.val_edp.loss


def test_from_state_handles_missing_val_edp_keys():
    """Regression: from_state() used to KeyError when val_edp.* keys were
    absent (which happens when loading runs that didn't have a val set)."""
    state = {
        'lr': 1e-3,
        'iter_idx': 10,
        'epoch_idx': 1,
        'batch_idx': 5,
        'train_edp.loss': 0.5,
        'train_edp.error': 0.5,
        'train_edp.accuracy': 0.9,
        'train_edp.f1': 0.9,
        'train_edp.recall': 0.9,
        'train_edp.precision': 0.9,
        # No val_edp.* keys at all
    }
    reconstructed = NNIterationDataPoint.from_state(state)
    assert reconstructed.val_edp is None
    assert reconstructed.train_edp.loss == 0.5


def test_from_state_handles_explicit_none_val_edp_keys():
    """Same regression, when val_edp.* keys are present but all None
    (which is what state() produces when val_edp was None at serialize time)."""
    state = {
        'lr': 1e-3,
        'iter_idx': 10,
        'epoch_idx': 1,
        'batch_idx': 5,
        'train_edp.loss': 0.5,
        'train_edp.error': 0.5,
        'train_edp.accuracy': 0.9,
        'train_edp.f1': 0.9,
        'train_edp.recall': 0.9,
        'train_edp.precision': 0.9,
        'val_edp.loss': None,
        'val_edp.error': None,
        'val_edp.accuracy': None,
        'val_edp.f1': None,
        'val_edp.recall': None,
        'val_edp.precision': None,
    }
    reconstructed = NNIterationDataPoint.from_state(state)
    assert reconstructed.val_edp is None
