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
