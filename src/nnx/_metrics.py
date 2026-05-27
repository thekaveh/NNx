"""Tiny internal helper for metric-fallback resolution.

`_resolve_metric` walks `(val_edp, train_edp)` taking `.error` first then
`.loss`, returning the first non-None float. Used by both
:class:`nnx.nn.nn_model.NNModel` (in `_step_scheduler` /
`_update_tqdm_postfix`) and :class:`nnx.trainer.trainer.Trainer` (same
two roles) — without a shared helper, the same six-line block was
copy-pasted four times and was prone to drift when the fallback order
changed.

Internal; not part of the public API.
"""
from __future__ import annotations

from typing import Optional

from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def _resolve_metric(
    val_edp: Optional[NNEvaluationDataPoint],
    train_edp: Optional[NNEvaluationDataPoint],
) -> Optional[float]:
    """Return the first non-None metric, walking val→train and error→loss.

    Custom `train_step_fn` factories may leave `.error` unset (the supervised
    error field is meaningless for diffusion / SimCLR / Mixup paradigms);
    `.loss` is the universal fallback. `val_edp` outranks `train_edp` because
    a metric-driven scheduler typically wants to track the validation signal.

    Returns None when neither edp has any signal — callers should treat that
    as "skip this step" (e.g., `ReduceLROnPlateau.step(None)` crashes inside
    `float()`).
    """
    for edp in (val_edp, train_edp):
        if edp is None:
            continue
        value = edp.error if edp.error is not None else edp.loss
        if value is not None:
            return value
    return None
