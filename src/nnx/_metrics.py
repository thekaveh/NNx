"""Tiny internal helpers shared across training-step code paths.

`_resolve_metric` walks `(val_edp, train_edp)` taking `.error` first then
`.loss`, returning the first non-None float. Used by both
:class:`nnx.nn.nn_model.NNModel` (in `_step_scheduler` /
`_update_tqdm_postfix`) and :class:`nnx.trainer.trainer.Trainer` (same
two roles) — without a shared helper, the same six-line block was
copy-pasted four times and was prone to drift when the fallback order
changed.

`classification_edp` is the shared classification step epilogue
(`NNEvaluationDataPoint.of` + loss + top-1 error) used by
`default_train_step` and the classification-shaped paradigm factories
(distillation, feature-KD, MoE) — previously the same five-line tail
was copy-pasted at four sites.

Internal; not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional

import torch

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


def classification_edp(
    *,
    Y: torch.Tensor,
    Y_hat: torch.Tensor,
    loss: float,
    extra_metrics: Optional[Mapping[str, Callable]] = None,
) -> NNEvaluationDataPoint:
    """Standard classification step epilogue.

    Builds the per-batch :class:`NNEvaluationDataPoint` (accuracy / f1 /
    recall / precision via ``NNEvaluationDataPoint.of``), attaches the
    scalar ``loss``, and computes top-1 error from the predictions. The
    caller supplies ``Y_hat`` explicitly (typically ``logits.argmax(-1)``,
    or ``_fwd_pass``'s predictions on the default path) so each step
    keeps its own prediction rule.
    """
    return (
        NNEvaluationDataPoint.of(
            Y=Y.cpu().numpy(),
            Y_hat=Y_hat.cpu().numpy(),
            extra_metrics=extra_metrics,
        )
        .with_loss(value=loss)
        .with_error(value=float(1 - (Y_hat == Y).sum().item() / Y.size(0)))
    )
