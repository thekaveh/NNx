"""Tiny internal helpers shared across training-step code paths.

`_resolve_metric` walks `(val_edp, train_edp)` taking `.error` first then
`.loss`, returning the first *finite* float. Used by both
:class:`nnx.nn.nn_model.NNModel` (in `_step_scheduler` /
`_update_tqdm_postfix`), :class:`nnx.trainer.trainer.Trainer` (same
two roles) and `nnx.nn.params.nn_run._best_err` (per-run and cross-run
BEST comparison) — without a shared helper, the same six-line block was
copy-pasted four times and was prone to drift when the fallback order
changed. `_resolve_metric_with_provenance` is the same walk, additionally
reporting which field was selected and which non-finite candidates were
rejected, so scheduler callers can warn with context.

`classification_edp` is the shared classification step epilogue
(`NNEvaluationDataPoint.of` + loss + top-1 error) used by
`default_train_step` and the classification-shaped paradigm factories
(distillation, feature-KD, MoE) — previously the same five-line tail
was copy-pasted at four sites.

Internal; not part of the public API.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Optional

import torch

from .nn.params.nn_evaluation_data_point import NNEvaluationDataPoint


def _resolve_metric_with_provenance(
    val_edp: Optional[NNEvaluationDataPoint],
    train_edp: Optional[NNEvaluationDataPoint],
) -> tuple[Optional[float], Optional[str], tuple[str, ...]]:
    """Walk val→train, error→loss and return ``(value, source, rejected)``.

    ``value`` is the first finite candidate (or None when there is none),
    ``source`` names it (``"val_edp.loss"``), and ``rejected`` lists every
    non-finite candidate met *before* it as ``"<split>.<field>=<value>"``.
    ``None`` fields are ordinary absences, not rejections. NaN and both
    infinities are skipped rather than returned: a NaN error would
    otherwise freeze BEST selection (``finite < nan`` is always False)
    and be fed to ``ReduceLROnPlateau`` even when the same data point
    carries a perfectly usable finite loss.
    """
    rejected: list[str] = []
    for split, edp in (("val_edp", val_edp), ("train_edp", train_edp)):
        if edp is None:
            continue
        for field in ("error", "loss"):
            value = getattr(edp, field)
            if value is None:
                continue
            if math.isfinite(float(value)):
                return value, f"{split}.{field}", tuple(rejected)
            rejected.append(f"{split}.{field}={value}")
    return None, None, tuple(rejected)


def _resolve_metric(
    val_edp: Optional[NNEvaluationDataPoint],
    train_edp: Optional[NNEvaluationDataPoint],
) -> Optional[float]:
    """Return the first finite metric, walking val→train and error→loss.

    Custom `train_step_fn` factories may leave `.error` unset (the supervised
    error field is meaningless for diffusion / SimCLR / Mixup paradigms);
    `.loss` is the universal fallback. `val_edp` outranks `train_edp` because
    a metric-driven scheduler typically wants to track the validation signal.
    A non-finite `.error` does not block the same edp's finite `.loss`.

    Returns None when neither edp has any finite signal — callers should
    treat that as "skip this step" (e.g., `ReduceLROnPlateau.step(None)`
    crashes inside `float()`). Use `_resolve_metric_with_provenance` when
    the caller needs to explain a rejection.
    """
    return _resolve_metric_with_provenance(val_edp, train_edp)[0]


def _resolve_scheduler_metric(
    val_edp: Optional[NNEvaluationDataPoint],
    train_edp: Optional[NNEvaluationDataPoint],
    *,
    epoch_idx: int,
) -> Optional[float]:
    """Resolve the metric a plateau scheduler should step on, warning once
    per call about rejected non-finite candidates and about a skipped
    step. Call it once per epoch (not once per scheduler, not from the
    progress-bar refresh) so the warning volume is bounded.

    Returns None when no finite signal exists — the caller must then skip
    ``ReduceLROnPlateau.step`` rather than feed it NaN/inf (which would
    poison its running best) or None (which crashes inside ``float()``).
    """
    value, source, rejected = _resolve_metric_with_provenance(val_edp, train_edp)
    if value is not None:
        if rejected:
            warnings.warn(
                f"epoch {epoch_idx}: ignoring non-finite metric(s) {', '.join(rejected)}; "
                f"using {source}={value} for ReduceLROnPlateau",
                RuntimeWarning,
                stacklevel=3,
            )
        return value
    if rejected:
        warnings.warn(
            f"epoch {epoch_idx}: skipping ReduceLROnPlateau step: every candidate metric is "
            f"non-finite ({', '.join(rejected)})",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"epoch {epoch_idx}: skipping ReduceLROnPlateau step: no metric available "
            "(error and loss are absent on both the validation and training data points)",
            RuntimeWarning,
            stacklevel=3,
        )
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
    edp = NNEvaluationDataPoint.of(Y=Y.cpu().numpy(), Y_hat=Y_hat.cpu().numpy(), extra_metrics=extra_metrics)
    return edp.with_loss(value=loss).with_error(value=float(1 - edp.accuracy))
