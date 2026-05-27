from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .nn_evaluation_data_point import NNEvaluationDataPoint


@dataclass(frozen=True, kw_only=True, slots=True)
class NNIterationDataPoint:
    """One row in the per-iteration training log.

    `train_edp` is computed from the current batch only. `val_edp` is the
    per-epoch validation evaluation — populated **only on the last idp of
    each epoch** (the idp at which the validation loop ran). Other idps in
    the same epoch have `val_edp=None`. When reading idps.csv, group by
    epoch_idx and take the row with val_edp set for per-epoch validation
    metrics.
    """

    lr: float
    iter_idx: int
    epoch_idx: int
    batch_idx: int
    train_edp: NNEvaluationDataPoint
    val_edp: Optional[NNEvaluationDataPoint] = None

    def with_val_edp(self, value: NNEvaluationDataPoint):
        return replace(self, val_edp=value)

    def state(self) -> dict:
        return dict(
            lr=self.lr,
            iter_idx=self.iter_idx,
            epoch_idx=self.epoch_idx,
            batch_idx=self.batch_idx,
            train_edp=self.train_edp.state(),
            val_edp=self.val_edp.state() if self.val_edp is not None else None,
        )

    @staticmethod
    def from_state(state: dict) -> NNIterationDataPoint:
        # Reassemble the `extra` dict from flattened CSV columns. After
        # NNRun.save, pd.json_normalize flattens nested {prefix: {name: v}}
        # into `<prefix>.extra.<name>` columns. We collect them back into
        # the inner state dict so NNEvaluationDataPoint.from_state can
        # populate the extra field correctly.
        def _collect_extra(prefix: str) -> dict:
            marker = f"{prefix}.extra."
            return {
                k[len(marker) :]: v
                for k, v in state.items()
                if k.startswith(marker)
                and v is not None
                # NaN values appear in CSV when other idps in the run had
                # the key set but this row didn't — filter via isna check.
                and not _is_nan(v)
            }

        val_edp = None
        if any(
            state.get(f"val_edp.{k}") is not None for k in ("loss", "error", "accuracy", "f1", "recall", "precision")
        ):
            val_edp = NNEvaluationDataPoint.from_state(
                dict(
                    loss=state.get("val_edp.loss"),
                    error=state.get("val_edp.error"),
                    accuracy=state.get("val_edp.accuracy"),
                    f1=state.get("val_edp.f1"),
                    recall=state.get("val_edp.recall"),
                    precision=state.get("val_edp.precision"),
                    extra=_collect_extra("val_edp"),
                )
            )
        return NNIterationDataPoint(
            lr=state["lr"],
            iter_idx=state["iter_idx"],
            epoch_idx=state["epoch_idx"],
            batch_idx=state["batch_idx"],
            train_edp=NNEvaluationDataPoint.from_state(
                dict(
                    loss=state["train_edp.loss"],
                    error=state["train_edp.error"],
                    accuracy=state["train_edp.accuracy"],
                    f1=state["train_edp.f1"],
                    recall=state["train_edp.recall"],
                    precision=state["train_edp.precision"],
                    extra=_collect_extra("train_edp"),
                )
            ),
            val_edp=val_edp,
        )


def _is_nan(v) -> bool:
    """True iff v is a float NaN. CSV → DataFrame → dict puts NaN for
    missing numeric cells; this catches them without depending on numpy."""
    try:
        return v != v  # NaN is the only value where this holds
    except TypeError:
        # Non-comparable types (e.g., uncomparable custom objects) — treat
        # as not-NaN. Narrow except so genuine programming errors surface.
        return False
