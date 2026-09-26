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

    def with_val_edp(self, value: Optional[NNEvaluationDataPoint]) -> NNIterationDataPoint:
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
        # Reassemble the `extra` / `metrics` dicts from flattened CSV
        # columns. After NNRun.save, pd.json_normalize flattens nested
        # {prefix: {name: v}} into `<prefix>.extra.<name>` (and
        # `<prefix>.metrics.<name>`) columns. We collect them back into the
        # inner state dict so NNEvaluationDataPoint.from_state can populate
        # both mappings correctly.
        def _collect(prefix: str, group: str) -> dict:
            marker = f"{prefix}.{group}."
            return {
                k[len(marker) :]: v
                for k, v in state.items()
                if k.startswith(marker)
                and v is not None
                # NaN values appear in CSV when other idps in the run had
                # the key set but this row didn't — filter via isna check.
                and not _is_nan(v)
            }

        def _field(key: str):
            # CSV round-trip (NNRun.load → pd.read_csv) yields NaN, not
            # None, for cells that were None at save time. Map both back
            # to None so loaded idps match what state() wrote.
            v = state.get(key)
            return None if v is None or _is_nan(v) else v

        def _edp_state(prefix: str) -> dict:
            edp_state = {name: _field(f"{prefix}.{name}") for name in _EDP_SCALARS}
            edp_state["extra"] = _collect(prefix, "extra")
            edp_state["metrics"] = _collect(prefix, "metrics")
            return edp_state

        # A validation record is present when any of its scalars is — an
        # all-masked task record (FEAT-002) has no loss but still has its
        # kind, count and status.
        val_state = _edp_state("val_edp")
        val_edp = None
        if any(val_state[name] is not None for name in _EDP_SCALARS) or val_state["metrics"]:
            val_edp = NNEvaluationDataPoint.from_state(val_state)
        return NNIterationDataPoint(
            lr=state["lr"],
            iter_idx=state["iter_idx"],
            epoch_idx=state["epoch_idx"],
            batch_idx=state["batch_idx"],
            train_edp=NNEvaluationDataPoint.from_state(_edp_state("train_edp")),
            val_edp=val_edp,
        )


# Scalar NNEvaluationDataPoint.state() keys, flattened to `<prefix>.<name>`
# CSV columns by NNRun.save (`extra` / `metrics` become `<prefix>.<group>.<key>`).
_EDP_SCALARS = ("loss", "error", "accuracy", "f1", "recall", "precision", "kind", "count", "status")


def _is_nan(v) -> bool:
    """True iff v is a float NaN. CSV → DataFrame → dict puts NaN for
    missing numeric cells; this catches them without depending on numpy."""
    try:
        return v != v  # NaN is the only value where this holds
    except TypeError:
        # Non-comparable types (e.g., uncomparable custom objects) — treat
        # as not-NaN. Narrow except so genuine programming errors surface.
        return False
