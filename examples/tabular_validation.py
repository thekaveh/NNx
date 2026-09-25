"""Tabular data admission — non-finite features and narrowing overflow
(FIX-023).

Demonstrates what ``NNTabularDataset`` refuses to admit, and that a valid
frame still trains:

  1. **Source infinity before an integer / bool cast.** ``±inf`` in a
     selected feature column is rejected in source precision, before the
     dtype conversion — an int64 cast would otherwise turn it into an INT64
     extreme and a bool cast into ``True``. Finite values outside an integer
     ``feature_dtype``'s range (``300`` for ``int8``) would wrap and are
     rejected the same way. The error names the column and the requested
     ``feature_dtype``.
  2. **Narrowing overflow.** A finite float64 value that ``float16`` cannot
     represent (``1e5``) overflows to ``inf`` during conversion; the dataset
     rejects it after conversion and before the split, for features and for
     a floating ``target_dtype`` target.
  3. **Only modelled columns are inspected.** An unrelated column full of
     ``inf`` / ``NaN`` does not block construction.
  4. **A valid frame still trains.** One classification step yields a
     finite loss equal to a hand-computed cross-entropy reference, and a
     regression dataset keeps ``(N, 1)`` targets for a one-output model.

A rejected construction leaves the caller's DataFrame and the global split
RNG untouched. Fully offline, CPU only.

Run:
    python examples/tabular_validation.py

The bounded ``tabular_validation_workflow()`` helper is executed by
``tests/test_examples_smoke.py -k tabular_validation`` in a temporary
working directory.
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

from nnx import (
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
    NNTabularDataset,
    set_seed,
)


def _rejected(df: pd.DataFrame, **kwargs) -> str:
    """Build a dataset that must be rejected; return the error message."""
    frame_before = df.copy(deep=True)
    rng_before = torch.default_generator.get_state()
    try:
        NNTabularDataset(df=df, target_col="y", **kwargs)
    except ValueError as exc:
        pd.testing.assert_frame_equal(df, frame_before)
        assert torch.equal(torch.default_generator.get_state(), rng_before), "a rejection must not consume the RNG"
        return str(exc)
    raise AssertionError(f"expected NNTabularDataset to reject {kwargs}")


def tabular_validation_workflow() -> dict:
    set_seed(0)
    inf = float("inf")
    messages = {}

    # 1. ±inf would be absorbed by an int64 / bool cast: rejected in source precision.
    for dtype in (torch.int64, torch.bool):
        for value in (inf, -inf):
            msg = _rejected(pd.DataFrame({"x": [1.0, value], "y": [0, 1]}), feature_cols=["x"], feature_dtype=dtype)
            assert "['x']" in msg and str(dtype) in msg, msg
        messages[str(dtype)] = msg

    # ...and finite values an integer dtype cannot hold would wrap (300 → 44 in int8).
    msg = _rejected(pd.DataFrame({"x": [1.0, 300.0], "y": [0, 1]}), feature_cols=["x"], feature_dtype=torch.int8)
    assert "outside the range" in msg and "torch.int8" in msg, msg
    messages["int8_range"] = msg

    # 2. Finite float64 values that overflow float16 during conversion.
    msg = _rejected(pd.DataFrame({"x": [1.0, 100000.0], "y": [0, 1]}), feature_cols=["x"], feature_dtype=torch.float16)
    assert "overflow" in msg and "torch.float16" in msg, msg
    messages["feature_float16"] = msg
    msg = _rejected(
        pd.DataFrame({"x": [1.0, 2.0], "y": [0.5, 100000.0]}), feature_cols=["x"], target_dtype=torch.float16
    )
    assert "overflow" in msg and "'y'" in msg, msg
    messages["target_float16"] = msg

    # 3. An unrelated non-finite column does not block construction.
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 0.2, 0.9, 0.1, 1.1],
            "b": [1.0, 0.0, 0.8, 0.1, 0.9, 0.2],
            "notes": [inf, float("nan"), -inf, 1.0, 2.0, 3.0],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    ds = NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", val_proportion=0.0, test_proportion=0.0)

    # 4. A valid dataset drives a finite classification step matching a reference.
    model = NNModel(
        net_params=NNParams(
            input_dim=ds.input_dim,
            output_dim=ds.output_dim,
            hidden_dims=[4],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    X, y = next(iter(ds.train_loader))
    logits = model.net(X)
    loss = model.loss_fn(logits, y)
    reference = F.cross_entropy(logits, y)
    assert torch.isfinite(loss) and torch.allclose(loss, reference), (loss, reference)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.net.parameters())

    # ...and regression keeps (N, 1) targets against a one-output model.
    reg = NNTabularDataset(
        df=frame.assign(y=[0.5, 1.5, 0.25, 1.25, 0.75, 1.0]),
        feature_cols=["a", "b"],
        target_col="y",
        target_dtype=torch.float32,
        val_proportion=0.0,
        test_proportion=0.0,
    )
    _, y_reg = next(iter(reg.train_loader))
    assert reg.output_dim == 1 and tuple(y_reg.shape) == (6, 1), y_reg.shape

    summary = {
        "rejections": len(messages),
        "reference_loss": float(reference.detach()),
        "regression_target_shape": tuple(y_reg.shape),
    }
    print(f"Tabular validation workflow: {summary}")
    return summary


def main() -> None:
    summary = tabular_validation_workflow()
    print("=" * 60)
    print("Non-finite and overflowing tabular inputs are rejected before any split")
    print(summary)


if __name__ == "__main__":
    main()
