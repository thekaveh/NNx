# Examples

Runnable scripts demonstrating common nnx patterns. Each is self-contained — no external data dependencies. CPU is sufficient for everything in here.

```bash
pip install -e ".[dev]"        # if you haven't already
python examples/01_synthetic_classification.py
```

| Example | What it demonstrates |
|---|---|
| `01_synthetic_classification.py` | Train a feed-forward classifier on random data; `EarlyStopping`, `LRMonitor`; load BEST checkpoint and predict. |
| `02_resume_training.py` | Warm-resume training from a prior run's LAST checkpoint with optimizer state preserved. |
| `03_custom_metrics.py` | Plug a custom `metric_fn(Y, Y_hat)` into `NNTrainParams.extra_metrics`; inspect `idp.train_edp.extra` and `idp.val_edp.extra`. |
| `04_onnx_export.py` | Export a trained model to ONNX, validate via `onnx.checker`. |
