# Quickstart

An end-to-end CPU example you can paste into a Python REPL. Trains a tiny feed-forward classifier on random data so you can verify the install in under five seconds.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    NNModel, NNParams, NNModelParams, NNTrainParams,
    NNOptimParams, NNSchedulerParams,
    Activations, Devices, Losses, Nets, Optims,
    EarlyStopping,
)

# 1. Data
X_train, y_train = torch.randn(256, 8), torch.randint(0, 3, (256,))
X_val,   y_val   = torch.randn(64,  8), torch.randint(0, 3, (64,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=32)

# 2. Model
net_params   = NNParams(input_dim=8, output_dim=3, hidden_dims=[32, 16],
                        dropout_prob=0.1, activation=Activations.RELU)
model_params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU,
                             loss=Losses.CROSS_ENTROPY)
model = NNModel(net_params=net_params, params=model_params)

# 3. Train
train_params = NNTrainParams(
    n_epochs=10,
    seed=42,                       # reproducibility
    train_loader=train_loader,
    val_loader=val_loader,
    optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2,
                        momentum=(0.9, 0.999), weight_decay=5e-5,
                        grad_clip_norm=1.0),
    scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5,
                                patience=3, cooldown=1, threshold=1e-3),
)
run = model.train(params=train_params, callbacks=[EarlyStopping(patience=5)])

# 4. Use it
print(f"trained {len(run.idps)} iterations; saved under runs/{run.id}/")
result = model.predict(X=X_val)
print(f"predicted {len(result.classes)} samples")
```

## Variations

### GPU / Apple Silicon

```python
from nnx import Devices
NNModelParams(net=Nets.FEED_FWD, device=Devices.get(), loss=Losses.CROSS_ENTROPY)
# Devices.get() picks MPS > CUDA > CPU.
```

### Mixed precision (CUDA)

```python
NNModelParams(..., mixed_precision=True)   # silently no-op on CPU/MPS
```

### Custom metrics

```python
from sklearn.metrics import roc_auc_score

NNTrainParams(
    ...,
    extra_metrics={
        "roc_auc": lambda y, y_hat: float(roc_auc_score(y, y_hat, multi_class="ovr")),
    },
)
# Every NNEvaluationDataPoint gets `.extra["roc_auc"]` populated.
```

### TensorBoard

```bash
pip install nnx[tensorboard]
```

```python
from nnx import TensorBoardCallback
model.train(params=..., callbacks=[TensorBoardCallback(log_dir="tb_logs")])
```

### Warm-resume training

```python
# Train round 1
run = model.train(params=NNTrainParams(n_epochs=10, ...))

# Train round 2 — pick up from where the last run's LAST checkpoint left off.
model.train(params=NNTrainParams(
    n_epochs=10,
    resume_from_run_id=run.id,
    resume_from_checkpoint="last",   # or "best"
    ...
))
```

### Loading a finished run

```python
from nnx import NNRun, NNCheckpoint, Checkpoints, NNModel

run  = NNRun.load(id="<md5>")                              # rehydrate idps + params
ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
model = NNModel.from_checkpoint(checkpoint=ckpt)
```

### Non-supervised paradigms (autoencoder, VAE, etc.)

For tasks where loss isn't `loss_fn(net(X), Y)` — autoencoder reconstruction, VAE composite loss, link prediction with negative sampling, recommendation pairwise loss, diffusion noise prediction — pass `train_step_fn=` to `train()`. See [Concepts → Custom training paradigms](concepts.md#custom-training-paradigms).
