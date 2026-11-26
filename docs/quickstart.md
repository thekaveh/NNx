# 2. Quickstart

An end-to-end CPU example you can paste into a Python REPL. Trains a tiny feed-forward classifier on random data so you can verify the install in under five seconds.

Install NNx first with `pip install thekaveh-nnx`. This guide describes the current source, and the newest PyPI release can be older. To install the source or choose extras, see the [README install section](../README.md#2-install).

## 1. Minimal example

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

Run ids are content-addressed, so running the same script again in the same directory raises `FileExistsError`. Delete `runs/` first, or pass `overwrite_existing=True` to `NNTrainParams`.

## 2. Common variations

### 2.1. GPU / Apple Silicon

```python
from nnx import Devices
NNModelParams(net=Nets.FEED_FWD, device=Devices.get(), loss=Losses.CROSS_ENTROPY)
# Devices.get() picks MPS > CUDA > CPU.
```

The other variations (sections 2.2 to 2.22) and the recipes beyond supervised classification (section 3) are in [Quickstart recipes](quickstart-recipes.md).
