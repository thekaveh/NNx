# Quickstart

An end-to-end CPU example you can paste into a Python REPL. Trains a tiny feed-forward classifier on random data so you can verify the install in under five seconds.

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

## 2. Common variations

### 2.1. GPU / Apple Silicon

```python
from nnx import Devices
NNModelParams(net=Nets.FEED_FWD, device=Devices.get(), loss=Losses.CROSS_ENTROPY)
# Devices.get() picks MPS > CUDA > CPU.
```

### 2.2. Mixed precision (CUDA)

```python
NNModelParams(..., mixed_precision=True)   # silently no-op on CPU/MPS
```

### 2.3. Warm-resume training

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

### 2.4. Loading a finished run

```python
from nnx import NNRun, NNCheckpoint, Checkpoints, NNModel

run  = NNRun.load(id="<md5>")                              # rehydrate idps + params
ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
model = NNModel.from_checkpoint(checkpoint=ckpt)
```

### 2.5. Custom metrics

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

### 2.6. Silencing the progress bar (CI / non-TTY)

The training loop draws a tqdm progress bar by default. Set `NNX_TQDM_DISABLE=1` in the environment to silence it — useful in CI, in non-TTY contexts, and in test suites:

```bash
NNX_TQDM_DISABLE=1 python your_train_script.py
```

`NNX_TQDM_DISABLE` is read by both `NNModel.train()` and `Trainer.train()`. Any value of `1` / `true` / `yes` (case-insensitive) disables the bar; anything else leaves it enabled.

### 2.7. TensorBoard

```bash
pip install nnx[tensorboard]
```

```python
from nnx import TensorBoardCallback
model.train(params=..., callbacks=[TensorBoardCallback(log_dir="tb_logs")])
```

### 2.8. LR finder pre-flight

Before a long training run, sweep learning rates exponentially and let the Smith-2017 steepest-descent heuristic pick a defensible `max_lr` for the real run. The sweep is non-destructive — model weights and the training-mode flag are snapshotted and restored on exit.

```python
import torch.nn.functional as F
from nnx import lr_finder

result = lr_finder(
    model.net, train_loader,
    loss_fn=F.cross_entropy,
    start_lr=1e-7, end_lr=10.0, num_iter=100,
)
print(f"Suggested max_lr: {result.suggested_lr:.2e}")
result.figure.show()  # Plotly: loss vs log(LR) with the suggestion marked

# Plug into the real training run:
NNTrainParams(..., optim=NNOptimParams(name=Optims.ADAM, max_lr=result.suggested_lr, ...))
```

See [Concepts → LR finder](concepts.md#131-lr-finder) for the algorithm details and divergence early-exit behavior.

## 3. Beyond supervised classification

For tasks where loss isn't `loss_fn(net(X), Y)` — autoencoder reconstruction, VAE composite loss, link prediction with negative sampling, recommendation pairwise loss, diffusion noise prediction — pass `train_step_fn=` to `train()`. See [Concepts → Custom training paradigms](concepts.md#6-custom-training-paradigms).

The same hook underpins the four specialization-paradigm pointers below.

### 3.1. Fine-tuning (transfer learning)

Load external pretrained weights, freeze layers by glob pattern, and (optionally) train them at different learning rates. See [Concepts → Fine-tuning](concepts.md#7-fine-tuning-transfer-learning) and [`examples/06_finetune_with_layer_freezing.py`](https://github.com/thekaveh/NNx/blob/main/examples/06_finetune_with_layer_freezing.py).

### 3.2. Multi-optimizer training (GANs, actor-critic)

When per-batch updates need multiple optimizers (G + D for GANs, policy + value for actor-critic), use `nnx.trainer.Trainer` — accepts one `NNModel` and a dict of `NNOptimParams`, scoped via `NNParamGroupSpec` globs. See [Concepts → Multi-optimizer training](concepts.md#8-multi-optimizer-training-gans-actor-critic) and [`examples/09_gan_with_trainer.py`](https://github.com/thekaveh/NNx/blob/main/examples/09_gan_with_trainer.py).

### 3.3. Diffusion (DDPM)

For DDPM-style diffusion: `nnx.diffusion.{NoiseSchedulers, DiffusionMLP, diffusion_train_step_factory, sample}`. The training step is a `train_step_fn` on `NNModel.train()` — no Trainer, no new params dataclass. See [Concepts → Diffusion](concepts.md#9-diffusion-ddpm) and [`examples/08_diffusion_2d_mixture.py`](https://github.com/thekaveh/NNx/blob/main/examples/08_diffusion_2d_mixture.py).

### 3.4. Training paradigms (KD, SimCLR, Mixup, CutMix)

`nnx.paradigms.{kd, simclr, mixup, cutmix}_train_step_factory` return `train_step_fn`s for `NNModel.train()`. Knowledge distillation freezes the teacher and mixes soft/hard losses; SimCLR runs NT-Xent on paired-view batches; Mixup / CutMix interpolate samples within a batch. See [Concepts → Training paradigms](concepts.md#10-training-paradigms) and [`examples/10_knowledge_distillation.py`](https://github.com/thekaveh/NNx/blob/main/examples/10_knowledge_distillation.py).

### 3.5. Parameter-efficient fine-tuning (LoRA, DoRA, IA3, Prefix, Prompt, Adapters)

`nnx.peft.{LoRALinear, apply_lora_to, save_lora_weights, load_lora_weights, AdapterLayer}` plus DoRA / IA3 / PrefixTuner / PromptTuner. LoRA wraps `nn.Linear` submodules with a frozen base + trainable low-rank residual; DoRA layers in a per-output magnitude vector; IA3 is a per-output scaling; PrefixTuner / PromptTuner attach learned prefixes to a frozen `TransformerNN`; `AdapterLayer` is a bottleneck residual the user inserts manually. See [Concepts → Parameter-efficient fine-tuning](concepts.md#11-parameter-efficient-fine-tuning-lora-dora-ia3-prefix-prompt-adapters) and [`examples/07_lora_finetuning.py`](https://github.com/thekaveh/NNx/blob/main/examples/07_lora_finetuning.py).
