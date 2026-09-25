# 2. Quickstart

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

`mixed_precision=True` activates only when the model runs on CUDA: the
default training step wraps the forward in `torch.amp.autocast("cuda")` and
the loop owns one `torch.amp.GradScaler("cuda")` — scale → backward →
`unscale_` (so `grad_clip_norm` clips true gradients) → `step` → `update`. On
CPU / MPS no scaler is built (`TrainStepContext.scaler is None`) and training
is plain FP32. A custom `train_step_fn` sees the scaler through the standard
`scale` / `unscale_` / `step` / `update` / `state_dict` protocol and never
needs to branch on its concrete class. The scaler's state dictionary travels
with every checkpoint's training-state sidecar, so a warm resume continues from
the saved scale factor and growth tracker, and a resume whose configuration
turns AMP on or off relative to the checkpoint is rejected
(`resume GradScaler presence mismatch`) rather than silently continuing:

```python
amp = NNModelParams(net=Nets.FEED_FWD, device=Devices.CUDA, loss=Losses.CROSS_ENTROPY, mixed_precision=True)
first = NNModel(net_params=net_params, params=amp).train(params=NNTrainParams(n_epochs=1, train_loader=train_loader))
state = NNCheckpoint.load_training_state(run=first.id, type=Checkpoints.LAST)
print(state["scaler"]["scale"])       # None on CPU/MPS; the current scale factor on CUDA
resumed = NNModel(net_params=net_params, params=amp).train(
    params=NNTrainParams(n_epochs=1, train_loader=train_loader, resume_from_run_id=first.id),
)
```

The scaler factory is why NNx declares the PyTorch floor it does — see the
[support matrix](external-contracts.md#21-pytorch-support-matrix) for the
tested torch / torchvision / Python combinations and which of them carry real
CUDA evidence. [`examples/02_resume_training.py`](https://github.com/thekaveh/NNx/blob/main/examples/02_resume_training.py)'s
`amp_resume_compatibility()` runs the CPU no-scaler cycle everywhere and the
enabled-AMP cycle only on a CUDA host.

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

Resume validates the checkpoint's optimizer topology and matching immutable
training-state generation before applying it. It restores scheduler/scaler,
completed epoch, loader generators, and Python/NumPy/PyTorch CPU/CUDA/MPS RNG
state; use `num_workers=0` when exact continuation matters. Any re-iterable batch
source works — a `DataLoader`, a list of `(X, Y)` batches, or the graph
full-batch list — and the worker warning only appears for a real loader with
`num_workers > 0`.

Probing for state is side-effect free: `NNCheckpoint.load_with_training_state(run=..., type=Checkpoints.LAST)`
returns `(None, None)` for a run that was never written and does not create
its directory, so a preflight check never blocks the first fit.

### 2.4. Loading a finished run

```python
from nnx import NNRun, NNCheckpoint, Checkpoints, NNModel

run  = NNRun.load(id="<md5>")                              # rehydrate idps + params
ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
model = NNModel.from_checkpoint(checkpoint=ckpt)
```

### 2.5. Tabular regression targets

`NNTabularDataset` treats targets as class labels by default. For regression,
request a floating target dtype and use a one-output network with a regression
loss:

```python
import pandas as pd
import torch
from nnx import Devices, Losses, Nets, NNModelParams, NNParams, NNTabularDataset

frame = pd.DataFrame({"rooms": [1, 2, 3], "price": [95.0, 150.0, 220.0]})
dataset = NNTabularDataset(
    df=frame,
    feature_cols=["rooms"],
    target_col="price",
    target_dtype=torch.float32,
    batch_sizes=(2, 1, 1),
)
net_params = NNParams(
    input_dim=dataset.input_dim,
    output_dim=dataset.output_dim,
    hidden_dims=[16],
    dropout_prob=0.1,
)
model_params = NNModelParams(
    net=Nets.FEED_FWD,
    device=Devices.CPU,
    loss=Losses.MEAN_SQUARED_ERROR,
)
```

Leave `target_dtype=None` for classification, where targets remain `torch.long`
and `output_dim` is the number of classes.

**Admission checks.** Only the selected `feature_cols` and `target_col` are
inspected — unrelated columns may hold anything. NaN in any of them is rejected
naming the columns and the requested dtypes. Non-finite features are never
admitted: `±inf` in a selected feature is rejected in source precision *before*
the `feature_dtype` cast, so an int64 or bool cast cannot quietly absorb it
(into an INT64 extreme or `True`); a non-finite target was already rejected.
For an integer `feature_dtype`, finite values outside its range (for example
`300` with `torch.int8`, `1e30` with `torch.int64`) are rejected in source
precision too, since the cast would wrap them. A finite value that overflows a
narrower floating or complex dtype during conversion (for example `1e5` with
`feature_dtype` or `target_dtype=torch.float16`) is rejected after conversion.
Errors name the columns and dtypes, and every admission check — including the
classification label check — runs before the train/val/test split, so the
caller's DataFrame and the global RNG are untouched. See
[`examples/tabular_validation.py`](https://github.com/thekaveh/NNx/blob/main/examples/tabular_validation.py).

**Batch sizes.** Every dataset wrapper (`NNDataset`, `NNTabularDataset`,
`NNPreferenceDataset`, `NNGraphDataset`) takes `batch_sizes=(train, val, test)`.
`None` — the default for every slot — means *one batch holding the complete
split*, so the default train loader performs **one optimizer step per epoch**;
pass an explicit train size for stochastic mini-batches:

```python
from torchvision import datasets, transforms
from nnx import NNDataset

dataset = NNDataset(
    ds_class=datasets.MNIST,
    transform=transforms.ToTensor(),
    batch_sizes=(128, None, None),   # 128-sample train batches; val/test = one full batch each
    val_proportion=0.1,
)
```

A positive integer is used verbatim (larger than the split → one smaller
batch); NumPy integers are normalized to `int`. Zero, `False`, negatives,
fractions, strings and anything but a 3-tuple raise a `ValueError` naming the
slot (`batch_sizes[0] (train)`) *before* the dataset factory, tokenizer or split
runs — zero never disables a split. Empty optional splits come from the
proportions: with `val_proportion=0.0` (tabular / preference: also
`test_proportion=0.0`) that loader is `None` and the resolved `batch_sizes`
carries a placeholder `1` for it. `NNGraphDataset(sampler="full")` rejects every
explicit size, since the whole graph is always one batch.

### 2.6. Custom metrics

```python
from sklearn.metrics import f1_score

NNTrainParams(
    ...,
    extra_metrics={
        # Called as fn(y_true, y_pred): truth first, decoded class labels second.
        # zero_division=0 keeps per-batch calls quiet when a batch lacks a class.
        "weighted_f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    },
)
# Recorded in `.extra["weighted_f1"]` wherever NNx computes metrics.
```

Each callable receives `(y_true, y_pred)` — the truth first and the *decoded*
class predictions second, never probabilities, so use label-based metrics
(ranking metrics such as ROC-AUC need scores that `extra_metrics` does not
see). The default classification training step calls them per batch (a batch
whose targets are all `ignore_index` has no metrics at all), `evaluate()` (and
so the default validation pass) calls them once on the aggregate predictions,
and a custom `train_step_fn` / `eval_step_fn` decides whether and how to call
them. The values persist in `idp.*_edp.extra` and
survive `NNRun.load`; see
[`examples/03_custom_metrics.py`](https://github.com/thekaveh/NNx/blob/main/examples/03_custom_metrics.py).

### 2.7. Silencing the progress bar (CI / non-TTY)

The training loop draws a tqdm progress bar by default. Set `NNX_TQDM_DISABLE=1` in the environment to silence it — useful in CI, in non-TTY contexts, and in test suites:

```bash
NNX_TQDM_DISABLE=1 python your_train_script.py
```

`NNX_TQDM_DISABLE` is read by both `NNModel.train()` and `Trainer.train()`. Any value of `1` / `true` / `yes` (case-insensitive) disables the bar; anything else leaves it enabled.

### 2.8. TensorBoard

```bash
pip install thekaveh-nnx[tensorboard]
```

```python
from nnx import TensorBoardCallback
model.train(params=..., callbacks=[TensorBoardCallback(log_dir="tb_logs")])
```

### 2.9. LR finder pre-flight

Before a long training run, sweep learning rates exponentially and let the Smith-2017 steepest-descent heuristic pick a defensible `max_lr` for the real run. The sweep is non-destructive: model state, mixed per-module modes, loader generators, and Python/NumPy/PyTorch RNG streams are restored on exit.

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

### 2.10. Native NLL instead of cross-entropy

```python
NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.NEGATIVE_LOG_LIKELIHOOD)
```

The network still emits raw logits and `predict().logits` stays raw. NNx applies
`log_softmax` internally before the native `torch.nn.NLLLoss` during training
and evaluation, so the reported loss equals cross-entropy from the same weights
and checkpoints saved with this descriptor evaluate under the same rule when
reloaded. A custom loss module or `train_step_fn` receives the raw logits.

### 2.11. Non-finite metrics, BEST and plateau scheduling

BEST checkpoints, `runs/best` and `ReduceLROnPlateau` all track the first
*finite* value in the order validation error → validation loss → training
error → training loss (lower is better). A custom `eval_step_fn` that reports
`error=nan` with a finite `loss` therefore still drives scheduling and BEST
through its loss; the rejection is reported once per epoch as a
`RuntimeWarning`. An epoch with no finite signal at all skips the plateau
step instead of feeding it NaN, and its checkpoint counts as an unavailable
baseline that the next finite epoch (or run) replaces. Raw NaN/inf
observations stay in the run history as diagnostics.

### 2.12. Graph datasets with an empty validation split

`NNGraphDataset` reads the split sizes from the graph's `train_mask` /
`val_mask` / `test_mask`. An empty optional mask is an *absent* split: that
loader is `None` (resolved `batch_sizes` entry `0`, `"0"` in `state()`), and
passing it to `train()` skips validation instead of scoring zero seed rows —
the same optional-loader contract `NNTabularDataset` and `NNPreferenceDataset`
use. An empty `train_mask` raises at construction. Fully offline, no neighbor
sampling backend needed:

```python
import torch
from torch_geometric.data import Data
from nnx import (
    Activations, Devices, EarlyStopping, Losses, Nets,
    NNGraphDataset, NNModel, NNModelParams, NNParams, NNTrainParams,
)

class TinySplitGraph:                      # (root, transform) ctor + dataset[0] surface
    num_features, num_classes = 4, 2
    def __init__(self, root, transform=None):
        train = torch.tensor([True, True, True, False, False, False])
        self.data = Data(
            x=torch.randn(6, 4), y=torch.tensor([0, 1, 0, 1, 0, 1]),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
            train_mask=train, val_mask=torch.zeros(6, dtype=torch.bool), test_mask=~train,
        )
    def __getitem__(self, idx):
        return self.data

dataset = NNGraphDataset(ds_class=TinySplitGraph, sampler="full")
assert dataset.val_loader is None and dataset.batch_sizes == (3, 0, 3)

model = NNModel(
    net_params=NNParams(input_dim=dataset.input_dim, output_dim=dataset.output_dim,
                        hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
    params=NNModelParams(net=Nets.GRAPH_CONV, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
)
run = model.train(
    params=NNTrainParams(n_epochs=5, train_loader=dataset.train_loader, val_loader=dataset.val_loader),
    callbacks=[EarlyStopping(monitor="train_edp.loss", patience=2)],   # no validation signal to monitor
)
assert all(idp.val_edp is None for idp in run.idps)                   # persisted as absence, not zeros
print(model.evaluate(loader=dataset.test_loader).accuracy)            # test split scored on its seed rows
```

`EarlyStopping`'s default monitor reads only validation data (`val_edp.error`,
or `val_edp.loss` when the evaluator reports no error). Here it would have
nothing to read — it would warn once and never stop — so point it at a train
metric. See
[`examples/graph_optional_splits.py`](https://github.com/thekaveh/NNx/blob/main/examples/graph_optional_splits.py)
for the runnable version (it also proves, with an `eval_step_fn` spy, that no
validation hook runs).

## 3. Beyond supervised classification

For tasks where loss isn't `loss_fn(net(X), Y)` — autoencoder reconstruction, VAE composite loss, link prediction with negative sampling, recommendation pairwise loss, diffusion noise prediction — pass `train_step_fn=` to `train()`. See [Concepts → Custom training paradigms](concepts.md#6-custom-training-paradigms).

The same hook underpins the four specialization-paradigm pointers below.

### 3.1. Fine-tuning (transfer learning)

Load external pretrained weights, freeze layers by glob pattern, and (optionally) train them at different learning rates. Parameter-group rules are matched in order and the first match wins (rules never merge), so specific rules go first:

```python
# For a net with `encoder` / `head` submodules (FeedFwdNN uses `layers.0`, `layers.1`, ...;
# check `named_parameters()` — a pattern that matches nothing is silently empty).
param_groups=[
    NNParamGroupSpec(name_pattern="encoder.*bias", lr_multiplier=0.01, weight_decay=0.0),
    NNParamGroupSpec(name_pattern="encoder.*",     lr_multiplier=0.01),
    NNParamGroupSpec(name_pattern="*.bias",        weight_decay=0.0),
]
```

See [Concepts → Fine-tuning](concepts.md#7-fine-tuning-transfer-learning) and [`examples/06_finetune_with_layer_freezing.py`](https://github.com/thekaveh/NNx/blob/main/examples/06_finetune_with_layer_freezing.py).

### 3.2. Multi-optimizer training (GANs, actor-critic)

When per-batch updates need multiple optimizers (G + D for GANs, policy + value for actor-critic), use `nnx.trainer.Trainer` — accepts one `NNModel` and a dict of `NNOptimParams`, scoped via `NNParamGroupSpec` globs. See [Concepts → Multi-optimizer training](concepts.md#8-multi-optimizer-training-gans-actor-critic) and [`examples/09_gan_with_trainer.py`](https://github.com/thekaveh/NNx/blob/main/examples/09_gan_with_trainer.py).

### 3.3. Diffusion (DDPM)

For DDPM-style diffusion: `nnx.diffusion.{NoiseSchedulers, DiffusionMLP, diffusion_train_step_factory, sample}`. The training step is a `train_step_fn` on `NNModel.train()` — no Trainer, no new params dataclass. See [Concepts → Diffusion](concepts.md#9-diffusion-ddpm) and [`examples/08_diffusion_2d_mixture.py`](https://github.com/thekaveh/NNx/blob/main/examples/08_diffusion_2d_mixture.py).

### 3.4. Training paradigms (KD, SimCLR, Mixup, CutMix)

`nnx.paradigms.{kd, simclr, mixup, cutmix}_train_step_factory` return `train_step_fn`s for `NNModel.train()`. Knowledge distillation freezes the teacher and mixes soft/hard losses; SimCLR runs NT-Xent on paired-view batches; Mixup / CutMix interpolate samples within a batch. See [Concepts → Training paradigms](concepts.md#10-training-paradigms) and [`examples/10_knowledge_distillation.py`](https://github.com/thekaveh/NNx/blob/main/examples/10_knowledge_distillation.py).

### 3.5. Parameter-efficient fine-tuning (LoRA, DoRA, IA3, Prefix, Prompt, Adapters)

`nnx.peft.{LoRALinear, apply_lora_to, save_lora_weights, load_lora_weights, AdapterLayer}` plus DoRA / IA3 / PrefixTuner / PromptTuner. LoRA wraps `nn.Linear` submodules with a frozen base + trainable low-rank residual; DoRA layers in a per-output magnitude vector; IA3 is a per-output scaling; PrefixTuner / PromptTuner attach learned prefixes to a frozen `TransformerNN`; `AdapterLayer` is a bottleneck residual the user inserts manually. See [Concepts → Parameter-efficient fine-tuning](concepts.md#11-parameter-efficient-fine-tuning-lora-dora-ia3-prefix-prompt-adapters) and [`examples/07_lora_finetuning.py`](https://github.com/thekaveh/NNx/blob/main/examples/07_lora_finetuning.py).
