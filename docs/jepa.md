# I-JEPA — Joint Embedding Predictive Architecture

NNx ships I-JEPA (Assran et al., CVPR 2023) as a `TrainStepFn` factory
alongside the other self-supervised paradigms. JEPA predicts in
**latent space** (no decoder, no pixel-reconstruction loss) and avoids
representation collapse via an EMA target encoder whose weights are
never touched by the optimizer.

The shipped path is sized for "verify-the-plumbing on a laptop, run a
ViT-S for a few epochs on 32x32 images" — not a SOTA reproduction.

## Public surface

| Symbol | Notes |
|---|---|
| `nnx.ViTNN` | Small Vision Transformer encoder. Patch-embed conv + learned pos embeds + CLS + N pre-norm blocks (RMSNorm + bidirectional MHA + SwiGLU). `forward(x, mask=None)` accepts an optional `BoolTensor[B, n_patches]` mask — True = keep — so masked patches never enter attention. |
| `nnx.ViTBlock` | Single pre-norm ViT block. Reused by `JEPAPredictor`. |
| `nnx.JEPAPredictor` | Small predictor module mapping `(context_embeds, context_positions, target_positions) -> predicted_target_embeds`. Uses its own (smaller) hidden width + position embeddings + a learned mask token. |
| `nnx.build_target_encoder(source)` | Deep-copy `source`, freeze every param (`requires_grad=False`), pin to `eval()`. The factory function freezes again defensively. |
| `nnx.update_ema(source, target, momentum)` | In-place EMA update: `target ← momentum · target + (1 - momentum) · source`. Name-keyed against the target's params so a source with extra submodules (the typical "predictor under model.net" idiom) is fine. |
| `nnx.random_block_mask(n_patches, grid_size, …)` | Sample one rectangular block as the prediction target. Returns `(context_mask, target_mask)` 1-D BoolTensors. |
| `nnx.jepa_train_step_factory(target_encoder, predictor, mask_fn, *, ema_momentum=0.996)` | Returns a `TrainStepFn` for `NNModel.train(..., train_step_fn=...)`. |

## How a step runs

```
1. mask_fn(n_patches, device) -> (ctx_mask_1d, tgt_mask_1d)
2. context_embeds = model.net(x, mask=ctx_mask)
3. with torch.no_grad():
       target_embeds = target_encoder(x)[:, target_positions, :]
4. predicted    = predictor(context_embeds, ctx_positions, tgt_positions)
5. loss         = MSE(predicted, target_embeds)
6. finalize_step(loss)              # NaN guard + grad-clip + optimizer.step()
7. update_ema(model.net, target_encoder, ema_momentum)
```

## Quickstart

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations, Devices, Losses, Nets,
    NNModel, NNModelParams, NNOptimParams, NNParams, NNSchedulerParams, NNTrainParams, Optims,
    ViTNN, JEPAPredictor,
    build_target_encoder, jepa_train_step_factory, random_block_mask, set_seed,
)

set_seed(0)

# Synthetic 32x32 batch — JEPA is self-supervised, labels are ignored.
loader = DataLoader(
    TensorDataset(torch.randn(64, 3, 32, 32), torch.zeros(64, dtype=torch.long)),
    batch_size=8,
)

# NNModel with a placeholder NNParams; the real net is the ViT below.
model = NNModel(
    net_params=NNParams(
        input_dim=3 * 32 * 32, output_dim=64, hidden_dims=[64],
        dropout_prob=0.0, activation=Activations.RELU,
    ),
    params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
)

# Swap in the trainable ViT context encoder.
model.net = ViTNN(image_size=32, patch_size=4, in_channels=3,
                  d_model=64, n_layers=4, n_heads=4).to(model.device)

# EMA target + predictor (registered under model.net so the optimizer
# trains both encoder and predictor jointly).
target_encoder = build_target_encoder(model.net)
predictor = JEPAPredictor(embed_dim=model.net.d_model,
                          n_patches=model.net.n_patches,
                          predictor_dim=32, n_layers=2, n_heads=2)
model.net.add_module("_jepa_predictor", predictor)

# One random block per step, shared across the batch.
GRID = 32 // 4
def mask_fn(n_p, device):
    return random_block_mask(n_patches=n_p, grid_size=GRID, device=device)

run = model.train(
    params=NNTrainParams(
        n_epochs=3, train_loader=loader,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=5e-4,
                            momentum=(0.9, 0.999), weight_decay=1e-4),
        scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5,
                                    patience=2, cooldown=1, threshold=1e-3),
    ),
    train_step_fn=jepa_train_step_factory(
        target_encoder=target_encoder,
        predictor=predictor,
        mask_fn=mask_fn,
        ema_momentum=0.996,
    ),
)
```

The full example with an optional CIFAR-10 download lives in
[`examples/16_ijepa_cifar10.py`](https://github.com/Kavehjr/NNx/blob/main/examples/16_ijepa_cifar10.py).

## Design notes

### Why a separate `ViTNN` instead of reusing `TransformerNN`

`TransformerNN` (SP-4) is a decoder-only LM stack. Its
`MultiHeadCausalAttention` hard-codes a causal mask + RoPE on Q/K;
both are wrong for vision tokens (bidirectional attention, learned
absolute positions). Rather than carve a parameter into the causal
path that vision callers would have to remember to flip, `ViTNN`
ships a sibling `_MultiHeadSelfAttention` and reuses the parts that
generalize (`RMSNorm`, `SwiGLU`).

### Why a single block mask per step

The reference I-JEPA samples 4 target blocks per image. The shipped
`random_block_mask` samples 1 — enough for the demo and forced to
share the mask across the batch so `ViTNN`'s forward can stack the
kept patches into a rectangular tensor. Users who want the 4-block
recipe can compose four calls inside a custom `mask_fn`; the only
hard constraint is that **kept-counts per batch row must agree** (or
the encoder will raise).

### EMA update is name-keyed, not positional

`update_ema` walks `target.named_parameters()` and looks each name
up on the source. Composing the predictor as a submodule of
`model.net` is the canonical idiom so a single optimizer picks up
both encoder and predictor params; the predictor's params then
appear on `source.named_parameters()` but not on the target, and
the EMA correctly leaves them alone. Mismatches the other direction
(target param missing on source) raise — that's a real bug.

### Loss reporting

JEPA has no classification metric. The returned
`NNEvaluationDataPoint` reports the L2 loss in both `.loss` and
`.error` so `BEST` checkpoint tracking and `ReduceLROnPlateau` have
a signal to lock onto. The other classification fields stay zero.

### Sharp edges

* **`NNModelParams.mixed_precision=True`** is rejected by the
  factory's `finalize_step` call — like every paradigm step factory
  — because AMP requires per-loss scaler bookkeeping that JEPA's
  custom step doesn't implement.
* **`NNOptimParams.accumulate_grad_batches != 1`** is also rejected
  for the same reason. JEPA's reference recipe uses large batches
  rather than accumulation; if you need accumulation, write a
  custom step that calls `update_ema` only at the cycle boundary.
* **Resume-from-checkpoint** only restores `model.net` — the EMA
  target encoder is not persisted on the standard checkpoint path.
  Use `torch.save(target_encoder.state_dict(), ...)` alongside the
  NNx checkpoint if you need exact-resume continuity.
