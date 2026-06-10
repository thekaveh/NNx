# Model surgery — `nnx.surgery`

The `nnx.surgery` subpackage ships five primitives that take a trained `nn.Module` and return a fresh module with a structural change applied. Three of the five (`widen`, `deepen`, `low_rank_factorize` at max rank) are **function-preserving** — the surged module's forward output equals the original's *before any training step*, so `NNModel.train()` can immediately resume refinement without an accuracy cliff. The remaining two (`drop_layer`, `low_rank_factorize` at lower rank) are chain-preserving but change the function the network computes; the surged module is meant to be refined via `NNModel.train()` to recover quality.

This is the unique compositional payoff of pairing surgery primitives with a training loop in the same toolkit: every primitive returns a fresh `nn.Module` instance, and that instance is a drop-in target for `NNModel.train()`.

## 1. The five primitives at a glance

| Primitive | Op | Function-preserving? | Returns |
|---|---|---|---|
| `widen(model, *, layer_name, new_width)` | Net2WiderNet — grow `out_features`, halve downstream weights | yes | fresh `nn.Module` |
| `deepen(model, *, after_layer_name)` | Net2DeeperNet — identity-init Linear after a ReLU | yes (ReLU only) | fresh `nn.Module` |
| `drop_layer(model, *, layer_name, importance=None)` | Replace named layer with `nn.Identity` | no — chain-preserving only | fresh `nn.Module` |
| `low_rank_factorize(linear, *, rank, method='svd')` | SVD truncation: Linear → `Sequential(Linear, Linear)` | yes at max rank, approximate below | fresh `nn.Sequential` |
| `expand_embedding(emb, *, new_num_embeddings, init=...)` | Resize Embedding; preserve original rows | yes on original token IDs | `(nn.Embedding, frozen_mask)` |

All primitives accept keyword-only arguments after the first positional and operate on a deep copy of the input so the caller's reference survives.

## 2. End-to-end: widen → refine → save

The canonical surgery workflow: load a trained checkpoint, apply a function-preserving edit, hand the surged net to `NNModel.train()` for a brief refinement pass, save the result.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    NNCheckpoint, NNModel, NNModelParams, NNOptimParams, NNParams,
    NNRun, NNSchedulerParams, NNTrainParams, Activations, Checkpoints,
    Devices, Losses, Nets, Optims, widen,
)

# 1. Load a previously trained run (or train one inline if you don't have one).
run  = NNRun.load(id="<md5>")
ckpt = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
model = NNModel.from_checkpoint(checkpoint=ckpt)

# 2. Widen the first hidden layer. The new model is a FeedFwdNN with
#    layers.0 expanded; the forward output equals model.net's exactly
#    (within FP rounding) before any training.
new_net = widen(model.net, layer_name="layers.0", new_width=64)

x = torch.randn(8, model.net.params.input_dim)
with torch.no_grad():
    assert torch.allclose(model.net(x), new_net(x), atol=1e-5)

# 3. Rewire NNModel around the wider net. The simplest path is to
#    construct a new NNModel with the wider NNParams and load the
#    surged state_dict; this preserves all of NNModel's training-loop
#    machinery (callbacks, schedulers, NNRun bookkeeping).
new_params = NNParams(
    input_dim=model.net.params.input_dim,
    output_dim=model.net.params.output_dim,
    hidden_dims=[64, *model.net.params.hidden_dims[1:]],
    dropout_prob=model.net.params.dropout_prob,
    activation=model.net.params.activation,
)
refined = NNModel(
    net_params=new_params,
    params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
)
refined.net.load_state_dict(new_net.state_dict())

# 4. Refine. Even a single epoch is enough to "absorb" the surgery —
#    function-preservation means the starting point is still good.
train_loader = DataLoader(TensorDataset(torch.randn(256, 8), torch.randint(0, 3, (256,))), batch_size=32, shuffle=True)
refined.train(params=NNTrainParams(
    n_epochs=3,
    train_loader=train_loader,
    optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=5e-5),
    scheduler=NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=3, cooldown=1, threshold=1e-3),
))
# The refined run is saved under runs/<new-id>/ via NNRun + NNCheckpoint as usual.
```

The same template works for every primitive — only the surgery line and the `NNParams` you rebuild around it change.

## 3. Parameter-count tables

Each table compares the original module against its surged form. The "delta" column is the bottom-line growth (negative for shrinking primitives).

### 3.1. `widen` on `nn.Linear(in=4, out=8)` followed by `nn.Linear(in=8, out=2)`

| Layer | Before | After (`new_width=16`) |
|---|---|---|
| First Linear (weight + bias) | `8·4 + 8 = 40` | `16·4 + 16 = 80` |
| Second Linear (weight + bias) | `2·8 + 2 = 18` | `2·16 + 2 = 34` |
| **Total** | **58** | **114** — **delta +56** |

The first layer grows by `q·(in + 1)` (new units · (incoming weight + bias)); the second's input fan-in grows by `q·out_down` (rescaled, not biased). The downstream bias is untouched.

### 3.2. `deepen` on `nn.Sequential(Linear(4,8), ReLU(), Linear(8,2))`

| Layer | Before | After (insert after the ReLU) |
|---|---|---|
| Linear(4, 8) | 40 | 40 |
| Identity-init Linear(8, 8) | — | `8·8 + 8 = 72` |
| Linear(8, 2) | 18 | 18 |
| **Total** | **58** | **130** — **delta +72** |

The inserted Linear has `dim·dim + dim` parameters and is identity-initialized (weight = I, bias = 0).

### 3.3. `low_rank_factorize` on `nn.Linear(in=64, out=32)`

| Form | Parameter count |
|---|---|
| Original `nn.Linear(64, 32)` | `32·64 + 32 = 2080` |
| Factored at `rank=8` | `8·64 + 32·8 + 32 = 800` — **delta −1280 (≈ 61% reduction)** |
| Factored at `rank=16` | `16·64 + 32·16 + 32 = 1568` — **delta −512 (≈ 25% reduction)** |
| Factored at `rank=32` (= max, exact) | `32·64 + 32·32 + 32 = 3104` — **delta +1024** (factored form is bigger past breakeven) |

The breakeven rank below which factoring saves parameters is `k* = (out·in) / (out + in)` — for a 64×32 Linear that's `32·64 / 96 ≈ 21`. At ranks below `k*`, factoring strictly reduces parameter count; at higher ranks the two-Linear sandwich actually carries more parameters than the original (but the rank-truncated weight still fits the original lower-rank structure).

### 3.4. `drop_layer` and `expand_embedding`

`drop_layer` replaces a submodule with `nn.Identity`, so parameter count drops by the entire dropped layer (e.g. dropping a square `Linear(d, d)` removes `d² + d` parameters). `expand_embedding` grows row count from `old_num` to `new_num`, so parameter count grows by `(new_num − old_num) · embedding_dim` — exactly the new-row weights, regardless of `init`.

## 4. The function-preservation contract

Every test in `tests/test_surgery_*.py` for a function-preserving primitive has, as its first assertion:

```python
assert torch.allclose(orig(x), surged(x), atol=1e-5), (
    f"surgery broke function-preservation: max diff "
    f"{(orig(x) - surged(x)).abs().max().item():.2e}"
)
```

If a future change to `widen`, `deepen`, or `low_rank_factorize` (at max rank) ever produces a `max diff` that exceeds `1e-5`, the surgery is broken — **do not relax the tolerance**. The whole point of these primitives is that the post-surgery model is *immediately* a good starting point for training; an accuracy cliff at step 0 defeats the construction.

## 5. When function-preservation doesn't hold

- `deepen` rejects any activation other than ReLU with an explicit `ValueError`. The identity-Linear trick only function-preserves through ReLU; for sigmoid / tanh / GELU networks, structurally similar insertions silently produce a drifted forward output.
- `drop_layer` is never function-preserving (with one degenerate exception: if the dropped layer was already the identity on its inputs — e.g. a ReLU fed strictly positive activations). The function is chain-preserving: dotted-name lookup, downstream shapes, and the forward pass still work.
- `low_rank_factorize` at `rank < min(out, in)` is an *approximation*. The Frobenius error of the truncation is bounded by the L2 norm of the discarded singular values (Eckart-Young) — that bound is asserted as a regression test in `tests/test_surgery_low_rank.py`.
- `expand_embedding` preserves the original rows exactly (so any token ID `< old_num` is unchanged) but introduces new rows that *must* be initialized — pick `init="zeros"` for a safe default, `init="copy_mean"` when you want the new rows to warm-start near the existing manifold.

## 6. Combining with `nnx.finetune` for the "freeze old, train new" pattern

`expand_embedding` returns a `frozen_mask` of bool shape `(new_num_embeddings,)` — `True` for rows that came from the original embedding, `False` for new rows. The mask is a hand-off to the caller's training step:

```python
new_emb, frozen_mask = expand_embedding(model.embed, new_num_embeddings=20_000, init="copy_mean")

# Register a gradient hook ONCE before training: hooks fire during
# backward(), i.e. before optimizer.step(), so frozen rows never
# receive an update. (Zeroing .grad AFTER default_train_step(ctx)
# would be too late — that helper already stepped the optimizer.)
keep = (~frozen_mask).unsqueeze(1)
new_emb.weight.register_hook(lambda g: g * keep.to(g.dtype))

model.train(params=train_params)  # default supervised step works as-is
```

`nnx.finetune.freeze` covers the simpler case of freezing entire parameter tensors via fnmatch globs; the `frozen_mask` covers the row-level case that `freeze` can't reach.
