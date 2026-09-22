# 4. Model surgery — `nnx.surgery`

The `nnx.surgery` subpackage ships five primitives that take a trained `nn.Module` and return a fresh module with a structural change applied. Four are **function-preserving** in some sense — `widen` and `deepen` unconditionally preserve the forward output (before any training step); `low_rank_factorize` is function-preserving at max rank and approximate below; `expand_embedding` preserves the forward output on original token IDs. Only `drop_layer` is purely chain-preserving — it changes the function the network computes; the surged module is meant to be refined via `NNModel.train()` to recover quality. See the table in §1 for the per-primitive breakdown.

This is the unique compositional payoff of pairing surgery primitives with a training loop in the same toolkit: every primitive returns a fresh `nn.Module` instance, and that instance is a drop-in target for `NNModel.train()`.

## 1. The five primitives at a glance

| Primitive | Op | Function-preserving? | Returns |
|---|---|---|---|
| `widen(model, *, layer_name, new_width)` | Net2WiderNet — grow `out_features`, rescale the consumer's incoming columns | yes — for a Linear inside `nn.Sequential` or `FeedFwdNN.layers` whose path to the next Linear holds only elementwise ops (activations, `Dropout`, `Identity`); eval-mode parity when dropout > 0; Softmax / LayerNorm / BatchNorm boundaries, other containers and aliased modules are rejected | fresh `nn.Module` |
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

# 3. Rewire NNModel around the wider net: build a NEW, correctly
#    described model and load the surged state_dict. Rebuild the
#    immutable params with `dataclasses.replace` so the new width and
#    every per-layer `activations` / `dropout_probs` override stay
#    aligned (a hand-built NNParams from scalar fields would silently
#    drop those overrides). Never assign the widened net beneath the
#    old `net_params`: ordinary checkpoint reconstruction reads the
#    descriptor, and a stale one no longer describes the network.
from dataclasses import replace

new_params = replace(model.net.params, hidden_dims=[64, *model.net.params.hidden_dims[1:]])
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

The inserted Linear has `dim·dim + dim` parameters and is identity-initialized (weight = I, bias = 0). On a `FeedFwdNN`, `deeper.params.hidden_dims` gains the matching entry and any `activations` / `dropout_probs` overrides stay aligned (dropout `0.0` at the new site).

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

- `widen` infers the consumer only where the execution order is proven: a target inside an `nn.Sequential` (walking the following siblings) or inside `FeedFwdNN.layers` (resolving the *effective* `activation_for(i)` / `dropout_for(i)`, not the scalar defaults). Every op between the target and the next Linear must be elementwise — the built-in activations except Softmax, `Dropout`, `Identity` — because duplicating units through a width-dependent op (Softmax's denominator, LayerNorm / BatchNorm statistics) changes the output. Width-dependent ops, other containers (registration order is not data flow), missing consumers and modules registered under more than one path (aliases) raise a `ValueError` naming the boundary *before* anything is allocated; the source model, its module identities and the RNG are untouched. With dropout configured the parity promise is for eval mode — a stochastic training forward is not identical by construction.
- `deepen` rejects any activation other than ReLU with an explicit `ValueError`. The identity-Linear trick only function-preserves through ReLU; for sigmoid / tanh / GELU networks, structurally similar insertions silently produce a drifted forward output. On a `FeedFwdNN` the check reads the activation actually applied at the insertion site — `params.activation_for(i)`, so a per-layer override wins over the scalar default in both directions — and the returned network carries updated immutable `params`: one more `hidden_dims` entry at the site, a ReLU inserted into an explicit `activations` list, and the new identity site's dropout set to `0.0` while every existing site keeps its own probability (a nonzero scalar dropout is materialized into a per-layer list rather than switched off). Adopt `deeper.params` as the `net_params` of a *fresh* `NNModel` when you rebuild for refinement — an outer model still holding the old descriptor cannot reconstruct the deeper checkpoint.
- `drop_layer` is never function-preserving (with one degenerate exception: if the dropped layer was already the identity on its inputs — e.g. a ReLU fed strictly positive activations). The function is chain-preserving: dotted-name lookup, downstream shapes, and the forward pass still work.
- `low_rank_factorize` at `rank < min(out, in)` is an *approximation*. The Frobenius error of the truncation is bounded by the L2 norm of the discarded singular values (Eckart-Young) — that bound is asserted as a regression test in `tests/test_surgery_low_rank.py`.
- `expand_embedding` preserves the original rows exactly (so any token ID `< old_num` is unchanged) but introduces new rows that *must* be initialized — pick `init="zeros"` for a safe default, `init="copy_mean"` when you want the new rows to warm-start near the existing manifold.

## 6. See it in practice

Worked end-to-end in [ml-eng-lab](https://github.com/thekaveh/ml-eng-lab):

- [model_surgery-mnist-ffnn-pytorch](https://github.com/thekaveh/ml-eng-lab/blob/main/notebooks/model_surgery-mnist-ffnn-pytorch/notebook.ipynb) — applies `nnx.surgery.widen` and `nnx.surgery.deepen` for function-preserving architectural edits on a trained MNIST `FeedFwdNN`.

## 7. Combining with `nnx.finetune` for the "freeze old, train new" pattern

Surgery preserves trainability and modes. Every primitive that rebuilds an *existing* Linear — `widen` for the target layer and its downstream consumer, `low_rank_factorize` for both SVD factors — gives each replacement tensor the `requires_grad` flag of the tensor it replaces (the target's weight flag and bias flag independently; the downstream layer's own flags, never the target's; both factors inherit the source weight's flag while only `up.bias` inherits the source bias's flag) and each replacement module the train/eval mode of the module it replaces, so a mixed root/child configuration survives unchanged. Build the optimizer *after* surgery: a frozen layer stays frozen through widening or factorization and never enters `build_param_groups(..., strict=True)`; if you want the new factors to learn, `unfreeze` them explicitly first. Genuinely new layers inserted by `deepen` are ordinary fresh modules (trainable, train mode). These flags and modes live on the returned object only — `state_dict()` carries neither, so they are not a cross-process freeze policy.


`expand_embedding` returns a `frozen_mask` of bool shape `(new_num_embeddings,)` — `True` for rows that came from the original embedding, `False` for new rows. The mask is a hand-off to the caller's training step:

```python
from nnx import NNParamGroupSpec, expand_embedding

new_emb, frozen_mask = expand_embedding(model.net.embed, new_num_embeddings=20_000, init="copy_mean")
model.net.embed = new_emb  # reattach — expand_embedding returns a NEW module

# Register a gradient hook ONCE before training: hooks fire during
# backward(), i.e. before optimizer.step(), so frozen rows receive
# zero gradient. (Zeroing .grad AFTER default_train_step(ctx) would
# be too late — that helper already stepped the optimizer.)
keep = (~frozen_mask).unsqueeze(1)
new_emb.weight.register_hook(lambda g: g * keep.to(g.dtype))

# One more trap: Adam applies weight decay INSIDE step() — after the
# hook — so the frozen rows would still drift under the default
# weight_decay. Give the embedding a decay-free param group:
optim = NNOptimParams(
    name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5,
    param_groups=[NNParamGroupSpec(name_pattern="embed.*", weight_decay=0.0)],
)
from dataclasses import replace

model.train(params=replace(train_params, optim=optim))  # default supervised step works as-is
```

(Verified: with the reattach + hook + decay-free group, frozen-row
drift is exactly 0.0 over training while the new rows learn.)

`nnx.finetune.freeze` covers the simpler case of freezing entire parameter tensors via fnmatch globs; the `frozen_mask` covers the row-level case that `freeze` can't reach.
