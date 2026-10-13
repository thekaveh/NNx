# 4. Model surgery — `nnx.surgery`

The `nnx.surgery` subpackage ships five primitives that take a trained `nn.Module` and return a fresh module with a structural change applied. Four are **function-preserving** in some sense:

- `widen` and `deepen` preserve the forward output (before any training step) within their supported sites (§5).
- `low_rank_factorize` is function-preserving at max rank and approximate below.
- `expand_embedding` preserves the forward output on original token IDs.

Only `drop_layer` is purely chain-preserving. It changes the function the network computes; the surged module is meant to be refined via `NNModel.train()` to recover quality. See the table in §1 for the per-primitive breakdown.

This is the unique compositional payoff of pairing surgery primitives with a training loop in the same toolkit. Every primitive returns a fresh module, and that module is a drop-in target for `NNModel.train()`. `expand_embedding` returns the new `nn.Embedding` with a mask of the original rows.

## 1. The five primitives at a glance

| Primitive | Op | Function-preserving? | Returns |
|---|---|---|---|
| `widen(model, *, layer_name, new_width, rng_seed=0)` | Net2WiderNet — grow `out_features`, rescale the consumer's incoming columns | yes — for a Linear inside `nn.Sequential` or `FeedFwdNN.layers` whose path to the next Linear holds only elementwise ops (activations, `Dropout`, `Identity`). Parity is in eval mode when dropout > 0. Softmax / LayerNorm / BatchNorm boundaries, other containers and aliased modules are rejected. | fresh `nn.Module` |
| `deepen(model, *, after_layer_name)` | Net2DeeperNet — identity-init Linear after a ReLU | yes (ReLU only) | fresh `nn.Module` |
| `drop_layer(model, *, layer_name, importance=None)` | Replace named layer with `nn.Identity` | no — chain-preserving only | fresh `nn.Module` |
| `low_rank_factorize(linear, *, rank, method='svd')` | SVD truncation: Linear → `Sequential(Linear, Linear)` | yes at max rank, approximate below | fresh `nn.Sequential` |
| `expand_embedding(emb, *, new_num_embeddings, init=...)` | Resize Embedding; preserve original rows | yes on original token IDs | `(nn.Embedding, frozen_mask)` |

All primitives accept keyword-only arguments after the first positional and operate on a deep copy of the input so the caller's reference survives.

## 2. End-to-end: widen → refine → save

The canonical surgery workflow has four steps. Load a trained checkpoint and apply a function-preserving edit. Hand the surged net to `NNModel.train()` for a brief refinement pass. Then save the result.

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
model.net.eval()  # compare in eval mode: dropout would break parity
new_net.eval()
with torch.no_grad():
    assert torch.allclose(model.net(x), new_net(x), atol=1e-5)

# 3. Rewire NNModel around the wider net: build a NEW, correctly
#    described model and load the surged state_dict. On a FeedFwdNN,
#    `widen` already returns aligned params (the new width, every
#    per-layer override kept), so adopt them. Never assign the widened
#    net beneath the old `net_params`: ordinary checkpoint
#    reconstruction reads the descriptor, and a stale one no longer
#    describes the network.
refined = NNModel(
    net_params=new_net.params,
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

The same template works for `widen` and `deepen`: only the surgery line changes, and both return aligned `params` on a `FeedFwdNN`. A low-rank factorization cannot be described by `NNParams`, so record it as a recipe (§8). `expand_embedding` needs a module with an embedding (§7). `drop_layer` on a `FeedFwdNN` leaves its layer keys and `params.hidden_dims` as they were, so `NNParams` cannot describe the result either.

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

A *named* `nn.Sequential` is built from an `OrderedDict`, at the root or nested below another module. In one, address the ReLU by its dotted key — `deepen(model, after_layer_name="body.relu")`. Then:

- Every original key is preserved in order.
- The two new modules are registered as `_nnx_deepen_linear_N` / `_nnx_deepen_relu_N` right after the site. `N` increments until both keys are unused, so repeated deepening never collides.
- A module registered in several slots keeps every slot pointing at one copied object.

A purely numeric container keeps its positional contract: `deeper[2]` is the new Linear and `deeper[3]` the new ReLU. It is rebuilt by position, so the modules after the site shift by two (`"2"` becomes `"4"`). In a named container, retained keys (numeric ones included) are not renumbered, and every surgery primitive addresses a registered key exactly. So there a later `widen(deeper, layer_name="1")` targets the module named `"1"` even when it no longer sits at position 1.

On a `FeedFwdNN`, `deeper.params.hidden_dims` gains the matching entry. Any `activations` / `dropout_probs` overrides stay aligned (dropout `0.0` at the new site).

### 3.3. `low_rank_factorize` on `nn.Linear(in=64, out=32)`

| Form | Parameter count |
|---|---|
| Original `nn.Linear(64, 32)` | `32·64 + 32 = 2080` |
| Factored at `rank=8` | `8·64 + 32·8 + 32 = 800` — **delta −1280 (≈ 61% reduction)** |
| Factored at `rank=16` | `16·64 + 32·16 + 32 = 1568` — **delta −512 (≈ 25% reduction)** |
| Factored at `rank=32` (= max, exact) | `32·64 + 32·32 + 32 = 3104` — **delta +1024** (factored form is bigger past breakeven) |

The breakeven rank below which factoring saves parameters is `k* = (out·in) / (out + in)`. For a 64×32 Linear that's `32·64 / 96 ≈ 21`. At ranks below `k*`, factoring strictly reduces parameter count. At higher ranks the two-Linear sandwich actually carries more parameters than the original. The rank-truncated weight still fits the original lower-rank structure.

### 3.4. `drop_layer` and `expand_embedding`

`drop_layer` replaces a submodule with `nn.Identity`, so parameter count drops by the entire dropped layer. For example, dropping a square `Linear(d, d)` removes `d² + d` parameters.

`expand_embedding` grows row count from `old_num` to `new_num`. Parameter count grows by `(new_num − old_num) · embedding_dim` — exactly the new-row weights, regardless of `init`.

## 4. The function-preservation contract

Every test in `tests/test_surgery_*.py` for a function-preserving primitive compares the surged module's output with the original's on the same input. The tolerance is `atol=1e-5` (and `rtol=1e-5` where it uses `torch.testing.assert_close`).

A change to `widen`, `deepen`, or `low_rank_factorize` (at max rank) can produce an output difference above `1e-5`. If it does, the surgery is broken — **do not relax the tolerance**. The whole point of these primitives is that the post-surgery model is *immediately* a good starting point for training. An accuracy cliff at step 0 defeats the construction.

## 5. When function-preservation doesn't hold

- `widen` infers the consumer only where the execution order is proven:
    - A target inside an `nn.Sequential` (walking the following siblings).
    - A target inside `FeedFwdNN.layers` (resolving the *effective* `activation_for(i)` / `dropout_for(i)`, not the scalar defaults).
- Every op between the `widen` target and the next Linear must be elementwise: the built-in activations except Softmax, `Dropout`, `Identity`. Duplicating units through a width-dependent op (Softmax's denominator, LayerNorm / BatchNorm statistics) changes the output.
- `widen` raises a `ValueError` naming the boundary *before* anything is allocated. The source model, its module identities and the RNG are untouched. It refuses:
    - width-dependent ops;
    - other containers (registration order is not data flow), and a container subclass with its own forward;
    - hooks on the container, or on the path from the layer to its consumer;
    - missing consumers;
    - aliases (modules registered under more than one path).
- With dropout configured, the `widen` parity promise is for eval mode. A stochastic training forward is not identical by construction.
- `deepen` rejects any activation other than ReLU: a `ValueError` at a `FeedFwdNN` site, a `TypeError` for a non-ReLU target in an `nn.Sequential`. The ReLU must be fed by an `nn.Linear` through elementwise or norm modules only; any other module in between is refused with a `ValueError`. The identity-Linear trick only function-preserves through ReLU. For sigmoid / tanh / GELU networks, structurally similar insertions silently produce a drifted forward output.
- `deepen` rebuilds an `nn.Sequential` as a plain `nn.Sequential` of the same children. It therefore refuses a subclass with its own `forward`, and a container with its own parameters, buffers or hooks. These would be dropped. It also refuses a target ReLU with hooks, and a `ModuleList` that is not the `layers` / `fcs` of a stock `FeedFwdNN` / `ConvNN`.
- On a `FeedFwdNN`, the `deepen` check reads the activation actually applied at the insertion site — `params.activation_for(i)`. So a per-layer override wins over the scalar default in both directions.
- On a `FeedFwdNN`, `deepen` returns a network with updated immutable `params`:
    - One more `hidden_dims` entry at the site.
    - A ReLU inserted into an explicit `activations` list.
    - The new identity site's dropout set to `0.0`, while every existing site keeps its own probability. A nonzero scalar dropout is materialized into a per-layer list rather than switched off.
- On a `FeedFwdNN`, `widen` likewise returns `params` whose `hidden_dims` record the new width, with every per-layer override kept.
- Adopt `deeper.params` (or the widened net's `params`) as the `net_params` of a *fresh* `NNModel` when you rebuild for refinement. An outer model still holding the old descriptor cannot reconstruct the deeper checkpoint.
- `drop_layer` is never function-preserving, with one degenerate exception. The exception is a dropped layer that was already the identity on its inputs (e.g. a ReLU fed strictly positive activations). The function is chain-preserving: dotted-name lookup, downstream shapes, and the forward pass still work.
- `low_rank_factorize` at `rank < min(out, in)` is an *approximation*. The Frobenius error of the truncation is bounded by the L2 norm of the discarded singular values (Eckart-Young). That bound is asserted as a regression test in `tests/test_surgery_low_rank.py`.
- `expand_embedding` preserves the original rows exactly, so any token ID `< old_num` is unchanged. But it introduces new rows that *must* be initialized. Pick `init="zeros"` for a safe default, or `init="copy_mean"` when you want the new rows to warm-start near the existing manifold.

## 6. See it in practice

Worked end-to-end in [ml-eng-lab](https://github.com/thekaveh/ml-eng-lab):

- [model_surgery-mnist-ffnn-pytorch](https://github.com/thekaveh/ml-eng-lab/blob/main/notebooks/model_surgery-mnist-ffnn-pytorch/notebook.ipynb) — applies `nnx.surgery.widen` and `nnx.surgery.deepen` for function-preserving architectural edits on a trained MNIST `FeedFwdNN`.

## 7. Combining with `nnx.finetune` for the "freeze old, train new" pattern

Surgery preserves trainability and modes. Two primitives rebuild an *existing* Linear: `widen` for the target layer and its downstream consumer, and `low_rank_factorize` for both SVD factors. Each gives every replacement tensor the `requires_grad` flag of the tensor it replaces:

- The target's weight flag and bias flag are carried independently.
- The downstream layer keeps its own flags, never the target's.
- Both factors inherit the source weight's flag, while only the second factor's bias (`1.bias` in the returned `nn.Sequential`) inherits the source bias's flag.

Each replacement module also takes the train/eval mode of the module it replaces, so a mixed root/child configuration survives unchanged.

`widen`, `low_rank_factorize` and `expand_embedding` refuse a module with a `weight_norm` / `spectral_norm` parametrization, which a rebuilt plain module would drop silently. Remove it first with `torch.nn.utils.parametrize.remove_parametrizations`.

Build the optimizer *after* surgery. A frozen layer stays frozen through widening or factorization and never enters `build_param_groups(..., strict=True)`. If you want the new factors to learn, `unfreeze` them explicitly first. Genuinely new layers inserted by `deepen` are ordinary fresh modules: trainable, and in the mode of the container they join. Existing layers keep their own modes. `expand_embedding` gives the new table the source table's `requires_grad` flag and mode.

These flags and modes live on the returned object only. `state_dict()` carries neither, so they are not a cross-process freeze policy.


`expand_embedding` returns a `frozen_mask` of bool shape `(new_num_embeddings,)` — `True` for rows that came from the original embedding, `False` for new rows. The mask is a hand-off to the caller's training step.

The example uses a custom module `net` with an `embed` attribute; no built-in net has one. As in §2, a fresh `NNModel` wraps the changed module, so its runs and checkpoints describe the expanded topology:

```python
from nnx import NNModel, NNOptimParams, NNParamGroupSpec, Optims, expand_embedding

new_emb, frozen_mask = expand_embedding(net.embed, new_num_embeddings=20_000, init="copy_mean")
net.embed = new_emb  # reattach — expand_embedding returns a NEW module
model = NNModel(module=net, params=model_params)  # a fresh model over the expanded net

# Register a gradient hook ONCE before training: hooks fire during
# backward(), i.e. before optimizer.step(), so frozen rows receive
# zero gradient. (Zeroing .grad AFTER default_train_step(ctx) would
# be too late — that helper already stepped the optimizer.)
keep = (~frozen_mask).unsqueeze(1)
new_emb.weight.register_hook(lambda g: g * keep.to(device=g.device, dtype=g.dtype))

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

With the reattach, the hook and the decay-free group together, the
frozen rows do not change during training while the new rows learn.

`nnx.finetune.freeze` covers the simpler case of freezing entire parameter tensors via fnmatch globs; the `frozen_mask` covers the row-level case that `freeze` can't reach.

## 8. Recorded recipes: surgery that checkpoints can rebuild

The primitives above edit a module and record nothing, so a model they changed cannot be rebuilt from its descriptor. `NNModel.train` refuses such a net before any run is reserved:

- unrecorded low-rank surgery;
- unbaked prune masks;
- `weight_norm` / `spectral_norm` parametrizations.

Remove masks with `torch.nn.utils.prune.remove` and parametrizations with `torch.nn.utils.parametrize.remove_parametrizations` (`remove_weight_norm` / `remove_spectral_norm` for the legacy hooks). Alternatively, train the net as `NNModel(module=...)`. NNx can replay two topology changes: LoRA on explicit `nn.Linear` targets and SVD low-rank replacement. For these, use a **recipe** (`nnx.transforms`):

```python
from nnx.optimizers import build_optimizer
from nnx.transforms import TransformRecipe, check_optimizer, lora, low_rank

recipe = TransformRecipe(
    [lora("layers.0", r=4, alpha=8.0), low_rank("layers.1", rank=4)],
    materialization="in_place",  # the default; "fresh" transforms a new, randomly initialized base instead
)
recipe.validate(model)          # mutates nothing; one RecipeError names every problem
recipe.materialize(model)       # transactional (model and random streams); the operations are recorded
optimizer = build_optimizer(model.net, optim_params)  # AFTER the recipe
check_optimizer(model, optimizer)
```

- **What is recorded.** Each operation's `id` (`"lora"` / `"low_rank"`), `version`, explicit `targets` and `config`, in order.
    - Targets are dotted path strings under `model.net`, any module name included. Glob characters `*?[]`, empty segments and non-string targets are refused.
    - The recipe copies its operations: changing the list they came from changes nothing. Its configs are read-only.
    - Operations and recipes hash, compare and pickle by value.
- **Which models.** A recipe needs a model NNx can rebuild: built-in `Nets` or a registered `ModelSpec`.
    - A runtime-only `module=` is refused, because nothing could replay the recipe on it. Use `nnx.peft` / `nnx.surgery` directly there.
    - A target that was an uninitialized lazy layer (`nn.LazyLinear`) when the model was built is also refused, since a rebuild starts lazy again.
- **Materialization.** `"in_place"` (the default) transforms the given model and keeps its trained weights. `"fresh"` builds a new plain `NNModel` from the descriptor — randomly initialized, keeping the batch adapter — and transforms that, leaving the given model untouched. A `"fresh"` recipe is validated against that base, not the given model.
- **Validation.** Before anything changes, `validate` / `materialize` reject each of these, named by operation index and target:
    - a missing target, or a target that is not exactly an `nn.Linear`;
    - a target registered under more than one name;
    - two targets of one operation that are the same or nested;
    - a target inside a subtree another operation (or an earlier recipe) transforms;
    - a rank above `min(in, out)`;
    - an unknown operation or version.

    They also reject any optimizer passed in `optimizers=` that already holds a target's parameters. The error instructs you to build the optimizer after the recipe.

- **What `validate` checks.** `validate` checks exactly what `materialize` would (the fresh base for `"fresh"`). In place, it also refuses a model that already differs from its descriptor plus its recorded recipe, since no checkpoint could rebuild it. This includes unrecorded surgery anywhere, such as an added, resized or bias-changed layer.
- **Optimizer check.** `check_optimizer(model, optimizer)` refuses an optimizer that is stale for the current topology, whenever it was built. A stale optimizer holds either of these:
    - parameters a low-rank operation replaced (or the parameters of the model a `"fresh"` materialization started from);
    - a LoRA target's frozen base weights but none of its adapter, so it would never train.

    It accepts an optimizer built afterwards over the trainable parameters. This covers every one, a subset, parameters outside `model.net` (a learnable loss temperature, say), or BitFit-style over unfrozen base tensors.

- **What carries the recipe.** The recorded operations travel with every run checkpoint (LAST, BEST, the phase tags and `ModelCheckpoint` snapshots). They also travel with `NNCheckpoint.to_file` in both formats, `save_pretrained`, and run bundles. `nnx.bundles.export_bundle` refuses any other recorded transform (a callback's own train-end operation, say), naming its index and id, because a bundle is rebuilt from data alone.
- **Reconstruction.** `NNModel.from_checkpoint` and `from_pretrained` build a fresh base and replay the operations. The replay puts LoRA wrappers around the fresh layers and allocates low-rank factors in their recorded two-Linear shape. Only then do they load the saved tensors. SVD is never rerun, and the ordered keys, shapes, trainability and predictions match the saved model. An unknown operation or version, or malformed recorded options, is refused before any tensor is loaded, naming the transform's index.
- **Training.** A recipe model trains through `NNModel.train` / `Trainer.train`, custom `train_step_fn` included.
    - Before any run is reserved, both check the live topology. It must be exactly the base plus its recipe: tensor names and shapes, and each target's module and configuration.
    - So surgery outside the recipe is refused. This includes a recipe target that unrecorded surgery added, because a rebuild from the base could not find it.
    - The recipe is part of the run id (runs without one keep theirs) and of `ExperimentManifest.for_model`.
- **Resume.** A resume, from any tag or snapshot, needs the resuming model to have materialized the same recipe — ids, versions, targets and configs. It says so otherwise. `ExperimentPlan` builds untransformed models and cannot express a recipe, so resume a recipe run through `NNModel.train` or `Trainer.train`.
- **Checkpoints during training.** The final `LAST` stores the weights once: a recipe is the live topology, not a train-end transform. A train-end transform such as a QAT conversion still keeps the pre-transform state for resuming.
    - A recipe is materialized before training. Before each checkpoint a run writes, and again after `on_train_end`, NNx refuses recipe operations a callback applied or declared during training.
    - On a recipe model, it also refuses unrecorded surgery at those points, since no checkpoint could rebuild or resume them.
    - The run's earlier checkpoints are kept.
- **What cannot rebuild a recipe.** A raw `export_state_dict()` file or an adapter-only `save_lora_weights` export carries no recipe.
    - Loading transformed weights without one through `NNModel.from_checkpoint` or a strict `from_pretrained` fails with an error saying so.
    - A non-strict load (`from_pretrained(strict=False)`, `nnx.finetune.load_pretrained`) keeps its partial-load contract and cannot tell.
    - Materialize the recipe on the model first and load into it, or keep a checkpoint / `save_pretrained` artifact.

The manual refinement loop in [`examples/20_low_rank_surgery_ffn.py`](https://github.com/thekaveh/NNx/blob/main/examples/20_low_rank_surgery_ffn.py) stays (unrecorded surgery); its `recipe_reconstruction_workflow()` helper shows the recorded path end to end.

