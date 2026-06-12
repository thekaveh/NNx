# NNx vs Lightning / HF / fastai / Composer

An honest, scope-explicit comparison of NNx against the four closest PyTorch training/specialization toolkits, organized so users can pick the right tool for their actual need.

## 1. Quick decision matrix

| If you need... | Reach for |
|---|---|
| Distributed multi-GPU training (DDP / FSDP / DeepSpeed) | **Lightning** or **Accelerate** |
| Production-grade LM fine-tuning + Hub model zoo | **HF Transformers + PEFT + TRL** |
| Production-scale diffusion (SD, SDXL, ControlNet) | **HF diffusers** |
| Algorithmic-methods benchmarking (SAM / BlurPool / SqueezeExcite) | **MosaicML Composer** |
| Opinionated high-level API + tabular / vision / collab stacks | **fastai** |
| GNN training/checkpoint integration | **NNx** (PyG-backed but NNx is the only toolkit treating GNNs as first-class) |
| Single-package breadth (graph + LM + diffusion + PEFT + surgery in one install) | **NNx** |
| Content-addressed run reproducibility (`run.id` = md5 of config) | **NNx** |
| Model surgery (Net2Net widen/deepen, low-rank, drop, embedding expansion) | **NNx** (no mainstream alternative) |
| Tight notebook research loop on a single GPU | **NNx** or **fastai** |

## 2. Landscape map

| Competitor | Overlap axis with NNx | Where they're stronger | Where NNx is stronger |
|---|---|---|---|
| **PyTorch Lightning + Fabric** | Generic training-loop toolkit | Distributed (DDP / FSDP / DeepSpeed), accelerator/strategy abstraction, callback integrations ecosystem, `LightningCLI`, community scale | Functional `train_step_fn` hook (vs class-method override), single-package breadth, content-addressed runs, GNN as first-class, tight core |
| **HF Transformers + Accelerate + PEFT + TRL** | LM / PEFT / preference fine-tuning | Model zoo + HF Hub, distributed + DeepSpeed integration, 12+ PEFT methods (QLoRA / AdaLoRA / LoHA / OFT / VeRA), beam search + constrained gen, production-scale RLHF / PPO | Single-package install (vs four), no Hub-flow lock-in, cleaner training-loop API, graph + diffusion + surgery in the same package, lower entry mass |
| **fastai** | High-level opinionated training, notebook UX | Built-in tabular + vision + collab-filtering stacks, learn-rate finder, progressive resizing, nbdev integration, large teaching community | PyTorch-native (no fastai abstraction layer), graph + LM + diffusion + PEFT in one package, content-addressed runs |
| **MosaicML Composer** | Algorithmic training methods + efficient training | BlurPool, SAM, SqueezeExcite, MixUp variants, more algorithmic recipes; production-scale benchmarks; distributed/sharded | Single-GPU notebook UX, broader specialization (PEFT + surgery + GNN + embeddings + LM in one), no Mosaic-cloud coupling |

## 3. Capability-axis comparison

Each row: what NNx ships today, the credible competitor on that axis, and the scope difference. **No "NNx is better" claims — just what each tool covers.**

### 3.1. Training loop core

| Aspect | NNx | Lightning |
|---|---|---|
| Loop abstraction | `NNModel.train(params, train_step_fn=...)` — functional injection hook | `LightningModule.training_step(self, batch, batch_idx)` — class method override |
| Callback bus | `Callback.on_{train,epoch}_{begin,end}` — 4 hooks | `Callback.on_*` — ~30 hooks |
| Auto-resume | Content-addressed: `resume_from_run_id=run.id` + `resume_from_checkpoint="last"` | Manual checkpoint-by-epoch-number |
| Custom step | `train_step_fn=...` kwarg | Subclass override |

### 3.2. Distributed / scale

| Aspect | NNx | Lightning + Accelerate |
|---|---|---|
| DDP | Not shipped | Built-in |
| FSDP | Not shipped | Built-in |
| DeepSpeed | Not shipped | Integrated |
| `torch.compile` | Not shipped (deferred) | Per-strategy opt-in |

If you need any of these, NNx is the wrong tool today.

### 3.3. PEFT methods

| Method | NNx | HF PEFT |
|---|---|---|
| LoRA | Yes | Yes |
| DoRA | Yes | Yes |
| IA3 | Yes | Yes |
| Prefix-Tuning | Yes | Yes |
| Prompt-Tuning | Yes | Yes |
| Adapters | Yes | Yes |
| QLoRA (4-bit base) | Not shipped | Yes |
| AdaLoRA | Not shipped | Yes |
| LoHA / LoKr / OFT / BOFT / VeRA | Not shipped | Yes |
| `merge_lora` (bake adapter into base) | Not shipped | Yes |

### 3.4. LM / generation

| Aspect | NNx | HF `generate` |
|---|---|---|
| Greedy / top-k / top-p / temperature / repetition penalty | Yes | Yes |
| KV cache | Yes (default-on; typically ≥1.2× CPU @ 128 tokens, up to ≈1.9× on unloaded CPU) | Yes |
| Beam search | Not shipped | Yes |
| Contrastive search | Not shipped | Yes |
| Constrained generation (vocab / regex / grammar) | Not shipped | Yes |
| Streaming | Not shipped | Yes (`TextStreamer`) |

### 3.5. Diffusion

| Aspect | NNx | HF diffusers |
|---|---|---|
| DDPM training step + reverse sampler | Yes (toy) | Yes |
| Noise schedules | Linear / cosine | Many |
| Denoiser | `DiffusionMLP` only | UNet / DiT / etc. |
| Stable Diffusion / SDXL / ControlNet | Not shipped | Yes |

NNx's `nnx.diffusion` is teaching/research-scoped. For production, use HF diffusers.

### 3.6. GNN

| Aspect | NNx | PyG (raw) |
|---|---|---|
| GCN / GraphSAGE / GAT | Yes | Yes |
| HGT / GraphTransformer / RGCN | Not shipped (planned) | Yes |
| Training-loop integration | Yes (via `NNModel`) | Not shipped (manual loops) |
| `NeighborLoader` batching | Yes (via `NNGraphDataset`) | Yes |

NNx's GNN value is the training-loop + checkpoint integration on top of PyG's primitives.

### 3.7. Model surgery

| Aspect | NNx | Anything else |
|---|---|---|
| Net2Net widen / deepen | Yes | None |
| `drop_layer` | Yes | None |
| `low_rank_factorize` (SVD truncation) | Yes | None |
| `expand_embedding` | Yes | None |

No mainstream alternative — NNx's `nnx.surgery` is unique.

### 3.8. Observability

| Aspect | NNx | Lightning loggers |
|---|---|---|
| TensorBoard | Yes (basic) | Yes (rich) |
| Weights & Biases | Yes (basic) | Yes (rich) |
| MLflow / Comet / Neptune / Aim | Not shipped | Yes |
| Custom Logger API | Partial (Callback subclass) | Yes (Logger protocol) |

### 3.9. Hub / model sharing

| Aspect | NNx | HF Hub ecosystem |
|---|---|---|
| Publish to HF Hub | Yes (via `PyTorchModelHubMixin`) | Yes |
| Load from HF Hub | Yes | Yes |
| Discoverable NNx-tagged model zoo | Not shipped | Yes |

NNx publishes to the same Hub HF uses; there's no separate NNx model zoo.

### 3.10. Training-loop diagnostics

| Aspect | NNx | fastai | Lightning |
|---|---|---|---|
| LR finder | Yes (`nnx.lr_finder`, Smith 2017) | Yes (`Learner.lr_find`) | Not shipped (`tuner.lr_find` removed in 2.0) |
| Per-layer gradient norms | Yes (`nnx.viz.gradient_flow`, Plotly bar chart) | Hook-based recipes | `track_grad_norm` callback |
| `_repr_html_` for runs in Jupyter | Yes (`NNRun._repr_html_`) | Notebook-native | Not shipped |
| PEP 561 `py.typed` marker | Yes (PR #32) | Not shipped | Yes |

NNx's recently-shipped diagnostics close the most visible UX gap vs fastai's notebook ergonomics.

## 4. When to use what

**Use NNx when** any combination of these matters:
- You need graph neural networks alongside LM / diffusion / PEFT in the same project.
- Reproducibility via `run.id` content-addressing has organizational value.
- You want model-surgery primitives (Net2Net, low-rank).
- You're running on a single GPU and don't need distributed.
- You prefer a tight, hold-in-your-head core over a deep ecosystem.

**Use Lightning when** you need distributed training, accelerator strategy abstraction, or the deep callback-integrations ecosystem.

**Use HF Transformers + PEFT + TRL when** you're doing production-scale LM work, you want the Hub model zoo, or you need QLoRA / RLHF / DeepSpeed integration.

**Use fastai when** you want strongly-opinionated defaults and built-in tabular / vision / collab-filtering stacks.

**Use Composer when** you need production-scale algorithmic-method benchmarking (BlurPool, SAM, SqueezeExcite) with sharded distributed.

## 5. Scope explicit

This page documents NNx's *current* coverage as of `main`. The roadmap explicitly defers distributed training, `torch.compile` integration, Lightning-style strategy abstraction, and a Lightning-CLI equivalent. If you need any of those, NNx today is the wrong tool.

NNx's planned near-term additions (QLoRA, beam search, more GNN nets, SWA / EMA / SAM, `merge_lora`) close the most visible gaps in §3.3 / §3.4 / §3.6 but do not address distributed. `nnx.lr_finder` and `nnx.viz.gradient_flow` already shipped — see §3.10.
