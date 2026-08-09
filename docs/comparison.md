# 13. NNx vs Lightning / HF / fastai / Composer

An evidence-oriented, scope-explicit comparison of NNx against nearby PyTorch
training and specialization toolkits. Competitor behavior was last checked
against official documentation on **2026-08-08**; follow the linked sources
before making a version-sensitive platform decision.

## 1. Quick decision matrix

| If you need... | Reach for |
|---|---|
| Distributed multi-GPU training (DDP / FSDP / DeepSpeed) | **Lightning** or **Accelerate** |
| Production-grade LM fine-tuning + Hub model zoo | **HF Transformers + PEFT + TRL** |
| Production-scale diffusion (SD, SDXL, ControlNet) | **HF diffusers** |
| Algorithmic-methods benchmarking (SAM / BlurPool / SqueezeExcite) | **MosaicML Composer** |
| Opinionated high-level API + tabular / vision / collab stacks | **fastai** |
| GNN training/checkpoint integration | **NNx** when a PyG-backed model should share the same NNx training and checkpoint contracts |
| Single-package breadth (graph + LM + diffusion + PEFT + surgery in one install) | **NNx** |
| Content-addressed run reproducibility (`run.id` = md5 of config) | **NNx** |
| Model surgery (Net2Net widen/deepen, low-rank, drop, embedding expansion) | **NNx** when these operations should compose directly with the NNx training loop |
| Tight notebook research loop on a single GPU | **NNx** or **fastai** |

## 2. Landscape map

| Competitor | Overlap axis with NNx | Where they're stronger | Where NNx is stronger |
|---|---|---|---|
| [**PyTorch Lightning + Fabric**](https://lightning.ai/docs/pytorch/stable/) | Generic training-loop toolkit | Distributed strategies, accelerator abstraction, callback integrations, and `LightningCLI` | Functional `train_step_fn` hook, content-addressed runs, and NNx's combined specialization modules |
| [**HF Transformers + Accelerate + PEFT + TRL**](https://huggingface.co/docs) | LM / PEFT / preference fine-tuning | Model and dataset ecosystem, distributed integrations, broad PEFT coverage, generation strategies, and production-oriented preference tooling | One NNx package combines its smaller LM surface with graph, diffusion, surgery, and experiment persistence |
| [**fastai**](https://docs.fast.ai/) | High-level opinionated training and notebook UX | Built-in application stacks, learning-rate finder, data blocks, and teaching ecosystem | Direct NNx/PyTorch configuration, graph/LM/diffusion/PEFT modules, and content-addressed runs |
| [**MosaicML Composer**](https://docs.mosaicml.com/projects/composer/en/stable/) | Algorithmic training methods and efficient training | A larger algorithm catalog, distributed training, and benchmark-oriented workflows | A smaller notebook-oriented core plus NNx-specific PEFT, surgery, GNN, embeddings, and LM modules |

## 3. Capability-axis comparison

Each row: what NNx ships today, the credible competitor on that axis, and the scope difference. **No "NNx is better" claims — just what each tool covers.**

### 3.1. Training loop core

| Aspect | NNx | Lightning |
|---|---|---|
| Loop abstraction | `NNModel.train(params, train_step_fn=...)` — functional injection hook | `LightningModule.training_step(self, batch, batch_idx)` — class method override |
| Callback bus | `Callback.on_{train,epoch}_{begin,end}` — 4 hooks | `Callback.on_*` — ~30 hooks |
| Auto-resume | Content-addressed: `resume_from_run_id=run.id` + `resume_from_checkpoint="last"` | `Trainer.fit(..., ckpt_path=path_or_last)` restores full training state |
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
| KV cache | Yes (default-on; regression-tested at ≥1.2× CPU for the fixed 128-token workload) | Yes |
| Beam search | Not shipped | Yes |
| Contrastive search | Not shipped | Yes |
| Constrained generation (vocab / regex / grammar) | Not shipped | Yes |
| Streaming | Token-ID callback via `generate(on_token=...)` | Text-oriented streamer objects such as `TextStreamer` |

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
| HGT / GraphTransformer / RGCN | Not shipped | Yes |
| Training-loop integration | Yes (via `NNModel`) | User-owned loop around PyG modules/loaders |
| `NeighborLoader` batching | Yes (via `NNGraphDataset`) | Yes |

NNx's GNN value is the training-loop + checkpoint integration on top of PyG's primitives.

### 3.7. Model surgery

| Aspect | NNx | Broader ecosystem |
|---|---|---|
| Net2Net widen / deepen | Yes | Available in research implementations and focused libraries; APIs vary |
| `drop_layer` | Yes | Can be implemented directly against PyTorch modules; no single comparison target |
| `low_rank_factorize` (SVD truncation) | Yes | Available through PyTorch linear algebra and compression libraries |
| `expand_embedding` | Yes | Can be implemented directly or through model-specific ecosystem helpers |

NNx keeps these surgery operations in one namespace and makes their results immediately composable with `NNModel.train()`.

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
| LR finder | Yes (`nnx.lr_finder`, Smith 2017) | Yes (`Learner.lr_find`) | Yes (`Tuner.lr_find`) |
| Per-layer gradient norms | Yes (`nnx.viz.gradient_flow`, Plotly bar chart) | Hook-based recipes | `grad_norm` utility from `on_before_optimizer_step` |
| `_repr_html_` for runs in Jupyter | Yes (`NNRun._repr_html_`) | Notebook-native displays | Rich notebook/logging integrations; no `NNRun` equivalent |
| PEP 561 `py.typed` marker | Yes | Check the installed fastai distribution/version | Yes |

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

This page documents NNx's current coverage as of `main`. Distributed training,
`torch.compile` integration, Lightning-style strategy abstraction, and a CLI
equivalent are not shipped. If you need those capabilities, NNx today is the
wrong tool.

This page does not promise untracked roadmap work. Future capabilities should
appear here only after they ship or when they have a linked, approved public
issue.

## 6. Primary comparison sources

- [Lightning checkpoint resume](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
  [Lightning `Tuner.lr_find`](https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.tuner.tuning.Tuner.html),
  and [Lightning gradient inspection](https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html)
- [Hugging Face generation](https://huggingface.co/docs/transformers/main_classes/text_generation),
  [streamers](https://huggingface.co/docs/transformers/internal/generation_utils),
  [PEFT](https://huggingface.co/docs/peft), and [TRL](https://huggingface.co/docs/trl)
- [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
- [fastai documentation](https://docs.fast.ai/)
- [Composer documentation](https://docs.mosaicml.com/projects/composer/en/stable/)
