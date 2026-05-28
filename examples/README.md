# Examples

Runnable scripts demonstrating common NNx patterns. Each is self-contained — no external data dependencies. CPU is sufficient for everything in here.

## 1. Run

```bash
pip install -e ".[dev]"        # if you haven't already
python examples/01_synthetic_classification.py
```

## 2. Catalog

Ordered from foundational to most specialized. Each numbered prefix on the filename matches the order below.

### 2.1. Core training loop

| Example | What it demonstrates |
|---|---|
| `01_synthetic_classification.py` | Train a feed-forward classifier on random data; `EarlyStopping`, `LRMonitor`; load BEST checkpoint and predict. |
| `02_resume_training.py` | Warm-resume training from a prior run's LAST checkpoint with optimizer state preserved. |
| `03_custom_metrics.py` | Plug a custom `metric_fn(Y, Y_hat)` into `NNTrainParams.extra_metrics`; inspect `idp.train_edp.extra` and `idp.val_edp.extra`. |
| `04_onnx_export.py` | Export a trained model to ONNX, validate via `onnx.checker`. |

### 2.2. `train_step_fn` hook

| Example | What it demonstrates |
|---|---|
| `05_custom_train_step_autoencoder.py` | Use `train_step_fn` to replace the supervised step with a reconstruction-loss step (tiny linear autoencoder). |

### 2.3. Fine-tuning

| Example | What it demonstrates |
|---|---|
| `06_finetune_with_layer_freezing.py` | Transfer learning: pretrain on distribution A, export weights, load into a fresh model, `freeze("layers.0.*", "layers.1.*")`, fine-tune the head on distribution B. |
| `07_lora_finetuning.py` | Parameter-efficient fine-tuning via LoRA: `apply_lora_to(net, "layers.*", r=4, alpha=8)`, fine-tune on a new distribution, verify every base parameter is bit-exactly unchanged, save a LoRA-only checkpoint and compare its size to the full state-dict. |

### 2.4. Alternative paradigms

| Example | What it demonstrates |
|---|---|
| `08_diffusion_2d_mixture.py` | DDPM-style diffusion on a 2D mixture of 4 Gaussians: `NoiseSchedulers.LINEAR` + `DiffusionMLP` + `diffusion_train_step_factory` + reverse-diffusion `sample()`. |
| `09_gan_with_trainer.py` | Multi-optimizer training via `nnx.trainer.Trainer` — a tiny GAN on a 1D mixture of Gaussians, with disjoint optimizers for `G` and `D` scoped via `NNParamGroupSpec`. |
| `10_knowledge_distillation.py` | Hinton-style KD: pretrain a wider teacher, then distill into a much smaller student (~4% of the teacher's parameters) via `kd_train_step_factory`. Verifies the teacher's weights are frozen across the student's training. |
| `14_moe_classifier.py` | Sparse top-k Mixture-of-Experts: a feed-forward classifier whose hidden layer is an `MoELinear(num_experts=4, top_k=2)` instead of `nn.Linear`. Trained via `moe_train_step_factory` (supervised loss + Switch-style load-balancing aux). Reports the param-count breakdown and verifies the aux loss decreases as routing balances out. |

### 2.5. Embeddings + FAISS export

| Example | What it demonstrates |
|---|---|
| `13_train_domain_embedder.py` | Train a tiny text embedder from scratch on synthetic `(sentence, paraphrase)` pairs via NT-Xent contrastive loss, embed a corpus, export to a FAISS index, query the index for self-similarity. End-to-end demo of `nnx.embeddings.train_contrastive` + `export_to_faiss`. Requires `pip install "nnx[embeddings]"`. |
| `11_tinystories_lm.py` | Decoder-only LM end-to-end: train a tiny BPE tokenizer, build a `TransformerNN`, train next-token prediction via a custom `train_step_fn`, then sample with `GenerativeNNModel.generate()`. CPU-friendly (uses an inline corpus by default; pass `--use-hf` to download TinyStories). |
