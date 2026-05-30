# Examples

Runnable scripts demonstrating common NNx patterns. Each is self-contained — no external data dependencies. CPU is sufficient for everything in here.

## 1. Run

```bash
pip install nnx                # core (covers examples 01–06)
python examples/01_synthetic_classification.py
```

A handful of examples depend on optional extras — install them as needed:

```bash
pip install "nnx[onnx]"             # 04_onnx_export.py
pip install "nnx[quantize]"         # 12_quantize_int8.py, 15_qat_classifier.py
pip install "nnx[embeddings]"       # 13_train_domain_embedder.py
pip install "nnx[lm]"               # 11_tinystories_lm.py, 17_export_transformer_to_gguf.py, 22_dpo_tinystories.py
pip install "nnx[gguf-write]"       # 17_export_transformer_to_gguf.py, 18_publish_to_ollama.py
pip install "nnx[viz]"              # 21_viz_attribute_xai.py
```

Working from a git checkout instead of PyPI? See [CONTRIBUTING.md §1](../CONTRIBUTING.md#1-getting-set-up) for the editable + dev install.

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

### 2.5. Quantization

| Example | What it demonstrates |
|---|---|
| `12_quantize_int8.py` | Post-training quantization (PTQ): train a feed-forward classifier, call `nnx.quantize.quantize_int8(model)` once, verify val accuracy is preserved and the quantized model still ONNX-exports. No calibration data, no retraining. Requires `pip install "nnx[quantize]"`. |
| `15_qat_classifier.py` | Quantization-aware training (QAT 8da4w via torchao): combine `qat_train_step_factory` and `QATLifecycleCallback` to fake-quant during training, then real-quant on convert. Reports the pre-/post-convert accuracy delta. Requires `pip install "nnx[quantize]"`. |

### 2.6. Embeddings + FAISS export

| Example | What it demonstrates |
|---|---|
| `13_train_domain_embedder.py` | Train a tiny text embedder from scratch on synthetic `(sentence, paraphrase)` pairs via NT-Xent contrastive loss, embed a corpus, export to a FAISS index, query the index for self-similarity. End-to-end demo of `nnx.embeddings.train_contrastive` + `export_to_faiss`. Requires `pip install "nnx[embeddings]"`. |

### 2.7. Language modeling

| Example | What it demonstrates |
|---|---|
| `11_tinystories_lm.py` | Decoder-only LM end-to-end: train a tiny BPE tokenizer, build a `TransformerNN`, train next-token prediction via a custom `train_step_fn`, then sample with `GenerativeNNModel.generate()` (KV-cache enabled by default). CPU-friendly (uses an inline corpus by default; pass `--use-hf` to download TinyStories). Requires `pip install "nnx[lm]"`. |

### 2.8. Self-supervised pretraining

| Example | What it demonstrates |
|---|---|
| `16_ijepa_cifar10.py` | I-JEPA on CIFAR-10: a small `ViTNN` context encoder predicts the latent of masked patches against an EMA target encoder. Demonstrates `jepa_train_step_factory` + `JEPAPredictor` + `build_target_encoder` + `update_ema` + `random_block_mask`. No pixel reconstruction, no strong augmentations. |

### 2.9. GGUF + Ollama export

| Example | What it demonstrates |
|---|---|
| `17_export_transformer_to_gguf.py` | Build a tiny `TransformerNN` + BPE tokenizer, write a `.gguf` via `nnx.interop.write_gguf`, round-trip via `gguf.GGUFReader`. Includes the shell-out recipe for sub-F16 quantization (`llama-quantize`). Requires `pip install "nnx[gguf-write]"`. |
| `18_publish_to_ollama.py` | Bundle `model.gguf` + a generated `Modelfile` (`FROM` / `PARAMETER` / `SYSTEM` / `TEMPLATE`) so `ollama create -f Modelfile` registers the model locally. Requires `pip install "nnx[gguf-write]"`. |

### 2.10. Pruning, surgery, quantization

| Example | What it demonstrates |
|---|---|
| `19_prune_mnist.py` | Magnitude prune a small classifier at 50% sparsity (`bake=True` keeps state_dict keys intact), evaluate the pruned accuracy, then briefly fine-tune to recover. Demonstrates `nnx.prune.magnitude_prune`. |
| `20_surgery_resnet.py` | Train a wide FFN, low-rank-factorize the widest Linear at rank=8 via `nnx.surgery.low_rank_factorize`, then refine to recover accuracy. Shows the caller is responsible for swapping the returned `nn.Sequential` back into the `ModuleList`. |

### 2.11. Explainability

| Example | What it demonstrates |
|---|---|
| `21_viz_attribute_xai.py` | Captum-backed input attribution via `nnx.viz.attribute(method=...)` — runs `integrated_gradients`, `saliency`, `input_x_gradient`, and `deep_lift` on a trained classifier. Requires `pip install "nnx[viz]"`. |

### 2.12. LM follow-ons

| Example | What it demonstrates |
|---|---|
| `22_dpo_tinystories.py` | DPO preference fine-tuning of a tiny `TransformerNN` against synthetic `(prompt, chosen, rejected)` triples using `dpo_train_step_factory`; reference policy frozen via `copy.deepcopy`. Requires `pip install "nnx[lm]"`. |

### 2.13. Distillation variants

| Example | What it demonstrates |
|---|---|
| `23_born_again_distillation.py` | Iterated self-distillation across G=3 generations via `born_again_train`; each generation distills from the previous via Hinton-style KD. Demonstrates the Furlanello et al. ICML 2018 result that successive generations often match or outperform the original. |
| `24_feature_kd.py` | FitNets-style feature distillation via `feature_kd_train_step_factory` with one paired teacher→student auxiliary layer (shape-matched: teacher `layers.1` output 32 → student `layers.0` output 32). |
