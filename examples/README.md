# 16. Examples

Runnable scripts demonstrating common NNx patterns. Each is self-contained — no external data dependencies. CPU is sufficient for everything in here.

## 1. Run

```bash
pip install thekaveh-nnx                # core (covers every example not listed below)
python examples/01_synthetic_classification.py
```

A handful of examples depend on optional extras — install them as needed:

```bash
pip install "thekaveh-nnx[onnx]"             # 04_onnx_export.py, 12_quantize_int8.py (Phase-5 export)
pip install "thekaveh-nnx[quantize]"         # 12_quantize_int8.py, 15_qat_classifier.py
pip install "thekaveh-nnx[onnx-dynamo]"      # 15_qat_classifier.py (Phase-6 dynamo export)
pip install "thekaveh-nnx[embeddings]"       # 13_train_domain_embedder.py
pip install "thekaveh-nnx[lm]"               # 11_tinystories_lm.py, 17_export_transformer_to_gguf.py, 18_export_ollama_bundle.py, 22_dpo_synthetic_preferences.py
pip install "thekaveh-nnx[gguf-write]"       # 17_export_transformer_to_gguf.py, 18_export_ollama_bundle.py
pip install "thekaveh-nnx[viz]"              # 21_viz_attribute_xai.py
```

Working from a git checkout instead of PyPI? See [CONTRIBUTING.md §1](../CONTRIBUTING.md#1-getting-set-up) for the editable + dev install.

## 2. Catalog

Ordered from foundational to most specialized. Each numbered prefix on the filename matches the order below.

### 2.1. Core training loop

| Example | What it demonstrates |
|---|---|
| `01_synthetic_classification.py` | Train a feed-forward classifier on random data; `EarlyStopping`, `LRMonitor`; load BEST checkpoint and predict — with `predict()` and with the opt-in `predict_proba(X, ProbabilitySpec(kind="categorical", class_axis=1, labels=...))`, asserting equal logits/classes, probability rows summing to 1, sample ids aligned with the input rows (also through a batched loader), an identical `VisUtils.confusion_matrix` from `class_indices` + `spec.labels`, and that Bernoulli indicators refuse `class_indices` (run end to end by `tests/test_examples_smoke.py`, CPU, temporary working directory). Its bounded `native_nll_workflow()` helper (executed by `tests/test_examples_smoke.py -k native_nll_workflow`, CPU, temporary working directory) trains one epoch with `Losses.NEGATIVE_LOG_LIKELIHOOD`, reloads BEST and checks the reported loss against an explicit log-softmax/NLL reference while `predict().logits` stays raw. |
| `02_resume_training.py` | Warm-resume training from a prior run's LAST checkpoint with optimizer, scheduler, scaler, epoch, and RNG state preserved. Its setup helpers (`_make_model_and_loader`, `_base_optim`, `_base_sched`) are also reused by `tests/test_atomic_write_text.py::test_resume_sample_text_save_failure`, which injects a text-save fault into a resumed run and checks that the source run stays loadable and resumable with no temporary file left behind. Its bounded `checkpoint_probe_first_fit()` helper (executed by `tests/test_examples_smoke.py -k checkpoint_probe_first_fit`, CPU, temporary working directory) probes missing training/optimizer/combined state for the prospective run, asserts no run directory was reserved, then fits once with overwrite disabled and reloads LAST. Its bounded `overwrite_best_recovery()` helper (`-k overwrite_best_recovery`, temporary root only) commits two runs, overwrites the winner with a worse result and verifies `runs/best` re-elects the survivor, whose BEST loads for inference. Its bounded `iterable_graph_resume()` helper (`-k iterable_graph_resume`; the graph branch needs `thekaveh-nnx[graph]` and is skipped without it) resumes from a reusable list of batches and from an in-memory five-node full-batch graph, checking resumed epoch numbers and seed-only evaluation. Its bounded `amp_resume_compatibility()` helper (`-k amp_resume_compatibility`) fits and resumes with `mixed_precision=True` on CPU — where AMP is a documented no-op, so the training-state sidecar records `scaler=None` — and, only when a CUDA device is present, repeats the cycle with AMP enabled and checks the restored `torch.amp.GradScaler` scale / growth tracker; without CUDA that branch is reported as skipped, never faked. Its bounded `callback_continuation()` helper (`-k callback_continuation`) splits a run with validation losses `[0.6, 0.4, 0.41, 0.42]` after the third epoch and shows the resumed `EarlyStopping` stopping at the same epoch as the uninterrupted run, reports `run.resume_status`, and warm-starts a second session with `resume_mode="weights_only"`. |
| `builder_branching.py` | Branching mutable params builders (FEAT-041): `copy()` splits a partial `NNOptimParams.builder()` base (clipping + a head param group) into AdamW and SGD variants, `from_params()` rebuilds an equal builder from a built value, and a shared named-`Trainer` base branches into two sibling CPU runs with distinct `data_id`s (one adds a StepLR scheduler) that train one update each, get distinct run ids and reload with `NNRun.load`; the shared `DataLoader` keeps its identity. Also round-trips an `NNTransformerParams` with per-layer `activations` / `dropout_probs` through `NNTransformerParamsBuilder.from_params` and `from_state`. Offline, CPU. Unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `builder_branching_workflow()` (`-k builder_branching`, temporary working directory). |
| `graph_optional_splits.py` | Full-graph (`sampler="full"`) node classification on a six-node in-memory PyG graph whose validation mask is empty: `NNGraphDataset` publishes `val_loader=None` (resolved `batch_sizes=(3, 0, 3)`, state `"0"`), `NNModel.train` skips validation (an `eval_step_fn` spy proves it), `val_edp=None` persists through `NNRun.load`, LAST is committed, `EarlyStopping` monitors `train_edp.loss`, and the test split is evaluated on its seed rows only. Synthetic data — no download, no neighbor sampling (no pyg-lib / torch-sparse); `torch_geometric` is a core dependency. Unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `graph_optional_splits_workflow()` (`-k graph_optional_splits`, temporary working directory). |
| `optimizer_factories.py` | Registered optimizer factories and AdamW (FEAT-013): `register_optimizer_factory` + `NNOptimFactoryParams(OptimizerFactorySpec(id, version, config))` give a custom SGD factory the `layers.0.*` group while a built-in `Optims.ADAMW` (decoupled weight decay) owns the disjoint `layers.1.*` group in one CPU `Trainer` update; asserts the factory is called once with one resolved group and that both groups moved, then unregisters it and reloads the run metadata offline with `NNRun.load`. Unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `optimizer_factories_workflow()` (`-k optimizer_factories`, temporary working directory). |
| `regression_task.py` | Task adapters (FEAT-002): `NNModelParams(task=TaskSpec.regression(2))` makes the default step, `evaluate()` and `predict()` regression-aware — a two-target fit with one NaN (masked) target, per-epoch validation, records with valid-target counts, `mse` / `mae` and no fabricated classification fields, run and checkpoint reload carrying the task, `predict()` returning continuous values and `predict_proba()` returning `probabilities=None`; plus a multilabel batch where subset accuracy (0.5) differs from element accuracy (0.75). Offline, CPU. Unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `regression_task_workflow()` (`-k regression_task`) and an end-to-end subprocess run. |
| `tabular_validation.py` | Tabular data admission (FIX-023): `NNTabularDataset` rejects `±inf` in a selected feature in source precision before an int64/bool cast (which would absorb it into an INT64 extreme or `True`) and finite values outside an integer dtype's range (`300` → `int8` would wrap), rejects finite values that overflow `float16` during conversion for both features and a floating `target_dtype` target, ignores non-finite values in unselected columns, and leaves the DataFrame and global RNG untouched on rejection; a valid frame then drives one classification step whose loss matches a hand-computed cross-entropy reference, and a regression dataset keeps `(N, 1)` targets. Offline, CPU. Unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `tabular_validation_workflow()` (`-k tabular_validation`, temporary working directory). |
| `03_custom_metrics.py` | Plug a custom `metric_fn(Y, Y_hat)` into `NNTrainParams.extra_metrics`; inspect `idp.train_edp.extra` and `idp.val_edp.extra`. Its bounded `named_monitor_workflow()` helper (also run by `main`; `tests/test_examples_smoke.py -k named_monitor_workflow`, CPU, temporary working directory) declares `nll` / `brier` / `accuracy` metrics and a validation-NLL `MonitorSpec` beside a decoded callable on uneven batches (`[16, 16, 16, 2]` training, `[8, 8, 5]` validation), shares the monitor with `EarlyStopping`, reloads the run and checks the reported value against an independent recomputation from the BEST model and that BEST is the last epoch the monitor marked as improved. |
| `04_onnx_export.py` | Export a trained model to ONNX, validate via `onnx.checker`. |
| `25_conv_classifier.py` | LeNet-style conv classifier via `NNConvParams` + `Nets.CONV`: conv-stack arithmetic helpers (`spatial_sizes()`/`flatten_dim()`), per-layer FC `activations`/`dropout_probs` overrides, image-vs-flat input equivalence, and a checkpoint round-trip through `resolve_from_state`. Bounded helper `conv_integral_schema_roundtrip()` (no training) builds the params from NumPy-integer shapes, checks they normalize to plain-`int` YAML-portable state with an identical rebuilt forward, and shows fractional/boolean counts rejected before allocation. Synthetic stripes/checkerboard imagery — no download. |
| `26_custom_eval_step.py` | Train a non-classification paradigm end-to-end: a regression `train_step_fn` (without a task the default step's argmax metrics do not fit continuous targets; plain regression can instead declare `TaskSpec.regression`, see `regression_task.py`) paired with `eval_step_fn(EvalStepContext) -> NNEvaluationDataPoint` — a custom MSE/MAE val pass whose metrics persist per-epoch in the run history (`idp.val_edp`, MAE riding in `extra`). Its bounded `nonfinite_metric_workflow()` helper (executed by `tests/test_examples_smoke.py -k nonfinite_metric_workflow`, CPU, temporary working directory) scripts NaN/inf/absent validation metrics and asserts the finite fallback: plateau inputs, the selected BEST, raw diagnostics, and CSV NaN→`None` readback. |

### 2.2. `train_step_fn` hook

| Example | What it demonstrates |
|---|---|
| `05_custom_train_step_autoencoder.py` | Use `train_step_fn` to replace the supervised step with a reconstruction-loss step (tiny linear autoencoder). |

### 2.3. Fine-tuning

| Example | What it demonstrates |
|---|---|
| `06_finetune_with_layer_freezing.py` | Transfer learning: pretrain on distribution A, export weights, load into a fresh model, `freeze("layers.0.*", "layers.1.*")`, fine-tune the head on distribution B. |
| `07_lora_finetuning.py` | Parameter-efficient fine-tuning via LoRA: `apply_lora_to(net, "layers.*", r=4, alpha=8)`, fine-tune on a new distribution, verify every base parameter is bit-exactly unchanged, save a LoRA-only checkpoint and compare its size to the full state-dict. Its bounded `dora_zero_row_composition(dtype=torch.float16)` helper (executed by `tests/test_examples_smoke.py -k dora_zero_row_composition` in FP16 and FP32, CPU) wraps a two-layer net with mixed zero/nonzero base rows via `apply_dora_to`, converts it to half after wrapping, checks a finite forward through the following projection, one optimizer step on a float32 loss reduction, the frozen base, and a full `state_dict()` reload including `magnitude`. Its bounded `peft_alias_preflight()` helper (`-k peft_alias_preflight`, CPU, offline) registers one `Linear` under two names beside an independent head, shows `apply_lora_to` rejecting both the wildcard and the second alias (naming both) with every weight, identity and `requires_grad` flag unchanged, then wraps only the head and takes one optimizer step that trains exactly its adapter. Its bounded `peft_eval_injection()` helper (`-k peft_eval_injection`, CPU, offline, temporary directory) puts the classifier in `eval()` *before* injecting nonzero-dropout LoRA adapters and loading trained adapter weights, then asserts bit-identical repeated `predict` output, an unchanged all-eval mode map and an adapter file with no mode field, and that a later `train()` re-activates the adapter dropout. Its bounded `peft_preconverted_base()` helper (`-k peft_preconverted_base`, CPU) converts a tiny base to float64 *before* `apply_lora_to`, takes one adapter-only SGD step with no second `.to()`, and asserts adapter placement and base immutability. Its bounded `lora_artifact_roundtrip()` helper (`-k lora_artifact_roundtrip`, CPU, temporary directory) wraps a net whose first layer is *named* `lora_A_projection`, inspects the exact keys `save_lora_weights` writes (adapter-owned tensors only, no `*.base.*`), reloads into a matching wrapped base and checks load count, output parity and base identity — an adapter file carries only the trainable delta, not a resumable model checkpoint. |

### 2.4. Alternative paradigms

| Example | What it demonstrates |
|---|---|
| `08_diffusion_2d_mixture.py` | DDPM-style diffusion on a 2D mixture of 4 Gaussians: `NoiseSchedulers.LINEAR` + `DiffusionMLP` + `diffusion_train_step_factory` + reverse-diffusion `sample()`. |
| `09_gan_with_trainer.py` | Multi-optimizer training via `nnx.trainer.Trainer` — a tiny GAN on a 1D mixture of Gaussians, with disjoint optimizers for `G` and `D` scoped via `NNParamGroupSpec`. Its bounded `trainer_builder_snapshot()` helper (executed by `tests/test_examples_smoke.py -k trainer_builder_snapshot`, CPU, temporary working directory) builds a G/D config, mutates the builder afterwards, runs the captured config for two minibatches and checks optimizer names, scoped updates and the saved descriptor. Its bounded `two_optimizer_resume()` helper (`-k two_optimizer_resume`) trains the GAN for one epoch, resumes it for one more in a freshly initialised model under a different seed through `NNTrainerParamsBuilder.resume_from`, and checks that the G / D weights match an uninterrupted two-epoch run, that `run.resume_status` lists the restored `EarlyStopping`, and that the final LAST checkpoint carries both named optimizers, both schedulers and the component; `gan_step` still owns every optimizer update. |
| `10_knowledge_distillation.py` | Hinton-style KD: pretrain a wider teacher, then distill into a much smaller student (~4% of the teacher's parameters) via `kd_train_step_factory`. Verifies the teacher's weights are frozen across the student's training. |
| `14_moe_classifier.py` | Sparse top-k Mixture-of-Experts as a first-class model type: `NNMoEParams(num_experts=4, top_k=2)` + `Nets.FEED_FWD_MOE` builds a `FeedFwdMoENN` (every hidden layer an `MoELinear`), trained via `moe_train_step_factory` (supervised loss + Switch-style load-balancing aux). Reports the param-count breakdown, verifies the aux loss decreases as routing balances out, and round-trips the MoE params through a checkpoint. |

### 2.5. Quantization

| Example | What it demonstrates |
|---|---|
| `12_quantize_int8.py` | Post-training quantization (PTQ): train a feed-forward classifier, call `nnx.quantize.quantize_int8(model)` once, verify val accuracy is preserved and the quantized model still ONNX-exports. No calibration data, no retraining. Requires `pip install "thekaveh-nnx[quantize,onnx]"`. Its bounded `quantized_generative_subtype()` helper (executed by `tests/test_examples_smoke.py -k quantized_generative_subtype`; needs `[quantize,lm]`, skipped otherwise; no ONNX) quantizes a one-layer `GenerativeNNModel` with a locally trained BPE tokenizer and generates two tokens on the cached and full-recompute paths — `quantize_int8` returns the same class it was given. |
| `15_qat_classifier.py` | Quantization-aware training (QAT 8da4w via torchao): combine `qat_train_step_factory` and `QATLifecycleCallback` to fake-quant during training, then real-quant on convert. Verifies the saved LAST checkpoint holds the CONVERTED int4 state (scales/zeros on disk) and round-trips it into a fresh prepare→convert net. Requires `pip install "thekaveh-nnx[quantize,onnx-dynamo]"`. |

### 2.6. Embeddings + FAISS export

| Example | What it demonstrates |
|---|---|
| `13_train_domain_embedder.py` | Train a tiny text embedder from scratch on synthetic `(sentence, paraphrase)` pairs via NT-Xent contrastive loss, embed a corpus, export to a FAISS index, query the index for self-similarity. End-to-end demo of `nnx.embeddings.train_contrastive` + `export_to_faiss`. Requires `pip install "thekaveh-nnx[embeddings]"`. |

### 2.7. Language modeling

| Example | What it demonstrates |
|---|---|
| `11_tinystories_lm.py` | Decoder-only LM end-to-end: train a tiny BPE tokenizer, build a `TransformerNN`, train next-token prediction via a custom `train_step_fn`, then sample with `GenerativeNNModel.generate()` (KV-cache enabled by default). CPU-friendly (uses an inline corpus by default; pass `--use-hf` to download TinyStories). Requires `pip install "thekaveh-nnx[lm]"`. Its bounded `manual_attention_lm_example(dtype=torch.bfloat16)` helper (executed by `tests/test_examples_smoke.py -k manual_attention_lm_example` in BF16 and FP32, CPU; skipped without `[lm]`) runs one attention-dropout training update on a one-layer `d_model=16` model with a four-token local batch and greedily generates two tokens on the cached and full paths. Its `build_lm_train_params(lr, ...)` helper exposes the CLI configuration, and the bounded `lm_config_rejects_nonfinite()` check (`-k lm_config_rejects_nonfinite`, no tokenizer/training/downloads) shows a NaN learning rate rejected with a `ValueError` naming `max_lr` while the finite control keeps its state. |

### 2.8. Self-supervised pretraining

| Example | What it demonstrates |
|---|---|
| `16_ijepa_image_plumbing.py` | I-JEPA image-path plumbing on synthetic images by default (`--cifar` opts into CIFAR-10): a small `ViTNN` context encoder predicts masked-patch latents against an EMA target encoder. Demonstrates `jepa_train_step_factory` + `JEPAPredictor` + `build_target_encoder` + `random_block_mask`. The smoke-tested `float64_vit_predictor_step()` helper runs one bounded double-precision forward/backward on tiny synthetic images (no download) and checks every gradient is finite FP64. |

### 2.9. Experimental GGUF export

| Example | What it demonstrates |
|---|---|
| `17_export_transformer_to_gguf.py` | Build a tiny `TransformerNN` + BPE tokenizer, write an NNx-tagged `.gguf`, and inspect it with `gguf.GGUFReader`. Includes the official llama.cpp source-build path for `llama-quantize`. Stock llama.cpp-derived runtimes do not implement the NNx architecture. Requires `pip install "thekaveh-nnx[gguf-write,lm]"`. |
| `18_export_ollama_bundle.py` | Generate `model.gguf` + a Modelfile (`FROM` / `PARAMETER` / `SYSTEM` / `TEMPLATE`) as an experimental bundle fixture. Stock Ollama cannot run `nnx_transformer`; use only with a compatible patched runtime. Requires `pip install "thekaveh-nnx[gguf-write,lm]"`. |

### 2.10. Pruning + surgery

| Example | What it demonstrates |
|---|---|
| `19_prune_synthetic_classifier.py` | Magnitude prune a small synthetic-data classifier at 50% sparsity (`bake=True` keeps state_dict keys intact), evaluate the pruned accuracy, then briefly fine-tune to recover. Demonstrates `nnx.prune.magnitude_prune`. |
| `20_low_rank_surgery_ffn.py` | Train a wide FFN, low-rank-factorize the widest Linear at rank=8 via `nnx.surgery.low_rank_factorize`, then refine to recover accuracy. Shows the caller is responsible for swapping the returned `nn.Sequential` back into the `ModuleList`. Its bounded `surgery_freeze_roles()` helper (executed by `tests/test_examples_smoke.py -k surgery_freeze_roles`, CPU, no files) factorizes a weight-frozen / bias-trainable layer, builds fresh strict parameter groups and takes one update, asserting the factors stay frozen and unchanged while only the intended bias and the untouched layer move. Its bounded `widen_supported_workflow()` helper (`-k widen_supported_workflow`, CPU, temporary working directory) shows eval-mode parity through ReLU, the rejection of a Softmax boundary with the source untouched, and the honest reconstruction path: widen → `dataclasses.replace` the params → fresh `NNModel` → one training epoch → BEST reload. Its bounded `deepen_override_workflow()` helper (`-k deepen_override_workflow`, CPU, temporary working directory) deepens a `FeedFwdNN` with explicit mixed dropout overrides, checks that the returned `params` describe the deeper topology (aligned overrides, zero dropout at the new site) with eval parity, rebuilds a fresh `NNModel` from `deeper.params`, fits one epoch and reloads BEST. Its bounded `named_deepen_workflow()` helper (`-k named_deepen_workflow`, CPU, no files) deepens a named `nn.Sequential` at the dotted key `relu`, checks the preserved original keys plus the generated `_nnx_deepen_*` keys, head-path lookup, `freeze` / parameter-group selection and a state-dict round trip. |

### 2.11. Explainability

| Example | What it demonstrates |
|---|---|
| `21_viz_attribute_xai.py` | Captum-backed input attribution via `nnx.viz.attribute(method=...)` — runs `integrated_gradients`, `saliency`, `input_x_gradient`, and `deep_lift` on a trained classifier. Requires `pip install "thekaveh-nnx[viz]"`. |

### 2.12. LM follow-ons

| Example | What it demonstrates |
|---|---|
| `22_dpo_synthetic_preferences.py` | DPO preference fine-tuning of a tiny `TransformerNN` against synthetic `(prompt, chosen, rejected)` triples using `dpo_train_step_factory`; reference policy frozen via `copy.deepcopy`. `build_preference_dataset()` wraps the `NNPreferenceDataset` construction; the bounded `dpo_sample_batch_sizes()` helper (offline counting stub tokenizer, no `lm` extra, no training) shows `batch_sizes`: `None` = one full-split batch, positive = explicit mini-batch, zero rejected before tokenization. Requires `pip install "thekaveh-nnx[lm]"` for `main()`. |

### 2.13. Distillation variants

| Example | What it demonstrates |
|---|---|
| `23_born_again_distillation.py` | Iterated self-distillation across G=3 generations via `born_again_train`; each generation distills from the previous via Hinton-style KD. Demonstrates the Furlanello et al. ICML 2018 result that successive generations often match or outperform the original. |
| `24_feature_kd.py` | FitNets-style feature distillation via `feature_kd_train_step_factory` with one paired teacher→student auxiliary layer (shape-matched: teacher `layers.1` output 32 → student `layers.0` output 32). |
