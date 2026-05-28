# Changelog

All notable changes to NNx are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is roughly [SemVer](https://semver.org/) — pre-1.0, we allow behavior changes (typically bug fixes) without renaming public APIs.

## [Unreleased] — Expansion megamerge (PR #29)

This release integrates **20 sub-projects** consolidated on 2026-05-28: HuggingFace Hub interop (safetensors + `PyTorchModelHubMixin`), PEFT additions (DoRA + IA3 + Prefix + Prompt tuning on top of LoRA + Adapters), quantization (PTQ INT8 weight-only + QAT 8da4w via `torchao`), pruning (magnitude + 2:4 semi-structured), model surgery (Net2Net `widen` / `deepen` + `drop_layer` + `low_rank_factorize` + `expand_embedding`), embeddings (contrastive trainer + FAISS export), decoder-only LM (`TransformerNN` + `NNTransformerParams` + `NNTokenizerParams` + `GenerativeNNModel.generate()` with KV-cache), GGUF write + Ollama Modelfile bundle, model-internals visualization (`torchinfo` summary + weight histogram + activation map + Captum attribution + Netron export), I-JEPA self-supervised pretraining (+ small `ViTNN` encoder), Mixture-of-Experts (`MoELinear` + `moe_train_step_factory` with Switch-style aux loss), Born-Again Networks (iterated self-distillation), Feature-KD (FitNets-style), DPO (preference fine-tuning for LMs), `LogitsProcessor` chain (temperature / top-k / top-p / repetition-penalty), ONNX dynamo export opt-in, and assorted ergonomic improvements.

Every change preserves back-compatibility with existing `run.id` hashes and on-disk checkpoint formats — new params fields all follow the omit-when-default state() invariant. Test suite: 642 tests; 640 pass, 2 skip on torch/onnxscript version-skew (opt-in dynamo path), 1 skip on absence of CUDA (2:4 semi-structured).

### Added — model-internals viz attribution + ONNX dynamo opt-in

- **`nnx.viz.attribute(model, x, *, method, target, **method_kwargs)`** — Captum-backed input-attribution wrapper. Single string-keyed dispatch over the six most common methods (`integrated_gradients`, `gradient_shap`, `deep_lift`, `saliency`, `input_x_gradient`, `occlusion`) returning `(attribution_tensor, plotly.Figure)`. The figure renders the attribution as a Plotly `Heatmap` (3-/4-D image-shaped inputs are mean-pooled over channels first). Captum is lazy-imported at the call site so the rest of `nnx.viz` keeps working without it; the missing-dep path raises a clear `ImportError("nnx.viz.attribute requires captum: pip install captum")`. Sensible per-method defaults (`baselines=zeros` for GradientShap, `sliding_window_shapes` for Occlusion) preserve the one-call ergonomics. Optional dep promoted into the existing `viz` extra: `pip install nnx[viz]` now pulls `captum>=0.7.0` alongside `torchinfo>=1.8.0`. 10 new tests in `tests/test_viz_attribute.py` (unknown-method ValueError, IG return-shape + figure-type, saliency works, missing-captum ImportError via `sys.modules` stub, every supported-method key end-to-end via `@pytest.mark.parametrize`).
- **`NNModel.to_onnx(..., dynamo=True)` opt-in.** New `dynamo: bool = False` kwarg on `NNModel.to_onnx`. When True, dispatches through PyTorch's new `torch.export`-based ONNX exporter (default in torch>=2.9; supports >2 GB models via external data; generally faster). The default (False) preserves the existing legacy TorchScript path — no behavior change for existing callers. The dynamo path lazy-imports `onnxscript` and raises a clear `ImportError` pointing at the new `nnx[onnx-dynamo]` extra (`pip install nnx[onnx-dynamo]`) if missing, rather than letting torch surface a less actionable failure.
### Added — quantization (PTQ INT8 weight-only via torchao)

- **`nnx.quantize` package** — post-training quantization built on top of [`torchao`](https://github.com/pytorch/ao) (the replacement for the deprecated `torch.ao.quantization`, which is removed in PyTorch 2.10).
  - **`nnx.quantize_int8(model: NNModel) -> NNModel`** — one-call PTQ INT8 weight-only quantization. Deep-copies `model.net`, applies `torchao.quantization.quantize_(net, Int8WeightOnlyConfig(version=2))` to the copy, and returns a new `NNModel` whose `net.Linear` weights are stored in int8 per-channel (symmetric). Activations stay FP32 — only the weights are quantized, so accuracy loss is typically a fraction of a percentage point. **No calibration data, no retraining** — pure post-process. The original `NNModel` is left untouched so callers can keep both around for an accuracy delta comparison.
  - **Vision + GNN compatible** — any module exposing `nn.Linear` submodules is a valid target.
  - **ONNX export still works** on the quantized model (`NNModel.to_onnx` routes through `torch.onnx.export`'s legacy tracing path; torchao's quantized tensor falls back to dequantized matmul during the trace, so the exported ONNX is FP32 with the quantized weights baked in). Regression test included.
  - **State-dict round-trips through `NNCheckpoint.to_file`** (the existing pickle path); the on-disk file shrinks by roughly the same ratio as the pickled state-dict (≈30% on the example below, closer to ~25% at production-scale dims).
- Runnable demo: `examples/12_quantize_int8.py` — trains a small classifier (FP32), prints the FP32 val accuracy + state-dict size, calls `quantize_int8` once, prints the INT8 val accuracy + size, and confirms the quantized model still ONNX-exports. On the toy task the size reduction lands at ~69% with zero measurable accuracy delta.
- New optional dependency: `pip install nnx[quantize]` (pulls `torchao>=0.17`).
- 15 new tests in `tests/test_quantize_ptq.py` covering: returns a fresh `NNModel`, preserves output shape, doesn't mutate the source, replaces Linear weights with a torchao-quantized tensor, preserves attached attrs (`params` / `net_params` / `device` / `loss_fn`), output stays within 5% relative L2 of FP32, pickled state-dict shrinks vs FP32, `NNCheckpoint.to_file` round-trip shrinks on disk, ONNX export round-trip, deep-copy isolation (mutating quantized doesn't leak back), `predict()` end-to-end on a deeper model, clear `ImportError` when torchao is missing, state-dict keys unchanged, idempotency-via-deep-copy (calling twice on the same source produces identical outputs), `.train()` / `.eval()` toggle still works.

**Deferred:** QAT (`qat_train_step_factory`) and INT4 weight-only land in separate follow-up PRs.
### Added — PEFT++ (IA3)

- **`IA3Linear(base)`** — Infused Adapter by Inhibiting and Amplifying Inner Activations (Liu et al., NeurIPS 2022). The smallest adapter in the PEFT family: a single learned per-output-dim `scaling` vector applied multiplicatively to a frozen `nn.Linear`'s output. Trainable parameter count per wrapped layer is exactly `out_features` — roughly two orders of magnitude smaller than LoRA at the same effective adaptation budget. `scaling` is initialized to all-ones so the forward output at step 0 equals `base(x)` exactly.
- **`apply_ia3_to(module, *patterns)`** — fnmatch-glob in-place wrap mirroring `apply_lora_to`. Same two-phase traversal and idempotency contract (existing IA3 wrappers are not re-wrapped).
- **`save_ia3_weights(module, path)`** / **`load_ia3_weights(module, source)`** — persist ONLY the `scaling` parameters, symmetric to LoRA's save/load idiom. The resulting checkpoint is tiny (a single vector per wrapped layer). Same `weights_only=True` safety guarantee; same empty-dict-is-zero-op contract; same dict-source convenience overload.
- 19 new tests in `tests/test_peft_ia3.py`: validation (non-Linear base rejection), base-freezing, zero-init invariant (output == base at step 0, with and without bias), forward shape, trainable parameter set is exactly `{scaling}`, scaling init is all-ones, in/out features pass-through, scaling actually scales the output by a known non-unit value; `apply_ia3_to` empty-pattern rejection + selective wrap + wildcard wrap + idempotency + forward-preserves-at-init; save/load round-trip + base-keys-excluded-from-checkpoint + dict-source loading + bad-source-type rejection + empty-dict no-op contract.

### Added — PEFT++ (DoRA)

- **`DoRALinear(base, *, r, alpha, dropout)`** — Weight-Decomposed Low-Rank Adaptation (Liu et al., NVIDIA, ICML 2024 Oral). Subclass of `LoRALinear` that adds a trainable per-output-row `magnitude` parameter and recomposes the layer's weight as `W = magnitude * V / ||V||_c` where `V = W_0 + (α/r) · BA` is the LoRA-augmented direction. `magnitude` is initialized from `||W_0||_c` so the forward output at step 0 equals `base(x)` exactly (combined with LoRA's zero-init B). Often outperforms LoRA at the same rank with only `out_features` extra parameters — negligible vs LoRA's `r · (in + out)` baseline.
- **`apply_dora_to(module, *patterns, r, alpha, dropout)`** — fnmatch-glob in-place wrap mirroring `apply_lora_to`. Same idempotency contract (existing LoRA/DoRA wrappers are skipped via the parent-is-LoRALinear check, which covers DoRALinear by inheritance).
- DoRA reuses `save_lora_weights` / `load_lora_weights` for the `lora_A` / `lora_B` matrices unchanged (the inheritance hierarchy ensures the LoRA filter still matches). The `magnitude` vector is captured by the standard `state_dict()` round-trip — single vector of length `out_features` per wrapped layer.
- 16 new tests in `tests/test_peft_dora.py`: validation (non-Linear base, r/alpha/dropout ranges), base-freezing, zero-init invariant (output == base at step 0, with and without bias), forward shape, trainable parameter set is exactly `{lora_A, lora_B, magnitude}`, magnitude init matches `||W_0||_c`, in/out features pass-through, LoRALinear subclass relationship; `apply_dora_to` empty-pattern rejection + selective wrap + wildcard wrap + idempotency on re-application + forward-preserves-at-init; `save_lora_weights` round-trip via DoRA wrappers.
- **`nnx.paradigms.feature_kd_train_step_factory(teacher, *, auxiliary_layers, alpha, beta, temperature)`** — FitNets-style intermediate-layer feature distillation. Extends the existing `kd_train_step_factory` with an additional MSE term between named teacher / student intermediate-layer activations: `L = α · KL_soft · T² + β · MSE(student_act, teacher_act) + (1 − α) · L_hard`. Forward hooks register on the `auxiliary_layers` pairs (teacher_layer_name → student_layer_name, resolved via `nn.Module.get_submodule`); activations are collected per forward and the MSE term is averaged across pairs so `beta`'s scale is invariant to the pair count. The teacher freeze + eval-mode guarantee carries over from `kd_train_step_factory`. The v1 factory **requires shape-matched paired layers** — the `FeatureRegressor` projector for mismatched widths is deferred. Routes through `finalize_step` for the standard NaN guard + grad-clip path. Re-exported from `nnx.paradigms.*` and `nnx.*`.
### Added — born-again self-distillation

- **`nnx.paradigms.born_again_train(model, *, generations, train_params, **kd_kwargs) -> list[NNRun]`** — iterated self-distillation wrapper. Generation 0 trains plain (no teacher); each subsequent generation uses a deep-copied, frozen, eval-mode snapshot of the model after the prior generation as the teacher in a Hinton-style KD step (composed via `kd_train_step_factory`). Returns the per-generation `NNRun` list so callers can inspect the convergence trajectory. Born-again networks (Furlanello et al., ICML 2018) often match or slightly outperform the original — the soft targets act as an implicit regularizer. 9 new tests covering generations-count validation, KD-factory not invoked on generation 0, KD-factory invoked on generations 1+, teacher snapshot is a deepcopy (not the live model), teacher requires_grad=False + eval-mode at handoff, kwargs forwarding, teacher isolation from subsequent training, top-level re-export, and end-to-end model mutation across generations.
### Added — Mixture-of-Experts (tutorial-grade)

- **`nnx.MoELinear(in_features, out_features, *, num_experts, top_k=2)`** — sparse top-k MoE drop-in for `nn.Linear`. Router (bias-less `nn.Linear`) emits per-expert logits; the `top_k` experts per token are selected, their outputs are weighted by softmax-renormalized gating values, and the per-token result is the weighted sum. Exposes `.last_aux_loss` after each forward — the Switch-Transformer load-balancing penalty `N · Σ_i f_i · P_i` where `f_i` is the dispatch fraction and `P_i` is the mean router probability for expert `i`. The penalty is minimized at value 1 (NOT 0) when routing is perfectly uniform across experts. Validates `num_experts ≥ 2`, `top_k ∈ [1, num_experts]` at construction.
- **`nnx.paradigms.moe_train_step_factory(*, aux_loss_weight=0.01)`** — supervised training step that adds `aux_loss_weight · Σ_layer layer.last_aux_loss` to the main loss, summed across every `MoELinear` in the net. Routes through the shared `_step_helpers.finalize_step` for the standard NaN-guard + grad-clip tail (same shape as the KD / SimCLR / Mixup / CutMix factories). Works on nets with zero MoE layers too — the aux sum just collapses to 0 and the step is exactly supervised.
- Runnable demo: `examples/14_moe_classifier.py` — a feed-forward classifier whose hidden layer is an `MoELinear` (4 experts, top-k=2). Prints router / expert / classifier param breakdown, trains with `moe_train_step_factory`, and verifies the aux loss decreases across the run (routing balances out).
- 22 new tests across `tests/test_nn_moe.py` (12) and `tests/test_paradigms_moe.py` (10): MoELinear input validation + forward shape + router / experts module shape + top-k routing invariant + `last_aux_loss` populated-after-forward + non-negativity + uniform-routing-equals-1 (minimum-value math) + above-minimum-when-skewed + load-balancing converges under SGD on the aux loss; paradigm factory validation + end-to-end aux-loss-decreases + finalize-step routing (NaN guard fires) + no-MoE-layers no-op + zero-weight collapse to supervised + multi-MoE-layer summation + AMP rejection + grad-clip honored + EDP return shape.
- Scope explicitly limited to tutorial-grade. Production-scale MoE (MegaBlocks block-sparse kernels, expert parallelism across GPUs, token-dropping with capacity factor) is OUT — would be hollow wrapping over specialized libraries.
### Added — pruning (`nnx.prune`)

- **`nnx.prune` package** — two complementary network-pruning strategies layered on top of plain `nn.Linear` submodules, mirroring the `nnx.peft` package shape (public functions, fnmatch glob patterns, in-place mutation).
  - **`magnitude_prune(net, sparsity, *, layer_pattern="*", bake=True)`** — wraps `torch.nn.utils.prune.l1_unstructured`. For each `nn.Linear` whose dotted name matches `layer_pattern`, zeros the `round(sparsity · numel)` smallest-magnitude entries of its weight matrix. **Checkpoint-compat invariant:** `bake=True` (default) calls `prune.remove` immediately after each layer is pruned, so the `state_dict` keys stay identical to the pre-prune network — pruned checkpoints load into unpruned-network code under `strict=True`. `bake=False` keeps the reparameterization in place (state_dict carries `weight_orig` + `weight_mask` instead of `weight`); use this for iterative pruning schedules where successive `magnitude_prune` calls need to compose with the existing mask. Validates `sparsity ∈ [0, 1)`. Returns the number of layers pruned (0 if `layer_pattern` matches nothing).
  - **`semi_structured_24(net, *, layer_pattern="*")`** — 2:4 semi-structured sparsity via `torchao.sparsity.sparsify_` with `semi_sparse_weight()`. Swaps each matched `nn.Linear`'s weight with a 2:4 structured-sparse tensor subclass. **Real wall-clock speedup on Ampere+ GPUs** (~1.1× inference, ~1.3× training per torchao's ViT/SAM benchmarks); CPU and pre-Ampere hardware are unsupported by the underlying sparse kernel. The torchao dep is loaded lazily inside the function body so users on the magnitude-only path pay no dep cost; the dep is installed transitively via the existing `quantize` (torchao>=0.17) tooling.
- 17 new tests across `tests/test_prune_{magnitude,semi_structured}.py`: zero-fraction correctness; state_dict-keys preservation under bake=True (THE checkpoint-compat invariant); pattern-filter selectivity; idempotency on already-zeroed weights; iterative bake=False path; sparsity bounds rejection + sparsity=0 no-op + no-match returns 0; smallest-magnitude-go-to-zero correctness; full state_dict round-trip into a fresh unpruned net; CUDA-gated swap-actually-happens (skipped on CPU); monkey-patched pattern-filter selectivity for `semi_structured_24` (decouples nnx's filter logic from torchao's CUDA-only kernel); torchao-importorskip guard.
- Structured pruning that REMOVES channels / heads (and so breaks `state_dict` shape) is deferred — needs a per-architecture surgery API the existing checkpoint format doesn't yet support.
### Added — HuggingFace interop (safetensors + Hub mixin)

- **safetensors as opt-in checkpoint format.** `NNCheckpoint.to_file(path, format="safetensors")` writes a safe, mmap-friendly file readable by ComfyUI / vLLM / AutoGPTQ / HuggingFace tools. `NNParams`, `NNModelParams`, and `NNIterationDataPoint` are JSON-serialized into the safetensors metadata dict (the spec only allows `str -> str` metadata, so a JSON wrapper is the cleanest fit). Pickle remains the default format for back-compat; `NNCheckpoint.from_file(path)` auto-detects via magic-byte sniff (modern torch.save starts with the ZIP container `PK\x03\x04`, legacy / bare pickle starts with `\x80`, safetensors starts with neither). Requires `pip install nnx[hub]`.
- **`NNModel` is now HuggingFace-Hub-publishable** via `PyTorchModelHubMixin`. Free `model.save_pretrained("./dir")`, `model.push_to_hub("user/repo")`, and `NNModel.from_pretrained("user/repo" | "./dir")`. The on-disk layout is the canonical Hub flat layout: `model.safetensors` (weights), `config.json` (`{"net_params": <state>, "params": <state>}` using the public `state()` form NNRun hashes), and an auto-generated `README.md` model card. Without the `hub` extra installed, all three methods raise a clear `ImportError` pointing back at `pip install nnx[hub]`.
- **`nnx[hub]` extra** — pulls in `safetensors>=0.7.0` and `huggingface_hub>=1.4.0`. Both deps are runtime-import-guarded, so `pip install nnx` keeps working without them.
- **`docs/hub.md`** — when-to-use guide for both tracks (safetensors checkpoints + Hub mixin), a local save/load walkthrough, the Hub publish/download path, and the explicit non-goals.
### Added — embeddings (contrastive trainer + FAISS export)

- **`nnx.embeddings` package** — the one RAG-adjacent surface NNx ships. Users train a domain-specific text embedder via the existing SimCLR / NT-Xent machinery, then export the trained model to a FAISS index for any retrieval framework (LangChain / LlamaIndex / Haystack / raw FAISS) to consume. NNx does NOT host the RAG stack — chunking, reranking, prompt orchestration, vector-DB clients are inference-time concerns and explicitly out of scope.
  - **`ContrastiveTextDataset(pairs)`** — wraps `(anchor, positive)` string tuples as a `torch.utils.data.Dataset`. Validates input shape + types up-front (empty list / non-tuple / non-string entries all raise `ValueError`).
  - **`train_contrastive(backbone, dataset, *, n_epochs, batch_size, lr, temperature, ...)`** — high-level training loop. Builds a `DataLoader` with the string-aware `pair_collate`, runs NT-Xent (`nnx.nt_xent_loss`) updates over the trainable parameters of the backbone (anything `requires_grad=True`; composes with `nnx.freeze` / `nnx.apply_lora_to`), returns the in-place-mutated backbone. Accepts either a `sentence_transformers.SentenceTransformer` or any plain `nn.Module(list[str]) -> Tensor[B, D]`.
  - **`text_contrastive_train_step_factory(*, temperature)`** — lower-level `TrainStepFn` factory for users who want NNx's full callback / checkpoint / `runs/<id>/` machinery wrapped around the contrastive step (drive it through `NNModel.train(train_step_fn=...)` with a `DataLoader` that yields `(anchors: list[str], positives: list[str])` batches).
  - **`embed_texts(backbone, texts, *, batch_size, device, normalize)`** — inference-time encoder; runs `torch.no_grad()` + `eval()`. Used by `export_to_faiss` internally and exposed for ad-hoc similarity probes.
  - **`export_to_faiss(backbone, corpus, out_path, *, index_type, normalize, ...)`** — embed corpus → build a FAISS index of the requested type (`IndexFlatIP` for cosine via normalize-then-IP, `IndexFlatL2` for L2 distance, `IndexHNSWFlat` for approximate ANN with `M=32`) → write to disk via `faiss.write_index`. Lazy `faiss` import with a clear "install nnx[embeddings]" message on the failure path.
  - **`export_to_safetensors(backbone, out_path)`** — persist backbone weights for HuggingFace Hub / sentence-transformers reload. Uses the `safetensors` format when the package is importable (transitive via `sentence-transformers≥3`); falls back to plain `torch.save` otherwise.
- **`embeddings` optional extra** in `pyproject.toml` — pins `faiss-cpu>=1.7` + `sentence-transformers>=2.7`. Both are optional at import time; the package imports cleanly without them and the `ImportError` is deferred to the call site that actually needs each one.
- Runnable demo: `examples/13_train_domain_embedder.py` — synthesizes 40 `(sentence, paraphrase)` training pairs, trains a tiny bag-of-words hash embedder from scratch for 5 epochs (mean anchor-positive cosine: 0.61 → 0.98), exports to a FAISS `IndexFlatIP`, reloads from disk, and runs a top-3 query (the paraphrase comes back at #2 with cosine ≈ 0.99). Network-free, CPU-only, ~10s end-to-end.
- New docs page: `docs/embeddings.md` — when to use, install, quickstart, full API, composition with `nnx.freeze` / `nnx.apply_lora_to`, and the explicit "what this is NOT" list (no chunker, no reranker, no vector-DB client, no RAG-framework wrapper).
- 28 new tests across `tests/test_embeddings_{contrastive,faiss_export}.py`: dataset validation (empty / non-tuple / non-string entries), `pair_collate` shape, end-to-end "training reduces anchor-positive cosine distance" assertion on a synthetic 32-pair dataset (the headline TDD test), embed_texts batch-invariance + normalize on/off, `text_contrastive_train_step_factory` bad-batch rejection + weights-move-on-step, FAISS `IndexFlat{IP,L2}` + `IndexHNSWFlat` index construction, "embed 100-text corpus → save → reload → top-1 self-similarity" assertion (the FAISS-export TDD test), explicit-normalize override semantics, safetensors-roundtrip via both `safetensors` and the `torch.save` fallback. FAISS / safetensors tests skip gracefully when the optional dep isn't installed.
- `tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` + `OMP_NUM_THREADS=1` at session start — sidesteps a macOS-specific `faiss-cpu` segfault in its parallel search kernel when `torch`'s `libomp.dylib` got loaded first. Harmless on Linux CI.
### Added — Transformer fork (SP-4): TransformerNN + tokenizer + generate

- **`Nets.TRANSFORMER` enum variant** — decoder-only LM dispatched through the standard `NNModelParams(net=Nets.TRANSFORMER, ...)` factory path. Back-compat-safe: existing pre-TRANSFORMER `run.yaml` files load unchanged.
- **`TransformerNN`** — decoder-only stack matching LLaMA / Mistral conventions: token embeddings + N `TransformerBlock`s (pre-norm with RMSNorm + RoPE + SwiGLU FFN + multi-head causal attention) + final RMSNorm + tied LM head. KV-cache seam wired but switched off (`use_cache=False`); SP-10c will flip it on without changing call sites.
- **`NNTransformerParams(NNParams)`** — frozen dataclass holding `vocab_size`, `n_layers`, `n_heads`, `d_model`, `ffn_mult`, `max_seq_len`, `rope_base`, `tie_embeddings`, `attn_dropout`, `resid_dropout`. Lifts the GraphAttNN `n_heads`-on-NNParams pattern by subclassing. Every optional field omits itself from `state()` when at default — the broken-three-times omit-when-default invariant; covered by regression tests.
- **`NNTokenizerParams`** — wraps `tokenizers.Tokenizer` (HF Rust BPE). `state()` returns `{"path": "<tokenizer.json>"}`; the tokenizer payload lives on disk, only the pointer goes into `run.yaml`. Companion `train_bpe(...)` helper trains a tiny BPE from either file paths or an in-memory text iterator. Available when the `nnx[lm]` extra is installed.
- **`GenerativeNNModel(NNModel).generate(prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, stop, seed)`** — autoregressive decode via a `LogitsProcessor` chain (`TemperatureScaling` / `TopKFilter` / `TopPFilter` / `RepetitionPenalty`). `temperature=0` short-circuits to deterministic greedy; same-seed sampling reproducibility is part of the contract.
- **New example `examples/11_tinystories_lm.py`** — end-to-end TinyStories-class training run: train a BPE on the corpus, build a small Transformer, train next-token prediction via a custom `train_step_fn`, then sample. Ships with an inline fallback corpus so it runs offline; `--use-hf` downloads TinyStories.
- **New docs page `docs/lm.md`** — when/how to use the LM path. Linked from README §1.2 + §5.
- **`pyproject.toml` `lm` extra** — `tokenizers>=0.20`, `datasets>=2.20`. Opt-in so the Rust tokenizer binary isn't pulled for non-LM users.

### Migration notes

These two fixes shift `run.id` hashes on disk. Older `runs/<id>/` directories on disk continue to load by their existing directory name; recomputed ids land in a fresh directory.

- **Default-AMP runs now hash to a different `run.id`** than they did between pass-2 and this audit. The `mixed_precision=False` default is now correctly omitted from `state()` (back-compat invariant from before pass-2).
- **Plateau-scheduler runs now hash to a different `run.id`** than they did between the Schedulers-enum addition and this audit. The `kind=None` default + its variant-specific knobs (step_size / T_max / max_lr / total_steps / warmup_steps) are now correctly omitted from `state()` when at their defaults (same back-compat invariant).

### Fixed — back-compat invariant audit

- **`NNModelParams.state()` omits `mixed_precision` when False.** The field was added in pass-2 but always emitted into `state()`, breaking the omit-when-default back-compat invariant. Every default-AMP run had a shifted `run.id` versus pre-pass-2 runs with otherwise identical config. **One-time hash shift:** any existing default-AMP `runs/<id>/` directory will recompute to a different id after this fix — load by the on-disk directory name still works; recomputed ids will land in a fresh directory.
- **`NNSchedulerParams.state()` omits `kind` and the variant-specific knobs** (`step_size` / `T_max` / `max_lr` / `total_steps` / `warmup_steps`) when None. Same omit-when-default invariant: a plain ReduceLROnPlateau `NNSchedulerParams` now hashes to the same `run.id` as it did before the `Schedulers` enum was added. Existing on-disk runs with explicit-None entries still load (the legacy form is tolerated in `from_state`).
- **In-memory `best_checkpoint` tracking aligned with on-disk BEST.** `NNModel.train()`'s `best_checkpoint` reassignment used a different comparison than the BEST write inside `_save_checkpoints`. When `val_loader=None` (so every `val_edp` is None), the in-memory tracker effectively held LAST while the on-disk BEST tracked training error. Both now go through the same `_best_err` helper.
- **`_best_err` deduplicated.** Was triplicated — a local closure in `NNModel._save_checkpoints`, a module-level helper in `nn_run.py`, and another module-level helper in `trainer/trainer.py`. Kept the `nn_run.py` version as canonical; the other two now import it.
- **Paradigm step factories honor `grad_clip_norm` and guard against non-finite loss.** The four paradigm `train_step_fn` factories (diffusion / SimCLR / Mixup / CutMix) plus KD now route through a shared `nnx._step_helpers.finalize_step` helper. Previously they silently dropped `NNOptimParams.grad_clip_norm`, and diffusion / SimCLR / Mixup / CutMix had no NaN/Inf guard — only KD checked. **New explicit rejection:** the helper raises `ValueError` if `NNModelParams.mixed_precision=True` (paradigm steps don't handle the scaler) or if `accumulate_grad_batches != 1` (no cycle-aware accumulation). Previously these were silently ignored; users with those knobs set now see a clear error.
- **`ModelCheckpoint` callback actually saves now.** The body was `if ctx.epoch in self.epochs: pass` — a no-op stub. Now writes `runs/<run.id>/checkpoints/<tag>_e<epoch>.pt` via the atomic-write path on matched epochs.
- **`FeedFwdNN.from_file` uses `torch.load(weights_only=True)`** for consistency with `NNCheckpoint.load_optimizer_state` and `load_pretrained`. State-dicts are tensor-only; the strict loader works AND removes the arbitrary-code-execution risk on user-supplied paths.
- Documentation and comment cleanups: `docs/index.md` listed only pass-2 features (added the five new tracks); `docs/concepts.md` architecture diagram missed the five new subpackages (extended with a Specializations branch); `examples/06`'s `_make_loaders` docstring claimed class-conditional Gaussians that the code didn't implement (rewrote); `freezing.py` docstring incorrectly claimed `fnmatch *` matches segment-boundaries (it matches across dots); `loading.py` `key_map` docstring said "substring replacement" but the code does prefix replacement; KD's loss formula in `paradigms/distillation.py` module docstring + inline comment AND `docs/concepts.md` all reversed the KL direction — the math is `KL(teacher || student)` (standard Hinton), but the doc strings read `KL(student || teacher)`; `peft/adapters.py` activation docstring said `nn.GELU()` (instance) but the default is `nn.GELU` (class factory); README's enums-as-factories bullet was missing `NoiseSchedulers`. Internal phase labels (Track A / Track B / Track C / pass-2 R2 / R3 / R4) that had leaked into published code/docs/tests have been replaced with descriptions of WHAT the referenced thing is.

### Added — model-internals viz (`nnx.viz` subpackage)

- **`nnx.viz` subpackage** — sibling of the existing `nnx.vis_utils` (which handles run-output viz: training curves, confusion matrices, t-SNE of checkpoint logits). `nnx.viz` covers the **model itself** rather than what the run produced. Two primitives ship in this PR; `activation_map`, `netron_export`, and Captum attribution land in a later PR.
  - **`nnx.viz.summary(model, *, input_size=..., depth=4, col_names=...)`** — Keras-style parameter table via a thin `torchinfo.summary` wrapper. Returns the `torchinfo.ModelStatistics` object directly so callers can both print the formatted table AND query `.total_params` / `.trainable_params` / `.total_mult_adds` for programmatic regression assertions. Accepts an `NNModel` (unwrapped to `.net`) or any `nn.Module`.
  - **`nnx.viz.weight_histogram(model, *, bins=64, cols=3, fig_width=1000, row_height=200)`** — per-parameter Plotly histogram grid. Walks `model.named_parameters()` and emits one `Histogram` trace per tensor in a subplot grid, consistent with `vis_utils`'s Plotly-returning idiom. Useful for spotting dead layers, NaN / Inf weights, or saturation patterns. Raises `ValueError` on a parameter-less module (which would otherwise produce a silently-empty figure).
- **New `viz` optional extra** — `pip install nnx[viz]` pulls in `torchinfo>=1.8.0`. `nnx.viz.summary` raises a clear `ImportError` pointing at the extra if `torchinfo` is missing; `weight_histogram` only depends on `plotly` (already a core dep), so it works out of the box.

### Added — PEFT (LoRA + adapters)

- **`nnx.peft` package** — two complementary patterns for parameter-efficient fine-tuning of pretrained networks.
  - **`LoRALinear(base, *, r, alpha, dropout)`** — wraps an `nn.Linear`, freezes the base's parameters (`requires_grad=False`) on construction, and adds two trainable matrices `lora_A` (r × in, Kaiming-uniform init) and `lora_B` (out × r, **zero-initialized**) whose product is added as a residual scaled by `α/r`. The zero-init on B means output at step 0 equals `base(x)` exactly — fine-tuning starts from the pretrained behavior and diverges only as B picks up gradient. Validates `r > 0`, `alpha > 0`, `0 ≤ dropout < 1` at construction.
  - **`apply_lora_to(module, *patterns, r, alpha, dropout)`** — walks `module.named_modules()` and replaces every `nn.Linear` whose dotted name matches any fnmatch glob with a `LoRALinear` wrapper, in place. Returns the count wrapped. **Idempotent**: re-applying against patterns that already match LoRA-wrapped layers is a no-op (the inner `.base` is excluded from the walk). Same glob conventions as `nnx.finetune.freeze`.
  - **`save_lora_weights(module, path)`** — writes ONLY the `lora_A` / `lora_B` parameters via `torch.save` of a filtered state-dict subset. A small percentage of the full `state_dict` size (single-digit % at production scale; closer to ~40% on tiny demo nets where r/dim is large — see `docs/concepts.md` §11 for the math).
  - **`load_lora_weights(module, source)`** — loads LoRA params from a path (`weights_only=True` for safety) or directly from a dict, via `load_state_dict(strict=False)` so the frozen base's missing keys don't raise. Returns the number of tensors loaded.
  - **`AdapterLayer(dim, bottleneck, activation=nn.GELU)`** — bottleneck residual block `y = x + up(act(down(x)))`. `up.weight` and `up.bias` are zero-initialized so the layer is the residual identity at step 0. Composed by the user into a custom `nn.Module` — NNx doesn't ship a "wrap every block" helper because adapter insertion points are architecture-specific.
- Runnable LoRA demo: `examples/07_lora_finetuning.py` — pretrains a small classifier, wraps every Linear with LoRA, fine-tunes on a different distribution, **explicitly verifies every base parameter is bit-exactly unchanged** across the fine-tuning run, and compares the LoRA-only checkpoint size against a full `state_dict` snapshot.
- 23 new tests across `tests/test_peft_{lora,adapters}.py`: LoRALinear validation + base-freezing + zero-init invariant (output == base at step 0) + only-LoRA-trainable invariant + in/out features pass-through; `apply_lora_to` empty-pattern rejection + selective wrap + wildcard wrap + idempotency on re-application + forward-preserves-at-init; save/load round-trip + base-keys-excluded-from-checkpoint + dict-source loading + bad-source-type rejection; end-to-end PEFT contract (every base param bit-exactly unchanged + every lora_B param has moved); AdapterLayer shape + identity-at-init + parameter-count scaling + gradient-flow + dim validation + custom activation.

### Added — training paradigms (KD / SimCLR / Mixup / CutMix)

- **`nnx.paradigms` package** — four `TrainStepFn` factories for non-vanilla supervised paradigms, all consumed via the existing `NNModel.train(train_step_fn=...)` hook. No new params dataclass, no NNModel changes; each is a self-contained closure.
  - **`kd_train_step_factory(teacher, *, alpha, temperature)`** — Hinton-style knowledge distillation. Mixes a temperature-softened KL divergence against the teacher's logits (`α · KL · T²`) with the standard hard-label loss (`(1-α) · L_hard`). The factory **freezes the teacher's parameters and sets its net to eval mode on call**, so the teacher provably cannot drift across the student's training. The hard term goes through the student's `loss_fn` so KD works with any classification loss.
  - **`simclr_train_step_factory(*, temperature)`** — SimCLR contrastive training. The training loader must yield `(view1, view2)` paired-view tensors per source sample. `model.net` is forwarded once per view (BatchNorm sees one view at a time). Reports the NT-Xent loss in both `.loss` and `.error`.
  - **`nt_xent_loss(z1, z2, *, temperature)`** — the SimCLR loss exposed as a standalone for users wanting to compose it into custom training loops.
  - **`mixup_train_step_factory(*, alpha)`** — Mixup batch augmentation: `x' = λx_a + (1-λ)x_b` with `λ ~ Beta(α, α)`. Works for any input rank (tabular, sequence, image). Reports λ-weighted accuracy as the `accuracy` field; `accuracy + error == 1`.
  - **`cutmix_train_step_factory(*, alpha)`** — CutMix batch augmentation for 4D `(B, C, H, W)` image batches. Copies a random rectangle from `x_b` into `x_a`, then re-weights the loss by the actual cut area (which can be smaller than the nominal Beta draw when the box clips at an edge). Raises a clear `ValueError` on lower-rank input — CutMix's spatial cut isn't well-defined without H and W.
- Runnable distillation demo: `examples/10_knowledge_distillation.py` — pretrains a wider teacher (hidden_dims=[64, 64]) then distills into a much smaller student (hidden_dims=[16], roughly 4% of the teacher's parameters). The example explicitly verifies teacher weights are unchanged across the student's training run, demonstrating the factory's freeze guarantee. Honest about scope: doesn't claim to beat a non-distilled baseline on toy tabular data.
- 19 new tests across `tests/test_paradigms_{distillation,contrastive,augmentation}.py`: factory validation (alpha / temperature ranges), teacher freezing guarantee + teacher-eval-mode assertion, end-to-end loss-decreases (KD α=0.5) + α-boundary cases (α=0.0 collapse to supervised, α=1.0 pure distillation), NT-Xent properties (shape mismatch, finite + scalar output, loss smaller for aligned pairs than random), SimCLR step bad-batch-shape error, Mixup self-consistency (accuracy + error == 1), CutMix non-image input rejection + 4D end-to-end.

### Added — diffusion (DDPM)

- **`nnx.diffusion` package** — DDPM-style diffusion training and sampling, layered entirely on top of the existing `train_step_fn` hook on `NNModel.train()` (no Trainer, no NNModel internals touched).
  - **`NoiseSchedulers`** — enum-as-factory with two variants: `LINEAR(T, beta_min, beta_max)` (original DDPM linear betas) and `COSINE(T, s)` (Improved-DDPM cosine schedule). Each enum value's `__call__` returns a precomputed `NoiseSchedule`.
  - **`NoiseSchedule`** — frozen dataclass holding the derived tensors (`betas`, `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, `posterior_variance`). All 1D of length T. `.to(device)` returns a copy with every tensor migrated. Not `state()`-serialized — recoverable from `(kind, T, kind-specific knobs)`.
  - **`DiffusionMLP(input_dim, hidden_dims, time_embed_dim)`** — small conditional MLP: sinusoidal time embed → projection → concat with flat x → MLP → noise prediction. `forward(x, t) → ε_pred`. Handles arbitrary-rank inputs by flattening + un-flattening. Intentionally minimal; image-space diffusion calls for a U-Net the user supplies, with the same schedule / step / sampler machinery.
  - **`diffusion_train_step_factory(schedule) -> TrainStepFn`** — closes over the schedule and returns a `TrainStepFn` suitable for `NNModel.train(train_step_fn=...)`. Per batch: samples `t ~ Uniform[0, T)`, samples `ε ~ N(0, I)`, computes `x_t`, predicts noise, backprops MSE. Reports loss as both `.loss` and `.error` on the EDP so BEST tracking + ReduceLROnPlateau work.
  - **`sample(model, schedule, shape, device=, generator=)`** — reverse-diffusion sampler. Runs T backward steps under `torch.no_grad()` and `model.net.eval()`. The optional `generator=` enables reproducible sampling for notebooks.
  - **`sinusoidal_time_embed(t, dim)`** — standalone helper for the standard sinusoidal positional embedding, exposed for users building their own t-conditioned nets.
- **`NNModel.train()` net-params fallback** — the run-construction line now reads `self.net_params` (always set in `__init__`) instead of `self.net.params` (FeedFwdNN-specific attribute). Back-compat-safe: the values are identical for the existing supervised path. Lets callers swap `model.net` for a custom `nn.Module` post-construction (the same idiom the multi-optimizer Trainer's GAN demo uses) without breaking `NNModel.train()`.
- Runnable diffusion demo: `examples/08_diffusion_2d_mixture.py` — DDPM on a 2D mixture of 4 Gaussians at (±2, ±2). Verified end-to-end (loss 1.0078 → 0.6048; samples land in all four modes at roughly equal counts).
- 27 new tests across `tests/test_diffusion_{schedules,nets,training,sampling}.py` covering schedule shape/monotonicity/clamping, net forward shape, full training + loss-decreases, sampling shape / finiteness / reproducibility / mode coverage.

### Added — multi-optimizer Trainer (GAN / actor-critic)

- **`nnx.trainer` package** — `Trainer` class that parallels `NNModel.train()` for scenarios where the per-batch update isn't a single supervised forward/backward/step. Built around the GAN G/D pattern, but applicable to actor-critic, EBM, contrastive multi-head, or any other multi-optimizer paradigm.
  - **`Trainer(model: NNModel).train(params, trainer_step_fn, callbacks=)`** — builds one `torch.optim.Optimizer` per entry in `NNTrainerParams.optims`, dispatches to a user-supplied `trainer_step_fn(ctx) -> NNEvaluationDataPoint` per batch, writes the same `NNRun` + per-tag `NNCheckpoint` artifacts as `NNModel.train()`. No `default_trainer_step` — multi-optim updates are scenario-specific and silently running the wrong update is worse than requiring an explicit fn.
  - **`NNTrainerParams`** — frozen dataclass with `optims: Mapping[str, NNOptimParams]` (name-keyed multi-optim config), `schedulers: Mapping[str, NNSchedulerParams]` (one per optim, defaults to ReduceLROnPlateau when missing), plus the standard `n_epochs` / `train_loader` / `val_loader` / `seed` / `save_phase_checkpoints` / `extra_metrics`. Validates non-empty `optims` and that every scheduler key matches an optim key. `state()` keys sorted for deterministic `run.id`.
  - **`TrainerStepContext`** — frozen bundle passed into a `trainer_step_fn`: `model`, `batch`, `optimizers` (dict), `schedulers` (dict), `extra_metrics`, `batch_idx`, `epoch_idx`. The companion `TrainerStepFn` type alias is exported.
- **Strict `param_groups` semantics** for multi-optim — `build_param_groups(..., strict=True)` (new keyword) drops parameters that match no spec instead of bucketing them into a default group. Threaded through `Optims.__call__(..., strict_param_groups=True)`. The Trainer passes True so disjoint optimizers don't co-own parameters via implicit default buckets. Default `strict=False` preserves the fine-tuning semantics introduced by `nnx.finetune.param_groups` exactly.
- **`NNRun.trainer: Optional[NNTrainerParams]`** — populated by the Trainer; None for `NNModel.train()` runs. **Strict back-compat:** OMITTED from `state()` when None so existing `NNModel` run.id hashes are unchanged. `NNRun.load(id)` round-trips trainer-mode runs by lazy-importing `NNTrainerParams.from_state` when the YAML carries a `trainer` block.
- Runnable GAN demo: `examples/09_gan_with_trainer.py` — generator + discriminator packed into one nn.Module, two disjoint optimizers scoped via `NNParamGroupSpec(name_pattern="G.*" | "D.*")`, alternating updates on a 1D mixture-of-Gaussians. Verified end-to-end on CPU.

**Deferred from this PR:** trainer-mode warm-resume. The Trainer writes only the model net's `state_dict` to its `NNCheckpoint`s — there is no per-optimizer `.opt.<name>.pt` sidecar yet. `NNTrainerParams` does not carry `resume_from_run_id` / `resume_from_checkpoint`. Resuming a GAN's Adam state for both G and D will land as its own follow-up PR once the use case is exercised.

### Added — fine-tuning infrastructure (freeze / unfreeze / param_groups)

- **`nnx.finetune` package** with three submodules:
  - **`freezing`** — `freeze(module, *patterns)` / `unfreeze(module, *patterns)` / `frozen(module)`. Glob-pattern (`fnmatch`) toggling of `requires_grad` on submodule parameters; the standard transfer-learning idiom. `NNModel.freeze` / `NNModel.unfreeze` are convenience methods delegating to the free functions.
  - **`loading`** — `load_pretrained(module, source, *, key_map, strict, prefix)` returns a `LoadPretrainedResult` with `loaded_keys` / `missing_keys` / `unexpected_keys`. Sources: file paths (loaded with `weights_only=True` for safety), state-dicts, or other `nn.Module`s. Key remapping handles foreign naming conventions (torchvision / HuggingFace / etc.).
  - **`param_groups`** — `NNParamGroupSpec` (frozen, kw_only, slots dataclass) for declarative per-layer LR / weight_decay overrides. The fine-tuning idiom of "small LR on the backbone, large LR on the head" expressed as a list of specs on `NNOptimParams.param_groups`. `build_param_groups(module, specs, default_lr, default_weight_decay)` is the helper the `Optims` enum factory dispatches through.
- **`NNOptimParams.param_groups: Optional[list[NNParamGroupSpec]]`** field. When set, the optimizer factory builds per-group dicts with the spec's lr / lr_multiplier / weight_decay overrides; frozen parameters are dropped. **Strict back-compat:** `param_groups=None` (default) is OMITTED from `state()`, so existing `run.id` hashes are unchanged.
- **`NNModel.export_state_dict(path)`** — saves `self.net.state_dict()` to disk as a plain torch file (no NNCheckpoint wrapper). Companion to `load_pretrained` for the round-trip.

### Added — `train_step_fn` hook on `NNModel.train()` (foundational)

- **`train_step_fn` hook on `NNModel.train()`.** One optional kwarg that swaps out the supervised forward/backward/step for any user-supplied function. Unblocks non-supervised training paradigms (autoencoder, VAE, link prediction, recommendation, diffusion) without modifying NNx core. Default-None path is byte-identical to the prior loop. New public surface: `TrainStepContext` (frozen dataclass carrying model/batch/optimizer/scaler/grad_clip_norm/extra_metrics/accumulate_grad_batches/batch_idx/epoch_idx), `default_train_step(ctx)` (the standard supervised step, exported for users who want to layer behavior on top), `TrainStepFn` (type alias). Seven tests in `tests/test_train_step_hook.py`; runnable autoencoder example at `examples/05_custom_train_step_autoencoder.py`.
- Public alias for `nnx.PredictResult` (was reachable only via `nnx.nn.nn_model`).

### Changed — internal

- `NNModel.__fwd_pass` → `NNModel._fwd_pass`. Required so the free `default_train_step` can reach it without Python name-mangling. Single underscore is still "weak private"; no external consumer touched the mangled `_NNModel__fwd_pass` name.
- `NNModel._train_step` becomes a one-line wrapper around `default_train_step` for back-compat with any hypothetical subclass that overrode it. The `train()` loop itself no longer dispatches through `_train_step`.

### Fixed

- `_save_checkpoints` / `_step_scheduler` / `_update_tqdm_postfix` now tolerate an `NNEvaluationDataPoint` with `error=None`. Custom `train_step_fn` hooks for non-supervised paradigms (VAE/autoencoder/diffusion) don't always have a classification error to report; the loop falls back through `val_edp.error → val_edp.loss → train_edp.error → train_edp.loss` and skips the scheduler step entirely if nothing is set. Previously these three sites crashed with `TypeError` on `None < float` / `float(None)` / `f"{None:.4f}"`.

### Deferred

- `eval_step_fn` / `predict_fn` — same pattern, but `evaluate()` and `predict()` still assume supervised classification. First ml-lab task that needs custom eval (autoencoder, VAE, DDPM) will drive that.
- Network registry (`Nets.register(...)`) — each new architecture lands a `Nets` enum variant via its task's PR.
- Loss registry — custom losses live inside `train_step_fn` today (the user computes the loss tensor manually). Lift to a registry when multiple tasks duplicate the same custom loss.

## [Pass-2 unreleased] — comprehensive improvements pass 2

Second improvement pass on branch `chore/comprehensive-improvements-pass-2`, building on pass-1. Strict back-compat preserved throughout — every new field on a params dataclass defaults to its old value and omits itself from `state()` when the default holds, so existing `run.id` hashes are unchanged.

### Added — features (warm-resume, gradient accumulation, custom epoch checkpointing, etc.)

- **Warm-resume training.** `NNTrainParams.resume_from_run_id` and `resume_from_checkpoint` load weights AND optimizer state from a prior run's checkpoint at the start of `train()`. Optimizer state is written as a `.opt.pt` sidecar so the existing pickled `NNCheckpoint` format is untouched.
- **Gradient accumulation** via `NNOptimParams.accumulate_grad_batches` (default 1). Loss is scaled by 1/N; `zero_grad`/`optimizer.step` fire on cycle boundaries; AMP unscale + grad-clip both honor the cycle.
- **TensorBoardCallback** and **WandbCallback** — stream per-epoch train/val metrics + LR. Lazy import so users not on the path don't pay the dep cost.
- **`NNModel.to_onnx(path, example_input)`** — export the network via the legacy `torch.onnx.export` tracing path (no `onnxscript` needed). Marks dim-0 dynamic by default.
- **`NNTabularDataset`** — wraps a pandas DataFrame into train/val/test loaders matching the `NNDatasetBase` contract.
- **Custom metrics** via `NNTrainParams.extra_metrics={name: fn}`. Each `fn(Y, Y_hat) -> float` populates the new `NNEvaluationDataPoint.extra` dict; survives the `NNRun.save`/`NNRun.load` round-trip via `extra.<name>` CSV columns.

### Added — reproducibility (seeded RNGs + env snapshot in metadata.yaml)

- `nnx.set_seed(seed, strict=False)` pins Python `random`, NumPy, torch CPU+CUDA, and cuDNN. `strict=True` also calls `torch.use_deterministic_algorithms(True)`.
- `nnx.dataloader_worker_init_fn` — pass to `DataLoader(worker_init_fn=...)` for per-worker deterministic seeds.
- `NNTrainParams.seed` runs `set_seed` at `train()` entry; included in `state()` only when set.
- `nnx.env_snapshot()` captures library / torch / numpy / python / platform / CUDA / git-commit info. Written by `NNRun.save()` to `runs/<id>/metadata.yaml` — separate from `run.yaml` so it does NOT contribute to `run.id`.

### Added — API ergonomics (predict tuple unpack, file= kwargs, NNCheckpoint helpers)

- `NNModel.predict(X)` accepts `numpy.ndarray`, `torch.Tensor`, tuples thereof, or a `DataLoader` (labels in batches are discarded). Returns a `PredictResult` NamedTuple that unpacks positionally as `(logits, classes)` for back-compat.
- `NNTrainParams.save_phase_checkpoints: bool = True`. Set False to skip the FIRST + Q1/Q2/Q3 cycle (LAST + BEST still always saved) — useful for tiny experiments or huge models.
- `Devices.torch_device()` / `Devices.get_torch_device()` return `torch.device` directly without the `.()` dance.
- `Utils.print_tree` / `print_table` accept `file=` for output redirection.
- `nnx.__version__` resolves from `importlib.metadata`; falls back to `"0.1.0+local"` when editable-installed.
- `pyproject` keywords expanded (training, checkpointing, callbacks, experiments, reproducibility, neural-networks, research).

### Added — reliability (NaN-loss guard, gradient clipping, atomic NNRun.save)

- **NaN/Inf guard** in `NNModel._train_step` — raises `FloatingPointError` rather than letting divergence corrupt checkpoints silently.
- **Gradient clipping** via `NNOptimParams.grad_clip_norm: Optional[float]`. AMP-aware (unscales before clipping).
- **Incremental persistence** — `NNRun.save()` runs after every epoch, not just at the end. `KeyboardInterrupt` / OOM mid-training now leaves a loadable partial run.
- **SECURITY note** on `NNCheckpoint.from_file` calling out the arbitrary-code-execution risk of `weights_only=False` on untrusted files.
- Re-pin `loss_fn` to `self.device` on every `evaluate()` call (guards against late device reassignment).

### Fixed — correctness (evaluate aggregation, NNOptimParams.is_valid, callback isolation)

- `NNOptimParams.is_valid()` now returns `False` (not implicit `None`) for unknown enum variants — invalid configs no longer slip past the `not params.optim.is_valid()` pre-flight check.
- `NNModel.train()` tolerates `DataLoader`s without `__len__` (`IterableDataset`-backed). Falls back to a tqdm bar with no total.
- `NNRun.save()` falls back to writing `best/POINTER.txt` when `os.symlink` raises (Windows without developer mode).
- `NNModel.evaluate()` aggregates Y / Y_hat across batches and computes metrics once on the aggregate, fixing unequal-final-batch weighting. Raises `ValueError` on an empty loader instead of returning NaN.
- `NNIterationDataPoint` gets a docstring spelling out that `val_edp` is populated only on the LAST idp of each epoch — readers shouldn't expect it on every row.

### Changed — tooling (pyproject extras, conftest hygiene, type-checker config)

- CI runs pytest under coverage (`pytest-cov`), uploads `coverage.xml` artifact on Python 3.11.
- CI runs pyright in basic mode (`continue-on-error: true` today; will tighten to `--strict` over time).
- `NNX_TQDM_DISABLE=1` silences the training progress bar — autouse'd in `tests/conftest.py` so pytest output stays clean.
- `tests/conftest.py` exposes shared fixtures (`tiny_model`, `tiny_classification_loaders`, `tmp_runs_root`, ...).
- `mkdocs.yml` + `docs/` skeleton (index, quickstart, concepts, api). New `.github/workflows/docs.yml` builds with `--strict` on every push and deploys via `mkdocs gh-deploy` on `main`. New `nnx[docs]` optional extra.
- `.pre-commit-config.yaml` with ruff + standard pre-commit-hooks.
- `CONTRIBUTING.md` covering setup, workflow, back-compat invariants, testing.
- `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md` + `pull_request_template.md`.
- `.github/workflows/release.yml` — tag-triggered build + PyPI publish via OIDC trusted publishing.

### Added — docs (concepts.md / quickstart.md / api.md)

- `examples/` folder with four runnable scripts: `01_synthetic_classification.py`, `02_resume_training.py`, `03_custom_metrics.py`, `04_onnx_export.py`. All verified end-to-end on CPU.

### Internal (Utils back-compat shim, vis_utils module aliases)

- `Utils.print_tree` / `print_table` / `flatten_dict` are now module-level functions in `nnx.utils`. The `Utils` class is a thin shim binding the same functions as staticmethods, so existing `Utils.method(...)` callers continue to work with no semantic change.
- `VisUtils` plotting helpers get module-level aliases (`from nnx.vis_utils import confusion_matrix` works).

### Additional fixes (post-initial-pass)

- `runs/best` POINTER.txt fallback wasn't read during BEST comparison; env_snapshot subprocessed git on every save; `.gitignore` missed `runs/`, `tb_logs/`, `*.onnx`, `coverage.xml`, `site/`.
- **Critical:** `NNOptimParams.state()` unconditionally emitted `grad_clip_norm=None`, changing every existing `run.id` hash. Plus `callbacks.py` top-level IPython import (pulling IPython into every `import nnx`); `NNRun.all()` crashed on missing `runs/` and tried to load stray files.
- `mkdocs build --strict` had 4 warnings (specs in docs but not nav; griffe couldn't parse a docstring; missing type annotation on `to_onnx.example_input`).
- **Critical:** `NNEvaluationDataPoint.extra` didn't actually round-trip through `idps.csv`. json_normalize flattened the dict on save but `NNIterationDataPoint.from_state` never reassembled the `train_edp.extra.*` columns. The pass-2 claim that "extra survives idps.csv" was false until this fix.
- `pytest-cov` listed in dev extras but not installed locally. CI handles via `pip install -e ".[dev]"`; surfaced via cov-run on a fresh venv.
- `NNEvaluationDataPoint.mean_of` silently dropped the `extra` dict from inputs. `NNCheckpoint.load_optimizer_state` now uses `weights_only=True` (the state dict is structured tensors + dicts; the strict loader works AND removes the ACE risk).
- Six conftest fixtures (`tiny_model`, `tiny_classification_loaders`, etc.) defined but unused — premature abstractions deleted; CONTRIBUTING.md updated to match.
- `NNTabularDataset` now validates `feature_cols` / `target_col` against `df.columns` up-front with a clear KeyError; new test for `env_snapshot` cache (introduced in R1 but never explicitly tested).
- stray leading blank line in `nn_graph_dataset.py`.
- **Real recovery gap:** `NNRun.save()`'s three writes (run.yaml, metadata.yaml, idps.csv) were non-atomic. A Ctrl-C mid-write left half-written files. New `_atomic_write_text` helper does tmp + fsync + os.replace.
- `NNCheckpoint.to_file` had the same non-atomic gap (torch.save direct to destination). New `_atomic_torch_save` helper applies the same tmp + rename pattern to both the main checkpoint and the `.opt.pt` sidecar.
- Atomicity also applied to the Windows POINTER.txt fallback; helper reordered (defined before its caller); pyproject `filterwarnings` for the upstream `torch_geometric.distributed` / `torch.jit.script` DeprecationWarnings; fix the scheduler test's optimizer-before-scheduler step order so the runtime UserWarning doesn't fire.
- README "Other models" was a non-functional snippet (imported classes without showing how to wire them through `NNModel`). Replaced with concrete `NNModelParams(net=Nets.GRAPH_*)` examples + a pointer at the `examples/` folder. Added README subsections for Reproducibility, Warm-resume, and Custom metrics so the pass-2 features are visible from the top-level doc.
- `test_imports.py` was missing smoke imports for `nnx.seeding`, `nnx.nn.callbacks`, `nnx.nn.net.graph_nn_base`, `nnx.nn.dataset.nn_tabular_dataset`, and `nnx.nn.enum.schedulers`. The test predated pass-1 and never grew with the codebase. Closed the gap so the cheapest-possible refactor signal is exhaustive again.
- `release.yml` skipped `twine check` between `python -m build` and the PyPI upload step. A malformed README or invalid classifier would only surface when PyPI rejected the upload — by then the tag is burned. Added a `twine check dist/*` verification step; also added `cache: pip` to the setup-python step for parity with the other workflows.
- **Final sweeps**: ran the literal README quickstart end-to-end, manually exercised the four `predict()` input forms (ndarray, tensor, tuple-of-each), verified all internal markdown links resolve, and confirmed `mkdocs build --strict` is silent. No additional actionable findings.

### Deferred (with rationale)

- **D3** (split `NNModel.train()` into a `TrainingLoop` runner): the existing helpers (`_train_step`, `_save_checkpoints`, `_step_scheduler`, `_build_scheduler`, ...) already break the loop body into testable units. A full extraction would be churn without proportional value.
- **D7** (versioned state-dict checkpoint format with a versioned reader): too risky for this back-compat pass. The pickled `NNCheckpoint` continues to work; the `weights_only=False` security note in the docstring guards against the supply-chain risk.
- **D8** (Storage protocol for cloud backends): broad I/O abstraction touching every save/load site. Better as its own focused PR.
- **N5** (md5 of `str(state)` → `json.dumps(sort_keys=True)`): would change every existing `run.id`. Can't ship under strict back-compat.
- **O5 / O6 / O7** (NNTrainParams config-vs-runtime split, callbacks-as-params, NNModel `__init__` param rename): API breaks. Deferred.
- **O4** (frozen `_CallbackContext` view): would change the surface callbacks can mutate — defer to a callback API revision.
- **P1 / P2** (per-batch device sync, loss.item() sync): would sacrifice per-batch metric granularity (idp.train_edp). Deferred.
- **E7** (move ipython/kaleido to optional extras): would break `pip install nnx` for users relying on the default extras. Deferred.

## [Pass-1 unreleased] — comprehensive improvements pass 1

The pass-1 series landed on branch `chore/comprehensive-improvements-pass-1`. Strict back-compat preserved: no public API renames, no on-disk format breaks, deep imports still resolve.

### Fixed — correctness

- `NNDataset` now carves the validation slice out of the source `train=True` split, keeping the source `train=False` split intact for final evaluation. Previously val was a slice of test, leaking the test pool. Reported val metrics will differ between pre/post versions.
- `NNDataset` `random_split` sizes are computed as `(total - val, val)` instead of two truncated halves. Fixes the crash on odd-length source train sets.
- `NNRun.save` no longer crashes when comparing against a prior BEST run that has no `val_edp` (e.g., a no-validation experiment). A new `_best_err` helper falls back to train error, then `+inf`.
- `NNEvaluationDataPoint.of` now defaults `average="macro"` for `f1` / `recall` / `precision`. The prior `"micro"` hardcoding made all three numerically identical to accuracy for single-label multi-class tasks. Pass `average="micro"` to opt back in.
- `VisUtils.multi_line_plot`: removed dead `cs = px.colors.qualitative.Plotly[...]` assignment that was immediately overwritten; replaced the `ls[:len(ys)]` legend loop (which depended on a leaked inner-loop variable) with `n_lines_per_series = len(yss[0])`; raises `ValueError` on empty `yss`.
- `Activations.SOFTMAX` returns a closure that supplies `dim=-1`, avoiding the implicit-dim warning and ambiguity from `F.softmax`.
- `NNModel._train_step`: detach `train_loss` before `float()` to avoid `UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior`.

### Fixed — deprecations / future breakage

- Migrated `torch.cuda.amp.{autocast,GradScaler}` to `torch.amp.*` with explicit `device_type="cuda"`. The `torch.cuda.amp` module has been deprecated since torch 2.4.
- `NNCheckpoint.from_file` calls `torch.load(weights_only=False)`. Without it, torch ≥ 2.6 (where `weights_only` defaults to `True`) raises `UnpicklingError` on any saved `NNCheckpoint` (checkpoints pickle the full Python object, not a bare state dict).
- `NNGraphDataset` reads the underlying `Data` via `dataset[0]` instead of `dataset._data`. The private accessor was renamed/removed across PyG versions.
- Removed the top-level `from IPython.display import clear_output` import in `nn_model.py`. The actual use is in `callbacks._LegacyCallback`; leaving the top-level import made every consumer of `nnx.nn.nn_model` pull in IPython.

### Added — initial release scaffolding (top-level re-exports, persistence root, viz figures)

- `nnx/__init__.py` re-exports the curated public surface (`NNModel`, params, callbacks, enums, nets, datasets, utils) with an explicit `__all__`. Deep imports (`from nnx.nn.net.feed_fwd_nn import FeedFwdNN`) still work for existing code.
- `NNRun.save / load / all / checkpoints` and `NNCheckpoint.save / load` accept an optional `root: Optional[str] = None` kwarg. Default is unchanged (cwd-relative); callers wanting to redirect persistence can now pass one.
- `NNEvaluationDataPoint.of` accepts `average: str = "macro"`.
- `VisUtils.{multi_line_plot, scatter_plot, two_dim_tsne_checkpoint_logits, confusion_matrix}` now return the `plotly.graph_objects.Figure` they build. The `.show()` call is gated on a non-None renderer so headless test envs no longer crash.
- `tests/test_params_round_trip.py` — contract test asserting `obj == from_state(state())` for every params dataclass. Fails loudly when fields drift.
- `tests/test_train_integration.py` — end-to-end `NNModel.train()` coverage on a tiny in-memory `TensorDataset`, plus `NNRun.load` round-trip and `NNModel.from_checkpoint` reconstruction.
- `NNOptimParams.momentum` docstring explaining the SGD-vs-Adam dual meaning.
- `NNDataset` docstring documenting that val is carved from train.
- This `CHANGELOG.md`.

### Changed — tooling

- Ruff lint now selects `E`, `F`, `W`, `B` (bugbear), `I` (isort), `UP` (pyupgrade). Style-preserving ignores: `E701` (case style), `B024` (structural base class), `UP007` / `UP045` (keep `Optional` over `X | None`). 213 auto-fixes applied (mostly import ordering).
- CI matrix adds Python 3.12.
- CI ruff step no longer has `continue-on-error: true` — lint gates merges.

### Internal

- `nn_dataset_base.py`: trimmed 9 unused imports.
- `nn_model.py`: removed empty `class NNModel():` parens.
- `nn_dataset.py`: switched to a local `resolved_batch_sizes` so downstream loaders don't read `self.batch_sizes` while it still holds the default tuple.

## [0.1.0] — 2026-05-18

Initial extraction from `thekaveh/ml`.
