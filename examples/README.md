# 19. Examples

Runnable scripts demonstrating common NNx patterns. Each is self-contained, with no external data dependencies, and CPU is sufficient unless a script's mode says otherwise. `decision_model_pilot.py --mode empirical` needs external resources and reports *blocked* without them. It also reports *blocked* with them, since its training runs outside this repository.

## 1. Run

```bash
pip install thekaveh-nnx                # core (covers every example not listed below)
python examples/01_synthetic_classification.py
```

**Re-running an example.** Run ids are content-addressed: the same configuration gets the same id. So most training examples that save under `runs/` in the working directory raise `FileExistsError` when re-run there (a few overwrite on purpose). Delete `runs/`, or run each example in a fresh directory.

The examples track the current source. If one needs a feature that your PyPI release does not have, install the source as shown in the [README](../README.md#2-install).

`01_synthetic_classification.py` and `25_conv_classifier.py` run on the core install alone (CI runs both on the core wheel and sdist).

A handful of examples depend on optional extras — install them as needed:

```bash
pip install "thekaveh-nnx[onnx]"             # 04_onnx_export.py, 12_quantize_int8.py (Phase-5 export)
pip install "thekaveh-nnx[onnx-runtime]"     # 04_onnx_export.py (executed level)
pip install "thekaveh-nnx[quantize]"         # 12_quantize_int8.py, 15_qat_classifier.py
pip install "thekaveh-nnx[onnx-dynamo]"      # 15_qat_classifier.py (Phase-6 dynamo export)
pip install "thekaveh-nnx[embeddings]"       # 13_train_domain_embedder.py
pip install "thekaveh-nnx[lm]"               # ordered_logits_pipeline.py, 11_tinystories_lm.py, 12_quantize_int8.py (quantized_generative_subtype), 17_export_transformer_to_gguf.py, 18_export_ollama_bundle.py, 22_dpo_synthetic_preferences.py
pip install "thekaveh-nnx[gguf-write]"       # 17_export_transformer_to_gguf.py, 18_export_ollama_bundle.py
pip install "thekaveh-nnx[viz]"              # 21_viz_attribute_xai.py
pip install "thekaveh-nnx[graph]"            # graph_classification_offline.py, graph_optional_splits.py, link_prediction_offline.py, the graph helper in 02_resume_training.py
pip install "thekaveh-nnx[vision]"           # preprocessing_offline.py, 16_ijepa_image_plumbing.py --cifar
pip install "thekaveh-nnx[optuna]"           # search_offline.py
pip install "thekaveh-nnx[jev]"              # decision_jev.py
```

Working from a git checkout instead of PyPI? See [CONTRIBUTING.md §1](../CONTRIBUTING.md#1-getting-set-up) for the editable + dev install.

**Which examples are evidence.** `tests/test_examples_smoke.py` imports every
example, runs many of them end to end and executes their bounded helpers.
Examples that are only imported are references, not evidence that they work.
Examples 01, 05 and 26 are the registered scenarios of the
[feature-composition matrix](../docs/feature-composition.md) (`EX-01`, `EX-05`,
`EX-26`). To check one:

```bash
python scripts/check_feature_composition.py --check   # EX-01's node ids are collected
python -m pytest "tests/test_examples_smoke.py::test_representative_examples_run_end_to_end[01_synthetic_classification.py]"
```

## 2. Catalog

Grouped by topic, from foundational to most specialized. §2.1 also holds the offline, decision, graph and evaluation examples that have no section of their own. The numbered prefixes record the order in which the examples were added, not their place below.

Each table gives a short summary per example. The notes after a table list each example's helpers, what they check and how the tests run them. In the notes, `-k <selector>` means `tests/test_examples_smoke.py -k <selector>`.

### 2.1. Core training loop

| Example | What it demonstrates |
|---|---|
| `01_synthetic_classification.py` | Train a feed-forward classifier on synthetic data (labels from a fixed random projection) with `EarlyStopping` and `LRMonitor`, then load the BEST checkpoint and predict. Helpers cover a native NLL loss and a precision option. |
| `02_resume_training.py` | Warm-resume training from a prior run's LAST checkpoint with optimizer, scheduler, scaler, epoch, and RNG state preserved. Bounded helpers cover checkpoint probing, BEST re-election, iterable and graph resume, AMP, callbacks and provenance. |
| `benchmark_compile.py` | Opt-in `torch.compile`, measured: times eager and compiled copies of the same weights and writes a profiler trace. No speedup is claimed. |
| `builder_branching.py` | Branching mutable params builders with `copy()` and `from_params()`, including a shared named-`Trainer` base that branches into two sibling runs. Offline, CPU. |
| `graph_classification_offline.py` | Graph-level classification with the registered recipe `graph_classifier_spec(...)` (GCN encoder → mean pooling → linear head) over synthetic graphs with stable ids. Offline, CPU. |
| `link_prediction_offline.py` | Leakage-aware link prediction with the registered recipe `link_predictor_spec(...)` (GCN encoder, dot decoder), selecting BEST by validation AUROC. Offline, CPU. |
| `graph_optional_splits.py` | Full-graph (`sampler="full"`) node classification on a six-node in-memory PyG graph whose validation mask is empty. Requires the `graph` extra. |
| `ddp_supervised.py` | Single-node data parallelism under `torchrun` with `train(distributed=DDP())`, including an injected-failure mode. CPU over Gloo. |
| `decision_benchmark_offline.py` | A reproducible decision-provider benchmark. Samples are collected once from a `FixedHeadProvider`, written to JSONL, then replayed offline into a sliced report. Offline, CPU. |
| `decision_fixed_head.py` | Typed decisions from a local model: deterministic `NNModel` heads answer a `Choice`, an ordinal `Score` and a `Boolean` through `FixedHeadProvider`. Offline, CPU. |
| `decision_jev.py` | Typed decisions on Jev models through the TypeSafe SDK, over an offline `httpx2.MockTransport`. Requires the `jev` extra; offline, no credentials, no client built at import. |
| `decision_jobs.py` | Deferred decision jobs: questions are collected, mapped and run in batched provider calls, with continuations, failure reporting, concurrency and cancellation. Offline, CPU. |
| `decision_model_pilot.py` | Label-conditioned decision-model research pilot (experimental), with a mechanics mode and an empirical mode. See `docs/decision-model-pilot.md`. |
| `decision_nli.py` | A local label-conditioned baseline: `NLIProvider` scores texts against candidate descriptions supplied at inference. Fully offline. |
| `custom_module.py` | Arbitrary modules and registered model factories: train from a registered `Encoder` factory, and train a wrapped keyword-only module in place. Offline, CPU. |
| `history_journal.py` | Bounded training history with an append journal (`HistoryJournal`), covering resume, reload, the notebook chart and CSV export. Offline, CPU. |
| `optimizer_factories.py` | Registered optimizer factories and AdamW: a custom factory and a built-in `Optims.ADAMW` each own a disjoint parameter group in one CPU `Trainer` update. |
| `abstention_offline.py` | Abstention policies from supplied arrays: select a threshold on validation, save and reload the policy, and apply it to test rows. Offline, CPU, no model needed. |
| `calibration_offline.py` | Fitted temperature calibration from supplied arrays, with save and reload and before/after reporting on held-out data. Offline, CPU, no model needed. |
| `preprocessing_offline.py` | Train-only preprocessing: a standardizer fitted on training rows only, and separate training and evaluation transforms over one base dataset. Offline, CPU. |
| `result_boundaries.py` | An opt-in `Result` at two fallible boundaries, each written with `try` / `except` and with `nnx.result`, with identical results both ways. Offline, CPU. |
| `run_bundle.py` | Portable, data-only run bundles: export, inspect, validate, reconstruct and resume a run. Offline, CPU; every run and bundle is written under a temporary directory. |
| `compare_seeds.py` | Multi-seed summaries and a paired comparison of two configurations, saved and reloaded as strict JSON. Offline, CPU. |
| `experiment_plan.py` | Immutable fluent experiment plans: branch, validate, probe, fit and resume an `ExperimentPlan`. Offline, CPU; every run is written under a temporary directory. |
| `prediction_stream.py` | Streaming prediction with `iter_predict` and mergeable metrics with `StreamingMetrics`, including a streaming validation step. Offline, CPU; runs under a temporary directory. |
| `ranking_offline.py` | Query-grouped ranking: retrieve candidates, train a pairwise scorer with `RankingTask.objective()` and evaluate held-out queries per complete query. Offline, CPU. |
| `regression_task.py` | Task adapters: `NNModelParams(task=TaskSpec.regression(2))` makes the default step, `evaluate()` and `predict()` regression-aware. Also shows a multilabel batch. Offline, CPU. |
| `scheduler_clocks.py` | Scheduler clocks: per-update and per-epoch stepping with gradient accumulation, non-finite skips and a resumed one-cycle schedule. Offline, CPU, temporary working directory. |
| `search_offline.py` | Budgeted experiment search with Optuna over an `ExperimentPlan`, with pruning and a stored study. Requires the `optuna` extra. |
| `split_replay.py` | Reproducible splits: `plan_split` manifests that are saved, reloaded and replayed, for grouped and time-ordered fixtures. Offline, CPU. |
| `tabular_validation.py` | Tabular data admission: `NNTabularDataset` rejects values that a dtype cast would corrupt, then computes a loss on a valid frame. Offline, CPU. |
| `03_custom_metrics.py` | Plug a custom `metric_fn(Y, Y_hat)` into `NNTrainParams.extra_metrics`; inspect `idp.train_edp.extra` and `idp.val_edp.extra`. A bounded helper adds named metrics and a monitor. |
| `04_onnx_export.py` | Export a trained model to ONNX and label what was shown: `checker-only` or `executed`. The recorded, tested profiles are `scripts/check_export_conformance.py` ([Export conformance](../docs/export-conformance.md)). |
| `25_conv_classifier.py` | LeNet-style conv classifier via `NNConvParams` + `Nets.CONV`, with a checkpoint round-trip through `resolve_from_state`. Synthetic stripes/checkerboard imagery — no download. |
| `26_custom_eval_step.py` | Train a non-classification paradigm end-to-end: a regression `train_step_fn` paired with `eval_step_fn(EvalStepContext) -> NNEvaluationDataPoint`, a custom MSE/MAE val pass. |

Notes for section 2.1:

**`01_synthetic_classification.py`**

- Predicts with `predict()` and with the opt-in `predict_proba(X, ProbabilitySpec(kind="categorical", class_axis=1, labels=...))`. It asserts equal logits/classes and probability rows summing to 1.
- It also asserts sample ids aligned with the input rows (also through a batched loader). It asserts an identical confusion matrix (`sklearn.metrics.confusion_matrix`) from `class_indices` + `spec.labels`, and that Bernoulli indicators refuse `class_indices`.
- Tests: run end to end by `tests/test_examples_smoke.py` (CPU, temporary working directory).
- Bounded `native_nll_workflow()` (executed by `tests/test_examples_smoke.py -k native_nll_workflow`, CPU, temporary working directory): trains one epoch with `Losses.NEGATIVE_LOG_LIKELIHOOD` and reloads BEST. It checks the reported loss against an explicit log-softmax/NLL reference while `predict().logits` stays raw.
- Precision option `precision_workflow(mode="fp32")` (also `--precision bf16` on the command line): trains under `PrecisionPolicy(mode)`. It prints the run's precision record and checks float32 predictions and parameters.
- Bounded CPU FP32 is the default (`tests/test_examples_smoke.py -k precision_workflow`). BF16 is gated on `precision_support`. It runs where the device verifies or supports it (CPU qualifies; `tests/test_examples_smoke.py -k precision_example`). Where it does not, it is skipped and reported.

**`02_resume_training.py`**

- Setup helpers `_make_model_and_loader`, `_base_optim` and `_base_sched` are also reused by `tests/test_atomic_write_text.py::test_resume_sample_text_save_failure`. That test injects a text-save fault into a resumed run. It checks that the source run stays loadable and resumable with no temporary file left behind.
- Bounded `checkpoint_probe_first_fit()` (executed by `tests/test_examples_smoke.py -k checkpoint_probe_first_fit`, CPU, temporary working directory): probes missing training/optimizer/combined state for the prospective run. It asserts no run directory was reserved, then fits once with overwrite disabled and reloads LAST.
- Bounded `overwrite_best_recovery()` (`-k overwrite_best_recovery`, temporary root only): commits two runs and overwrites the winner with a worse result. It verifies `runs/best` re-elects the survivor, whose BEST loads for inference.
- Bounded `iterable_graph_resume()` (`-k iterable_graph_resume`): resumes from a reusable list of batches and from an in-memory five-node full-batch graph. It checks resumed epoch numbers and seed-only evaluation. The graph branch needs `thekaveh-nnx[graph]` and is skipped without it.
- Bounded `amp_resume_compatibility()` (`-k amp_resume_compatibility`): fits and resumes with `mixed_precision=True` on CPU. There AMP is a documented no-op, so the training-state sidecar records `scaler=None`.
- Only when a CUDA device is present, `amp_resume_compatibility()` repeats the cycle with AMP enabled. It then checks the restored `torch.amp.GradScaler` scale / growth tracker. Without CUDA that branch is reported as skipped, never faked.
- Bounded `callback_continuation()` (`-k callback_continuation`): splits a run with validation losses `[0.6, 0.4, 0.41, 0.42]` after the third epoch. The resumed `EarlyStopping` stops at the same epoch as the uninterrupted run. It reports `run.resume_status` and warm-starts a second session with `resume_mode="weights_only"`.
- Opt-in, bounded `provenance_mode()` (`-k provenance_mode`): records an `ExperimentManifest` (a verified content digest of the training tensors) on a parent fit and its resume. It asserts the parent's `provenance.json` is byte-identical after the resume. The child attempt is distinct and links the parent attempt and checkpoint generation, and both share one fingerprint.

**`benchmark_compile.py`**

- Trains a small classifier one epoch with `train(compile=CompileSpec())` (inductor). It reloads its LAST checkpoint eagerly (no `_orig_mod.` keys, identical logits).
- `compare_compile` times eager and compiled copies of the same weights. It reports the first call (compilation included) separately from warmed mean / median / stdev and throughput.
- `profile_forward` writes a finite-schedule `torch.profiler` trace to `--output-dir`. The script prints separate `compile`, `warmed` and `profile` sections.
- No speedup is claimed. It covers FP32 built-in nets only; not with DDP or quantization.
- Run `python examples/benchmark_compile.py --device cpu --warmup 3 --repeats 10`. It needs a working C++ toolchain for inductor.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`, which runs exactly that command.

**`builder_branching.py`**

- `copy()` splits a partial `NNOptimParams.builder()` base (clipping + a head param group) into AdamW and SGD variants. `from_params()` rebuilds an equal builder from a built value.
- A shared named-`Trainer` base branches into two sibling CPU runs with distinct `data_id`s; one adds a StepLR scheduler. Each trains one update, gets a distinct run id and reloads with `NNRun.load`. The shared `DataLoader` keeps its identity.
- Also round-trips an `NNTransformerParams` with per-layer `activations` / `dropout_probs` through `NNTransformerParamsBuilder.from_params` and `from_state`.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `builder_branching_workflow()` (`-k builder_branching`, temporary working directory).

**`graph_classification_offline.py`**

- Builds a `GraphCollection` of synthetic 2–9-node graphs with stable graph ids, one flagged unlabeled. It is split by id into train / validation / test.
- Trains with `TaskSpec.categorical(..., ignore_index=IGNORE)`, so records count labeled graphs.
- Reloads the BEST checkpoint through the recipe and predicts the test graphs. Output has one row per graph, keyed by graph id, identical from a shuffled loader. Accuracy is checked against `evaluate()`.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `graph_classification_workflow()` (`-k graph_classification`) and an end-to-end subprocess run.

**`link_prediction_offline.py`**

- `split_links` turns a synthetic two-community graph into a replayable edge-split manifest. It uses canonical undirected edges and fixed held-out negatives from the complement.
- `LinkTask` builds candidate batches over the training topology only. The recipe trains with the task's objective and selects BEST by `MonitorSpec("auroc")` over exact, materialised validation AUROC.
- The model reloads from the run and scores the test candidates with stable ids. An injected leak (a test positive in the message graph) is refused.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `link_prediction_workflow()` (`-k link_prediction`) and an end-to-end subprocess run.

**`graph_optional_splits.py`**

- `NNGraphDataset` publishes `val_loader=None` (resolved `batch_sizes=(3, 0, 3)`, state `"0"`). `NNModel.train` skips validation, and an `eval_step_fn` spy proves it.
- `val_edp=None` persists through `NNRun.load`, and LAST is committed. `EarlyStopping` monitors `train_edp.loss`, and the test split is evaluated on its seed rows only.
- Synthetic data: no download and no neighbor sampling (no pyg-lib / torch-sparse).
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `graph_optional_splits_workflow()` (`-k graph_optional_splits`, temporary working directory).

**`ddp_supervised.py`**

- Run `torchrun --standalone --nproc_per_node=2 examples/ddp_supervised.py --device cpu --epochs 2 --output ddp-out`. It trains six rows with stable ids (0-5) through `nnx.distributed.train_loader` / `validation_loader` and `train(distributed=DDP())`.
- On macOS, if the host name resolves to an unreachable address, use the loopback instead: `GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 examples/ddp_supervised.py --device cpu --epochs 2 --output ddp-out`.
- Each rank prints its validation ids (together 0-5 exactly once), the shared run id and whether it owns the artifacts (rank 0 only). It then leaves the process group with `nnx.distributed.shutdown()`.
- `--inject-failure` makes rank 1 raise in epoch 1. The launch exits non-zero within the process-group timeout, leaving no worker behind.
- CUDA claims need a real two-GPU run.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (imported, and run under torchrun with and without `--inject-failure`).

**`decision_benchmark_offline.py`**

- A three-species `Choice` over feature rows, its perturbations and a held-out "farm" family are collected once. They come from a `FixedHeadProvider` with an explicit `Budget` and are written to JSONL.
- The perturbations are reordered options, new descriptions, and a distractor outside the head's label space. Each keeps its original's grouping unit.
- The replay runs with sockets disabled and no provider. Records join samples by sample id and question digest.
- The report shows every slice (in-family, held-out, family, perturbation) with coverage counts and unsupported reasons. It gives accuracy, macro-F1, exact NLL, Brier and ECE. It adds a grouped bootstrap interval with its seed, and JSON / CSV exports.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `decision_benchmark_offline_workflow()`, `-k decision_benchmark`).

**`decision_fixed_head.py`**

- A deterministic three-class `NNModel` (explicit weights, `TaskSpec` labels) answers a `Choice`. The `Choice`'s bookkeeping ids (`sp-cat` …) map onto the head's labels through `FixedHeadProvider(option_map=...)`.
- Prints `model_view()` (no ids), the digest and the validated per-photo distributions in the question's order.
- Shows an unseen label rejected before the model is called, and `validate_response` reordering another provider's keyed output.
- Then answers an ordinal `Score` from a second head whose class order is the level order (`expected_index`), and a `Boolean` from a one-logit head.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `decision_fixed_head_workflow()`, `-k decision_fixed_head`).

**`decision_jev.py`**

- Real SDK clients answer a `Choice`, a `Boolean` and a `Score` about each text in one `system_one` request per text. Answers are realigned to each question's option order.
- Each result records the model the service resolved for the requested alias, the request id and token usage. A Choice's or Score's result also records `provider_confidence`: Jev's certainty, not a correctness probability.
- A rate-limited request surfaces once as `JevRateLimited` with its request id. The SDK's `RetryPolicy` owns retries.
- The same questions run through the async client.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (bounded helper `decision_jev_workflow()`, which skips without the extra, `-k decision_jev`).

**`decision_jobs.py`**

- Five independent `Boolean` questions over the same texts are collected and mapped with zero provider calls. A deterministic local provider answers two questions per call, so they run in `ceil(5 / 2) = 3` calls in collection order.
- A `then` continuation maps the topic answers into follow-up state (only the sports texts) and asks one more question once.
- A failing call raises `JobFailed` with the completed, failed and skipped ids. The continuation waiting on it never runs.
- `arun` repeats the run with two calls at once. A `cancel` event returns a cancelled result with nothing sent.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `decision_jobs_workflow()`, `-k decision_jobs`).

**`decision_model_pilot.py`**

- The model is a registered candidate scorer (shared encoder, shared scalar head; no new `Nets` member). It scores `(state, question, candidate description)`.
- `--mode mechanics` (default, offline, CPU) runs the pre-execution manifest and trains on the fit families only. It selects on the select families and scores held-out families once.
- Mechanics mode also writes replay records with candidate ids and the schema digest, and reloads the checkpoint through the registry. It writes a stop record when the run reaches the compute cap (the bounded helper caps it at 48 updates). It ends with a **no-go** report (mechanics evidence only).
- `--mode empirical` writes and validates its manifest, checks its resources and writes a **blocked** report. The report names any missing resource; with all present, it says the empirical runs are outside this repository.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (bounded helper `pilot_workflow()`, `-k decision_model_pilot`).

**`decision_nli.py`**

- The NLI model and tokenizer are caller-supplied. Here they are a deterministic word-overlap stand-in, fully offline.
- A `Choice` normalizes entailment logits across the candidates (`log(3)` / `0` → `(0.75, 0.25)`).
- A `Boolean` takes each text's own contradiction-vs-entailment softmax (`0` / `log(4)` → `0.8`).
- Reports accuracy, macro-F1 and categorical NLL / Brier on a small labelled split. The report records the split, the model revision and the provider's settings (`calibrated: false`).
- A `Score` (unsupported) and an invalid request are refused with the call counter at zero.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `decision_nli_workflow()`, `-k decision_nli`).

**`custom_module.py`**

- Registers an `Encoder` factory and trains `NNModel(params=NNModelParams(net=ModelSpec("examples.encoder", 1, {"width": 16}, seed=7)))`.
- Prints the run (descriptor instead of built-in dims) and reloads it with `NNRun.load`. It rebuilds the LAST checkpoint through the registry with identical predictions.
- Shows that an unregistered factory raises `MissingModelFactoryError`.
- Then wraps a keyword-only `KeywordEncoder` instance with `KeywordInputs(("features", "mask"), target="labels")` and trains it in place (`model.net is encoder`). It reloads its runtime-only (`reconstructible=False`) checkpoint into a supplied module.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `custom_module_workflow()`, `-k custom_module`).

**`history_journal.py`**

- Runs three epochs of five batches with `HistoryJournal(retention=3, chunk_size=2)`. `ctx.idps` and the returned `NNRun.idps` hold the last three records, and the journal holds all fifteen. `idps.csv` is not written.
- A continuation resumed from the run's LAST reads no history and writes its own run directory and chunks. The source stays byte-for-byte unchanged.
- `NNRun.load` reads only the committed tail, and the notebook chart reads per-epoch rows.
- `export_history_csv` writes the legacy CSV layout. `lineage=True` puts the parent's committed prefix before the child's records (25 rows, each epoch once).
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `history_journal_workflow()`, `-k history_journal`).

**`optimizer_factories.py`**

- `register_optimizer_factory` + `NNOptimFactoryParams(OptimizerFactorySpec(id, version, config))` give a custom SGD factory the `layers.0.*` group. A built-in `Optims.ADAMW` (decoupled weight decay) owns the disjoint `layers.1.*` group.
- Asserts the factory is called once with one resolved group and that both groups moved.
- Then unregisters it and reloads the run metadata offline with `NNRun.load`.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `optimizer_factories_workflow()` (`-k optimizer_factories`, temporary working directory).

**`abstention_offline.py`**

- `select_threshold(kind="max_probability", risk_ceiling=0.1, split_id="val", test_split_id="test")` evaluates every candidate threshold on validation only. It prints each one's accepted / incorrect counts, ending in the curve-only accept-none endpoint.
- The chosen policy (field, threshold, labels, model id, tuning split) is saved to JSON in a temporary directory. It is reloaded unchanged, then applied to test rows.
- Prints the accepted ids, coverage over every row and selective risk over the accepted ones. It also prints each abstained row's prediction, score and reason.
- An all-rejected set has its risk reported as unavailable, never 0.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `abstention_offline_workflow()`, `-k abstention_offline`).

**`calibration_offline.py`**

- In the calibration split, 3 of every 4 rows are right at logit margin 2: the model says 0.88 where 0.75 is right.
- `fit_temperature(split_id="calibration", train_split_id="train", test_split_id="test")` fits `T = 2 / ln 3 ≈ 1.82`. It records that ids cannot prove disjointness.
- The calibrator (labels, model id, fit config, split ids) is saved as JSON in a temporary directory. It is reloaded with identical probabilities.
- The reloaded calibrator reports held-out NLL / Brier / reliability bins before and after (`improved`). A held-out set the model already got right is reported as `worsened`, never as success.
- A reordered label list is refused before any transform. A different model id is accepted only under a named override that is recorded.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `calibration_offline_workflow()`, `-k calibration_offline`).

**`preprocessing_offline.py`**

- Fixture S `[[0,5],[2,5],[1000,5]]` goes through `NNTabularDataset(split=..., standardize=True)`. It fits on rows 0–1 only (mean `[1,5]`, scale `[1,1]`; the constant column gets scale 1).
- The held-out `1000` therefore standardizes to `999`, and the source DataFrame is unchanged.
- A tiny model trains on the prepared features. The standardizer is saved, reloaded without refitting and applied exactly once to raw rows. This reproduces the training features bit for bit, because `NNModel.predict` takes model-ready input.
- Fixture V: `NNDataset(train_transform=..., eval_transform=...)` gives a random-offset training view and a `+1` evaluation view over one untouched base dataset. Its `transform` stays `None`, labels pass through, and evaluation reads repeat exactly.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `preprocessing_offline_workflow()`, `-k preprocessing_offline`).

**`result_boundaries.py`**

- A batch of plain-data decision requests includes one with duplicate option ids and one with an empty prompt. `validate_decision_request_result(...).bind(decide_result(...))` carries the invalid ones as `Err(invalid_decision_request)`. The provider is called once, for the valid request only.
- Three handed-over bundle paths (valid, missing, malformed) are summarized by `inspect_bundle_result(...).map(...).recover(...)`. `artifact_missing` / `bundle_invalid` come back as data, never unpickling a checkpoint.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `result_boundaries_workflow()`, `-k result_boundaries`).

**`run_bundle.py`**

- A built-in classifier (labelled categorical task) and a temperature calibrator fitted on its weights are exported with `export_bundle`. `inspect_bundle` summarizes the bundle and `validate_bundle` checks it.
- `reconstruct_bundle` rebuilds a model that predicts bit-for-bit like the trained one. The calibrator keeps its labels and weight fingerprint. `resume()` for one epoch equals the uninterrupted three-epoch run.
- A registered module's bundle names its missing factory before building anything. It then rebuilds from caller-supplied factories and resumes likewise.
- A flipped byte in a payload is refused before any tensor is read.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `run_bundle_workflow()`, `-k run_bundle`).

**`compare_seeds.py`**

- A stratified `plan_split` fixes train / validation / test membership once. Its digest is recorded in each run's provenance, and the test split is never read.
- Three seeds of two configurations (a narrow and a wide hidden layer) each train in their own run root. Each uses a fresh model, optimizer, callbacks and loader generator. One run is interrupted after its first epoch, a failed attempt.
- `observations_from_runs` reads the validation loss at the last committed epoch back without loading a model.
- The `ComparisonReport` shows each configuration's count, mean and sample standard deviation. The failed attempt has no value in the statistics; it is listed and counted as failed.
- The report also shows the paired `wide - narrow` deltas per seed and a bootstrap interval labelled seed variability. It then saves and reloads as strict JSON.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `compare_seeds_workflow()`, `-k compare_seeds`).

**`experiment_plan.py`**

- An `ExperimentPlan` holds network, model and training parameters, data factories and a seed. `with_epochs(1)` / `with_epochs(3)` branch a base plan and leave it and each other unchanged.
- `validate()` reports every problem of a broken plan at once (`net`, `train`, `data.train`, `seed`) without reading a loader or building a model.
- `probe(batch)` runs one no-grad forward pass on a temporary model and restores the ambient RNG.
- `fit()` matches the equally seeded imperative `set_seed(7); NNModel(...).train(...)` on weights and history.
- Two fits of one plan are two attempts with two run directories. `resuming(parent_id)` continues in a child run that records the parent as its lineage. The parent's artifacts stay byte-for-byte intact.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `experiment_plan_workflow()`, `-k experiment_plan`).

**`prediction_stream.py`**

- `iter_predict(loader)` streams a categorical task's predictions in `[2, 2, 1]`-row batches, in loader order. Concatenated, they equal `predict_proba()`, sample ids included. A regression task's stream equals `predict()`'s continuous values.
- Leaving the `with` block after one batch closes the stream and restores the network's training mode. The loader stays reusable.
- `StreamingMetrics` accumulates NLL and Brier score per batch. Two halves merged in either order equal one pass.
- `train(eval_step_fn=streaming_eval_step)` builds each validation record from counts and sums. It matches the default record on uneven, partly masked regression batches.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `prediction_stream_workflow()`, `-k prediction_stream`).

**`ranking_offline.py`**

- Retrieves a top-12 candidate set per query by inner product: FAISS `IndexFlatIP`, or the same search in NumPy without `faiss`. FAISS **positions** map to stable document **IDs**.
- Trains a small pairwise scorer with `RankingTask.objective()` and selects BEST by `MonitorSpec("ndcg_at_5")` over `task.metric_specs()`. It reloads the BEST checkpoint; the task configuration is checkpointed.
- Evaluates held-out queries per complete query, ID-aligned. Shuffled rows give identical MRR@k / Recall@k / NDCG@k.
- Queries whose candidate set holds no relevant document are excluded and counted. The record says the candidate sets were sampled.
- Prints the re-ranked metrics next to the index's own order.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `ranking_offline_workflow()` (`-k ranking_offline`) and an end-to-end subprocess run.

**`regression_task.py`**

- A two-target fit with one NaN (masked) target runs per-epoch validation. Records carry valid-target counts, `mse` / `mae` and no fabricated classification fields.
- The run and checkpoint reload carry the task. `predict()` returns continuous values and `predict_proba()` returns `probabilities=None`.
- In a multilabel batch, subset accuracy (0.5) differs from element accuracy (0.75).
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `regression_task_workflow()` (`-k regression_task`) and an end-to-end subprocess run.

**`scheduler_clocks.py`**

- Two epochs of five microbatches run with `accumulate_grad_batches=2`. A `StepLR` steps 6 times on `NNSchedulerParams(clock="optimizer_update")` and twice on the default epoch clock.
- `LRMonitor.history` keeps one LR per epoch, and `LRMonitor.update_history` the per-update LRs.
- An objective with `nonfinite="skip"` drops one non-finite window, so its scheduler steps 5 times.
- A one-cycle schedule over 12 updates, split 2 + 2 epochs across a resume, follows the uninterrupted run's LR sequence exactly.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `scheduler_clocks_workflow()`, `-k scheduler_clocks`).

**`search_offline.py`**

- A base `ExperimentPlan` has a validation-loss monitor and data as factories. `with_lr(plan, params)` returns a new plan per trial.
- `nnx.search.search(..., budget=SearchBudget(trials=3))` asks a grid of three learning rates. A `ThresholdPruner` prunes the one whose first validation loss is above 1.0.
- The best completed trial names its run and BEST checkpoint. Searching the stored SQLite study again asks no new trial.
- Prints where Optuna (`study.db`) and NNx (`runs/`) artifacts live.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (bounded helper `search_workflow()`, `-k search_offline`).

**`split_replay.py`**

- Fixture G has ids `s0..s5`, groups `a,a,b,b,c,c`, proportions `(1/3, 1/3, 1/3)` and seed 7. `plan_split` gives three two-row splits with one group each.
- The manifest is saved, reloaded and fed to `NNTabularDataset(split=..., id_col="id")` on the rows in reverse order. Every id stays in its split, with no `random_split` and no global RNG draw.
- Fixture T has times `[1,2,2,3,4,5]` and cutoffs `(3, 5)`. It gives `3/2/1` rows with gap 0 and `1/1/1` with gap 1 (excluded `s1, s2, s4`).
- The reloaded manifest has the same digest, which `ExperimentManifest(splits={...})` records.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `split_replay_workflow()`, `-k split_replay`).

**`tabular_validation.py`**

- `NNTabularDataset` rejects `±inf` in a selected feature in source precision before an int64/bool cast. The cast would absorb it into an INT64 extreme or `True`.
- It rejects finite values outside an integer dtype's range (`300` → `int8` would wrap).
- It rejects finite values that overflow `float16` during conversion, for both features and a floating `target_dtype` target.
- It ignores non-finite values in unselected columns. On rejection, it leaves the DataFrame and global RNG untouched.
- A valid frame then gives a classification loss that matches cross-entropy computed by hand from the logits. A regression dataset keeps `(N, 1)` targets.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py`; bounded helper `tabular_validation_workflow()` (`-k tabular_validation`, temporary working directory).

**`03_custom_metrics.py`**

- Bounded `named_monitor_workflow()` (also run by `main`; `tests/test_examples_smoke.py -k named_monitor_workflow`, CPU, temporary working directory): declares `nll` / `brier` / `accuracy` metrics. It adds a validation-NLL `MonitorSpec` beside a decoded callable.
- It runs on uneven batches (`[16, 16, 16, 2]` training, `[8, 8, 5]` validation) and shares the monitor with `EarlyStopping`.
- It reloads the run and checks the reported value against an independent recomputation from the BEST model. It also checks that BEST is the last epoch the monitor marked as improved.

**`04_onnx_export.py`**

- The `conformance(model, path, X)` helper always runs `onnx.checker`. Without `onnxruntime` it reports `checker-only`: well formed, **no parity claimed**.
- It reports `executed` when ONNX Runtime ran the graph on CPU and compared every output with the model at `rtol=1e-4, atol=1e-5`.
- Bounded `registered_module_variant()` (also run by `main`; `tests/test_examples_smoke.py -k "registered_module_variant or onnx_sample"`, CPU): does the same for a model built from a registered factory (`ModelSpec`).

**`25_conv_classifier.py`**

- Shows conv-stack arithmetic helpers (`spatial_sizes()`/`flatten_dim()`), per-layer FC `activations`/`dropout_probs` overrides and image-vs-flat input equivalence.
- Bounded `conv_integral_schema_roundtrip()` (no training): builds the params from NumPy-integer shapes. It checks they normalize to plain-`int` YAML-portable state with an identical rebuilt forward. It shows fractional/boolean counts rejected before allocation.

**`26_custom_eval_step.py`**

- Without a task, the default step's argmax metrics do not fit continuous targets. Plain regression can instead declare `TaskSpec.regression`; see `regression_task.py`.
- The custom val pass's metrics persist per-epoch in the run history (`idp.val_edp`, MAE riding in `extra`).
- Bounded `nonfinite_metric_workflow()` (executed by `tests/test_examples_smoke.py -k nonfinite_metric_workflow`, CPU, temporary working directory): scripts NaN/inf/absent validation metrics and asserts the finite fallback. It covers plateau inputs, the selected BEST, raw diagnostics, and CSV NaN→`None` readback.

### 2.2. `train_step_fn` hook

| Example | What it demonstrates |
|---|---|
| `05_custom_train_step_autoencoder.py` | Use `train_step_fn` to replace the supervised step with a reconstruction-loss step (a tiny 16 → 4 → 16 autoencoder). |

### 2.3. Fine-tuning

| Example | What it demonstrates |
|---|---|
| `06_finetune_with_layer_freezing.py` | Transfer-learning mechanics: pretrain on one random-label draw, export the weights and load them into a fresh model. `freeze("layers.0.*", "layers.1.*")` then fine-tunes only the head on a second draw. Nothing is learnable, so this shows the recipe, not a gain. |
| `07_lora_finetuning.py` | Parameter-efficient fine-tuning via LoRA: `apply_lora_to(net, "layers.*", r=4, alpha=8)`, fine-tune on a new distribution and verify every base parameter is bit-exactly unchanged. Save a LoRA-only checkpoint and compare its size to the full state-dict. |

Notes for section 2.3:

**`07_lora_finetuning.py`**

- Bounded `dora_zero_row_composition(dtype=torch.float16)` (executed by `tests/test_examples_smoke.py -k dora_zero_row_composition` in FP16 and FP32, CPU): wraps a two-layer net with mixed zero/nonzero base rows via `apply_dora_to`. It converts the net to half after wrapping.
- It checks a finite forward through the following projection and one optimizer step on a float32 loss reduction. It also checks the frozen base and a full `state_dict()` reload including `magnitude`.
- Bounded `peft_alias_preflight()` (`-k peft_alias_preflight`, CPU, offline): registers one `Linear` under two names beside an independent head. `apply_lora_to` rejects both the wildcard and the second alias, naming both. Every weight, identity and `requires_grad` flag stays unchanged.
- `peft_alias_preflight()` then wraps only the head and takes one optimizer step that trains exactly its adapter.
- Bounded `peft_eval_injection()` (`-k peft_eval_injection`, CPU, offline, temporary directory): puts the classifier in `eval()` *before* injecting nonzero-dropout LoRA adapters and loading trained adapter weights.
- `peft_eval_injection()` asserts bit-identical repeated `predict` output, an unchanged all-eval mode map and an adapter file with no mode field. A later `train()` re-activates the adapter dropout.
- Bounded `peft_preconverted_base()` (`-k peft_preconverted_base`, CPU): converts a tiny base to float64 *before* `apply_lora_to`. It takes one adapter-only SGD step with no second `.to()`, and asserts adapter placement and base immutability.
- Bounded `lora_artifact_roundtrip()` (`-k lora_artifact_roundtrip`, CPU, temporary directory): wraps a net whose first layer is *named* `lora_A_projection`. It inspects the exact keys `save_lora_weights` writes: adapter-owned tensors only, no `*.base.*`.
- `lora_artifact_roundtrip()` reloads into a matching wrapped base and checks load count, output parity and base identity. An adapter file carries only the trainable delta, not a resumable model checkpoint.

### 2.4. Alternative paradigms

| Example | What it demonstrates |
|---|---|
| `08_diffusion_2d_mixture.py` | DDPM-style diffusion on a 2D mixture of 4 Gaussians: `NoiseSchedulers.LINEAR` + `DiffusionMLP` + `diffusion_train_step_factory` + reverse-diffusion `sample()`. `--objective` trains with `diffusion_objective` instead. |
| `09_gan_with_trainer.py` | Multi-optimizer training via `nnx.trainer.Trainer`: a tiny GAN on a 1D mixture of Gaussians, with disjoint optimizers for `G` and `D` scoped via `NNParamGroupSpec`. The ten-epoch `main()` is not run by the smoke tests. |
| `10_knowledge_distillation.py` | Hinton-style KD: pretrain a wider teacher, then distill into a much smaller student (~4% of the teacher's parameters) via `kd_train_step_factory`. Verifies the teacher's weights are frozen across the student's training. |
| `14_moe_classifier.py` | Sparse top-k Mixture-of-Experts as a first-class model type. `NNMoEParams(num_experts=4, top_k=2)` + `Nets.FEED_FWD_MOE` builds a `FeedFwdMoENN` (every hidden layer an `MoELinear`), trained via `moe_train_step_factory` (supervised loss + Switch-style load-balancing aux). Reports the param-count breakdown and round-trips the MoE params through a checkpoint. Prints the aux loss before and after training as a routing diagnostic (not a guarantee on a tiny problem). |

Notes for section 2.4:

**`08_diffusion_2d_mixture.py`**

- Bounded `objective_mode()` (`tests/test_examples_smoke.py -k objective_mode`, CPU): accumulates uneven microbatches through the objective. It checks that a `sample` preview with its own generator leaves the objective's generator and the net's mode untouched.

**`09_gan_with_trainer.py`**

- Bounded `trainer_builder_snapshot()` (executed by `tests/test_examples_smoke.py -k trainer_builder_snapshot`, CPU, temporary working directory): builds a G/D config and mutates the builder afterwards. It runs the captured config for two minibatches and checks optimizer names, scoped updates and the saved descriptor.
- Bounded `two_optimizer_resume()` (`-k two_optimizer_resume`): trains the GAN for one epoch. It resumes for one more in a freshly initialised model under a different seed through `NNTrainerParamsBuilder.resume_from`. `gan_step` still owns every optimizer update.
- `two_optimizer_resume()` checks that the G / D weights match an uninterrupted two-epoch run and that `run.resume_status` lists the restored `EarlyStopping`. It also checks that the final LAST checkpoint carries both named optimizers, both schedulers and the component.
- Bounded `manual_scheduler_ownership()` (`-k manual_scheduler_ownership`): trains with `auto_step_schedulers=False`, an `optimizer_update`-clock schedule for G and an epoch-clock one for D. The step reports every update, yet only its own `scheduler.step()` calls advance a schedule.

**`10_knowledge_distillation.py`**

- Bounded `objective_mode()` (also run by `main`; `tests/test_examples_smoke.py -k objective_mode`, CPU): distills through `kd_objective`. It uses uneven microbatches `[2, 1]` and `accumulate_grad(4)`, a window cut short by the epoch's end.
- It checks the single committed update against a full-batch reference step, and the teacher's tensors against their snapshot.

### 2.5. Quantization

| Example | What it demonstrates |
|---|---|
| `12_quantize_int8.py` | Post-training quantization (PTQ): train a feed-forward classifier, call `nnx.quantize.quantize_int8(model)` once, print the val-accuracy change and the state-dict sizes, and export the quantized model to ONNX. No calibration data, no retraining. Requires `pip install "thekaveh-nnx[quantize,onnx]"`. |
| `15_qat_classifier.py` | Quantization-aware training (QAT 8da4w via torchao): combine `qat_train_step_factory` and `QATLifecycleCallback` to fake-quant during training, then real-quant on convert. Verifies the saved LAST checkpoint holds the CONVERTED int4 state (scales/zeros on disk) and reloads it with `NNModel.from_checkpoint`, which replays the recorded prepare→convert. Requires `pip install "thekaveh-nnx[quantize,onnx-dynamo]"`. |

Notes for section 2.5:

**`12_quantize_int8.py`**

- Bounded `quantized_generative_subtype()` (executed by `tests/test_examples_smoke.py -k quantized_generative_subtype`; needs `[quantize,lm]`, skipped otherwise; no ONNX): quantizes a one-layer `GenerativeNNModel` with a locally trained BPE tokenizer.
- It generates two tokens on the cached and full-recompute paths. `quantize_int8` returns the same class it was given.

### 2.6. Embeddings + FAISS export

| Example | What it demonstrates |
|---|---|
| `13_train_domain_embedder.py` | Train a tiny text embedder from scratch on synthetic `(sentence, paraphrase)` pairs via NT-Xent contrastive loss. Embed a corpus, export to a FAISS index, and query the index for self-similarity. End-to-end demo of `nnx.embeddings.train_contrastive` + `export_to_faiss`. Requires `pip install "thekaveh-nnx[embeddings]"`. |

### 2.7. Language modeling

| Example | What it demonstrates |
|---|---|
| `11_tinystories_lm.py` | Decoder-only LM end-to-end: train a tiny BPE tokenizer and build a `TransformerNN`. Train next-token prediction with the causal-LM task, then sample with `GenerativeNNModel.generate()` (KV-cache enabled by default). CPU-friendly. Requires `pip install "thekaveh-nnx[lm]"`. |
| `ordered_logits_pipeline.py` | Ordered logits pipelines: stage order, immutable edits, serialization and seeded generation. Requires the `lm` extra; offline, CPU. |

Notes for section 2.7:

**`ordered_logits_pipeline.py`**

- The order oracle: on logits `[2, 1, 0]`, temperature 0.5 then top-p 0.8 keeps one finite logit, and the reverse keeps two. Each equals `apply_chain` over the same list.
- Immutable `append` / `prepend`: a duplicated temperature equals one combined stage. A zero-temperature stage is refused anywhere but last.
- A versioned `state()` / `from_state()` round trip serializes a custom stage through a registered `LogitsStageCodec`.
- Seeded `generate(logits_pipeline=...)` on a tiny local-tokenizer transformer matches the explicit processor list on the cached and uncached paths.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (bounded helper `ordered_logits_pipeline_workflow()`, `-k ordered_logits_pipeline`). It is not in the extras-free end-to-end list, since step 4 needs the `lm` extra.

**`11_tinystories_lm.py`**

- `CausalLMTask(alignment="pre_shifted")` supplies the objective: cross-entropy over valid tokens, normalized by the window's valid-token count.
- The task's validation step reports unsmoothed token NLL, perplexity and token accuracy, with no placeholder classification metrics.
- It uses an inline corpus by default; pass `--use-hf` to download TinyStories.
- Bounded `manual_attention_lm_example(dtype=torch.bfloat16)` (executed by `tests/test_examples_smoke.py -k manual_attention_lm_example` in BF16 and FP32, CPU; skipped without `[lm]`): runs one attention-dropout training update. It uses a one-layer `d_model=16` model with a four-token local batch, then greedily generates two tokens on the cached and full paths.
- `build_lm_train_params(lr, ...)` exposes the CLI configuration. The bounded `lm_config_rejects_nonfinite()` check (`-k lm_config_rejects_nonfinite`, no tokenizer/training/downloads) shows a NaN learning rate rejected with a `ValueError` naming `max_lr`. The finite control keeps its state.
- Bounded `causal_lm_task_workflow()` (`-k causal_lm_task_workflow`, offline, no `--use-hf`): trains one epoch on inline-corpus `pre_shifted` windows with validation. It reloads the run and its LAST checkpoint (task configuration and token count included) and generates greedily from the reloaded model.

### 2.8. Self-supervised pretraining

| Example | What it demonstrates |
|---|---|
| `16_ijepa_image_plumbing.py` | I-JEPA image-path plumbing on synthetic images by default (`--cifar` opts into CIFAR-10): a small `ViTNN` context encoder predicts masked-patch latents against an EMA target encoder. Demonstrates `jepa_train_step_factory` + `JEPAPredictor` + `build_target_encoder` + `random_block_mask`. |

Notes for section 2.8:

**`16_ijepa_image_plumbing.py`**

- The smoke-tested `float64_vit_predictor_step()` helper runs one bounded double-precision forward/backward on tiny synthetic images (no download). It checks every gradient is finite FP64.
- `--objective` trains with `jepa_objective` instead. The bounded `objective_mode()` helper (smoke-tested, CPU) accumulates uneven microbatches and checks the EMA target advances once per committed update.

### 2.9. Experimental GGUF export

| Example | What it demonstrates |
|---|---|
| `17_export_transformer_to_gguf.py` | Build a tiny `TransformerNN` + BPE tokenizer, write an NNx-tagged `.gguf`, and inspect it with `gguf.GGUFReader`. Points to a `llama-quantize` from a llama.cpp revision or patch that supports `nnx_transformer` (docs/gguf.md §4) for sub-F16 quantization. Stock llama.cpp-derived runtimes do not implement the NNx architecture. Requires `pip install "thekaveh-nnx[gguf-write,lm]"`. |
| `18_export_ollama_bundle.py` | Generate `model.gguf` + a Modelfile (`FROM` / `PARAMETER` / `SYSTEM` / `TEMPLATE`) as an experimental bundle fixture. Stock Ollama cannot run `nnx_transformer`; use only with a compatible patched runtime. Requires `pip install "thekaveh-nnx[gguf-write,lm]"`. |

### 2.10. Pruning + surgery

| Example | What it demonstrates |
|---|---|
| `19_prune_synthetic_classifier.py` | Magnitude prune a small synthetic-data classifier at 50% sparsity (`bake=True` keeps state_dict keys intact), evaluate the pruned accuracy, then briefly fine-tune to recover. Demonstrates `nnx.prune.magnitude_prune`. |
| `20_low_rank_surgery_ffn.py` | Train a wide FFN, low-rank-factorize the widest Linear at rank=8 via `nnx.surgery.low_rank_factorize`, then refine to recover accuracy. Shows the caller is responsible for swapping the returned `nn.Sequential` back into the `ModuleList`. |

Notes for section 2.10:

**`20_low_rank_surgery_ffn.py`**

- Bounded `surgery_freeze_roles()` (executed by `tests/test_examples_smoke.py -k surgery_freeze_roles`, CPU, no files): factorizes a weight-frozen / bias-trainable layer. It builds fresh strict parameter groups and takes one update. It asserts the factors stay frozen and unchanged while only the intended bias and the untouched layer move.
- Bounded `widen_supported_workflow()` (`-k widen_supported_workflow`, CPU, temporary working directory): shows eval-mode parity through ReLU, and the rejection of a Softmax boundary with the source untouched. It then shows the honest reconstruction path: widen → `dataclasses.replace` the params → fresh `NNModel` → one training epoch → BEST reload.
- Bounded `deepen_override_workflow()` (`-k deepen_override_workflow`, CPU, temporary working directory): deepens a `FeedFwdNN` with explicit mixed dropout overrides. It checks that the returned `params` describe the deeper topology (aligned overrides, zero dropout at the new site) with eval parity.
- `deepen_override_workflow()` then rebuilds a fresh `NNModel` from `deeper.params`, fits one epoch and reloads BEST.
- Bounded `named_deepen_workflow()` (`-k named_deepen_workflow`, CPU, no files): deepens a named `nn.Sequential` at the dotted key `relu`. It checks the preserved original keys plus the generated `_nnx_deepen_*` keys and head-path lookup. It also checks `freeze` / parameter-group selection and a state-dict round trip.
- Bounded `recipe_reconstruction_workflow()` (`-k recipe_reconstruction_workflow`, CPU, temporary files only) records the change instead. Its `TransformRecipe` applies LoRA on `layers.0`, then a rank-4 factorization of `layers.1`.
- The recipe refuses an optimizer built before it. It is materialized and trains a few steps with an optimizer built after it. `NNModel.from_checkpoint` rebuilds it from a checkpoint with SVD patched to fail: same keys, trainability and predictions.
- The manual loop in `main()` remains: unrecorded surgery is still refused before persistent training.

### 2.11. Explainability

| Example | What it demonstrates |
|---|---|
| `21_viz_attribute_xai.py` | Captum-backed input attribution via `nnx.viz.attribute(method=...)` — runs `integrated_gradients`, `saliency`, `input_x_gradient`, and `deep_lift` on a trained classifier. Requires `pip install "thekaveh-nnx[viz]"`. |

### 2.12. LM follow-ons

| Example | What it demonstrates |
|---|---|
| `22_dpo_synthetic_preferences.py` | DPO preference fine-tuning of a tiny `TransformerNN` against synthetic `(prompt, chosen, rejected)` triples using `dpo_train_step_factory`; reference policy frozen via `copy.deepcopy`. Requires `pip install "thekaveh-nnx[lm]"` for `main()`. |

Notes for section 2.12:

**`22_dpo_synthetic_preferences.py`**

- `build_preference_dataset()` wraps the `NNPreferenceDataset` construction.
- Bounded `dpo_sample_batch_sizes()` (offline counting stub tokenizer, no `lm` extra, no training): shows `batch_sizes`. `None` = one full-split batch, positive = explicit mini-batch, zero rejected before tokenization.

### 2.13. Distillation variants

| Example | What it demonstrates |
|---|---|
| `23_born_again_distillation.py` | Iterated self-distillation across G=3 generations via `born_again_train`. Each generation distills from the previous via Hinton-style KD (Furlanello et al., ICML 2018) and prints its validation accuracy. A 5-epoch synthetic run does not reproduce the paper's result. |
| `24_feature_kd.py` | FitNets-style feature distillation via `feature_kd_train_step_factory` with one paired teacher→student auxiliary layer (shape-matched: teacher `layers.1` output 32 → student `layers.0` output 32). |
| `offline_teacher_distillation.py` | Offline teacher distributions: stored teacher probabilities are written to JSONL, aligned in a `TeacherDataset` and used to train and evaluate a student. Offline, CPU. |

Notes for section 2.13:

**`offline_teacher_distillation.py`**

- A teacher's stored probabilities over three candidates are exported in two different candidate orders. Each export carries sample and question ids, teacher revision, schema digest, semantics and declared training rights.
- They are written to and read back from JSONL. A `TeacherDataset` joins them to the student's inputs by sample id and aligns them by candidate id.
- The student trains with `dataset.objective(alpha=0.7)`: `KL(teacher ‖ student)` plus a hard cross-entropy at temperature 1, no `T²`. `dataset.eval_step()` validates each epoch.
- The LAST checkpoint is reloaded. `evaluate_offline` reports teacher agreement, imitation loss and labelled accuracy / NLL, each with its own denominator, beside how often the teacher itself was right.
- Tests: unnumbered, so it is registered explicitly in `tests/test_examples_smoke.py` (end-to-end run and bounded helper `offline_teacher_distillation_workflow()`, `-k offline_teacher`).
