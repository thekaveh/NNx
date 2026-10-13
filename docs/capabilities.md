# 17. Capability reference

This page lists the main capabilities of NNx, grouped by task. Each entry names
its entry points and links to the guide or example that shows it. It describes
the current source; an older PyPI release can lack some entries (see
[Release or source](../README.md#21-release-or-source)). For installation and a
first trained model, see the [README](../README.md#2-install) and the
[Quickstart](quickstart.md). Extras named here are installed as
`pip install "thekaveh-nnx[<extra>]"`.

## 1. Training loop

- **Supervised loop.** `NNModel` builds the network from frozen configuration
  objects and runs `train()`, `evaluate()` and `predict()`. Callbacks, early
  stopping, gradient clipping, gradient accumulation and seeded reproducibility
  are built in.
- **Schedulers.** The `Schedulers` enum provides `REDUCE_LR_ON_PLATEAU`,
  `STEP`, `COSINE_ANNEALING`, `ONE_CYCLE` and `LINEAR_WARMUP_DECAY`. A
  scheduler steps once per epoch by default, or once per committed optimizer
  update with `NNSchedulerParams(clock="optimizer_update")`.
- **Precision.** `PrecisionPolicy` selects FP32, FP16 or BF16 explicitly.
- **Metrics and monitors.** `NNTrainParams.extra_metrics` accepts any
  `callable(y_true, y_pred) -> float`. Declared metrics (`MetricSpec`) and one
  named monitor (`MonitorSpec`) drive BEST selection, plateau scheduling and
  an `EarlyStopping` given the same monitor.
- **Objectives.** `LossTerm`s with explicit denominators run over one shared
  update engine, which gives accumulation, clipping and the precision policy to
  `NNModel` and `Trainer` alike. Built-in objectives: `supervised_objective`,
  `kd_objective`, `diffusion_objective` and `jepa_objective`.
- **Custom steps.** The `train_step_fn` hook replaces the per-batch supervised
  step with your own function. Use it for autoencoders, VAEs, recommendation
  or other methods without changing NNx internals.
- **Multi-optimizer `Trainer`.** `nnx.trainer.Trainer` parallels
  `NNModel.train()` for disjoint optimizers, such as GAN generator and
  discriminator or actor-critic. Each entry of its name-keyed optimizer
  dictionary is scoped by an `NNParamGroupSpec`; the groups must partition the
  parameters.
- **Optimizers.** `Optims` provides SGD, Nesterov SGD, Adam, AMSGrad and AdamW.
  Any other `torch.optim.Optimizer` is a registered factory
  (`nnx.register_optimizer_factory` with `NNOptimFactoryParams`). `run.yaml`
  stores only the factory's id, version and config, so runs reload offline. See
  [Concepts §3.1](concepts.md#31-optimizers-adamw-registered-factories-and-the-shared-build-hook).
- **Callbacks.** The `Callback` base class has `on_{train,epoch}_{begin,end}`
  hooks. Built in: `EarlyStopping`, `LRMonitor`, `ModelCheckpoint` (custom
  epoch tags), `TensorBoardCallback` (`tensorboard` extra) and `WandbCallback`
  (`wandb` extra). A legacy `Callable[[List[IDP]], None]` is still accepted.
- **Data-parallel training.** `model.train(params, distributed=nnx.distributed.DDP())` runs
  one process per device under `torchrun` on one machine. NNx partitions the
  data and keeps every update equal to the union batch's, and one rank writes
  the run. See
  [Quickstart §3.10](quickstart.md#310-data-parallel-training-on-one-machine).
- **Opt-in compilation.** `model.train(params, compile=CompileSpec())` compiles
  the built-in step's FP32 forward pass; checkpoints, export and later
  inference stay eager. It records requested and effective state and never
  guarantees a speedup. `nnx.benchmarking` separates first-call compile cost
  from warmed latency. It is not available with DDP or quantization. See
  [Concepts §28](concepts.md#28-opt-in-compilation-and-benchmarks-nnxcompilation-nnxbenchmarking).

## 2. Runs, checkpoints and resume

- **Content-addressed runs.** `NNRun` saves `run.yaml`, `idps.csv` and
  `metadata.yaml` under `runs/<id>/`, where `id` is the MD5 of the
  configuration's `state()`.
- **Checkpoints.** `NNCheckpoint` saves the FIRST, Q1, Q2, Q3, LAST and BEST
  tags. LAST is the epoch commit marker: a failed LAST write rolls the history
  back, and `NNRun.load()` drops history newer than LAST after an interrupted
  process.
- **Warm resume.** Each checkpoint names a generation-addressed training-state
  sidecar. A resume restores the optimizer, scheduler, scaler, loader
  generators and every RNG stream; a torn or corrupted resume point is refused
  before anything is restored. See [Concepts §14](concepts.md#14-resuming-training).
- **Bounded history.** `history=nnx.history.HistoryJournal(retention, chunk_size)` keeps a
  bounded window in memory and appends each record once to
  `runs/<id>/history/`, with lazy tail reads and CSV export.
- **Experiment plans.** An immutable `ExperimentPlan` branches a configuration
  (`with_epochs`, `with_seed`, `with_data`, ...). `validate()` reports every
  problem without reading data. `probe(batch)` runs an eval-mode no-grad
  forward pass, plus a train-mode loss pass when the batch has a target. It
  then restores the ambient state. `fit()` trains a distinct attempt. See
  `examples/experiment_plan.py`.
- **Budgeted search** (`optuna` extra). `nnx.search.search(plan, space, ...)`
  runs a sequential Optuna study over an experiment plan; each trial fits a new
  plan in its own run directory. See
  [Concepts §29](concepts.md#29-budgeted-experiment-search-nnxsearch).
- **Provenance.** `train(..., provenance=ExperimentManifest(...))` records the
  declared task, labels, model, data and split identities with a SHA-256
  fingerprint and one attempt record per fit. `nnx.provenance.compare` diffs
  two plans without loading models.
- **Run bundles and the Hub.** `nnx.bundles` exports a checkpoint as a
  verified, portable bundle. With the `hub` extra, `NNModel` supports
  `save_pretrained`, `push_to_hub` and `from_pretrained`, and checkpoints can
  be saved as safetensors. See [HuggingFace Hub](hub.md).

## 3. Tasks

- **Task adapters.** `NNModelParams(task=TaskSpec.regression(n))`,
  `.multilabel(n, threshold=...)` or `.categorical(n, ignore_index=...)`
  validates batches before any update and masks NaN or ignored targets out of
  the loss and metrics. The loss is averaged over valid targets. See
  `examples/regression_task.py`.
- **Ranking.** `nnx.ranking.RankingTask` trains a scorer with a pairwise
  logistic objective within each query. It reports MRR@k, Recall@k and NDCG@k
  per complete query. See `examples/ranking_offline.py`.
- **Graph classification.** `nnx.graph_tasks` classifies whole graphs from a
  `GraphCollection` with stable graph ids and mean or sum pooling. See
  `examples/graph_classification_offline.py` (`graph` extra).
- **Link prediction.** `nnx.link_tasks` splits edges into a replayable
  manifest, passes messages over training edges only and reports exact AUROC
  and AP. A held-out positive edge in a message graph is refused before any
  update. See `examples/link_prediction_offline.py` (`graph` extra).
- **Your own modules.** `NNModel` trains any `torch.nn.Module`. Register a
  factory (`register_model_factory`, `ModelSpec`) for a model that rebuilds
  from checkpoints, safetensors and the Hub, or pass an instance with
  `NNModel(module=m, ...)` for runtime-only use. See `examples/custom_module.py`.

## 4. Training methods

- **Diffusion (DDPM).** `nnx.diffusion` provides LINEAR and COSINE noise
  schedules and a conditional MLP denoiser (`DiffusionMLP`). Train with a step
  factory or `diffusion_objective`, and sample with `sample`.
- **Distillation.** `kd_train_step_factory` mixes Hinton-style soft and hard
  losses with a frozen teacher. `feature_kd_train_step_factory` adds an MSE
  term between named intermediate activations. `born_again_train` iterates
  self-distillation over generations.
- **Offline teacher distributions.** `nnx.paradigms.offline_distillation`
  distills from stored teacher probabilities, aligned by sample and candidate
  id, without a live teacher.
- **Contrastive and augmentation.** SimCLR (NT-Xent), Mixup and CutMix step
  factories.
- **Mixture of Experts.** Sparse top-k routing with a Switch-style
  load-balancing loss, and a drop-in `nnx.MoELinear`.
- **I-JEPA.** Masked-patch latent prediction against an EMA target encoder,
  with `JEPAPredictor`, `build_target_encoder`, `update_ema`,
  `random_block_mask` and a small `ViTNN` encoder. See [I-JEPA](jepa.md).
- **DPO.** Preference-pair fine-tuning against a frozen reference policy. See
  [DPO](dpo.md).

Paradigm and diffusion step factories share a NaN guard and gradient
clipping.

## 5. Fine-tuning and model transforms

- **Transfer learning.** `nnx.finetune` freezes layers by glob pattern, loads
  external state dicts with optional key remapping and sets per-group learning
  rates through `NNOptimParams.param_groups`.
- **Parameter-efficient fine-tuning.** `nnx.peft` provides LoRA, DoRA, IA3,
  prefix tuning, prompt tuning and adapter layers. Each `save_*_weights`
  function saves only the trainable delta (DoRA's `magnitude` needs the full
  `state_dict()`). LoRA starts from the pretrained
  behavior because its second factor is zero-initialized.
- **Quantization** (`quantize` extra). `nnx.quantize.quantize_int8(model)`
  applies torchao INT8 weight-only post-training quantization without
  calibration data and returns the same model class. QAT 8da4w uses
  `qat_train_step_factory` with `QATLifecycleCallback`.
- **Pruning.** `nnx.prune.magnitude_prune` applies checkpoint-safe
  unstructured masks; `semi_structured_24` applies 2:4 sparsity through
  torchao.
- **Model surgery.** `nnx.surgery` provides `widen` and `deepen`
  (function-preserving Net2Net edits on validated topologies), `drop_layer`,
  `low_rank_factorize` (SVD) and `expand_embedding`. See
  [Model surgery](surgery.md).
- **Recorded recipes.** `nnx.transforms.TransformRecipe` records LoRA and SVD
  low-rank transforms in every checkpoint and Hub save, so `from_checkpoint`
  and `from_pretrained` rebuild the transformed topology.

## 6. Networks, language models and export

- **Networks.** `FeedFwdNN`, `FeedFwdMoENN`, `ConvNN`, `GraphConvNN`,
  `GraphSageNN`, `GraphAttNN`, `TransformerNN` and `ViTNN` (`Nets.VIT` with
  `NNViTParams`). The graph networks need the `graph` extra.
- **Causal language-model task.** `nnx.lm_tasks.CausalLMTask` provides the
  next-token objective and evaluation for a `TransformerNN`, reporting token
  NLL, perplexity and accuracy. See
  [Quickstart §3.6](quickstart.md#36-causal-language-models).
- **Language modeling.** `TransformerNN` is a decoder-only model with
  RMSNorm, RoPE, SwiGLU, tied embeddings and a KV cache.
  `GenerativeNNModel.generate()` supports greedy, top-k, top-p and
  repetition-penalty sampling. The `lm` extra adds the BPE tokenizer. See
  [Language modeling](lm.md).
- **KV-cache scope.** A CPU regression test requires cached decoding to be at
  least 1.2× faster than full recomputation for its fixed 256-token workload.
  Results on other hardware and workloads vary. Past `max_seq_len`, the cache
  rebuilds every step.
- **Decoding order.** `LogitsChain.builder()` sorts logit processors into the
  canonical order. `OrderedLogitsPipeline` runs stages exactly as declared,
  through `generate(logits_pipeline=...)`.
- **ONNX export.** `NNModel.to_onnx(path, example_input)` uses the legacy
  `torch.onnx.export`, which needs the `onnx` extra. `dynamo=True` uses the
  `torch.export`-based exporter, which also needs `onnxscript` (`onnx-dynamo`
  extra). `nnx.export_conformance` checks numerical parity in ONNX Runtime
  (`onnx-runtime` extra). See [Export conformance](export-conformance.md).
- **Experimental GGUF** (`gguf-write` extra). `nnx.interop.write_gguf` writes
  an NNx-tagged GGUF file. Stock llama.cpp, Ollama and LM Studio do not
  implement its architecture. See [Experimental GGUF export](gguf.md).
- **Embeddings** (`embeddings` extra). `nnx.embeddings` trains a text embedder
  from `(anchor, positive)` pairs and exports a FAISS index for downstream
  retrieval. See [Embeddings](embeddings.md).

## 7. Data, evaluation and decisions

- **Datasets.** `NNDataset` (torchvision, `vision` extra), `NNGraphDataset`
  (PyTorch Geometric, `graph` extra), `NNTabularDataset` (pandas) and
  `NNPreferenceDataset` (DPO preference triples).
- **Reproducible splits.** `nnx.data_splits.plan_split` plans group,
  chronological and stratified splits into a JSON `SplitManifest`, which
  `NNTabularDataset` replays by sample id. See `examples/split_replay.py`.
- **Train-only preprocessing.** `nnx.preprocessing.Standardizer` fits on the
  training rows only and reloads from JSON. See
  `examples/preprocessing_offline.py`.
- **Probabilities.** `NNModel.predict_proba(X, ProbabilitySpec(...))` returns
  probabilities for an explicitly declared task; `predict()` is unchanged.
- **Streaming prediction and metrics.** `NNModel.iter_predict(loader)` yields
  predictions batch by batch with bounded memory, and `nnx.streaming`
  provides mergeable metrics. See
  [Concepts §21](concepts.md#21-streaming-prediction-and-mergeable-metrics-nnxstreaming).
- **Calibration.** `nnx.calibration.fit_temperature` fits one temperature on a
  named calibration split and reports NLL, Brier and reliability before and
  after. See `examples/calibration_offline.py`.
- **Abstention.** `nnx.abstention` accepts a prediction only above a
  probability or margin threshold; `select_threshold` tunes it against a risk
  ceiling. See `examples/abstention_offline.py`.
- **Multi-seed comparisons.** `nnx.comparison` summarizes repeated runs
  without pooling incompatible observations and pairs two configurations by
  replicate key. See `examples/compare_seeds.py`.
- **Typed decisions.** `nnx.decisions` asks `Choice`, `Boolean` or `Score`
  questions and returns validated, labelled probabilities. Providers serve a
  trained classifier, a local NLI model or hosted Jev models (`jev` extra). See
  [Typed decisions](decisions.md).

## 8. Configuration, diagnostics and typing

- **Configuration objects.** `NNParams`, `NNModelParams`, `NNTrainParams`,
  `NNOptimParams`, `NNSchedulerParams` and `NNTrainerParams` are frozen
  dataclasses that round-trip through `state()` and `from_state()`. A field at
  its default is left out of `state()`, so adding a field does not change
  existing run ids.
- **Builders.** `NNOptimParams.builder()`, `NNSchedulerParams.builder()`,
  `NNTransformerParams.builder()` and `NNTrainerParams.builder()` expose one
  method per valid variant, such as `.adam(...)` or `.one_cycle(...)`, so an
  invalid combination cannot be built. `.copy()` branches a Builder and
  `Builder.from_params(params)` rebuilds an existing value. See
  `examples/builder_branching.py`.
- **Enum factories.** `Nets`, `Losses`, `Optims`, `Schedulers`, `Activations`,
  `Devices` and `NoiseSchedulers` construct their objects when called.
  `Checkpoints` names the checkpoint tags.
- **Reproducibility.** `nnx.set_seed(seed)` seeds every RNG and cuDNN;
  `nnx.dataloader_worker_init_fn` seeds loader workers.
- **Learning-rate finder** (`plots` extra). `nnx.lr_finder` runs an
  exponential sweep, suggests a one-cycle `max_lr` and restores the model and
  RNG state afterwards.
- **Run plots** (`plots` extra). `VisUtils` returns Plotly figures (confusion
  matrix, line and scatter plots, t-SNE of checkpoint logits) and a
  classification-report DataFrame.
- **Model internals** (`viz` extra). `nnx.viz` provides `summary`,
  `weight_histogram`, `activation_map`, `attribute` (Captum), `gradient_flow`
  and `netron_export`, which also needs the `onnx` extra.
- **Errors as values.** `nnx.result` provides opt-in `Ok` and `Err` values
  for code that prefers to branch on a value; every NNx API still raises. See
  [Concepts §27](concepts.md#27-errors-as-values-nnxresult).
- **Typing.** NNx ships a PEP 561 `py.typed` marker, so `pyright` and `mypy`
  use its annotations.
