# 11. API Reference

Generated from runtime signatures and source docstrings as portable Markdown so the same reference is readable in the repository, on the documentation site, and in the wiki. Sections are ordered from foundational APIs to specialized modules.

## 1. Top-level package

#### `nnx`

```python
module nnx
```

nnx — lightweight PyTorch training / eval / visualization toolkit.

**Details**

```text
The package is organized under `nnx.nn` (model, params, datasets, enums, nets,
callbacks) and two top-level helpers (`nnx.utils.Utils`, `nnx.vis_utils.VisUtils`).
The curated re-exports below give a flat surface for the most common imports
without forbidding the deep paths existing notebook code relies on.
```

##### `nnx.__version__`

```python
nnx.__version__ = '0.2.3'
```

Exported value.

##### `nnx.set_seed`

```python
nnx.set_seed(seed: 'int', strict: 'bool' = False) -> 'None'
```

Pin every RNG that affects training and toggle cuDNN deterministic.

**Details**

```text
Args:
    seed: integer seed shared across Python `random`, NumPy, and PyTorch
        (CPU + CUDA). Also written to `os.environ["PYTHONHASHSEED"]`
        so DataLoader workers started via the `spawn` method (default
        on Windows + macOS/Py3.8+) inherit a deterministic hash seed.
        Note: the current Python interpreter's hash state was fixed at
        startup — this assignment only affects spawned subprocesses.
        For full hash determinism in the current process, set
        `PYTHONHASHSEED=<N>` in the shell BEFORE invoking Python.
    strict: when True also calls torch.use_deterministic_algorithms(True)
        and sets CUBLAS_WORKSPACE_CONFIG. Slower and may raise on ops
        that lack a deterministic CUDA implementation; opt in only when
        full bit-for-bit reproducibility matters.
```

##### `nnx.dataloader_worker_init_fn`

```python
nnx.dataloader_worker_init_fn(worker_id: 'int') -> 'None'
```

DataLoader `worker_init_fn` that pins each worker's numpy/python seed deterministically from the worker_id + the parent torch seed.

**Details**

```text
Pass as: `DataLoader(..., worker_init_fn=dataloader_worker_init_fn)`.
```

##### `nnx.env_snapshot`

```python
nnx.env_snapshot(force_refresh: 'bool' = False) -> 'dict'
```

Capture a snapshot of the runtime environment for reproducibility.

**Details**

```text
Returned dict is JSON-serializable. Includes Python / torch / numpy
versions, GPU info if any, OS, and the git commit hash if running
inside a git repo. Safe to call from anywhere — failures degrade to
`None` per field rather than raising.

Result is memoized within the process (versions/hardware don't
change between calls). Caveat: the ``git_commit`` / ``git_dirty``
fields are frozen at first call too, so a long session that commits
mid-run records the session-start git state in later runs'
metadata.yaml. Pass ``force_refresh=True`` to re-compute — useful
in tests that mutate the environment, or to re-stamp git state.
```

##### `nnx.LRFinderResult`

```python
class nnx.LRFinderResult(*, lrs: 'list[float]', losses: 'list[float]', suggested_lr: 'float', figure: 'go.Figure') -> 'None'
```

Result of an :func:`lr_finder` sweep.

**Details**

```text
Attributes:
    lrs: list of learning rates actually exercised. Length matches
        ``losses``. May be shorter than ``num_iter`` if the sweep
        early-exited due to loss divergence.
    losses: list of loss values, one per LR.
    suggested_lr: the recommended ``max_lr`` for a subsequent
        real training run — the LR at the steepest-descent
        point of the smoothed loss curve.
    figure: Plotly ``Figure`` plotting loss vs log(LR) with the
        suggested LR marked.
```

##### `nnx.lr_finder`

```python
nnx.lr_finder(model: 'nn.Module', train_loader: 'DataLoader', *, loss_fn: 'Callable[[torch.Tensor, torch.Tensor], torch.Tensor]', optimizer_cls: 'type[torch.optim.Optimizer]' = torch.optim.adam.Adam, start_lr: 'float' = 1e-07, end_lr: 'float' = 10.0, num_iter: 'int' = 100, diverge_threshold: 'float' = 4.0, device: 'Optional[torch.device]' = None, ema_alpha: 'float' = 0.5) -> 'LRFinderResult'
```

Sweep LRs exponentially from ``start_lr`` to ``end_lr`` and suggest a one-cycle ``max_lr``.

**Details**

```text
Args:
    model: the network to sweep against. ``model.train()`` is
        called internally; the original training-mode state, the
        weights, AND the RNG state are all restored on exit.
    train_loader: a DataLoader yielding ``(X, Y)`` batches the
        model can forward and ``loss_fn`` can score against. The
        loader is iterated, and if the sweep exceeds one epoch
        the loader is re-iterated from the start.
    loss_fn: callable ``(y_hat, Y) -> scalar Tensor`` for the
        per-batch loss. Same shape contract as torch loss
        functions.
    optimizer_cls: optimizer class. Adam by default; SGD also
        works for the sweep.
    start_lr: low end of the sweep range. Must be > 0.
    end_lr: high end of the sweep range. Must be > start_lr.
    num_iter: number of training iterations to run. Must be >= 2.
    diverge_threshold: stop the sweep early when the EMA-smoothed
        loss exceeds ``diverge_threshold * smoothed_min`` (the
        minimum EMA-smoothed loss observed so far). Default 4. The
        smoothed check matches fastai's lr_find heuristic — using
        the raw ``min(losses)`` would let a single anomalous low
        first-batch loss pull the threshold too tight and abort
        the sweep prematurely.
    device: device to move batches to. If None, inferred from
        the first model parameter.
    ema_alpha: smoothing coefficient for the loss curve before
        the steepest-descent search. Default 0.5.

Returns:
    :class:`LRFinderResult` with the raw sweep data, the suggested
    max_lr, and a Plotly figure of loss vs log(LR).

Raises:
    ValueError: on invalid arguments (``num_iter < 2``,
        ``start_lr <= 0``, ``end_lr <= start_lr``).
```


## 2. Orchestrators

### 2.1. NNModel — supervised orchestrator

#### `nnx.nn.nn_model.NNModel`

```python
class nnx.nn.nn_model.NNModel(net_params: 'NNParams', params: 'NNModelParams')
```

Top-level training/eval/predict wrapper around an ``nn.Module``.

**Details**

```text
Inherits from :class:`huggingface_hub.PyTorchModelHubMixin` (when the
``thekaveh-nnx[hub]`` extra is installed) to gain ``save_pretrained`` /
``push_to_hub`` / ``from_pretrained``. Without the extra installed,
those three methods raise a clear ImportError pointing at the extra;
no other NNModel functionality is affected.
```

##### `nnx.nn.nn_model.NNModel.to_onnx`

```python
nnx.nn.nn_model.NNModel.to_onnx(self, path: 'str', example_input: 'Union[torch.Tensor, tuple, np.ndarray]', input_names: 'Optional[list[str]]' = None, output_names: 'Optional[list[str]]' = None, dynamic_batch: 'bool' = True, opset_version: 'int' = 17, dynamo: 'bool' = False) -> 'str'
```

Export the underlying network to ONNX format.

**Details**

```text
Args:
    path: output filename (e.g., "model.onnx").
    example_input: a tensor (or tuple of tensors for multi-input
        nets) with realistic shape/dtype used to trace the network.
    input_names: optional list of human-readable input port names.
    output_names: optional list of human-readable output port names.
    dynamic_batch: when True (default), marks dim 0 as dynamic so
        the exported model accepts any batch size at inference.
    opset_version: ONNX opset to target. 17 is broadly supported
        by current runtimes.
    dynamo: when False (default), uses the legacy TorchScript-based
        `torch.onnx.export` path — plain `pip install onnx` is
        enough. When True, dispatches to PyTorch's new
        `torch.export`-based exporter (default in torch>=2.9,
        supports >2 GB models via external data, faster). The
        dynamo path requires `onnxscript`; install via
        `pip install thekaveh-nnx[onnx-dynamo]`.

Returns the path written. Network is put in eval mode for tracing.
```

##### `nnx.nn.nn_model.NNModel.from_checkpoint`

```python
nnx.nn.nn_model.NNModel.from_checkpoint(checkpoint: 'NNCheckpoint', device: 'Optional[Devices]' = None, **model_kwargs: 'Any') -> 'Self'
```

Rebuild a model, replay topology transforms, and load its weights.

**Details**

```text
Ordinary and legacy FP32 checkpoints have no transforms. Converted
QAT checkpoints replay their persisted torchao recipe before state
loading; unsupported recipes fail explicitly rather than constructing
a model with the wrong topology.
```

##### `nnx.nn.nn_model.NNModel.freeze`

```python
nnx.nn.nn_model.NNModel.freeze(self, *patterns: 'str') -> 'int'
```

Freeze parameters under ``self.net`` matching any of ``patterns`` (fnmatch globs against the dotted parameter name). Returns the number of parameters newly frozen.

**Details**

```text
Convenience wrapper around :func:`nnx.finetune.freezing.freeze`
— use the standalone function when freezing a module that isn't
``self.net`` (e.g., a custom decoder hanging off this model).
```

##### `nnx.nn.nn_model.NNModel.unfreeze`

```python
nnx.nn.nn_model.NNModel.unfreeze(self, *patterns: 'str') -> 'int'
```

Mirror of :meth:`freeze` — set ``requires_grad=True`` on matching parameters.

##### `nnx.nn.nn_model.NNModel.export_state_dict`

```python
nnx.nn.nn_model.NNModel.export_state_dict(self, path: 'str') -> 'str'
```

Save just ``self.net.state_dict()`` to ``path``.

**Details**

```text
The file is a plain ``torch.save`` of a state-dict — loadable by
any torch consumer without nnx installed, and by
:func:`nnx.finetune.load_pretrained` for the fine-tuning round-trip.
Companion to the NNCheckpoint format, which carries the params +
idp wrapper alongside the weights; ``export_state_dict`` strips
all of that and leaves just the weights.

Returns ``path`` so calls can be chained.
```

##### `nnx.nn.nn_model.NNModel.train`

```python
nnx.nn.nn_model.NNModel.train(self, params: 'NNTrainParams', callbacks: 'Optional[list[CallbackLike]]' = None, train_step_fn: 'Optional[TrainStepFn]' = None, eval_step_fn: 'Optional[EvalStepFn]' = None, salt: 'Optional[str]' = None) -> 'NNRun'
```

Train the model and return its persisted run history.

**Details**

```text
Args:
    params: Required loaders, optimizer/scheduler configuration,
        epoch count, persistence controls, and optional resume source.
    callbacks: Lifecycle callbacks invoked around training and epochs.
    train_step_fn: Optional per-batch override; the default performs
        supervised forward, loss, backward, and optimizer stepping.
    eval_step_fn: Optional once-per-epoch validation override that
        receives the complete validation loader.
    salt: Optional string folded into the run.id hash so identical
        (model, net, train) configs run as distinct experiments
        without altering modeled params. ``None`` (the default)
        preserves existing run.id hashes exactly.

Returns:
    The completed :class:`NNRun`, persisted with run metadata,
    iteration history, and configured checkpoints.

Raises:
    ValueError: If required training inputs are missing or invalid,
        the model is fully frozen, or resume state is incompatible.
    FileExistsError: If the content-addressed run already exists and
        ``overwrite_existing`` is false.
    FloatingPointError: If the default step encounters non-finite loss.

The run lease prevents another process using ``overwrite_existing``
from deleting or interleaving artifacts until final persistence ends.
```

##### `nnx.nn.nn_model.NNModel.evaluate`

```python
nnx.nn.nn_model.NNModel.evaluate(self, loader: 'DataLoader', extra_metrics=None) -> 'NNEvaluationDataPoint'
```

Aggregate predictions across all batches in `loader` and compute a single NNEvaluationDataPoint. Aggregating (rather than averaging per-batch metrics) gives correct sample-weighted f1/precision/recall when the final batch is short.

**Details**

```text
Raises ValueError if the loader yields zero batches — previously
produced NaN metrics silently from np.mean over an empty list.
```

##### `nnx.nn.nn_model.NNModel.predict`

```python
nnx.nn.nn_model.NNModel.predict(self, X) -> 'PredictResult'
```

Run the network in eval mode and return logits + argmax classes.

**Details**

```text
Accepts any of:

- ``np.ndarray`` (single input tensor) — historical API.
- ``tuple[np.ndarray, ...]`` — for multi-input networks.
- ``torch.Tensor`` / ``tuple[torch.Tensor, ...]`` — skips the numpy
  conversion when callers already have tensors.
- ``DataLoader`` — iterates the loader, runs predictions per batch,
  concatenates and returns the full result. Y labels in the batch
  (if present) are ignored.

Returns a ``PredictResult`` (a ``NamedTuple`` of (logits, classes))
that unpacks like the original 2-tuple.

Non-destructive: ``self.net.training`` is snapshotted before
switching to ``eval()`` and restored on exit (matches
``NNModel.evaluate``, ``nnx.viz.activation_map``, and
``nnx.lr_finder``). Without this, a caller doing the common
train → predict → train-more pattern silently leaves the net
in ``.eval()`` mode.
```


#### `nnx.nn.nn_model.PredictResult`

```python
class nnx.nn.nn_model.PredictResult(logits: 'np.ndarray', classes: 'np.ndarray')
```

Structured result of NNModel.predict().

**Details**

```text
Unpacks positionally as ``(logits, classes)`` so callers doing
``log, hat = model.predict(X)`` keep working after the upgrade from
the original 2-tuple. Field access (``result.logits``, ``result.classes``)
is preferred for new code.
```


#### `nnx.nn.nn_model.TrainStepContext`

```python
class nnx.nn.nn_model.TrainStepContext(model: 'NNModel', batch: 'Any', optimizer: 'torch.optim.Optimizer', scaler: 'Optional[torch.amp.GradScaler]', grad_clip_norm: 'Optional[float]', extra_metrics: 'Optional[Mapping[str, Callable]]', accumulate_grad_batches: 'int', batch_idx: 'int', epoch_idx: 'int', is_last_batch: 'bool' = False, accumulation_state: 'Optional[GradientAccumulationState]' = None) -> 'None'
```

Frozen bundle of state passed into a training-step function.

**Details**

```text
The default `default_train_step` runs the standard supervised
forward/backward/step. Users can pass their own
`train_step_fn: Callable[[TrainStepContext], NNEvaluationDataPoint]`
to NNModel.train() for non-supervised paradigms (autoencoder, VAE,
link prediction, recommendation, diffusion, etc.). The custom step
is fully responsible for forward, backward, optimizer.step,
gradient accumulation, AMP scale/unscale, grad clipping, and the
NaN/Inf guard — the context tells it what knobs are set; honoring
them is on the caller.
```


#### `nnx.nn.nn_model.TrainStepFn`

```python
type alias nnx.nn.nn_model.TrainStepFn
```

Public type alias.


#### `nnx.nn.nn_model.default_train_step`

```python
nnx.nn.nn_model.default_train_step(ctx: 'TrainStepContext') -> 'NNEvaluationDataPoint'
```

Standard supervised training step: forward → loss → backward → step.

**Details**

```text
This is the body that `NNModel.train()` runs when no custom
`train_step_fn` is supplied. It honors:
  - gradient accumulation (zero_grad at cycle start, step at cycle
    end). A trailing partial cycle is stepped at the epoch boundary;
    gradients use each loss's effective normalization weight.
  - AMP (unscales before grad clip; scaler.step + update at cycle end)
  - grad clipping by L2 norm
  - the NaN/Inf guard (raises FloatingPointError on divergent loss)
  - extra_metrics injection on the returned NNEvaluationDataPoint

Custom training-step functions can call this directly to layer on
behavior (e.g., extra logging) without reimplementing the standard
forward/backward dance.
```


#### `nnx.nn.nn_model.EvalStepContext`

```python
class nnx.nn.nn_model.EvalStepContext(model: 'NNModel', val_loader: 'DataLoader', extra_metrics: 'Optional[Mapping[str, Callable]]', epoch_idx: 'int') -> 'None'
```

Frozen bundle of state passed into a validation-step function (#86).

**Details**

```text
Mirrors :class:`TrainStepContext` for the per-epoch VALIDATION pass: users
can pass ``eval_step_fn: Callable[[EvalStepContext], NNEvaluationDataPoint]``
to ``NNModel.train()`` to replace the built-in classification ``evaluate()``
for non-classification paradigms (next-token LM perplexity, DPO margins,
regression MAE, ...). The step runs under ``torch.no_grad()`` and its
returned EDP becomes ``val_edp`` — recorded on the epoch's last idp and
persisted through the incremental run save like any built-in val metric.
```


#### `nnx.nn.nn_model.EvalStepFn`

```python
type alias nnx.nn.nn_model.EvalStepFn
```

Public type alias.


### 2.2. GenerativeNNModel — decoder-only LM orchestrator

#### `nnx.nn.generative_nn_model.GenerativeNNModel`

```python
class nnx.nn.generative_nn_model.GenerativeNNModel(net_params: 'NNParams', params: 'NNModelParams', tokenizer: 'Optional[NNTokenizerParams]' = None)
```

Language model with an autoregressive ``generate()`` method.

**Details**

```text
``tokenizer`` is held as a regular instance attribute (not a
constructor-arg of NNModel) so existing NNModel callers don't
have to know about it. It's required for ``generate()`` but
optional at construction — train-time you can build the model
first and attach the tokenizer later.
```

##### `nnx.nn.generative_nn_model.GenerativeNNModel.generate`

```python
nnx.nn.generative_nn_model.GenerativeNNModel.generate(self, prompt: 'str', *, max_new_tokens: 'int' = 64, temperature: 'float' = 1.0, top_k: 'Optional[int]' = None, top_p: 'Optional[float]' = None, repetition_penalty: 'float' = 1.0, stop: 'Optional[list[str]]' = None, seed: 'Optional[int]' = None, use_cache: 'bool' = True, logits_chain: 'Optional[LogitsChain]' = None, on_token: 'Optional[Callable[[int], None]]' = None) -> 'str'
```

Autoregressive decode from ``prompt``.

**Details**

```text
Args:
    prompt: input text. Encoded via ``self.tokenizer``.
    max_new_tokens: hard cap on new tokens emitted. Generation
        also stops if the context window (max_seq_len) would be
        exceeded and the model can't shrink the window further,
        or if a ``stop`` string is decoded.
    temperature: 0 means greedy (argmax). Higher values produce
        more diverse output. Routes through TemperatureScaling.
    top_k: keep only the top-k logits. None disables.
    top_p: nucleus (top-p) cutoff. None disables.
    repetition_penalty: divide previously-seen tokens' positive
        logits by this. 1.0 is no-op (default).
    stop: list of stop strings — generation halts once any of
        them appears in the decoded CONTINUATION (the prompt
        itself is not searched, so a prompt containing a stop
        string doesn't halt generation immediately; a stop
        string straddling the prompt/continuation boundary is
        likewise not detected — matching the generated-text-only
        convention HF uses).
    seed: when set, sampling is reproducible — two calls with
        the same seed + prompt + model produce identical output.
    use_cache: when True (default), uses an incremental KV
        cache — each new token only re-runs attention on the
        last position, not the whole prefix. When False, falls
        back to the full-recompute path (kept for regression
        testing). Both paths produce the same tokens for greedy
        decoding (sampling paths agree given the same seed).
    logits_chain: optional pre-built ``LogitsChain`` (see
        ``nnx.LogitsChain.builder()``). When provided, the
        inline chain construction from ``temperature`` /
        ``top_k`` / ``top_p`` / ``repetition_penalty`` kwargs
        is skipped — the supplied chain is used as-is.
        Power-user path for custom logit processors (e.g.,
        logit-bias for forbidden tokens). When ``None`` (the
        default), behavior is unchanged.
    on_token: optional callback invoked with each newly
        generated token id immediately after it is appended.
        Lets callers stream partial output or drive progress
        reporting without re-running decode over the public forward /
        apply_chain / sample_next_token primitives. ``None``
        (default) is a no-op so existing callers are
        unaffected. Fires only for newly generated tokens
        (not prompt tokens) and on both the cached and
        no-cache decode paths. Fires before any ``stop``
        string check so the callback observes every emitted
        token including the one that triggers a stop.

Returns:
    The full decoded string (prompt + generated continuation).

Non-destructive: ``self.net.training`` is snapshotted before
switching to ``eval()`` and restored on exit (including the
exception path via ``try/finally``). Matches the convention
used by ``NNModel.predict`` / ``NNModel.evaluate``,
``nnx.diffusion.sample``, ``nnx.embeddings.embed_texts``,
``nnx.viz.activation_map``, and ``nnx.lr_finder``.
```


### 2.3. Trainer — multi-optimizer orchestrator

#### `nnx.trainer.trainer.Trainer`

```python
class nnx.trainer.trainer.Trainer(model: 'NNModel')
```

Multi-optimizer training orchestrator.

**Details**

```text
Constructed around a single NNModel. At train() time, builds one
torch.optim.Optimizer per entry in NNTrainerParams.optims (each
scoped to its sub-net via NNOptimParams.param_groups) and invokes
the user-supplied trainer_step_fn for each batch.

Same NNRun + per-tag NNCheckpoint cadence as NNModel.train(),
with the extra `trainer` block on NNRun preserving the multi-optim
configuration on disk.
```

##### `nnx.trainer.trainer.Trainer.train`

```python
nnx.trainer.trainer.Trainer.train(self, params: 'NNTrainerParams', trainer_step_fn: 'TrainerStepFn', callbacks: 'Optional[list[CallbackLike]]' = None, salt: 'Optional[str]' = None) -> 'NNRun'
```

Run the multi-optimizer training loop and return the resulting NNRun.

**Details**

```text
Args:
    params: NNTrainerParams — train_loader + n_epochs + optims dict +
        (optional) schedulers dict + (optional) val_loader, seed,
        save_phase_checkpoints, extra_metrics. Schedulers step once
        per epoch by default; set auto_step_schedulers=False when the
        custom step function owns scheduler timing.
    trainer_step_fn: required. `Callable[[TrainerStepContext],
        NNEvaluationDataPoint]`. The function owns the entire per-batch
        update — including which optimizers to step, in what order, and
        with what loss(es). There is no supervised fallback.
    callbacks: optional list of Callback instances. The callback
        context exposes `ctx.optimizer` (primary, sorted-first), plus
        a `ctx.optimizers` dict and `ctx.trainer` reference for
        trainer-aware callbacks.
    salt: mirrors ``NNModel.train()``'s ``salt`` parameter — an
        optional string folded into the run.id hash so identical
        (model, net, train) configs run as distinct experiments
        without altering modeled params. ``None`` (the default)
        preserves existing run.id hashes exactly.

Returns:
    NNRun with per-iteration idps, persisted under runs/<run.id>/
    alongside the standard FIRST/Q1/Q2/Q3/LAST/BEST checkpoints.

Raises:
    ValueError: when params is None, params.train_loader is None,
        trainer_step_fn is None, or any optim's
        NNOptimParams.is_valid() returns False.
```


#### `nnx.trainer.trainer.TrainerStepContext`

```python
class nnx.trainer.trainer.TrainerStepContext(model: 'NNModel', batch: 'Any', optimizers: 'Mapping[str, torch.optim.Optimizer]', schedulers: 'Mapping[str, Any]', extra_metrics: 'Optional[Mapping[str, Callable]]', batch_idx: 'int', epoch_idx: 'int') -> 'None'
```

Per-batch state passed into a trainer_step_fn.

**Details**

```text
Mirrors TrainStepContext from NNModel.train() but with `optimizer`
(singular) replaced by `optimizers` (name-keyed dict) and `schedulers`
threaded through alongside for inspection. Step functions should only
call schedulers directly when ``auto_step_schedulers=False``.

`model` is the single NNModel the Trainer was constructed with;
`model.net` carries the actual nn.Module (which may itself be a
composite, e.g., a GAN-style wrapper exposing G and D as submodules).
```


#### `nnx.trainer.trainer.TrainerStepFn`

```python
type alias nnx.trainer.trainer.TrainerStepFn
```

Public type alias.


#### `nnx.trainer.params.NNTrainerParams`

```python
class nnx.trainer.params.NNTrainerParams(*, n_epochs: 'int', optims: 'Mapping[str, NNOptimParams]', schedulers: 'Mapping[str, NNSchedulerParams]' = <factory>, seed: 'Optional[int]' = None, data_id: 'Optional[str]' = None, save_phase_checkpoints: 'bool' = True, auto_step_schedulers: 'bool' = True, overwrite_existing: 'bool' = False, train_loader: 'Optional[DataLoader]' = None, val_loader: 'Optional[DataLoader]' = None, extra_metrics: 'Optional[Mapping[str, Callable]]' = None) -> 'None'
```

Configuration for `Trainer.train()` — the multi-optimizer parallel to `NNModel.train()` / `NNTrainParams`.

**Details**

```text
`optims` is a name-keyed mapping of NNOptimParams; each entry
produces a distinct torch Optimizer. Use `NNOptimParams.param_groups`
on each entry (the fine-tuning hook from :mod:`nnx.finetune`) to scope an optimizer
to a subset of the model's parameters — e.g., one optim for the
generator sub-net (`name_pattern="G.*"`), one for the discriminator
(`name_pattern="D.*"`) inside a single combined NNModel.

`schedulers` is similarly keyed and indexes the same names. Missing
entries default to ReduceLROnPlateau with the same defaults
NNTrainParams uses, so callers only have to populate schedulers for
the optims they want to customize.

`seed`, `save_phase_checkpoints`, `extra_metrics`, `train_loader`,
`val_loader` mirror NNTrainParams. By default Trainer steps every
scheduler once after each epoch; set `auto_step_schedulers=False` when
the custom step function owns scheduler timing.
```

##### `nnx.trainer.params.NNTrainerParams.with_train_loader`

```python
nnx.trainer.params.NNTrainerParams.with_train_loader(self, value: 'DataLoader') -> 'NNTrainerParams'
```

No public description is currently available.

##### `nnx.trainer.params.NNTrainerParams.with_val_loader`

```python
nnx.trainer.params.NNTrainerParams.with_val_loader(self, value: 'DataLoader') -> 'NNTrainerParams'
```

No public description is currently available.

##### `nnx.trainer.params.NNTrainerParams.state`

```python
nnx.trainer.params.NNTrainerParams.state(self)
```

No public description is currently available.

##### `nnx.trainer.params.NNTrainerParams.from_state`

```python
nnx.trainer.params.NNTrainerParams.from_state(state: 'dict') -> 'NNTrainerParams'
```

No public description is currently available.

##### `nnx.trainer.params.NNTrainerParams.builder`

```python
nnx.trainer.params.NNTrainerParams.builder() -> 'NNTrainerParamsBuilder'
```

Return a composite multi-optim builder. See `NNTrainerParamsBuilder`. Composes `NNOptimParams.builder()` + `NNSchedulerParams.builder()`.


#### `nnx.trainer.params_builder.NNTrainerParamsBuilder`

```python
class nnx.trainer.params_builder.NNTrainerParamsBuilder() -> 'None'
```

Composite builder for `NNTrainerParams`.

**Details**

```text
Reach via `NNTrainerParams.builder()`. The required setter is
`.n_epochs(N)`; at least one `.optimizer(name, params)` call is
also required (`NNTrainerParams.__post_init__` rejects empty
optims). Schedulers, seed, loaders, etc. are all chained optionals.
```

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.n_epochs`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.n_epochs(self, n: 'int') -> 'NNTrainerParamsBuilder'
```

Number of training epochs. Required.

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.optimizer`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.optimizer(self, name: 'str', params: 'NNOptimParams') -> 'NNTrainerParamsBuilder'
```

Register one optimizer under `name`. Each name gets its own torch.optim.Optimizer at Trainer.train() time. Use `NNOptimParams.builder()` (Plan 2) to construct `params`.

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.scheduler`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.scheduler(self, name: 'str', params: 'NNSchedulerParams') -> 'NNTrainerParamsBuilder'
```

Register one scheduler under `name`. The name must match a previously-registered `.optimizer(name, ...)` call — `.build()` enforces the subset invariant.

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.seed`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.seed(self, value: 'int') -> 'NNTrainerParamsBuilder'
```

Seed for reproducibility. None at default (no seeding via params; the caller's `set_seed()` is the only path).

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.save_phase_checkpoints`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.save_phase_checkpoints(self, value: 'bool') -> 'NNTrainerParamsBuilder'
```

Whether to write phase checkpoints (FIRST / Q1 / Q2 / Q3 / LAST / BEST). Default True. The fluent contract is "last call wins" — a prior `.save_phase_checkpoints(False)` followed by `.save_phase_checkpoints(True)` leaves the dataclass at the default (which `state()` then omits).

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.auto_step_schedulers`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.auto_step_schedulers(self, value: 'bool') -> 'NNTrainerParamsBuilder'
```

Choose whether Trainer steps every scheduler after each epoch.

**Details**

```text
Disable this when the custom step function owns scheduler timing.
```

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.train_loader`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.train_loader(self, loader: 'DataLoader') -> 'NNTrainerParamsBuilder'
```

Training DataLoader. Optional at Builder time (can be wired later via NNTrainerParams.with_train_loader).

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.val_loader`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.val_loader(self, loader: 'DataLoader') -> 'NNTrainerParamsBuilder'
```

Validation DataLoader. Optional at Builder time (can be wired later via NNTrainerParams.with_val_loader).

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.extra_metrics`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.extra_metrics(self, metrics: 'Mapping[str, Callable]') -> 'NNTrainerParamsBuilder'
```

Extra metrics callables, name-keyed. Each is called with (y_pred, y_true) at every validation step.

##### `nnx.trainer.params_builder.NNTrainerParamsBuilder.build`

```python
nnx.trainer.params_builder.NNTrainerParamsBuilder.build(self) -> 'NNTrainerParams'
```

Validate the key-subset invariant, then construct the dataclass.

**Details**

```text
`schedulers.keys() ⊆ optims.keys()` is the contract
`NNTrainerParams.__post_init__` enforces. We check here so the
user sees the violation at the Builder boundary — e.g., they
called `.scheduler("d", ...)` without first calling
`.optimizer("d", ...)` — rather than at the dataclass ctor.

`n_epochs` has no meaningful default — call `.n_epochs(N)` before
`.build()`. Caught here too, for the same Builder-boundary reason.

Raises:
    ValueError: if `.n_epochs(N)` was not called before
        `.build()`, OR if a `.scheduler(name, ...)` was
        attached for a name that has no corresponding
        `.optimizer(name, ...)`. Both messages name the
        Builder methods to call so the user can fix the chain
        without consulting the dataclass schema.
```


## 3. Params

#### `nnx.nn.params.nn_params.NNParams`

```python
class nnx.nn.params.nn_params.NNParams(*, dropout_prob: 'float', n_heads: 'Optional[int]' = None, activation: 'Optional[Activations]' = leaky_relu, activations: 'Optional[list[Activations]]' = None, dropout_probs: 'Optional[list[float]]' = None, input_dim: 'int', output_dim: 'int', hidden_dims: 'Optional[list[int]]' = None) -> 'None'
```

NNParams(*, dropout_prob: 'float', n_heads: 'Optional[int]' = None, activation: 'Optional[Activations]' = leaky_relu, activations: 'Optional[list[Activations]]' = None, dropout_probs: 'Optional[list[float]]' = None, input_dim: 'int', output_dim: 'int', hidden_dims: 'Optional[list[int]]' = None)

##### `nnx.nn.params.nn_params.NNParams.dims`

```python
property nnx.nn.params.nn_params.NNParams.dims
```

No public description is currently available.

##### `nnx.nn.params.nn_params.NNParams.activation_for`

```python
nnx.nn.params.nn_params.NNParams.activation_for(self, layer_idx: 'int') -> 'Activations'
```

The activation for hidden layer ``layer_idx`` — the per-layer entry when `activations` is set, else the net-wide scalar (#85).

##### `nnx.nn.params.nn_params.NNParams.dropout_for`

```python
nnx.nn.params.nn_params.NNParams.dropout_for(self, layer_idx: 'int') -> 'float'
```

The dropout prob for hidden layer ``layer_idx`` — per-layer entry when `dropout_probs` is set, else the net-wide scalar (#85).

##### `nnx.nn.params.nn_params.NNParams.state`

```python
nnx.nn.params.nn_params.NNParams.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_params.NNParams.from_state`

```python
nnx.nn.params.nn_params.NNParams.from_state(state: 'dict') -> 'NNParams'
```

No public description is currently available.

##### `nnx.nn.params.nn_params.NNParams.resolve_from_state`

```python
nnx.nn.params.nn_params.NNParams.resolve_from_state(state: 'dict') -> 'NNParams'
```

Dispatch to the params subclass that wrote ``state``.

**Details**

```text
``NNTransformerParams.state()`` always emits its required
architectural keys (``vocab_size`` among them); base
``NNParams.state()`` never does. Without this dispatch a
transformer state is silently downgraded to base ``NNParams`` —
the subclass keys are dropped, the reloaded run re-hashes to a
different id, and net rebuilding crashes. Every loader
(``NNRun.load``, the ``NNCheckpoint`` readers, hub
``from_pretrained``) resolves through here.
```


#### `nnx.nn.params.nn_model_params.NNModelParams`

```python
class nnx.nn.params.nn_model_params.NNModelParams(*, net: 'Nets', device: 'Devices' = cpu, loss: 'Losses' = cross_entropy, mixed_precision: 'bool' = False) -> 'None'
```

NNModelParams(*, net: 'Nets', device: 'Devices' = cpu, loss: 'Losses' = cross_entropy, mixed_precision: 'bool' = False)

##### `nnx.nn.params.nn_model_params.NNModelParams.is_valid`

```python
nnx.nn.params.nn_model_params.NNModelParams.is_valid(self) -> 'bool'
```

No public description is currently available.

##### `nnx.nn.params.nn_model_params.NNModelParams.state`

```python
nnx.nn.params.nn_model_params.NNModelParams.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_model_params.NNModelParams.from_state`

```python
nnx.nn.params.nn_model_params.NNModelParams.from_state(state: 'dict') -> 'NNModelParams'
```

No public description is currently available.


#### `nnx.nn.params.nn_train_params.NNTrainParams`

```python
class nnx.nn.params.nn_train_params.NNTrainParams(*, n_epochs: 'int', scheduler: 'NNSchedulerParams' = NNSchedulerParams(min_lr=1e-07, factor=0.95, patience=8, cooldown=2, threshold=0.001, kind=None, step_size=None, T_max=None, max_lr=None, total_steps=None, warmup_steps=None), optim: 'NNOptimParams' = NNOptimParams(name=adam, max_lr=0.01, weight_decay=5e-05, momentum=(0.9, 0.999), grad_clip_norm=None, accumulate_grad_batches=1, param_groups=None), seed: 'Optional[int]' = None, data_id: 'Optional[str]' = None, save_phase_checkpoints: 'bool' = True, train_loader: 'Optional[DataLoader]' = None, val_loader: 'Optional[DataLoader]' = None, extra_metrics: 'Optional[Mapping[str, Callable]]' = None, resume_from_run_id: 'Optional[str]' = None, resume_from_checkpoint: 'Optional[str]' = 'last', parent_run_id: 'Optional[str]' = None, overwrite_existing: 'bool' = False) -> 'None'
```

Training configuration.

**Details**

```text
`seed` pins every RNG that affects training (Python random, NumPy,
torch CPU+CUDA, cuDNN) when NNModel.train() runs. None disables
seeding (default).

To preserve back-compat with previously-saved runs, `seed` is included
in state() ONLY when set — so existing runs with no seed continue to
hash to the same `run.id`.
```

##### `nnx.nn.params.nn_train_params.NNTrainParams.with_train_loader`

```python
nnx.nn.params.nn_train_params.NNTrainParams.with_train_loader(self, value: 'DataLoader') -> 'NNTrainParams'
```

No public description is currently available.

##### `nnx.nn.params.nn_train_params.NNTrainParams.with_val_loader`

```python
nnx.nn.params.nn_train_params.NNTrainParams.with_val_loader(self, value: 'DataLoader') -> 'NNTrainParams'
```

No public description is currently available.

##### `nnx.nn.params.nn_train_params.NNTrainParams.state`

```python
nnx.nn.params.nn_train_params.NNTrainParams.state(self)
```

No public description is currently available.

##### `nnx.nn.params.nn_train_params.NNTrainParams.from_state`

```python
nnx.nn.params.nn_train_params.NNTrainParams.from_state(state: 'dict') -> 'NNTrainParams'
```

No public description is currently available.


#### `nnx.nn.params.nn_optim_params.NNOptimParams`

```python
class nnx.nn.params.nn_optim_params.NNOptimParams(*, name: 'Optims', max_lr: 'float', weight_decay: 'float', momentum: 'Union[float, tuple[float, float]]', grad_clip_norm: 'Optional[float]' = None, accumulate_grad_batches: 'int' = 1, param_groups: 'Optional[list[NNParamGroupSpec]]' = None) -> 'None'
```

Optimizer config.

**Details**

```text
`momentum` is overloaded by optimizer kind:
  - For SGD / SGD_NESTEROV: a single float, the SGD momentum coefficient.
  - For ADAM / ADAM_AMSGRAD: a (beta1, beta2) tuple, passed as the
    Adam `betas=` argument. The name is retained for backwards
    compatibility — `is_valid()` enforces the per-optim shape.

`grad_clip_norm` clips gradients by global L2 norm before optimizer.step().
None = no clipping (back-compat default). Typical values: 1.0 for
transformers, 5.0 for RNNs.

`accumulate_grad_batches` enables gradient accumulation — the effective
batch size becomes batch_size * accumulate_grad_batches. The loss is
scaled by 1/N so the accumulated gradient is the mean across N batches.
Default 1 (back-compat: step every batch).

`param_groups` enables per-layer-group LR / weight_decay overrides — the
fine-tuning idiom of "small LR on the backbone, large LR on the head."
None = single-group behavior (every parameter at `max_lr` / `weight_decay`).
When set, the optimizer factory dispatches via
:func:`nnx.finetune.param_groups.build_param_groups` to construct
per-group dicts.
```

##### `nnx.nn.params.nn_optim_params.NNOptimParams.state`

```python
nnx.nn.params.nn_optim_params.NNOptimParams.state(self)
```

No public description is currently available.

##### `nnx.nn.params.nn_optim_params.NNOptimParams.from_state`

```python
nnx.nn.params.nn_optim_params.NNOptimParams.from_state(state: 'dict') -> 'NNOptimParams'
```

No public description is currently available.

##### `nnx.nn.params.nn_optim_params.NNOptimParams.is_valid`

```python
nnx.nn.params.nn_optim_params.NNOptimParams.is_valid(self) -> 'bool'
```

No public description is currently available.

##### `nnx.nn.params.nn_optim_params.NNOptimParams.builder`

```python
nnx.nn.params.nn_optim_params.NNOptimParams.builder() -> 'NNOptimParamsBuilder'
```

Return a variant-aware builder. See `NNOptimParamsBuilder`.


#### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder`

```python
class nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder() -> 'None'
```

Variant-aware builder for `NNOptimParams`.

**Details**

```text
Reach via `NNOptimParams.builder()`. Pick exactly one variant
method (`adam`, `adam_amsgrad`, `sgd`, `sgd_nesterov`), then chain
optional methods (`grad_clip`, `accumulate_grad`, `param_groups`),
then `.build()`. Method-call order is independent — a modifier
called before a variant survives the variant call, and the last
variant always wins.
```

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.adam`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.adam(self, *, max_lr: 'float', betas: 'tuple[float, float]' = (0.9, 0.999), weight_decay: 'float' = 0.0) -> 'NNOptimParamsBuilder'
```

torch.optim.Adam. `betas` is PyTorch's name for the (beta1, beta2) tuple; the Builder maps it onto the underlying `NNOptimParams.momentum` field (which holds the tuple for Adam variants).

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.adam_amsgrad`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.adam_amsgrad(self, *, max_lr: 'float', betas: 'tuple[float, float]' = (0.9, 0.999), weight_decay: 'float' = 0.0) -> 'NNOptimParamsBuilder'
```

torch.optim.Adam with `amsgrad=True`. Same `betas` mapping as `adam()`.

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.sgd`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.sgd(self, *, max_lr: 'float', momentum: 'float' = 0.9, weight_decay: 'float' = 0.0) -> 'NNOptimParamsBuilder'
```

torch.optim.SGD. The float momentum stays as `momentum` (no rename) — `betas` is an Adam-family term.

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.sgd_nesterov`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.sgd_nesterov(self, *, max_lr: 'float', momentum: 'float' = 0.9, weight_decay: 'float' = 0.0) -> 'NNOptimParamsBuilder'
```

torch.optim.SGD with `nesterov=True`. Same momentum shape as `sgd()`.

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.grad_clip`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.grad_clip(self, norm: 'float') -> 'NNOptimParamsBuilder'
```

Global-L2 gradient-norm clipping. None = no clipping (the dataclass default; this method is the opt-in path).

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.accumulate_grad`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.accumulate_grad(self, batches: 'int') -> 'NNOptimParamsBuilder'
```

Accumulate gradients over `batches` mini-batches before stepping. Default (no call) leaves the dataclass at 1.

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.param_groups`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.param_groups(self, groups: 'list[NNParamGroupSpec]') -> 'NNOptimParamsBuilder'
```

Per-layer-group LR / weight_decay overrides (the fine-tuning idiom). Default (no call) leaves the dataclass at None (single-group behavior).

##### `nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.build`

```python
nnx.nn.params.nn_optim_params_builder.NNOptimParamsBuilder.build(self) -> 'NNOptimParams'
```

Construct the dataclass from the fields the user touched.

**Details**

```text
Pre-empts the dataclass's missing-required-argument TypeError
with an actionable Builder-level ValueError naming the variant
methods — matches the [[builder-pattern-shape]] §11b convention
that PR #52 established on NNTrainerParamsBuilder.

Forwards only the keys present in `self._fields` so the
dataclass defaults govern every untouched optional field —
that's what preserves the omit-when-default state() invariant.

Raises:
    ValueError: if no variant method (`.adam`, `.adam_amsgrad`,
        `.sgd`, `.sgd_nesterov`) was called before `.build()`.
        The message names the four methods so the user can
        fix the chain without consulting the dataclass schema.
```


#### `nnx.nn.params.nn_scheduler_params.NNSchedulerParams`

```python
class nnx.nn.params.nn_scheduler_params.NNSchedulerParams(*, min_lr: 'float', factor: 'float', patience: 'int', cooldown: 'int', threshold: 'float', kind: 'Optional[Schedulers]' = None, step_size: 'Optional[int]' = None, T_max: 'Optional[int]' = None, max_lr: 'Optional[float]' = None, total_steps: 'Optional[int]' = None, warmup_steps: 'Optional[int]' = None) -> 'None'
```

NNSchedulerParams(*, min_lr: 'float', factor: 'float', patience: 'int', cooldown: 'int', threshold: 'float', kind: 'Optional[Schedulers]' = None, step_size: 'Optional[int]' = None, T_max: 'Optional[int]' = None, max_lr: 'Optional[float]' = None, total_steps: 'Optional[int]' = None, warmup_steps: 'Optional[int]' = None)

##### `nnx.nn.params.nn_scheduler_params.NNSchedulerParams.state`

```python
nnx.nn.params.nn_scheduler_params.NNSchedulerParams.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_scheduler_params.NNSchedulerParams.from_state`

```python
nnx.nn.params.nn_scheduler_params.NNSchedulerParams.from_state(state: 'dict') -> 'NNSchedulerParams'
```

No public description is currently available.

##### `nnx.nn.params.nn_scheduler_params.NNSchedulerParams.builder`

```python
nnx.nn.params.nn_scheduler_params.NNSchedulerParams.builder() -> 'NNSchedulerParamsBuilder'
```

Return a variant-aware builder. See `NNSchedulerParamsBuilder`.


#### `nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder`

```python
class nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder() -> 'None'
```

Variant-aware builder for `NNSchedulerParams`.

**Details**

```text
Reach this via `NNSchedulerParams.builder()`. Each variant method
is self-contained — the user calls exactly one of them per builder
instance. Calling a second variant overwrites the first (last
write wins); `.build()` produces the dataclass.
```

##### `nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.reduce_on_plateau`

```python
nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.reduce_on_plateau(self, *, min_lr: 'float', factor: 'float', patience: 'int', cooldown: 'int', threshold: 'float') -> 'NNSchedulerParamsBuilder'
```

ReduceLROnPlateau — the default scheduler.

**Details**

```text
Sets the five plateau fields. `kind` is left at None (the
dataclass default), which preserves the omit-when-default
state() invariant for callers who used the original pre-enum
config.
```

##### `nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.step`

```python
nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.step(self, *, step_size: 'int', min_lr: 'float', factor: 'float', patience: 'int', cooldown: 'int', threshold: 'float') -> 'NNSchedulerParamsBuilder'
```

torch.optim.lr_scheduler.StepLR — decay LR by `factor` every `step_size` epochs. The plateau-shape fields (`min_lr`, `patience`, `cooldown`, `threshold`) are not consumed by StepLR but are required by the underlying NNSchedulerParams dataclass and serialised for back-compat.

##### `nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.cosine_annealing`

```python
nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.cosine_annealing(self, *, T_max: 'int', min_lr: 'float', factor: 'float', patience: 'int', cooldown: 'int', threshold: 'float') -> 'NNSchedulerParamsBuilder'
```

torch.optim.lr_scheduler.CosineAnnealingLR — anneal LR over `T_max` steps.

##### `nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.one_cycle`

```python
nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.one_cycle(self, *, max_lr: 'float', total_steps: 'int', min_lr: 'float', factor: 'float', patience: 'int', cooldown: 'int', threshold: 'float') -> 'NNSchedulerParamsBuilder'
```

torch.optim.lr_scheduler.OneCycleLR — Smith one-cycle schedule with peak LR `max_lr` over `total_steps` steps.

##### `nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.linear_warmup_decay`

```python
nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.linear_warmup_decay(self, *, warmup_steps: 'int', total_steps: 'int', min_lr: 'float', factor: 'float', patience: 'int', cooldown: 'int', threshold: 'float') -> 'NNSchedulerParamsBuilder'
```

Linear warm-up to `max_lr` over `warmup_steps`, linear decay to 0 over the remaining `total_steps - warmup_steps`. Used by most transformer training recipes.

##### `nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.build`

```python
nnx.nn.params.nn_scheduler_params_builder.NNSchedulerParamsBuilder.build(self) -> 'NNSchedulerParams'
```

Construct the dataclass from the fields the user touched.

**Details**

```text
Pre-empts the dataclass's missing-required-argument TypeError
with an actionable Builder-level ValueError naming the variant
methods — matches the [[builder-pattern-shape]] §11b convention
that PR #52 established on NNTrainerParamsBuilder.

Forwards only the keys present in `self._fields` so the
dataclass defaults govern every untouched field — that's what
preserves the omit-when-default state() invariant.

Raises:
    ValueError: if no variant method (`.reduce_on_plateau`,
        `.step`, `.cosine_annealing`, `.one_cycle`,
        `.linear_warmup_decay`) was called before `.build()`.
        The message names the five methods so the user can
        fix the chain without consulting the dataclass schema.
```


#### `nnx.nn.params.nn_transformer_params.NNTransformerParams`

```python
class nnx.nn.params.nn_transformer_params.NNTransformerParams(*, dropout_prob: 'float', n_heads: 'Optional[int]' = None, activation: 'Optional[Activations]' = leaky_relu, activations: 'Optional[list[Activations]]' = None, dropout_probs: 'Optional[list[float]]' = None, input_dim: 'int', output_dim: 'int', hidden_dims: 'Optional[list[int]]' = None, vocab_size: 'int', n_layers: 'int', d_model: 'int', max_seq_len: 'int', ffn_mult: 'int' = 4, rope_base: 'float' = 10000.0, tie_embeddings: 'bool' = True, attn_dropout: 'float' = 0.0, resid_dropout: 'float' = 0.0) -> 'None'
```

NNTransformerParams(*, dropout_prob: 'float', n_heads: 'Optional[int]' = None, activation: 'Optional[Activations]' = leaky_relu, activations: 'Optional[list[Activations]]' = None, dropout_probs: 'Optional[list[float]]' = None, input_dim: 'int', output_dim: 'int', hidden_dims: 'Optional[list[int]]' = None, vocab_size: 'int', n_layers: 'int', d_model: 'int', max_seq_len: 'int', ffn_mult: 'int' = 4, rope_base: 'float' = 10000.0, tie_embeddings: 'bool' = True, attn_dropout: 'float' = 0.0, resid_dropout: 'float' = 0.0)

##### `nnx.nn.params.nn_transformer_params.NNTransformerParams.state`

```python
nnx.nn.params.nn_transformer_params.NNTransformerParams.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_transformer_params.NNTransformerParams.from_state`

```python
nnx.nn.params.nn_transformer_params.NNTransformerParams.from_state(state: 'dict') -> 'NNTransformerParams'
```

No public description is currently available.

##### `nnx.nn.params.nn_transformer_params.NNTransformerParams.builder`

```python
nnx.nn.params.nn_transformer_params.NNTransformerParams.builder() -> 'NNTransformerParamsBuilder'
```

Return a fluent LM-path builder. See `NNTransformerParamsBuilder`.


#### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder`

```python
class nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder() -> 'None'
```

Builder for `NNTransformerParams`.

**Details**

```text
Reach via `NNTransformerParams.builder()`. The six methods can be
chained in any order; `.build()` collects them, fills in the
LM-path defaults for the dead parent-NNParams fields, and
constructs the dataclass.
```

##### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.vocab`

```python
nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.vocab(self, size: 'int') -> 'NNTransformerParamsBuilder'
```

Set the vocabulary size. Mirrors into both `input_dim` and `output_dim` on the parent NNParams (the LM convention).

##### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.layers`

```python
nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.layers(self, *, n: 'int', heads: 'int', d_model: 'int') -> 'NNTransformerParamsBuilder'
```

Set depth (`n_layers`), attention head count (`n_heads`), and hidden dimension (`d_model`). Enforces `d_model % heads == 0` immediately — this is the Builder's safety value-add over the direct-kwarg ctor, which only catches the mismatch at __post_init__ time after all kwargs have already been typed.

##### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.ffn`

```python
nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.ffn(self, *, mult: 'int') -> 'NNTransformerParamsBuilder'
```

FFN expansion ratio. Default is 4 (the SwiGLU-friendly ratio); only call this method to override.

##### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.context`

```python
nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.context(self, *, max_seq_len: 'int', rope_base: 'Optional[float]' = None) -> 'NNTransformerParamsBuilder'
```

Context-length and RoPE base. `max_seq_len` is required; `rope_base=None` is the sentinel for "use the dataclass default (10000.0, the LLaMA / GPT convention)". The fluent contract is "last call wins": `.context(rope_base=500000.0).context(max_seq_len=128)` resets `rope_base` to the default — the second call's implicit `rope_base=None` drops the prior override.

##### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.dropout`

```python
nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.dropout(self, *, attn: 'float' = 0.0, resid: 'float' = 0.0) -> 'NNTransformerParamsBuilder'
```

Attention and residual dropout rates. Defaults are both 0.0 (modern LLM convention; regularization comes from data scale, not dropout).

**Details**

```text
Like `.context()`, a `dropout()` call specifies BOTH rates
together — each call fully replaces the pair, and a rate left
at its 0.0 default is reset, not carried over from a prior
call. So `.dropout(resid=0.3).dropout(attn=0.5)` yields
`attn=0.5, resid=0.0` (the second call's implicit `resid=0.0`
drops the prior override); call `.dropout(attn=0.5, resid=0.3)`
once to set both. Same-field last-call-wins still holds:
`.dropout(attn=0.5).dropout(attn=0.0)` resets to `attn=0.0`.
The dataclass's omit-when-default `state()` then handles
run.id stability automatically.
```

##### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.tied_embeddings`

```python
nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.tied_embeddings(self, value: 'bool') -> 'NNTransformerParamsBuilder'
```

Toggle weight-tying between input embeddings and LM head. Default is True. The fluent contract is "last call wins" — a prior `.tied_embeddings(False)` followed by `.tied_embeddings(True)` leaves the dataclass at the default (which `state()` then omits).

##### `nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.build`

```python
nnx.nn.params.nn_transformer_params_builder.NNTransformerParamsBuilder.build(self) -> 'NNTransformerParams'
```

Construct the dataclass.

**Details**

```text
Pre-empts the dataclass's missing-required-argument TypeError
with an actionable Builder-level ValueError naming the setter
methods that haven't been called yet — matches the
[[builder-pattern-shape]] §11b convention that PR #52
established on NNTrainerParamsBuilder.

Fills in the dead parent-NNParams fields the TransformerNN
net never reads but the parent dataclass requires at
construction. `activation` mirrors the parent NNParams's
default (`Activations.LEAKY_RELU`); a Builder-default
mismatch here previously produced a different `state()` /
`run.id` than the direct-kwarg ctor.

Raises:
    ValueError: if `.vocab(size=...)`, `.layers(n=..., heads=...,
        d_model=...)`, or `.context(max_seq_len=...)` was not
        called before `.build()`. The message names the
        specific setter methods that are still missing so the
        user can complete the chain without consulting the
        dataclass schema.
```


#### `nnx.nn.params.nn_conv_params.NNConvParams`

```python
class nnx.nn.params.nn_conv_params.NNConvParams(*, dropout_prob: 'float', n_heads: 'Optional[int]' = None, activation: 'Optional[Activations]' = leaky_relu, activations: 'Optional[list[Activations]]' = None, dropout_probs: 'Optional[list[float]]' = None, input_dim: 'int', output_dim: 'int', hidden_dims: 'Optional[list[int]]' = None, conv_channels: 'list[int]', in_channels: 'int' = 1, kernel_size: 'int' = 5, stride: 'int' = 1, padding: 'int' = 0, pool_size: 'int' = 2) -> 'None'
```

Parameters for a ConvNN with a required conv-block activation.

##### `nnx.nn.params.nn_conv_params.NNConvParams.image_side`

```python
nnx.nn.params.nn_conv_params.NNConvParams.image_side(self) -> 'int'
```

Spatial side of the (square) input image.

##### `nnx.nn.params.nn_conv_params.NNConvParams.spatial_sizes`

```python
nnx.nn.params.nn_conv_params.NNConvParams.spatial_sizes(self) -> 'list[int]'
```

Feature-map side after each Conv→Pool block (floor arithmetic, matching Conv2d/MaxPool2d).

##### `nnx.nn.params.nn_conv_params.NNConvParams.flatten_dim`

```python
nnx.nn.params.nn_conv_params.NNConvParams.flatten_dim(self) -> 'int'
```

Input width of the first FC layer: last block's channels × side².

##### `nnx.nn.params.nn_conv_params.NNConvParams.state`

```python
nnx.nn.params.nn_conv_params.NNConvParams.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_conv_params.NNConvParams.from_state`

```python
nnx.nn.params.nn_conv_params.NNConvParams.from_state(state: 'dict') -> 'NNConvParams'
```

No public description is currently available.


#### `nnx.nn.params.nn_moe_params.NNMoEParams`

```python
class nnx.nn.params.nn_moe_params.NNMoEParams(*, dropout_prob: 'float', n_heads: 'Optional[int]' = None, activation: 'Optional[Activations]' = leaky_relu, activations: 'Optional[list[Activations]]' = None, dropout_probs: 'Optional[list[float]]' = None, input_dim: 'int', output_dim: 'int', hidden_dims: 'Optional[list[int]]' = None, num_experts: 'int', top_k: 'int' = 2) -> 'None'
```

Serializable parameters for an expert-bearing feed-forward MoE.

**Details**

```text
Unlike base :class:`NNParams`, ``hidden_dims`` must contain at least one
layer because only hidden layers are replaced by ``MoELinear`` modules.
```

##### `nnx.nn.params.nn_moe_params.NNMoEParams.state`

```python
nnx.nn.params.nn_moe_params.NNMoEParams.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_moe_params.NNMoEParams.from_state`

```python
nnx.nn.params.nn_moe_params.NNMoEParams.from_state(state: 'dict') -> 'NNMoEParams'
```

No public description is currently available.


#### `nnx.nn.params.nn_tokenizer_params.NNTokenizerParams`

```python
class nnx.nn.params.nn_tokenizer_params.NNTokenizerParams(*, path: 'str', tokenizer: 'object') -> 'None'
```

Frozen dataclass holding a tokenizer + its on-disk pointer.

**Details**

```text
The dataclass is frozen so it can sit alongside NNTransformerParams /
NNModelParams in an NNRun without inviting in-place mutation. The
actual ``tokenizers.Tokenizer`` object is held in a repr=False field
so it doesn't bloat the str() output.
```

##### `nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.of`

```python
nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.of(tokenizer: 'object', path: 'str') -> 'NNTokenizerParams'
```

Construct from a live Tokenizer instance and persist it to ``path``.

**Details**

```text
This is the train-time entry point: train a tokenizer, then call
``NNTokenizerParams.of(tk, path="runs/tok.json")`` to wrap it
with a paired on-disk artifact.
```

##### `nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.from_state`

```python
nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.from_state(state: 'dict') -> 'NNTokenizerParams'
```

Load from a state dict produced by :meth:`state`. The single required key is ``path``; the tokenizer is reconstructed from the file the path points to.

**Details**

```text
The path is stored exactly as the caller gave it to :meth:`of` —
typically cwd-relative — so loading from a different working
directory requires the same relative layout. That's deliberate:
storing an absolute path would break run portability across
machines, which is the more common need.
```

##### `nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.state`

```python
nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.state(self) -> 'dict'
```

Return the serializable view — only the path goes into run.yaml.

##### `nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.vocab_size`

```python
property nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.vocab_size
```

No public description is currently available.

##### `nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.encode`

```python
nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.encode(self, text: 'str') -> 'list[int]'
```

No public description is currently available.

##### `nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.decode`

```python
nnx.nn.params.nn_tokenizer_params.NNTokenizerParams.decode(self, ids: 'list[int]', skip_special_tokens: 'bool' = True) -> 'str'
```

No public description is currently available.


#### `nnx.nn.params.nn_tokenizer_params.train_bpe`

```python
nnx.nn.params.nn_tokenizer_params.train_bpe(files: 'Optional[list[str]]' = None, *, vocab_size: 'int' = 8192, texts: 'Optional[list[str]]' = None, special_tokens: 'Optional[list[str]]' = None, min_frequency: 'int' = 2) -> 'TokenizerType'
```

Train a BPE tokenizer on either a list of files or a list of texts.

**Details**

```text
Mirrors the HF "quick BPE" recipe — Whitespace pre-tokenizer + BPE
model + BpeTrainer. Returns the trained Tokenizer instance; the
caller is responsible for persisting via
``NNTokenizerParams.of(tk, path=...)``.

Args:
    files: paths to plaintext files (one corpus line per file row).
        If None, ``texts`` is consulted instead.
    vocab_size: target vocab. Actual size may be smaller for tiny
        corpora.
    texts: in-memory list of training strings — useful for unit
        tests and the examples without writing a temp file.
    special_tokens: e.g. ``["<pad>", "<bos>", "<eos>"]``. Included
        in the vocab and not split during tokenization.
    min_frequency: minimum pair frequency to merge — higher values
        give smaller, more conservative vocabs.

Returns:
    Tokenizer: a trained ``tokenizers.Tokenizer`` ready for encode/decode + save.
```


#### `nnx.nn.params.nn_run.NNRun`

```python
class nnx.nn.params.nn_run.NNRun(*, net: 'NNParams', train: 'NNTrainParams', model: 'NNModelParams', trainer: 'Optional[NNTrainerParams]' = None, salt: 'Optional[str]' = None, idps: 'Optional[list[NNIterationDataPoint]]' = None) -> 'None'
```

NNRun(*, net: 'NNParams', train: 'NNTrainParams', model: 'NNModelParams', trainer: 'Optional[NNTrainerParams]' = None, salt: 'Optional[str]' = None, idps: 'Optional[list[NNIterationDataPoint]]' = None)

##### `nnx.nn.params.nn_run.NNRun.id`

```python
property nnx.nn.params.nn_run.NNRun.id
```

No public description is currently available.

##### `nnx.nn.params.nn_run.NNRun.state`

```python
nnx.nn.params.nn_run.NNRun.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_run.NNRun.with_idps`

```python
nnx.nn.params.nn_run.NNRun.with_idps(self, value: 'list[NNIterationDataPoint]') -> 'NNRun'
```

No public description is currently available.

##### `nnx.nn.params.nn_run.NNRun.ensure_writable`

```python
nnx.nn.params.nn_run.NNRun.ensure_writable(self, root: 'Optional[str]' = None, *, overwrite: 'bool' = False) -> 'None'
```

No public description is currently available.

##### `nnx.nn.params.nn_run.NNRun.writable_lease`

```python
nnx.nn.params.nn_run.NNRun.writable_lease(self, root: 'Optional[str]' = None, *, overwrite: 'bool' = False) -> 'Iterator[None]'
```

Reserve this run ID and hold exclusive ownership until training ends.

##### `nnx.nn.params.nn_run.NNRun.checkpoints`

```python
nnx.nn.params.nn_run.NNRun.checkpoints(self, root: 'Optional[str]' = None) -> 'list[Optional[NNCheckpoint]]'
```

Load this run's five phase checkpoints, in cadence order (FIRST, Q1, Q2, Q3, LAST). Entries are None when the tag was never written — e.g. runs trained with ``save_phase_checkpoints=False`` write only LAST and BEST.

**Details**

```text
BEST is deliberately excluded: it duplicates whichever phase
checkpoint won, so including it would double-count. Load it
directly via ``NNCheckpoint.load(run=run.id,
type=Checkpoints.BEST)``.
```

##### `nnx.nn.params.nn_run.NNRun.save`

```python
nnx.nn.params.nn_run.NNRun.save(self, root: 'Optional[str]' = None, *, update_best: 'bool' = True) -> 'NNRun'
```

No public description is currently available.

##### `nnx.nn.params.nn_run.NNRun.load`

```python
nnx.nn.params.nn_run.NNRun.load(id: 'str', root: 'Optional[str]' = None) -> 'NNRun'
```

No public description is currently available.

##### `nnx.nn.params.nn_run.NNRun.all`

```python
nnx.nn.params.nn_run.NNRun.all(root: 'Optional[str]' = None) -> 'list[NNRun]'
```

List every saved NNRun under the runs root, skipping the `best` pointer. Returns [] when the runs/ directory doesn't exist yet. Non-directory entries (stray files, .DS_Store) are filtered out so they don't trigger spurious NNRun.load failures.


#### `nnx.nn.params.nn_checkpoint.NNCheckpointTransform`

```python
class nnx.nn.params.nn_checkpoint.NNCheckpointTransform(*, name: 'str', version: 'int' = 1, options: 'dict[str, Any]' = <factory>) -> 'None'
```

A versioned recipe for rebuilding a checkpoint's module topology.

##### `nnx.nn.params.nn_checkpoint.NNCheckpointTransform.state`

```python
nnx.nn.params.nn_checkpoint.NNCheckpointTransform.state(self) -> 'dict[str, Any]'
```

No public description is currently available.

##### `nnx.nn.params.nn_checkpoint.NNCheckpointTransform.from_state`

```python
nnx.nn.params.nn_checkpoint.NNCheckpointTransform.from_state(state: 'dict[str, Any]') -> 'NNCheckpointTransform'
```

No public description is currently available.


#### `nnx.nn.params.nn_checkpoint.NNCheckpoint`

```python
class nnx.nn.params.nn_checkpoint.NNCheckpoint(*, net_params: 'NNParams', net_state: 'dict[str, Any]', model_params: 'NNModelParams', idp: 'NNIterationDataPoint', transforms: 'tuple[NNCheckpointTransform, ...]' = (), training_state_id: 'Optional[str]' = None, training_state_present: 'Optional[bool]' = None) -> 'None'
```

Model state plus the recipes needed to rebuild its module topology.

**Details**

```text
``transforms`` is empty for ordinary and legacy checkpoints. Training
callbacks that replace modules at ``on_train_end`` can persist ordered,
versioned recipes here; :meth:`NNModel.from_checkpoint` replays recognized
recipes before loading ``net_state``.
```

##### `nnx.nn.params.nn_checkpoint.NNCheckpoint.to_file`

```python
nnx.nn.params.nn_checkpoint.NNCheckpoint.to_file(self, path: 'str', format: "Literal['pickle', 'safetensors']" = 'pickle') -> 'None'
```

Atomically write this NNCheckpoint to ``path``.

**Details**

```text
Args:
    path: destination path. Parent directory is created if missing.
    format: one of:

        - ``"pickle"`` (default): a ``torch.save`` of the whole
          NNCheckpoint dataclass. Bit-exact round-trip including
          the OrderedDict state and the dataclass identity. The
          on-disk format NNx has always written; back-compat
          default for existing callers.
        - ``"safetensors"``: a ``.safetensors`` file with the
          net's tensors as the data section and
          NNParams + NNModelParams + NNIterationDataPoint + transform
          recipes
          JSON-serialized into the metadata dict (str→str only,
          per the safetensors spec). Safe to mmap, readable by
          ComfyUI/vLLM/AutoGPTQ/HF tools, and proof against
          arbitrary-code-execution on load. Requires the
          ``thekaveh-nnx[hub]`` extra.

Both formats write to ``<path>.tmp`` first and rename into place
so a KeyboardInterrupt during the underlying save can never leave
a half-written checkpoint at the destination — matching the
atomicity guarantee NNRun.save offers for YAML/CSV.
```

##### `nnx.nn.params.nn_checkpoint.NNCheckpoint.save`

```python
nnx.nn.params.nn_checkpoint.NNCheckpoint.save(self, run: 'str', type: 'Checkpoints', root: 'Optional[str]' = None, optimizer_state: 'Optional[dict[str, Any]]' = None, scheduler_state: 'Optional[dict[str, Any]]' = None, scaler_state: 'Optional[dict[str, Any]]' = None, rng_state: 'Optional[dict[str, Any]]' = None, completed_epoch: 'Optional[int]' = None, resume_net_state: 'Optional[dict[str, Any]]' = None, optimizer_type: 'Optional[str]' = None, scheduler_type: 'Optional[str]' = None, optimizer_topology: 'Optional[list[list[dict[str, Any]]]]' = None) -> 'None'
```

Save the checkpoint to disk atomically.

**Details**

```text
When `optimizer_state` is supplied, a generation-addressed sibling
file holds the training state, plus a fixed-name compatibility copy.
This sidecar is used by NNModel.train(resume_from=...) to warm-resume
with the prior optimizer momentum / Adam state.

The immutable generation sidecar is committed first and the checkpoint
second. The checkpoint names the sidecar it owns, so interruption
between replacements leaves the previous generation resumable.
```

##### `nnx.nn.params.nn_checkpoint.NNCheckpoint.load_training_state`

```python
nnx.nn.params.nn_checkpoint.NNCheckpoint.load_training_state(run: 'str', type: 'Checkpoints', root: 'Optional[str]' = None, map_location: 'Any' = 'cpu') -> 'Optional[dict[str, Any]]'
```

Load and validate the resumable optimizer/scheduler/scaler bundle.

**Details**

```text
Legacy optimizer-only sidecars are normalized into the new mapping so
checkpoints written by older NNx versions remain resumable.
```

##### `nnx.nn.params.nn_checkpoint.NNCheckpoint.load_with_training_state`

```python
nnx.nn.params.nn_checkpoint.NNCheckpoint.load_with_training_state(run: 'str', type: 'Checkpoints', root: 'Optional[str]' = None, map_location: 'Any' = 'cpu') -> 'tuple[Optional[NNCheckpoint], Optional[dict[str, Any]]]'
```

Atomically load a checkpoint and its matching training-state bundle.

##### `nnx.nn.params.nn_checkpoint.NNCheckpoint.load_optimizer_state`

```python
nnx.nn.params.nn_checkpoint.NNCheckpoint.load_optimizer_state(run: 'str', type: 'Checkpoints', root: 'Optional[str]' = None) -> 'Optional[dict[str, Any]]'
```

Load the optimizer state sidecar for a checkpoint. Returns None when no sidecar exists (e.g., checkpoints written before resume support was added).

**Details**

```text
Loaded with ``weights_only=True`` — the optimizer state-dict
contains only tensors and standard scalar/dict/list types, so the
strict loader works AND it removes the arbitrary-code-execution
risk that the main NNCheckpoint.from_file documents.
```

##### `nnx.nn.params.nn_checkpoint.NNCheckpoint.from_file`

```python
nnx.nn.params.nn_checkpoint.NNCheckpoint.from_file(path: 'str', map_location: 'Any' = 'cpu') -> 'Optional[NNCheckpoint]'
```

Load an NNCheckpoint from disk, auto-detecting pickle vs safetensors.

**Details**

```text
Returns ``None`` if the path doesn't exist or the loaded pickle
object isn't an NNCheckpoint instance.

Dispatch is by magic bytes:

- ``torch.save`` writes a ZIP archive in modern PyTorch
  (``_use_new_zipfile_serialization=True`` is the default since
  PyTorch 1.6), so the file starts with ``b"PK\x03\x04"``.
- Legacy ``torch.save`` (with the zipfile serialization disabled)
  and bare pickle files begin with ``\x80`` (the pickle PROTO
  opcode for protocol >= 2).
- safetensors files begin with a little-endian u64 header length
  followed by a JSON object — byte 8 is always ``{``. The u64's
  LOW byte can legitimately be ``0x80`` (any header length
  ≡ 128 mod 256), which would collide with the pickle PROTO
  opcode — so safetensors is positively identified by byte 8
  BEFORE the ```` pickle check. The ZIP magic is checked
  first of all (a ZIP's byte 8 is the compression method, never
  ``{``; a torch-LEGACY pickle has the fixed magic byte ``0xf9``
  at offset 8, and a protocol ≥ 4 bare pickle has a frame-length
  byte there, ``0x00`` for any file under a terabyte. A
  protocol-2/3 *bare* pickle's byte 8 is content-dependent, but
  NNx never produces bare pickles and such a file failed under
  the old routing too).

Anything matching none of the positive sniffs falls through to
the safetensors loader, whose error on a genuinely corrupt file
is clearer than a misleading unpickle attempt.

SECURITY: the pickle branch calls ``torch.load(weights_only=False)``,
which unpickles arbitrary Python objects. NEVER call this on a
checkpoint file from an untrusted source — a malicious .pt file
can execute arbitrary code at load time. The default
``./runs/<id>/checkpoints/`` layout assumes the files were
produced locally by NNCheckpoint.save. For untrusted sources,
use the safetensors path on save and load: safetensors has no
arbitrary-code path.
```

##### `nnx.nn.params.nn_checkpoint.NNCheckpoint.load`

```python
nnx.nn.params.nn_checkpoint.NNCheckpoint.load(run: 'str', type: 'Checkpoints', root: 'Optional[str]' = None, map_location: 'Any' = 'cpu') -> 'Optional[NNCheckpoint]'
```

No public description is currently available.


#### `nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint`

```python
class nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint(*, lr: 'float', iter_idx: 'int', epoch_idx: 'int', batch_idx: 'int', train_edp: 'NNEvaluationDataPoint', val_edp: 'Optional[NNEvaluationDataPoint]' = None) -> 'None'
```

One row in the per-iteration training log.

**Details**

```text
`train_edp` is computed from the current batch only. `val_edp` is the
per-epoch validation evaluation — populated **only on the last idp of
each epoch** (the idp at which the validation loop ran). Other idps in
the same epoch have `val_edp=None`. When reading idps.csv, group by
epoch_idx and take the row with val_edp set for per-epoch validation
metrics.
```

##### `nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint.with_val_edp`

```python
nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint.with_val_edp(self, value: 'Optional[NNEvaluationDataPoint]') -> 'NNIterationDataPoint'
```

No public description is currently available.

##### `nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint.state`

```python
nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint.from_state`

```python
nnx.nn.params.nn_iteration_data_point.NNIterationDataPoint.from_state(state: 'dict') -> 'NNIterationDataPoint'
```

No public description is currently available.


#### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint`

```python
class nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint(*, f1: 'float', recall: 'float', accuracy: 'float', precision: 'float', loss: 'Optional[float]' = None, error: 'Optional[float]' = None, extra: 'Mapping[str, float]' = <factory>) -> 'None'
```

Per-batch / per-epoch evaluation metrics.

**Details**

```text
The four core fields (f1, recall, accuracy, precision) are computed by
`of()` via sklearn. `loss` and `error` are typically attached after the
fact by NNModel during training / evaluation.

`extra` is a free-form dict of user-supplied custom metric names to
floats. Populated when NNTrainParams.extra_metrics or evaluate(metrics=)
is set; empty by default (and omitted from state() when empty so that
pre-extra runs hash to the same run.id and pre-extra YAML loads cleanly).
```

##### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.with_loss`

```python
nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.with_loss(self, value: 'float')
```

No public description is currently available.

##### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.with_error`

```python
nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.with_error(self, value: 'float')
```

No public description is currently available.

##### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.with_extra`

```python
nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.with_extra(self, name: 'str', value: 'float') -> 'NNEvaluationDataPoint'
```

No public description is currently available.

##### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.of`

```python
nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.of(Y: 'np.ndarray', Y_hat: 'np.ndarray', average: 'str' = 'macro', extra_metrics: 'Optional[Mapping[str, Callable]]' = None)
```

Compute per-batch evaluation metrics.

**Details**

```text
`average` controls how f1/precision/recall reduce across classes.
Default "macro" treats all classes equally — the right choice for
multi-class classification and the only one that makes f1/precision/
recall mathematically distinct from accuracy. Pass "micro" to
recover the legacy behavior (numerically identical to accuracy for
single-label multi-class). Accuracy itself is not affected.

`extra_metrics` is a {name -> callable(Y, Y_hat) -> float} map of
user-supplied custom metrics. Each is invoked once on the aggregate
predictions and stored in the returned object's `extra` dict.
```

##### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.mean_of`

```python
nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.mean_of(edps: 'list[NNEvaluationDataPoint]') -> 'NNEvaluationDataPoint'
```

Unweighted-mean reduce a list of EDPs across every metric.

**Details**

```text
.. warning::

    This is a **simple mean across edps**, NOT a sample-weighted
    mean. With unequal batch sizes (the common case), the result
    is statistically incorrect — a 1024-sample batch counts the
    same as an 8-sample tail batch. For correct sample-weighted
    metrics across batches, use :meth:`NNModel.evaluate`, which
    concatenates predictions across the loader and computes once
    on the full sample.

    ``mean_of`` is kept for back-compat with callers that already
    depend on the unweighted-mean semantics; new code should
    prefer :meth:`NNModel.evaluate` unless the unweighted form is
    specifically what's wanted (e.g., averaging across runs, not
    across batches within a run).

An ``extra`` key present on some but not all edps is averaged over
the edps where it IS present (skipped on the rest).
```

##### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.state`

```python
nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.from_state`

```python
nnx.nn.params.nn_evaluation_data_point.NNEvaluationDataPoint.from_state(state: 'dict') -> 'NNEvaluationDataPoint'
```

No public description is currently available.


## 4. Networks

#### `nnx.nn.net.feed_fwd_nn.FeedFwdNN`

```python
class nnx.nn.net.feed_fwd_nn.FeedFwdNN(params: 'NNParams')
```

Base class for all neural network modules.

**Details**

```text
Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool
```

##### `nnx.nn.net.feed_fwd_nn.FeedFwdNN.forward`

```python
nnx.nn.net.feed_fwd_nn.FeedFwdNN.forward(self, X: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.nn.net.feed_fwd_nn.FeedFwdNN.unpack_batch`

```python
nnx.nn.net.feed_fwd_nn.FeedFwdNN.unpack_batch(self, batch)
```

No public description is currently available.

##### `nnx.nn.net.feed_fwd_nn.FeedFwdNN.to_file`

```python
nnx.nn.net.feed_fwd_nn.FeedFwdNN.to_file(self, path: 'str') -> 'None'
```

No public description is currently available.

##### `nnx.nn.net.feed_fwd_nn.FeedFwdNN.from_file`

```python
nnx.nn.net.feed_fwd_nn.FeedFwdNN.from_file(path: 'str', params: 'NNParams', map_location='cpu') -> 'FeedFwdNN'
```

No public description is currently available.

##### `nnx.nn.net.feed_fwd_nn.FeedFwdNN.from_state`

```python
nnx.nn.net.feed_fwd_nn.FeedFwdNN.from_state(state_dict: 'dict', params: 'NNParams') -> 'FeedFwdNN'
```

No public description is currently available.


#### `nnx.nn.net.feed_fwd_moe_nn.FeedFwdMoENN`

```python
class nnx.nn.net.feed_fwd_moe_nn.FeedFwdMoENN(params: 'NNMoEParams')
```

Base class for all neural network modules.

**Details**

```text
Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool
```

##### `nnx.nn.net.feed_fwd_moe_nn.FeedFwdMoENN.forward`

```python
nnx.nn.net.feed_fwd_moe_nn.FeedFwdMoENN.forward(self, X: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.nn.net.feed_fwd_moe_nn.FeedFwdMoENN.unpack_batch`

```python
nnx.nn.net.feed_fwd_moe_nn.FeedFwdMoENN.unpack_batch(self, batch)
```

No public description is currently available.


#### `nnx.nn.net.conv_nn.ConvNN`

```python
class nnx.nn.net.conv_nn.ConvNN(params: 'NNConvParams')
```

Base class for all neural network modules.

**Details**

```text
Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool
```

##### `nnx.nn.net.conv_nn.ConvNN.forward`

```python
nnx.nn.net.conv_nn.ConvNN.forward(self, X: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.nn.net.conv_nn.ConvNN.unpack_batch`

```python
nnx.nn.net.conv_nn.ConvNN.unpack_batch(self, batch)
```

No public description is currently available.


#### `nnx.nn.net.graph_nn_base.GraphNNBase`

```python
class nnx.nn.net.graph_nn_base.GraphNNBase(params: 'NNParams')
```

Abstract base for GNN architectures.

**Details**

```text
Subclasses must implement `_build_layers()` returning an `nn.ModuleList`
of PyG message-passing layers. The forward loop applies all-but-last
layers with the configured activation + dropout, then a bare final layer.
```

##### `nnx.nn.net.graph_nn_base.GraphNNBase.forward`

```python
nnx.nn.net.graph_nn_base.GraphNNBase.forward(self, X: 'torch.Tensor', E: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.nn.net.graph_nn_base.GraphNNBase.unpack_batch`

```python
nnx.nn.net.graph_nn_base.GraphNNBase.unpack_batch(self, batch) -> 'tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]'
```

No public description is currently available.

##### `nnx.nn.net.graph_nn_base.GraphNNBase.seed_count`

```python
nnx.nn.net.graph_nn_base.GraphNNBase.seed_count(self, batch) -> 'Optional[int]'
```

Number of seed rows at the head of a NeighborLoader subgraph.

**Details**

```text
NeighborLoader puts the ``batch_size`` seed nodes first and
appends their sampled neighbors — which can belong to *other*
splits. Loss and metrics must be computed on the seed rows only;
scoring neighbor rows leaks val/test labels into the training
loss and train labels into val metrics.

Returns None (no slicing) for anything that isn't a
NeighborLoader subgraph: plain full-graph ``Data`` has no
``batch_size``, and a multi-graph ``Batch.from_data_list``
collation DOES carry ``batch_size`` (= ``num_graphs``) but no
``input_id`` — slicing there would truncate node-level output
to the graph count. ``input_id`` is the NeighborLoader-specific
marker (the seed indices), so it gates the slice.
```


#### `nnx.nn.net.graph_conv_nn.GraphConvNN`

```python
class nnx.nn.net.graph_conv_nn.GraphConvNN(params: 'NNParams')
```

Abstract base for GNN architectures.

**Details**

```text
Subclasses must implement `_build_layers()` returning an `nn.ModuleList`
of PyG message-passing layers. The forward loop applies all-but-last
layers with the configured activation + dropout, then a bare final layer.
```


#### `nnx.nn.net.graph_sage_nn.GraphSageNN`

```python
class nnx.nn.net.graph_sage_nn.GraphSageNN(params: 'NNParams')
```

Abstract base for GNN architectures.

**Details**

```text
Subclasses must implement `_build_layers()` returning an `nn.ModuleList`
of PyG message-passing layers. The forward loop applies all-but-last
layers with the configured activation + dropout, then a bare final layer.
```


#### `nnx.nn.net.graph_att_nn.GraphAttNN`

```python
class nnx.nn.net.graph_att_nn.GraphAttNN(params: 'NNParams')
```

Abstract base for GNN architectures.

**Details**

```text
Subclasses must implement `_build_layers()` returning an `nn.ModuleList`
of PyG message-passing layers. The forward loop applies all-but-last
layers with the configured activation + dropout, then a bare final layer.
```


#### `nnx.nn.net.transformer_nn.TransformerNN`

```python
class nnx.nn.net.transformer_nn.TransformerNN(params: 'NNTransformerParams')
```

Base class for all neural network modules.

**Details**

```text
Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool
```

##### `nnx.nn.net.transformer_nn.TransformerNN.forward`

```python
nnx.nn.net.transformer_nn.TransformerNN.forward(self, tokens: 'torch.Tensor') -> 'torch.Tensor'
```

Args: tokens: (batch, seq) long tensor of token ids.

**Details**

```text
Returns:
    (batch, seq, vocab_size) logits — pre-softmax.
```

##### `nnx.nn.net.transformer_nn.TransformerNN.forward_with_cache`

```python
nnx.nn.net.transformer_nn.TransformerNN.forward_with_cache(self, tokens: 'torch.Tensor', past_kvs: 'Optional[list[LayerKV]]' = None) -> 'tuple[torch.Tensor, list[LayerKV]]'
```

Cache-threading forward used by ``GenerativeNNModel.generate``.

**Details**

```text
Behaves like ``forward`` but additionally accepts a per-layer
list of (k, v) caches (or ``None`` entries on the first call)
and returns the updated per-layer caches alongside the logits.

The total attended-to length per layer is
``past_kv_len + tokens.shape[1]`` — the caller is responsible
for ensuring that this stays within ``max_seq_len`` (the
generate loop slides a window when it would otherwise overflow).

Args:
    tokens: (batch, seq) long tensor of token ids. During
        incremental decode, ``seq == 1``; on the prefill step
        the prompt's full length is fed in one shot.
    past_kvs: list of length ``n_layers`` with each entry a
        ``(k, v)`` tuple or ``None``. ``None`` means "no
        history for this layer" (i.e., first call).

Returns:
    A tuple ``(logits, new_kvs)`` where ``logits`` is
        ``(batch, seq, vocab)`` — the *new* tokens' logits (with
        ``past_kvs != None`` and ``seq=1`` the returned
        ``logits[:, -1, :]`` is the next-token distribution
        conditioned on the full cached prefix) — and ``new_kvs``
        is a list of length ``n_layers`` of updated ``(k, v)``
        tuples; pass this back in for the next step.
```

##### `nnx.nn.net.transformer_nn.TransformerNN.unpack_batch`

```python
nnx.nn.net.transformer_nn.TransformerNN.unpack_batch(self, batch)
```

Make TransformerNN compatible with the standard supervised NNModel training loop.

**Details**

```text
For an LM the canonical batch is ``(tokens, targets)`` where
``targets = tokens[:, 1:]`` shifted by one. We don't shift here —
the caller assembles the tuple — but we accept either a 2-tuple
``(X, Y)`` or a plain tensor of tokens (next-token loss is then
computed in the train step).
```

##### `nnx.nn.net.transformer_nn.TransformerNN.to_file`

```python
nnx.nn.net.transformer_nn.TransformerNN.to_file(self, path: 'str') -> 'None'
```

No public description is currently available.

##### `nnx.nn.net.transformer_nn.TransformerNN.from_file`

```python
nnx.nn.net.transformer_nn.TransformerNN.from_file(path: 'str', params: 'NNTransformerParams', map_location='cpu') -> 'TransformerNN'
```

No public description is currently available.

##### `nnx.nn.net.transformer_nn.TransformerNN.from_state`

```python
nnx.nn.net.transformer_nn.TransformerNN.from_state(state_dict: 'dict', params: 'NNTransformerParams') -> 'TransformerNN'
```

No public description is currently available.


#### `nnx.nn.net.vit_nn.ViTNN`

```python
class nnx.nn.net.vit_nn.ViTNN(*, image_size: 'int' = 32, patch_size: 'int' = 4, in_channels: 'int' = 3, d_model: 'int' = 64, n_layers: 'int' = 4, n_heads: 'int' = 4, ffn_mult: 'int' = 4, attn_dropout: 'float' = 0.0, resid_dropout: 'float' = 0.0)
```

Small Vision Transformer encoder.

**Details**

```text
Forward contract:

  ``forward(x: (B, C, H, W), mask: Optional[BoolTensor[B, n_patches]]=None)``
  → ``(B, T_kept + 1, d_model)`` if ``mask`` provided (T_kept = mask.sum())
  → ``(B, n_patches + 1, d_model)`` otherwise.

The leading token is the learned CLS. Patches are flattened in
raster order (row-major over the patch grid). The optional ``mask``
is the I-JEPA "context" mask: True positions are kept, False ones
are dropped before any attention runs, so gradients do not flow
through masked patches.

``__init__`` requires ``image_size``, ``patch_size``, and
``in_channels`` for the patch-embedding convolution. ``image_size``
must be divisible by ``patch_size`` — validated at construction.
```

##### `nnx.nn.net.vit_nn.ViTNN.patch_positions`

```python
nnx.nn.net.vit_nn.ViTNN.patch_positions(self) -> 'torch.Tensor'
```

Return ``LongTensor[n_patches]`` of patch-token positions in the full sequence (i.e., ``arange(1, n_patches + 1)`` — CLS is position 0).

**Details**

```text
Exposed so the I-JEPA step factory can derive its context /
target position indices by boolean-masking this tensor instead
of rebuilding the arange (see ``jepa_train_step_factory``).
```

##### `nnx.nn.net.vit_nn.ViTNN.forward`

```python
nnx.nn.net.vit_nn.ViTNN.forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]' = None) -> 'torch.Tensor'
```

Run the encoder.

**Details**

```text
Args:
    x: (B, C, H, W) input image tensor.
    mask: optional BoolTensor of shape ``(B, n_patches)`` —
        True positions are kept, False ones are dropped *before*
        attention. Per-sample masks may have different
        ``True``-counts, but the resulting batch must have the
        same kept-count per row (asserted). I-JEPA's typical
        context mask is uniform across the batch (same set of
        patches kept on every sample in a step).

Returns:
    ``(B, T_kept + 1, d_model)`` where T_kept is the number of
    kept patches (or ``n_patches`` when mask is None). The +1
    is the CLS token at position 0.
```

##### `nnx.nn.net.vit_nn.ViTNN.unpack_batch`

```python
nnx.nn.net.vit_nn.ViTNN.unpack_batch(self, batch)
```

Standard ``(X-tuple, Y)`` adapter. JEPA doesn't use Y but the supervised linear-probe path on top of a frozen ViTNN does.


#### `nnx.nn.net.vit_nn.ViTBlock`

```python
class nnx.nn.net.vit_nn.ViTBlock(d_model: 'int', n_heads: 'int', ffn_mult: 'int' = 4, attn_dropout: 'float' = 0.0, resid_dropout: 'float' = 0.0)
```

Pre-norm ViT block: ``x = x + attn(RMSNorm(x)); x = x + ffn(RMSNorm(x))``.

**Details**

```text
Same shape as :class:`nnx.nn.net.transformer_layers.TransformerBlock`
but with bidirectional attention instead of causal. SwiGLU is
reused unchanged.
```

##### `nnx.nn.net.vit_nn.ViTBlock.forward`

```python
nnx.nn.net.vit_nn.ViTBlock.forward(self, x: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```


#### `nnx.nn.moe.MoELinear`

```python
class nnx.nn.moe.MoELinear(in_features: 'int', out_features: 'int', *, num_experts: 'int', top_k: 'int' = 2)
```

Sparse top-k Mixture-of-Experts drop-in for :class:`nn.Linear`.

**Details**

```text
Forward pass:

  1. Router (a bias-less :class:`nn.Linear`) projects input
     ``(B, in_features) → (B, num_experts)`` logits.
  2. ``top_k`` largest logits per row are kept; a softmax over
     those ``k`` values produces the per-expert gating weight.
  3. Each token is dispatched to its top-``k`` experts; expert
     outputs are weighted by the gating weights and summed into
     the output tensor.
  4. ``self.last_aux_loss`` is populated with the Switch-style
     load-balancing penalty
     ``num_experts · Σ_i f_i · P_i``. This is a scalar tensor with
     gradients wired to the router so optimization of the main
     loss + this term pushes routing toward uniform expert usage.

Args:
    in_features: input feature dimension (matches ``nn.Linear``).
    out_features: output feature dimension (matches ``nn.Linear``).
    num_experts: number of expert sub-networks. Must be ≥ 2.
        (``num_experts=1`` collapses to a plain linear with extra
        book-keeping; the layer rejects it to surface the misuse.)
    top_k: number of experts each input is routed through. Must
        be ≥ 1 and ≤ ``num_experts``. Defaults to 2 — the
        Switch-Transformer paper uses ``k=1``, but ``k=2`` is the
        broader MoE convention and tolerates a single misrouted
        expert without losing the entire token.

Attributes:
    router: bias-less :class:`nn.Linear` of shape
        ``(in_features, num_experts)``.
    experts: :class:`nn.ModuleList` of ``num_experts``
        :class:`nn.Linear` layers, each ``(in_features, out_features)``.
    top_k: how many experts run per token.
    num_experts: total expert count.
    last_aux_loss: scalar ``torch.Tensor`` set after each
        :meth:`forward`. ``None`` before the first forward.

Raises:
    ValueError: if ``in_features <= 0``, ``out_features <= 0``,
        ``num_experts <= 1``, ``top_k <= 0``, or ``top_k > num_experts``.
```

##### `nnx.nn.moe.MoELinear.forward`

```python
nnx.nn.moe.MoELinear.forward(self, x: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.nn.moe.MoELinear.extra_repr`

```python
nnx.nn.moe.MoELinear.extra_repr(self) -> 'str'
```

Return the extra representation of the module.

**Details**

```text
To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.
```


## 5. Datasets

#### `nnx.nn.dataset.nn_dataset_base.NNDatasetBase`

```python
class nnx.nn.dataset.nn_dataset_base.NNDatasetBase() -> 'None'
```

NNDatasetBase()

##### `nnx.nn.dataset.nn_dataset_base.NNDatasetBase.state`

```python
nnx.nn.dataset.nn_dataset_base.NNDatasetBase.state(self) -> 'dict'
```

No public description is currently available.


#### `nnx.nn.dataset.nn_dataset.NNDataset`

```python
class nnx.nn.dataset.nn_dataset.NNDataset(*, ds_class: 'type[VisionDataset]', root_dir: 'str' = './data', download: 'bool' = True, transform: 'Optional[Callable]' = None, batch_sizes: 'tuple[Optional[int], Optional[int], Optional[int]]' = (None, None, None), val_proportion: 'float' = 0.1, seed: 'Optional[int]' = None) -> 'None'
```

Vision dataset wrapper. `val_proportion` carves a validation slice out of the source `train=True` split (NOT out of the test split, which stays untouched for final evaluation).


#### `nnx.nn.dataset.nn_graph_dataset.NNGraphDataset`

```python
class nnx.nn.dataset.nn_graph_dataset.NNGraphDataset(*, ds_class: 'type[Dataset]', n_neighbors: 'Optional[list[int]]' = None, root_dir: 'str' = './data', transform: 'Optional[Callable]' = None, n_workers: 'int' = 4, batch_sizes: 'tuple[Optional[int], Optional[int], Optional[int]]' = (None, None, None), seed: 'Optional[int]' = None, sampler: "Literal['neighbor', 'full']" = 'neighbor') -> 'None'
```

NNGraphDataset(*, ds_class: 'type[Dataset]', n_neighbors: 'Optional[list[int]]' = None, root_dir: 'str' = './data', transform: 'Optional[Callable]' = None, n_workers: 'int' = 4, batch_sizes: 'tuple[Optional[int], Optional[int], Optional[int]]' = (None, None, None), seed: 'Optional[int]' = None, sampler: "Literal['neighbor', 'full']" = 'neighbor')


#### `nnx.nn.dataset.nn_tabular_dataset.NNTabularDataset`

```python
class nnx.nn.dataset.nn_tabular_dataset.NNTabularDataset(*, df: 'pd.DataFrame', feature_cols: 'list[str]', target_col: 'str', batch_sizes: 'tuple[Optional[int], Optional[int], Optional[int]]' = (None, None, None), val_proportion: 'float' = 0.15, test_proportion: 'float' = 0.15, name_override: 'Optional[str]' = None, feature_dtype: 'torch.dtype' = torch.float32, target_dtype: 'Optional[torch.dtype]' = None, seed: 'Optional[int]' = None) -> 'None'
```

Wrap a pandas DataFrame as train/val/test DataLoaders.

**Details**

```text
`feature_cols` columns are stacked into the input tensor; `target_col`
is the target column. By default, targets are coerced to int64 (long)
and validated as contiguous integer classes 0..K-1 (classification);
the loaders yield 1-D class-index targets `(batch,)` (the
`CrossEntropyLoss` convention). Set `target_dtype` to a floating-point
dtype (e.g. `torch.float32`) to skip the integer cast and contiguity
check and fix `output_dim=1` for regression; the loaders then yield
targets of shape `(batch, 1)` so they line up with a model whose
final linear layer has one output. Integer dtypes are rejected —
leave `target_dtype` unset (`None`) for classification.
```


#### `nnx.nn.dataset.nn_preference_dataset.NNPreferenceDataset`

```python
class nnx.nn.dataset.nn_preference_dataset.NNPreferenceDataset(*, prompts: 'list[str]', chosen: 'list[str]', rejected: 'list[str]', tokenizer: 'object', max_prompt_len: 'int' = 64, max_response_len: 'int' = 64, pad_token_id: 'int' = 0, batch_sizes: 'tuple[Optional[int], Optional[int], Optional[int]]' = (None, None, None), val_proportion: 'float' = 0.1, test_proportion: 'float' = 0.1, name_override: 'Optional[str]' = None, seed: 'Optional[int]' = None) -> 'None'
```

Wrap parallel lists of (prompt, chosen, rejected) strings as DPO loaders.

**Details**

```text
Tokenizes every triple through ``tokenizer.encode`` once at
construction, pads/truncates to fixed lengths, then splits into
train / val / test ``DataLoader``\ s with the same shape as the
rest of :class:`NNDatasetBase` (so callbacks and the standard
training loop work unchanged).

Each batch yielded is ``(prompt_ids, chosen_ids, rejected_ids)``
where each entry is ``(B, T_*)`` ``torch.LongTensor``.
```


## 6. Enums

#### `nnx.nn.enum.activations.Activations`

```python
class nnx.nn.enum.activations.Activations(Enum)
```

Enum values: `ELU`, `SELU`, `TANH`, `RELU`, `SOFTMAX`, `SIGMOID`, `SOFTPLUS`, `LEAKY_RELU`.

##### `nnx.nn.enum.activations.Activations.ELU`

```python
nnx.nn.enum.activations.Activations.ELU = 'elu'
```

Enum value `elu`.

##### `nnx.nn.enum.activations.Activations.SELU`

```python
nnx.nn.enum.activations.Activations.SELU = 'selu'
```

Enum value `selu`.

##### `nnx.nn.enum.activations.Activations.TANH`

```python
nnx.nn.enum.activations.Activations.TANH = 'tanh'
```

Enum value `tanh`.

##### `nnx.nn.enum.activations.Activations.RELU`

```python
nnx.nn.enum.activations.Activations.RELU = 'relu'
```

Enum value `relu`.

##### `nnx.nn.enum.activations.Activations.SOFTMAX`

```python
nnx.nn.enum.activations.Activations.SOFTMAX = 'softmax'
```

Enum value `softmax`.

##### `nnx.nn.enum.activations.Activations.SIGMOID`

```python
nnx.nn.enum.activations.Activations.SIGMOID = 'sigmoid'
```

Enum value `sigmoid`.

##### `nnx.nn.enum.activations.Activations.SOFTPLUS`

```python
nnx.nn.enum.activations.Activations.SOFTPLUS = 'softplus'
```

Enum value `softplus`.

##### `nnx.nn.enum.activations.Activations.LEAKY_RELU`

```python
nnx.nn.enum.activations.Activations.LEAKY_RELU = 'leaky_relu'
```

Enum value `leaky_relu`.


#### `nnx.nn.enum.checkpoints.Checkpoints`

```python
class nnx.nn.enum.checkpoints.Checkpoints(Enum)
```

Enum values: `Q1`, `Q2`, `Q3`, `BEST`, `LAST`, `FIRST`.

##### `nnx.nn.enum.checkpoints.Checkpoints.Q1`

```python
nnx.nn.enum.checkpoints.Checkpoints.Q1 = 'q1'
```

Enum value `q1`.

##### `nnx.nn.enum.checkpoints.Checkpoints.Q2`

```python
nnx.nn.enum.checkpoints.Checkpoints.Q2 = 'q2'
```

Enum value `q2`.

##### `nnx.nn.enum.checkpoints.Checkpoints.Q3`

```python
nnx.nn.enum.checkpoints.Checkpoints.Q3 = 'q3'
```

Enum value `q3`.

##### `nnx.nn.enum.checkpoints.Checkpoints.BEST`

```python
nnx.nn.enum.checkpoints.Checkpoints.BEST = 'best'
```

Enum value `best`.

##### `nnx.nn.enum.checkpoints.Checkpoints.LAST`

```python
nnx.nn.enum.checkpoints.Checkpoints.LAST = 'last'
```

Enum value `last`.

##### `nnx.nn.enum.checkpoints.Checkpoints.FIRST`

```python
nnx.nn.enum.checkpoints.Checkpoints.FIRST = 'first'
```

Enum value `first`.


#### `nnx.nn.enum.devices.Devices`

```python
class nnx.nn.enum.devices.Devices(Enum)
```

Enum values: `CPU`, `MPS`, `CUDA`.

##### `nnx.nn.enum.devices.Devices.CPU`

```python
nnx.nn.enum.devices.Devices.CPU = 'cpu'
```

Enum value `cpu`.

##### `nnx.nn.enum.devices.Devices.MPS`

```python
nnx.nn.enum.devices.Devices.MPS = 'mps'
```

Enum value `mps`.

##### `nnx.nn.enum.devices.Devices.CUDA`

```python
nnx.nn.enum.devices.Devices.CUDA = 'cuda'
```

Enum value `cuda`.

##### `nnx.nn.enum.devices.Devices.torch_device`

```python
nnx.nn.enum.devices.Devices.torch_device(self) -> 'torch.device'
```

Explicit alias for ``self()`` — more readable in code that mixes the enum and torch.device usage.

##### `nnx.nn.enum.devices.Devices.get`

```python
nnx.nn.enum.devices.Devices.get() -> 'Devices'
```

No public description is currently available.

##### `nnx.nn.enum.devices.Devices.get_torch_device`

```python
nnx.nn.enum.devices.Devices.get_torch_device() -> 'torch.device'
```

Convenience: auto-detect and return the corresponding torch.device directly. Equivalent to ``Devices.get().torch_device()``.


#### `nnx.nn.enum.losses.Losses`

```python
class nnx.nn.enum.losses.Losses(Enum)
```

Enum values: `CROSS_ENTROPY`, `MEAN_SQUARED_ERROR`, `BINARY_CROSS_ENTROPY`, `NEGATIVE_LOG_LIKELIHOOD`.

##### `nnx.nn.enum.losses.Losses.CROSS_ENTROPY`

```python
nnx.nn.enum.losses.Losses.CROSS_ENTROPY = 'cross_entropy'
```

Enum value `cross_entropy`.

##### `nnx.nn.enum.losses.Losses.MEAN_SQUARED_ERROR`

```python
nnx.nn.enum.losses.Losses.MEAN_SQUARED_ERROR = 'mean_squared_error'
```

Enum value `mean_squared_error`.

##### `nnx.nn.enum.losses.Losses.BINARY_CROSS_ENTROPY`

```python
nnx.nn.enum.losses.Losses.BINARY_CROSS_ENTROPY = 'binary_cross_entropy'
```

Enum value `binary_cross_entropy`.

##### `nnx.nn.enum.losses.Losses.NEGATIVE_LOG_LIKELIHOOD`

```python
nnx.nn.enum.losses.Losses.NEGATIVE_LOG_LIKELIHOOD = 'negative_log_likelihood'
```

Enum value `negative_log_likelihood`.


#### `nnx.nn.enum.nets.Nets`

```python
class nnx.nn.enum.nets.Nets(Enum)
```

Enum values: `CONV`, `FEED_FWD`, `FEED_FWD_MOE`, `GRAPH_ATT`, `GRAPH_CONV`, `GRAPH_SAGE`, `TRANSFORMER`.

##### `nnx.nn.enum.nets.Nets.CONV`

```python
nnx.nn.enum.nets.Nets.CONV = 'conv'
```

Enum value `conv`.

##### `nnx.nn.enum.nets.Nets.FEED_FWD`

```python
nnx.nn.enum.nets.Nets.FEED_FWD = 'feed_fwd'
```

Enum value `feed_fwd`.

##### `nnx.nn.enum.nets.Nets.FEED_FWD_MOE`

```python
nnx.nn.enum.nets.Nets.FEED_FWD_MOE = 'feed_fwd_moe'
```

Enum value `feed_fwd_moe`.

##### `nnx.nn.enum.nets.Nets.GRAPH_ATT`

```python
nnx.nn.enum.nets.Nets.GRAPH_ATT = 'graph_att'
```

Enum value `graph_att`.

##### `nnx.nn.enum.nets.Nets.GRAPH_CONV`

```python
nnx.nn.enum.nets.Nets.GRAPH_CONV = 'graph_conv'
```

Enum value `graph_conv`.

##### `nnx.nn.enum.nets.Nets.GRAPH_SAGE`

```python
nnx.nn.enum.nets.Nets.GRAPH_SAGE = 'graph_sage'
```

Enum value `graph_sage`.

##### `nnx.nn.enum.nets.Nets.TRANSFORMER`

```python
nnx.nn.enum.nets.Nets.TRANSFORMER = 'transformer'
```

Enum value `transformer`.


#### `nnx.nn.enum.optims.Optims`

```python
class nnx.nn.enum.optims.Optims(Enum)
```

Enum values: `SGD`, `ADAM`, `ADAM_AMSGRAD`, `SGD_NESTEROV`.

##### `nnx.nn.enum.optims.Optims.SGD`

```python
nnx.nn.enum.optims.Optims.SGD = 'sgd'
```

Enum value `sgd`.

##### `nnx.nn.enum.optims.Optims.ADAM`

```python
nnx.nn.enum.optims.Optims.ADAM = 'adam'
```

Enum value `adam`.

##### `nnx.nn.enum.optims.Optims.ADAM_AMSGRAD`

```python
nnx.nn.enum.optims.Optims.ADAM_AMSGRAD = 'adam_amsgrad'
```

Enum value `adam_amsgrad`.

##### `nnx.nn.enum.optims.Optims.SGD_NESTEROV`

```python
nnx.nn.enum.optims.Optims.SGD_NESTEROV = 'sgd_nesterov'
```

Enum value `sgd_nesterov`.


#### `nnx.nn.enum.schedulers.Schedulers`

```python
class nnx.nn.enum.schedulers.Schedulers(Enum)
```

Enum values: `REDUCE_LR_ON_PLATEAU`, `STEP`, `COSINE_ANNEALING`, `ONE_CYCLE`, `LINEAR_WARMUP_DECAY`.

##### `nnx.nn.enum.schedulers.Schedulers.REDUCE_LR_ON_PLATEAU`

```python
nnx.nn.enum.schedulers.Schedulers.REDUCE_LR_ON_PLATEAU = 'reduce_lr_on_plateau'
```

Enum value `reduce_lr_on_plateau`.

##### `nnx.nn.enum.schedulers.Schedulers.STEP`

```python
nnx.nn.enum.schedulers.Schedulers.STEP = 'step'
```

Enum value `step`.

##### `nnx.nn.enum.schedulers.Schedulers.COSINE_ANNEALING`

```python
nnx.nn.enum.schedulers.Schedulers.COSINE_ANNEALING = 'cosine_annealing'
```

Enum value `cosine_annealing`.

##### `nnx.nn.enum.schedulers.Schedulers.ONE_CYCLE`

```python
nnx.nn.enum.schedulers.Schedulers.ONE_CYCLE = 'one_cycle'
```

Enum value `one_cycle`.

##### `nnx.nn.enum.schedulers.Schedulers.LINEAR_WARMUP_DECAY`

```python
nnx.nn.enum.schedulers.Schedulers.LINEAR_WARMUP_DECAY = 'linear_warmup_decay'
```

Enum value `linear_warmup_decay`.


## 7. Callbacks

#### `nnx.nn.callbacks.Callback`

```python
class nnx.nn.callbacks.Callback()
```

Base class for training callbacks. Override any subset of the hooks.

##### `nnx.nn.callbacks.Callback.on_train_begin`

```python
nnx.nn.callbacks.Callback.on_train_begin(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.

##### `nnx.nn.callbacks.Callback.on_epoch_begin`

```python
nnx.nn.callbacks.Callback.on_epoch_begin(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.

##### `nnx.nn.callbacks.Callback.on_epoch_end`

```python
nnx.nn.callbacks.Callback.on_epoch_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.

##### `nnx.nn.callbacks.Callback.on_train_end`

```python
nnx.nn.callbacks.Callback.on_train_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.

##### `nnx.nn.callbacks.Callback.checkpoint_transforms`

```python
nnx.nn.callbacks.Callback.checkpoint_transforms(self) -> 'tuple[NNCheckpointTransform, ...]'
```

Completed topology transforms to persist on the final checkpoint.


#### `nnx.nn.callbacks.EarlyStopping`

```python
class nnx.nn.callbacks.EarlyStopping(monitor: 'str' = 'val_edp.error', patience: 'int' = 10, min_delta: 'float' = 0.0, mode: 'str' = 'min')
```

Stop training when the monitored metric stops improving.

**Details**

```text
Args:
    monitor: which IDP field to track. "val_edp.error" (default), "val_edp.loss",
             "train_edp.error", or "train_edp.loss".
    patience: epochs with no improvement before stopping.
    min_delta: minimum change to qualify as improvement.
    mode: "min" (default) for loss/error; "max" for accuracy/f1.
```

##### `nnx.nn.callbacks.EarlyStopping.on_train_begin`

```python
nnx.nn.callbacks.EarlyStopping.on_train_begin(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.

##### `nnx.nn.callbacks.EarlyStopping.on_epoch_end`

```python
nnx.nn.callbacks.EarlyStopping.on_epoch_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.


#### `nnx.nn.callbacks.LRMonitor`

```python
class nnx.nn.callbacks.LRMonitor()
```

Logs the current LR each epoch. History exposed at `.history`.

##### `nnx.nn.callbacks.LRMonitor.on_epoch_end`

```python
nnx.nn.callbacks.LRMonitor.on_epoch_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.


#### `nnx.nn.callbacks.ModelCheckpoint`

```python
class nnx.nn.callbacks.ModelCheckpoint(epochs: 'Optional[list[int]]' = None, tag: 'str' = 'custom')
```

Save a custom-tagged checkpoint at user-specified epochs.

**Details**

```text
The standard train() loop already saves FIRST / Q1 / Q2 / Q3 / LAST / BEST
via the Checkpoints enum. This callback adds ad-hoc save points outside
that cycle — useful for sampling at fixed milestones (e.g., epoch 10,
20, 50) for downstream inspection.

Each match writes ``<cwd>/runs/<run.id>/checkpoints/<tag>_e<epoch>.pt``
— cwd-relative, matching what :meth:`NNRun.save` and :class:`NNCheckpoint`
use when called from inside :meth:`NNModel.train` (the train() entry
point doesn't accept a ``root=`` parameter). The epoch suffix
prevents successive matches from overwriting each other when
``epochs`` has multiple entries.

Args:
    epochs: list of 0-indexed epoch numbers at which to save. Empty /
        None means the callback never fires (and never saves anything).
    tag: prefix in the filename, defaults to ``"custom"``.
```

##### `nnx.nn.callbacks.ModelCheckpoint.on_epoch_end`

```python
nnx.nn.callbacks.ModelCheckpoint.on_epoch_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.


#### `nnx.nn.callbacks.TensorBoardCallback`

```python
class nnx.nn.callbacks.TensorBoardCallback(log_dir: 'Optional[str]' = None, flush_each_epoch: 'bool' = True)
```

Stream train/val metrics + LR to a TensorBoard SummaryWriter.

**Details**

```text
Requires `tensorboard` to be installed — imported lazily so users who
don't use this callback don't pay the dependency cost.

Args:
    log_dir: directory passed to SummaryWriter. None lets TensorBoard
        pick its default (runs/<datetime>).
    flush_each_epoch: when True (default), calls writer.flush() so
        partial training is visible in TB even if the process crashes.
```

##### `nnx.nn.callbacks.TensorBoardCallback.on_epoch_end`

```python
nnx.nn.callbacks.TensorBoardCallback.on_epoch_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.

##### `nnx.nn.callbacks.TensorBoardCallback.on_train_end`

```python
nnx.nn.callbacks.TensorBoardCallback.on_train_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.


#### `nnx.nn.callbacks.WandbCallback`

```python
class nnx.nn.callbacks.WandbCallback(project: 'Optional[str]' = None, wandb_run=None, **init_kwargs)
```

Stream train/val metrics + LR to Weights & Biases.

**Details**

```text
Requires `wandb` — lazily imported. Pass `project=` to start a new run,
or `wandb_run=` to attach to an externally-managed run.
```

##### `nnx.nn.callbacks.WandbCallback.on_epoch_end`

```python
nnx.nn.callbacks.WandbCallback.on_epoch_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.

##### `nnx.nn.callbacks.WandbCallback.on_train_end`

```python
nnx.nn.callbacks.WandbCallback.on_train_end(self, ctx: '_CallbackContext') -> 'None'
```

No public description is currently available.


## 8. Fine-tuning (`nnx.finetune`)

#### `nnx.finetune.freezing.freeze`

```python
nnx.finetune.freezing.freeze(module: 'nn.Module', *patterns: 'str') -> 'int'
```

Set ``requires_grad=False`` on every parameter under ``module`` whose dotted name matches any of ``patterns``.

**Details**

```text
Patterns use ``fnmatch`` shell-glob semantics: ``*`` matches any
sequence of characters **including dots** (not just one path segment),
``?`` matches a single character, ``[seq]`` matches one character
from the set. Match is against the parameter's full dotted name,
e.g., ``encoder.layer.5.weight``. So ``"encoder.*"`` matches every
parameter under the encoder subtree, including deeply nested ones
like ``encoder.layer.5.weight``.

Args:
    module: any ``nn.Module``.
    *patterns: one or more fnmatch globs. If no patterns are
        given, raises ``ValueError`` (freeze-all-by-default is too
        dangerous to be the no-arg behavior).

Returns:
    The number of parameters newly frozen (i.e., previously had
    ``requires_grad=True``). Useful for assertion / logging.
```


#### `nnx.finetune.freezing.unfreeze`

```python
nnx.finetune.freezing.unfreeze(module: 'nn.Module', *patterns: 'str') -> 'int'
```

Mirror of :func:`freeze` — set ``requires_grad=True`` on matching parameters. Returns the count newly unfrozen.


#### `nnx.finetune.freezing.frozen`

```python
nnx.finetune.freezing.frozen(module: 'nn.Module') -> 'list[str]'
```

List the dotted parameter names currently frozen under ``module``.

**Details**

```text
Returned list is sorted by name for stable test assertions. Useful
for logging at ``train()`` entry so users can see exactly which
parameters are excluded from training.
```


#### `nnx.finetune.loading.load_pretrained`

```python
nnx.finetune.loading.load_pretrained(module: 'nn.Module', source: 'Union[str, Path, dict, nn.Module]', *, key_map: 'Optional[dict[str, str]]' = None, strict: 'bool' = False, prefix: 'Optional[str]' = None) -> 'LoadPretrainedResult'
```

Load weights into ``module`` from an external source.

**Details**

```text
The source can be:
  - a path (str or Path) to a ``.pt`` / ``.pth`` file holding a
    state-dict (loaded with ``weights_only=True`` for safety);
  - a state-dict (``dict``) already in memory;
  - another ``nn.Module``, in which case its state-dict is used.

Args:
    module: target module to load into. Mutated in place.
    source: see above.
    key_map: optional remapping from source keys to target keys,
        applied AFTER ``prefix`` stripping and BEFORE matching.
        Each entry is a **prefix** substitution: for the first
        key in ``key_map`` whose prefix matches the source key,
        that prefix is replaced with the mapped value. E.g.,
        ``{"backbone.": "net."}`` rewrites ``backbone.conv1.weight``
        to ``net.conv1.weight``; later occurrences of ``backbone.``
        mid-string are NOT touched. First-match-wins; subsequent
        entries don't fire once a key has been remapped.
    strict: when True, raise if any source key has no target match
        OR any target key has no source. Default False (fine-tuning
        commonly partial-loads).
    prefix: optional prefix to strip from source keys before
        matching. E.g., ``prefix="model."`` turns ``model.layer.0``
        into ``layer.0``. Applied BEFORE ``key_map``.

Returns:
    :class:`LoadPretrainedResult` with the loaded / missing /
    unexpected key sets.
```


#### `nnx.finetune.loading.LoadPretrainedResult`

```python
class nnx.finetune.loading.LoadPretrainedResult(loaded_keys: 'list[str]', missing_keys: 'list[str]', unexpected_keys: 'list[str]') -> 'None'
```

Outcome of a :func:`load_pretrained` call.

**Details**

```text
Compared with :meth:`torch.nn.Module.load_state_dict`, this gives
you back not just the missing/unexpected keys but also the list
of keys actually applied (after any remapping) — useful for
confirming the load did what you intended.
```


#### `nnx.finetune.param_groups.NNParamGroupSpec`

```python
class nnx.finetune.param_groups.NNParamGroupSpec(*, name_pattern: 'str', lr: 'Optional[float]' = None, lr_multiplier: 'Optional[float]' = None, weight_decay: 'Optional[float]' = None) -> 'None'
```

One row in :attr:`NNOptimParams.param_groups`.

**Details**

```text
Matches parameters whose dotted name matches ``name_pattern``
(fnmatch glob) and applies the specified ``lr`` (absolute) or
``lr_multiplier`` (multiplied by ``NNOptimParams.max_lr``) and
optional ``weight_decay`` override.

Exactly one of ``lr`` and ``lr_multiplier`` may be set. If both are
None the matched parameters use the optimizer's default LR — handy
when you only want to override ``weight_decay`` for a group.

Example:
    # Freeze nothing, but train the backbone at 1/100th the head's LR
    # and disable weight_decay on every bias term.
    NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=5e-4,
        param_groups=[
            NNParamGroupSpec(name_pattern="encoder.*", lr_multiplier=0.01),
            NNParamGroupSpec(name_pattern="*.bias", weight_decay=0.0),
        ],
    )
```

##### `nnx.finetune.param_groups.NNParamGroupSpec.state`

```python
nnx.finetune.param_groups.NNParamGroupSpec.state(self) -> 'dict'
```

No public description is currently available.

##### `nnx.finetune.param_groups.NNParamGroupSpec.from_state`

```python
nnx.finetune.param_groups.NNParamGroupSpec.from_state(state: 'dict') -> 'NNParamGroupSpec'
```

No public description is currently available.


#### `nnx.finetune.param_groups.build_param_groups`

```python
nnx.finetune.param_groups.build_param_groups(module: 'nn.Module', specs: 'list[NNParamGroupSpec]', *, default_lr: 'float', default_weight_decay: 'float', strict: 'bool' = False) -> 'list[dict]'
```

Walk ``module``'s parameters, bucket them by the first matching spec (or into a fallback default group), and return the list of param-group dicts the optimizer expects.

**Details**

```text
Parameters with ``requires_grad=False`` are dropped — they're
frozen, the optimizer doesn't need to see them. (Without this, the
optimizer would still hold them in its state but they'd never
update; harmless but wasteful and confusing in `optimizer.param_groups`.)

Args:
    module: source of parameters to bucket.
    specs: list of :class:`NNParamGroupSpec` in priority order.
        The first spec whose ``name_pattern`` matches a parameter's
        dotted name wins.
    default_lr: LR for parameters that don't match any spec, or
        for specs that omit both ``lr`` and ``lr_multiplier``.
    default_weight_decay: WD for parameters that don't match any
        spec's ``weight_decay`` override.
    strict: when False (default, fine-tuning semantics), parameters
        that match no spec go into a default group at ``default_lr``
        so every trainable parameter ends up in the optimizer. When
        True (multi-optimizer Trainer semantics), unmatched parameters
        are DROPPED from the optimizer entirely — the contract is
        "this optimizer owns only what the specs explicitly select",
        which is what allows disjoint optimizers in
        :class:`nnx.trainer.Trainer`.

Returns:
    A list of dicts suitable for ``torch.optim.Optimizer(
    params, ...)`` — each entry has ``"params"`` plus any overrides.
```


## 9. Parameter-efficient fine-tuning (`nnx.peft`)

LoRA + DoRA + IA3 + Prefix-Tuning + Prompt-Tuning + Adapters. All methods share the same in-place wrap + save/load idiom (per-method `save_*_weights` / `load_*_weights` persist only the trainable delta).

### 9.1. LoRA

#### `nnx.peft.lora.LoRALinear`

```python
class nnx.peft.lora.LoRALinear(base: 'nn.Linear', *, r: 'int' = 8, alpha: 'float' = 16.0, dropout: 'float' = 0.0)
```

Linear layer wrapped with a LoRA low-rank residual.

**Details**

```text
The original :class:`nn.Linear` lives at ``self.base`` with its
parameters frozen (``requires_grad=False``) on construction.
``lora_A`` and ``lora_B`` are trainable; ``lora_A`` uses
Kaiming-uniform init and ``lora_B`` is zero-initialized so the
layer's output at step 0 equals the base layer's output exactly
— fine-tuning starts from the pretrained behavior and diverges
only as B picks up gradient.

The wrapper preserves the base layer's ``in_features`` /
``out_features``, so consumers that read ``base.weight.shape`` or
pass tensors through the layer don't change.
```

##### `nnx.peft.lora.LoRALinear.in_features`

```python
property nnx.peft.lora.LoRALinear.in_features
```

No public description is currently available.

##### `nnx.peft.lora.LoRALinear.out_features`

```python
property nnx.peft.lora.LoRALinear.out_features
```

No public description is currently available.

##### `nnx.peft.lora.LoRALinear.forward`

```python
nnx.peft.lora.LoRALinear.forward(self, x: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.peft.lora.LoRALinear.extra_repr`

```python
nnx.peft.lora.LoRALinear.extra_repr(self) -> 'str'
```

Return the extra representation of the module.

**Details**

```text
To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.
```


#### `nnx.peft.lora.apply_lora_to`

```python
nnx.peft.lora.apply_lora_to(module: 'nn.Module', *name_patterns: 'str', r: 'int' = 8, alpha: 'float' = 16.0, dropout: 'float' = 0.0) -> 'int'
```

Wrap every :class:`nn.Linear` submodule whose dotted name matches any of ``name_patterns`` with a :class:`LoRALinear`. Returns the number of layers wrapped.

**Details**

```text
Patterns use shell-style globs (``fnmatch``) against the dotted
submodule name as it appears in ``module.named_modules()`` — e.g.,
``"layers.0"``, ``"encoder.*"``, ``"*"`` for every Linear.

The wrap is in-place: each matched layer is removed from its parent
and replaced with a :class:`LoRALinear` wrapping it. The base
layer's parameters end up frozen as a side effect of LoRALinear's
construction; the LoRA parameters (``lora_A`` / ``lora_B``) are
trainable by default.

Args:
    module: root module to walk. The function mutates ``module``
        in place.
    name_patterns: at least one fnmatch glob. Empty raises.
    r: LoRA rank — passed through to :class:`LoRALinear`.
    alpha: LoRA scaling numerator — passed through.
    dropout: dropout on the LoRA path — passed through.

Returns:
    The count of layers wrapped (may be 0 if no patterns match).

Raises:
    ValueError: if ``name_patterns`` is empty.

**Idempotency note:** if a layer is already a :class:`LoRALinear`,
its inner ``.base`` is skipped — re-applying ``apply_lora_to``
against the same patterns is a no-op for layers that already
carry a LoRA wrapper. The function returns the count of NEW wraps.
```


#### `nnx.peft.lora.save_lora_weights`

```python
nnx.peft.lora.save_lora_weights(module: 'nn.Module', path: 'Union[str, Path]') -> 'str'
```

Save ONLY the LoRA parameters of ``module`` to ``path``.

**Details**

```text
The output is a plain ``torch.save`` of a dict-subset of the full
state_dict, containing only keys with ``lora_A`` or ``lora_B`` in
them. Loadable via :func:`load_lora_weights`.

Args:
    module: any module that has been processed by
        :func:`apply_lora_to`. If no LoRA params exist, an empty
        dict is saved (the caller decides whether that's an error).
    path: destination file path.

Returns:
    The path written (so calls can be chained).
```


#### `nnx.peft.lora.load_lora_weights`

```python
nnx.peft.lora.load_lora_weights(module: 'nn.Module', source: 'Union[str, Path, dict]') -> 'int'
```

Load LoRA parameters into ``module`` from ``source``.

**Details**

```text
Args:
    module: must already have :class:`LoRALinear` wrappers in the
        same positions as the source — apply_lora_to FIRST, then
        call this. Otherwise the keys won't match and 0 params load.
    source: either a path to a file produced by
        :func:`save_lora_weights`, or a state-dict dict directly.

Returns:
    The number of parameter tensors loaded.

Loads via ``module.load_state_dict(..., strict=False)`` so the
base layer's frozen weights — which are NOT in the LoRA-only
checkpoint — don't trigger a missing-keys error.
```


### 9.2. DoRA

#### `nnx.peft.dora.DoRALinear`

```python
class nnx.peft.dora.DoRALinear(base: 'nn.Linear', *, r: 'int' = 8, alpha: 'float' = 16.0, dropout: 'float' = 0.0)
```

Linear layer wrapped with a DoRA weight decomposition.

**Details**

```text
Subclasses :class:`LoRALinear` to inherit the frozen-base + trainable
low-rank residual machinery (``lora_A``, ``lora_B``, alpha/r scaling,
optional dropout, base-freeze-on-construction). Adds a trainable
per-output-row ``magnitude`` parameter (shape: ``out_features``)
initialized from the column-wise L2 norm of the base weight.

The forward composes the LoRA residual into a combined weight
``V = W_0 + (α/r) · BA``, normalizes ``V`` row-wise, then re-scales
by the trainable magnitude:

    ``W = magnitude.unsqueeze(1) * V / ||V||_c``
    ``y = W · x + b``

At step 0, ``B`` is zero-initialized (inherited from LoRALinear)
so ``V = W_0`` and ``||V||_c == magnitude``, giving ``W == W_0``
exactly — fine-tuning starts from the pretrained behavior.

Args:
    base: the :class:`nn.Linear` to wrap. Its parameters are frozen
        on construction (inherited from LoRALinear).
    r: low-rank dim for the LoRA residual. Must be positive.
    alpha: scaling numerator. Effective LoRA scale is ``alpha / r``.
    dropout: dropout on the LoRA update path. Range ``[0, 1)``.
```

##### `nnx.peft.dora.DoRALinear.forward`

```python
nnx.peft.dora.DoRALinear.forward(self, x: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.peft.dora.DoRALinear.extra_repr`

```python
nnx.peft.dora.DoRALinear.extra_repr(self) -> 'str'
```

Return the extra representation of the module.

**Details**

```text
To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.
```


#### `nnx.peft.dora.apply_dora_to`

```python
nnx.peft.dora.apply_dora_to(module: 'nn.Module', *name_patterns: 'str', r: 'int' = 8, alpha: 'float' = 16.0, dropout: 'float' = 0.0) -> 'int'
```

Wrap every :class:`nn.Linear` submodule whose dotted name matches any of ``name_patterns`` with a :class:`DoRALinear`. Returns the number of layers wrapped.

**Details**

```text
Mirrors :func:`nnx.peft.apply_lora_to` — same fnmatch glob conventions,
same two-phase (collect-then-mutate) traversal, same idempotency
contract (existing DoRA/LoRA wrappers are not re-wrapped — the
parent-is-LoRALinear check covers DoRALinear by inheritance).

Args:
    module: root module to walk. Mutated in place.
    name_patterns: at least one fnmatch glob.
    r: LoRA rank — passed through.
    alpha: LoRA scaling numerator — passed through.
    dropout: dropout on the LoRA update path — passed through.

Returns:
    The count of layers wrapped (may be 0 if no patterns match
    or every match is already wrapped).

Raises:
    ValueError: if ``name_patterns`` is empty.
```


### 9.3. IA3

#### `nnx.peft.ia3.IA3Linear`

```python
class nnx.peft.ia3.IA3Linear(base: 'nn.Linear')
```

Linear layer wrapped with an IA3 per-output-dim scaling vector.

**Details**

```text
The original :class:`nn.Linear` lives at ``self.base`` with its
parameters frozen (``requires_grad=False``) on construction.
``scaling`` is the only trainable parameter: a length-``out_features``
vector initialized to all-ones so the layer's output at step 0
equals the base layer's output exactly.

Forward: ``y = base(x) * scaling`` (broadcast over the trailing dim).

Args:
    base: the :class:`nn.Linear` to wrap.
```

##### `nnx.peft.ia3.IA3Linear.in_features`

```python
property nnx.peft.ia3.IA3Linear.in_features
```

No public description is currently available.

##### `nnx.peft.ia3.IA3Linear.out_features`

```python
property nnx.peft.ia3.IA3Linear.out_features
```

No public description is currently available.

##### `nnx.peft.ia3.IA3Linear.forward`

```python
nnx.peft.ia3.IA3Linear.forward(self, x: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.peft.ia3.IA3Linear.extra_repr`

```python
nnx.peft.ia3.IA3Linear.extra_repr(self) -> 'str'
```

Return the extra representation of the module.

**Details**

```text
To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.
```


#### `nnx.peft.ia3.apply_ia3_to`

```python
nnx.peft.ia3.apply_ia3_to(module: 'nn.Module', *name_patterns: 'str') -> 'int'
```

Wrap every :class:`nn.Linear` submodule whose dotted name matches any of ``name_patterns`` with an :class:`IA3Linear`. Returns the number of layers wrapped.

**Details**

```text
Mirrors :func:`nnx.peft.apply_lora_to` — same fnmatch glob conventions,
same two-phase (collect-then-mutate) traversal, same idempotency
contract (existing IA3 wrappers are skipped via the parent-is-IA3Linear
check).

Args:
    module: root module to walk. Mutated in place.
    name_patterns: at least one fnmatch glob.

Returns:
    The count of layers wrapped (may be 0 if no patterns match
    or every match is already wrapped).

Raises:
    ValueError: if ``name_patterns`` is empty.
```


#### `nnx.peft.ia3.save_ia3_weights`

```python
nnx.peft.ia3.save_ia3_weights(module: 'nn.Module', path: 'Union[str, Path]') -> 'str'
```

Save ONLY the IA3 ``scaling`` parameters of ``module`` to ``path``.

**Details**

```text
The output is a ``torch.save`` of a dict-subset of the full
state_dict, containing only keys whose name includes ``scaling``.
Loadable via :func:`load_ia3_weights`.

Args:
    module: any module that has been processed by
        :func:`apply_ia3_to`. If no IA3 params exist, an empty
        dict is saved.
    path: destination file path.

Returns:
    The path written (so calls can be chained).
```


#### `nnx.peft.ia3.load_ia3_weights`

```python
nnx.peft.ia3.load_ia3_weights(module: 'nn.Module', source: 'Union[str, Path, dict]') -> 'int'
```

Load IA3 ``scaling`` parameters into ``module`` from ``source``.

**Details**

```text
Args:
    module: must already have :class:`IA3Linear` wrappers in the
        same positions as the source — apply_ia3_to FIRST, then
        call this. Otherwise the keys won't match and 0 params load.
    source: either a path to a file produced by
        :func:`save_ia3_weights`, or a state-dict dict directly.

Returns:
    The number of parameter tensors loaded.

Loads via ``module.load_state_dict(..., strict=False)`` so the
base layer's frozen weights — which are NOT in the IA3-only
checkpoint — don't trigger a missing-keys error.
```


### 9.4. Prefix Tuning

#### `nnx.peft.prefix.PrefixTuner`

```python
class nnx.peft.prefix.PrefixTuner(model: 'TransformerNN', *, n_prefix: 'int' = 10, n_layers: 'Optional[int]' = None)
```

Wrap a :class:`TransformerNN` with learnable per-layer K/V prefixes.

**Details**

```text
Freezes every parameter of the wrapped model on construction and
registers ``n_layers`` pairs of ``(n_prefix, n_heads, head_dim)``
K / V tensors as the only trainable parameters.

Args:
    model: a :class:`TransformerNN` instance. Its parameters are
        mutated in place (set to ``requires_grad=False``); the
        attention forward of each targeted block is monkey-patched.
    n_prefix: number of virtual prefix tokens per layer. Must be > 0.
    n_layers: number of leading transformer blocks to attach a prefix
        to. ``None`` (default) targets every block in
        ``model.blocks``. When set, the first ``n_layers`` blocks
        are targeted; later blocks run un-prefixed.

Note on shape: the prefix uses ``n_heads`` and ``head_dim`` taken
from the model's ``params`` — there's no per-block override, since
every block in a TransformerNN shares the same attention shape.

Raises:
    TypeError: if ``model`` is not a :class:`TransformerNN`.
    ValueError: if ``n_prefix`` or ``n_layers`` is out of range, or
        if ``model`` is already prefix-tuned — a second tuner would
        silently hijack the patched forwards (they read the MHA's
        ``_nnx_prefix_tuner`` ref, which the second tuner overwrites,
        so the first tuner's parameters stop receiving gradients).
```

##### `nnx.peft.prefix.PrefixTuner.forward`

```python
nnx.peft.prefix.PrefixTuner.forward(self, *args, **kwargs)
```

Delegate to the wrapped model. The prefix injection happens inside each block's monkey-patched MHA forward.

##### `nnx.peft.prefix.PrefixTuner.trainable_parameters`

```python
nnx.peft.prefix.PrefixTuner.trainable_parameters(self) -> 'Iterator[nn.Parameter]'
```

Yield only the learned prefix tensors.

**Details**

```text
The wrapped model's parameters are frozen on construction; this
is the iterator you hand to an optimizer.
```

##### `nnx.peft.prefix.PrefixTuner.prefix_state_dict`

```python
nnx.peft.prefix.PrefixTuner.prefix_state_dict(self) -> 'dict'
```

Return a state-dict containing only the prefix tensors, keyed for round-trip via :meth:`load_prefix_weights`.

**Details**

```text
The keys are the same as ``self.state_dict()`` filtered to the
prefix entries — i.e., ``prefix_keys.0``, ``prefix_values.0``,
``prefix_keys.1``, …
```


#### `nnx.peft.prefix.save_prefix_weights`

```python
nnx.peft.prefix.save_prefix_weights(tuner: 'PrefixTuner', path: 'Union[str, Path]') -> 'str'
```

Save ONLY the prefix tensors of ``tuner`` to ``path``.

**Details**

```text
Args:
    tuner: a :class:`PrefixTuner` instance.
    path: destination file path.

Returns:
    The path written, so calls can be chained.
```


#### `nnx.peft.prefix.load_prefix_weights`

```python
nnx.peft.prefix.load_prefix_weights(tuner: 'PrefixTuner', source: 'Union[str, Path, dict]') -> 'int'
```

Load prefix tensors into ``tuner`` from ``source``.

**Details**

```text
Args:
    tuner: must already have the same prefix shape as the source
        (same n_prefix, n_heads, head_dim, n_layers). Otherwise
        ``load_state_dict`` will surface the mismatch.
    source: a path to a file produced by :func:`save_prefix_weights`,
        or a state-dict dict directly.

Returns:
    The number of parameter tensors loaded.
```


### 9.5. Prompt Tuning

#### `nnx.peft.prompt.PromptTuner`

```python
class nnx.peft.prompt.PromptTuner(model: 'TransformerNN', *, n_prompt_tokens: 'int' = 20)
```

Wrap a :class:`TransformerNN` with a learnable soft prompt.

**Details**

```text
Freezes every base parameter and allocates an
``(n_prompt_tokens, d_model)`` embedding tensor. The wrapper's
forward prepends the prompt to the token embeddings, runs the
stack, then trims the prompt positions off the logits before
returning.

Args:
    model: a :class:`TransformerNN` instance. Its parameters are
        mutated in place (set to ``requires_grad=False``).
    n_prompt_tokens: number of soft-prompt slots. Must be > 0.

The soft prompt is initialized with ``nn.init.normal_(std=0.02)``
— the same scale Lester et al. use as their "random init" baseline.
```

##### `nnx.peft.prompt.PromptTuner.effective_max_seq_len`

```python
property nnx.peft.prompt.PromptTuner.effective_max_seq_len
```

Window available for REAL tokens: the soft prompt occupies ``n_prompt_tokens`` of the wrapped model's ``max_seq_len`` slots. ``GenerativeNNModel.generate`` reads this so its sliding window never overflows the wrapped model mid-generation.

##### `nnx.peft.prompt.PromptTuner.forward`

```python
nnx.peft.prompt.PromptTuner.forward(self, tokens: 'torch.Tensor') -> 'torch.Tensor'
```

Run the wrapped model with the soft prompt prepended.

**Details**

```text
Args:
    tokens: (batch, seq) long tensor of token ids.

Returns:
    (batch, seq, vocab_size) logits over the REAL token
    positions only. The soft-prompt positions are scaffolding
    and their logits are discarded.

Raises:
    ValueError: if ``seq + n_prompt_tokens`` exceeds the
        wrapped model's ``max_seq_len``. The soft prompt
        consumes positions in the RoPE table just like real
        tokens do.
```

##### `nnx.peft.prompt.PromptTuner.trainable_parameters`

```python
nnx.peft.prompt.PromptTuner.trainable_parameters(self) -> 'Iterator[nn.Parameter]'
```

Yield only the soft-prompt tensor.

**Details**

```text
The wrapped model's parameters are frozen on construction; this
is the iterator you hand to an optimizer.
```

##### `nnx.peft.prompt.PromptTuner.prompt_state_dict`

```python
nnx.peft.prompt.PromptTuner.prompt_state_dict(self) -> 'dict'
```

Return a state-dict containing only the soft-prompt tensor, keyed for round-trip via :meth:`load_prompt_weights`.


#### `nnx.peft.prompt.save_prompt_weights`

```python
nnx.peft.prompt.save_prompt_weights(tuner: 'PromptTuner', path: 'Union[str, Path]') -> 'str'
```

Save ONLY the soft-prompt tensor of ``tuner`` to ``path``.

**Details**

```text
Args:
    tuner: a :class:`PromptTuner` instance.
    path: destination file path.

Returns:
    The path written, so calls can be chained.
```


#### `nnx.peft.prompt.load_prompt_weights`

```python
nnx.peft.prompt.load_prompt_weights(tuner: 'PromptTuner', source: 'Union[str, Path, dict]') -> 'int'
```

Load the soft-prompt tensor into ``tuner`` from ``source``.

**Details**

```text
Args:
    tuner: must already have the same prompt shape as the source
        (same n_prompt_tokens, d_model). A shape mismatch is
        surfaced by ``load_state_dict``.
    source: a path to a file produced by :func:`save_prompt_weights`,
        or a state-dict dict directly.

Returns:
    The number of parameter tensors loaded.
```


### 9.6. Adapters

#### `nnx.peft.adapters.AdapterLayer`

```python
class nnx.peft.adapters.AdapterLayer(dim: 'int', bottleneck: 'int', activation: 'Callable[[], nn.Module]' = torch.nn.modules.activation.GELU)
```

Bottleneck residual block: ``y = x + up(act(down(x)))``.

**Details**

```text
``up.weight`` and ``up.bias`` are zero-initialized so at step 0
the layer's output equals its input exactly. Gradient flow
through ``up`` and ``down`` is unblocked from the first step;
only the magnitude of the residual starts at zero.

Args:
    dim: input and output feature dimension. The adapter is
        shape-preserving.
    bottleneck: hidden dimension. Typically much smaller than
        ``dim`` (e.g., dim=768 → bottleneck=64 in the original
        Houlsby setup). Lower bottleneck = fewer params,
        potentially less expressive.
    activation: ``nn.Module`` factory (called with no args inside
        ``__init__`` to produce the activation module). Defaults to
        ``torch.nn.GELU`` — the modern adapter choice; ``nn.ReLU``
        works too.
```

##### `nnx.peft.adapters.AdapterLayer.forward`

```python
nnx.peft.adapters.AdapterLayer.forward(self, x: 'torch.Tensor') -> 'torch.Tensor'
```

Define the computation performed at every call.

**Details**

```text
Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.
```

##### `nnx.peft.adapters.AdapterLayer.extra_repr`

```python
nnx.peft.adapters.AdapterLayer.extra_repr(self) -> 'str'
```

Return the extra representation of the module.

**Details**

```text
To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.
```


## 10. Pruning (`nnx.prune`)

#### `nnx.prune.magnitude.magnitude_prune`

```python
nnx.prune.magnitude.magnitude_prune(net: 'nn.Module', sparsity: 'float', *, layer_pattern: 'str' = '*', bake: 'bool' = True) -> 'int'
```

Zero the smallest-magnitude entries of every matched layer's weight.

**Details**

```text
For each :class:`nn.Linear` submodule of ``net`` whose dotted name
matches ``layer_pattern`` (fnmatch glob), call
:func:`torch.nn.utils.prune.l1_unstructured` with
``amount=sparsity``. PyTorch's implementation zeros
``round(sparsity · weight.numel())`` entries — the ones with the
smallest absolute value — per layer.

Args:
    net: root module to walk. The function mutates ``net`` in place.
    sparsity: fraction of weights to zero, in ``[0, 1)``. ``0.0``
        is a valid no-op; ``1.0`` is rejected here — a fully-zeroed
        Linear is never useful (torch itself would accept
        ``amount=1.0`` and silently zero the whole weight).
    layer_pattern: fnmatch glob against dotted submodule name.
        ``"*"`` (the default) matches every :class:`nn.Linear`.
    bake: when ``True`` (default), call
        :func:`torch.nn.utils.prune.remove` immediately after each
        layer is pruned. The mask is baked into a plain ``weight``
        tensor, the reparameterization is dropped, and the
        ``state_dict`` keys stay identical to the pre-prune layout.
        When ``False``, the reparameterization stays in place — the
        ``state_dict`` carries ``<name>.weight_orig`` +
        ``<name>.weight_mask`` instead of ``<name>.weight``. Use
        ``False`` for iterative pruning schedules (e.g., 10% per
        epoch for N epochs) where successive ``magnitude_prune``
        calls need to compose with the existing mask.

Returns:
    The number of :class:`nn.Linear` submodules that were pruned.
    ``0`` if ``layer_pattern`` matched nothing.

Raises:
    ValueError: if ``sparsity`` is outside ``[0, 1)``.

**Idempotency note:** calling ``magnitude_prune`` twice at the
same ``sparsity`` is a no-op for the second call — l1_unstructured
picks the smallest-magnitude entries, which after the first prune
are exactly the already-zeroed positions. The zero count stays the
same; nothing is double-pruned.
```


#### `nnx.prune.semi_structured.semi_structured_24`

```python
nnx.prune.semi_structured.semi_structured_24(net: 'nn.Module', *, layer_pattern: 'str' = '*') -> 'int'
```

Swap each matched :class:`nn.Linear`'s weight with a 2:4 semi-structured sparse tensor via :func:`torchao.sparsity.sparsify_`.

**Details**

```text
Args:
    net: root module to walk. The function mutates ``net`` in place.
    layer_pattern: fnmatch glob against dotted submodule name.
        ``"*"`` (the default) matches every :class:`nn.Linear`.

Returns:
    The number of :class:`nn.Linear` submodules that were swapped.
    ``0`` if ``layer_pattern`` matched nothing (in which case the
    underlying ``torchao.sparsity.sparsify_`` is NOT invoked — this
    avoids an unnecessary torchao dispatch and the CUDA-only kernel
    error on CPU runners with no Linear targets to swap).

Raises:
    ImportError: if ``torchao`` isn't installed. The import happens
        inside the function body so :mod:`nnx.prune` doesn't pull
        torchao at package-import time; users on the magnitude-only
        path pay no dep cost.
    RuntimeError: surfaced from the underlying
        ``torch.sparse.SparseSemiStructuredTensor`` constructor on
        unsupported hardware (CPU / pre-Ampere GPU) or on weights
        whose inner dimension isn't a multiple of 4. The error
        originates in torch / torchao; we don't intercept it.

**Pattern semantics:** same fnmatch convention as
:func:`nnx.peft.apply_lora_to` and
:func:`nnx.prune.magnitude_prune` — dotted submodule names against
shell wildcards. Only :class:`nn.Linear` submodules are eligible
(Conv2d / BatchNorm / Embedding / etc. are skipped even under a
wildcard pattern).

**Note on weights:** ``torchao.sparsity.sparsify_`` does NOT enforce
the 2:4 mask before the swap. Callers are expected to either
(a) magnitude-prune the weight to a valid 2:4 pattern beforehand
(via :func:`magnitude_prune` or a custom mask), or
(b) accept whatever 2:4 approximation
:func:`torch.sparse.to_sparse_semi_structured` picks (which keeps
the top-2-by-absolute-value entries per 4-group). For training
workflows, the standard recipe is to pre-mask, then train the
surviving entries.
```


## 11. Model surgery (`nnx.surgery`)

Walkthrough at [Model surgery](surgery.md). Every primitive returns a fresh `nn.Module` and composes with `NNModel.train()` for the "load checkpoint → surgery → refine" loop.

#### `nnx.surgery.widen.widen`

```python
nnx.surgery.widen.widen(model: 'nn.Module', *, layer_name: 'str', new_width: 'int', rng_seed: 'Optional[int]' = 0) -> 'nn.Module'
```

Net2WiderNet: grow a Linear's ``out_features`` to ``new_width``.

**Details**

```text
Returns a deep copy of ``model`` with the named layer expanded and
the downstream Linear's ``in_features`` adjusted so the overall
forward output is preserved exactly (within FP rounding).

Args:
    model: any :class:`nn.Module`. The function deep-copies it so
        the caller's reference survives.
    layer_name: dotted name (as produced by ``named_modules()``) of
        the :class:`nn.Linear` to widen. Must be a Linear, must
        have an immediately downstream Linear, otherwise raises.
    new_width: desired ``out_features``. Must be strictly greater
        than the current ``out_features``.
    rng_seed: seed for the unit-duplication choices. Pass an int
        for deterministic surgery; ``None`` to seed the local
        generator non-deterministically (fresh entropy — the global
        torch RNG is never read or advanced). Defaults to ``0`` so
        the primitive is deterministic by default.

Returns:
    A new :class:`nn.Module` (same class as ``model``) with the
    widened Linear in place. Forward output equals the original's
    within ``atol=1e-5`` (typically much tighter).

Raises:
    KeyError: if ``layer_name`` is not a submodule of ``model``.
    TypeError: if the named submodule is not :class:`nn.Linear`.
    ValueError: if ``new_width`` is not strictly greater than the
        current ``out_features``, or if no downstream Linear exists.
```


#### `nnx.surgery.deepen.deepen`

```python
nnx.surgery.deepen.deepen(model: 'nn.Module', *, after_layer_name: 'str') -> 'nn.Module'
```

Net2DeeperNet: insert an identity-initialized Linear after the named layer. Function-preserving on ReLU networks only.

**Details**

```text
Args:
    model: any :class:`nn.Module`. Deep-copied so the caller's
        reference survives.
    after_layer_name: dotted name (as in ``named_modules()``) of
        the insertion site. Either:

          * an :class:`nn.ReLU` inside a parent :class:`nn.Sequential`
            — the primitive splices ``Linear(I) → ReLU`` in after it.
          * an :class:`nn.Linear` inside a parent :class:`nn.ModuleList`
            whose grandparent module declares ReLU as its activation
            (the FeedFwdNN contract) — the primitive inserts a new
            identity-init Linear into the ModuleList right after.

Returns:
    A fresh :class:`nn.Module` whose forward output matches the
    original within ``atol=1e-5``.

Raises:
    KeyError: if ``after_layer_name`` is not a submodule.
    TypeError: if the layer is neither a ReLU-in-Sequential nor a
        Linear-in-FeedFwdNN-like ModuleList.
    ValueError: if the parent's activation is anything other than
        ReLU. Sigmoid / tanh / GELU break function-preservation.
```


#### `nnx.surgery.drop_layer.drop_layer`

```python
nnx.surgery.drop_layer.drop_layer(model: 'nn.Module', *, layer_name: 'Union[str, list[str]]', importance: 'Callable[[nn.Module], float] | None' = None) -> 'nn.Module'
```

Replace a named layer with :class:`nn.Identity`.

**Details**

```text
Args:
    model: any :class:`nn.Module`. Deep-copied so the caller's
        reference survives.
    layer_name: either a dotted submodule name, or a list of
        dotted names to choose from. When a list is given,
        ``importance`` must be provided as well.
    importance: optional callable ``fn(submodule) -> float``. When
        ``layer_name`` is a list, the candidate with the *minimum*
        importance score is dropped (lowest = least informative =
        safest to remove). Rejected when ``layer_name`` is a single
        string because there is no candidate-selection step to score.

Returns:
    A fresh :class:`nn.Module` with the chosen layer replaced by
    :class:`nn.Identity`. Forward shape contract is preserved iff
    the dropped layer was shape-preserving (e.g. an activation or
    a square Linear); otherwise calling forward on the surged
    module will raise — by design, since silently corrupting the
    shape would be worse than a loud failure.

Raises:
    KeyError: if any candidate name is missing.
    ValueError: if ``layer_name`` is an empty list, or a list
        without ``importance``, or a single string with
        ``importance``.
```


#### `nnx.surgery.low_rank.low_rank_factorize`

```python
nnx.surgery.low_rank.low_rank_factorize(linear: 'nn.Linear', *, rank: 'int', method: 'str' = 'svd') -> 'nn.Sequential'
```

Factor a Linear into two smaller Linears via rank-``k`` SVD truncation.

**Details**

```text
Args:
    linear: an :class:`nn.Linear` to factorize. Its weights are
        read but not mutated — the returned Sequential is a fresh
        pair of Linears.
    rank: the truncation rank ``k``. Must be in ``[1, min(out, in)]``.
        When ``k == min(out, in)`` the factorization is exact.
    method: ``"svd"`` (the only option in v1). Reserved for the
        future ``"activation_svd"`` / ``"fisher"`` variants.

Returns:
    :class:`nn.Sequential` of two Linears whose composition
    approximates the input layer. The first Linear has
    ``bias=False`` (Sx@V.T has no native bias term); the second
    Linear carries the original bias verbatim.

Raises:
    TypeError: if ``linear`` is not :class:`nn.Linear`.
    ValueError: if ``rank`` is out of range, or ``method`` unknown.
```


#### `nnx.surgery.embedding.expand_embedding`

```python
nnx.surgery.embedding.expand_embedding(emb: 'nn.Embedding', *, new_num_embeddings: 'int', init: 'InitStrategy' = 'zeros') -> 'tuple[nn.Embedding, torch.Tensor]'
```

Return a larger Embedding whose first rows match ``emb`` exactly.

**Details**

```text
Args:
    emb: the source embedding. Its weights are read but not
        mutated.
    new_num_embeddings: the desired ``num_embeddings`` for the
        returned layer. Must be strictly greater than the current.
    init: how to initialize the new rows. ``"zeros"`` — fill with
        zeros (default; deterministic, safe). ``"copy_mean"`` —
        fill each new row with the per-column mean of the original
        rows.

Returns:
    ``(new_emb, frozen_mask)`` where ``new_emb`` is a fresh
    :class:`nn.Embedding` with the original rows preserved, and
    ``frozen_mask`` is a bool tensor of shape
    ``(new_num_embeddings,)`` marking the original rows (``True``)
    as candidates for freezing during refinement.

Raises:
    TypeError: if ``emb`` is not :class:`nn.Embedding`.
    ValueError: if ``new_num_embeddings`` is not strictly greater,
        or if ``init`` is unknown.
```


## 12. Quantization (`nnx.quantize`)

PTQ INT8 weight-only + QAT 8da4w via [`torchao`](https://github.com/pytorch/ao) (the replacement for the removed `torch.ao.quantization`). Opt-in via `pip install "thekaveh-nnx[quantize]"`.

#### `nnx.quantize.ptq.quantize_int8`

```python
nnx.quantize.ptq.quantize_int8(model: 'NNModel') -> 'NNModel'
```

Return a new :class:`NNModel` with int8 weight-only quantized ``net``.

**Details**

```text
Deep-copies ``model.net`` and applies
``torchao.quantization.quantize_(net, Int8WeightOnlyConfig())`` to
the copy. Every ``nn.Linear`` submodule of the copy has its weight
parameter replaced with an :class:`AffineQuantizedTensor` (int8
per-channel, symmetric). Activations stay FP32 — only the weights
are stored in int8.

The original ``model`` is untouched. The returned ``NNModel`` shares
every other attribute (``params``, ``net_params``, ``device``,
``loss_fn``) with the original — only ``net`` is the quantized copy.

Args:
    model: a trained :class:`NNModel`. PTQ has no training step;
        this function is a pure post-process.

Returns:
    a new :class:`NNModel` instance whose ``net`` is the quantized
    deep-copy of ``model.net``. The new model can be used for
    ``predict`` / ``evaluate`` / ``to_onnx`` exactly like the
    original; ``train`` on the quantized model is not supported
    (QAT lands in a separate module).

Raises:
    ImportError: if ``torchao`` is not installed. Install with
        ``pip install thekaveh-nnx[quantize]``.
```


#### `nnx.quantize.qat.qat_train_step_factory`

```python
nnx.quantize.qat.qat_train_step_factory(base_step: 'Optional[TrainStepFn]' = None, qat_config: 'str' = '8da4w') -> 'TrainStepFn'
```

Return a :class:`TrainStepFn` that runs ``base_step`` against a fake-quantized model.

**Details**

```text
The returned step is the *same* as ``base_step`` (or
:func:`default_train_step` when ``base_step`` is None) — fake-quant
insertion happens once, via :class:`QATLifecycleCallback`, on
``on_train_begin``. The per-batch forward/backward then exercises
those fake-quant ops automatically through the standard module
forward.

Why split the work between a factory and a callback?

- The factory validates ``qat_config`` early (at construction time)
  so misconfigurations surface before the data loader spins up.
- The callback owns the lifecycle: ``prepare`` at start, ``convert``
  at end. Bundling that into the per-batch step would re-check the
  module state every iteration and complicate gradient flow.

Both pieces are needed in :meth:`NNModel.train`::

    callback = QATLifecycleCallback(qat_config="8da4w")
    step_fn  = qat_train_step_factory(qat_config="8da4w")
    model.train(params=..., callbacks=[callback], train_step_fn=step_fn)

Args:
    base_step: optional underlying training step to wrap. ``None``
        (the default) uses :func:`default_train_step` — the standard
        supervised forward/backward. Pass a custom step here to
        combine QAT with e.g. knowledge distillation or mixup; the
        fake-quant ops live in the model graph, so any standard
        step picks them up transparently.
    qat_config: shortcut for the torchao QAT recipe. Currently only
        ``"8da4w"`` is supported (int8 dynamic activations + int4
        grouped weights). Validated eagerly so a typo doesn't
        propagate to the callback.

Returns:
    a :class:`TrainStepFn` ready to pass to
    ``NNModel.train(..., train_step_fn=...)``.

Raises:
    ValueError: if ``qat_config`` is not in
        :data:`_SUPPORTED_CONFIGS`.
    ImportError: if ``torchao`` is not installed.
```


#### `nnx.quantize.qat.QATLifecycleCallback`

```python
class nnx.quantize.qat.QATLifecycleCallback(qat_config: 'str' = '8da4w', *, groupsize: 'int' = 32)
```

Manage the torchao ``prepare`` / ``convert`` lifecycle around training.

**Details**

```text
Add to ``callbacks=[...]`` in :meth:`NNModel.train`. On train begin,
swaps every eligible :class:`torch.nn.Linear` in ``model.net`` for
its fake-quantized counterpart (the model now learns to be robust
to int4/int8 rounding). On train end, the fake-quantized linears
are converted to actually-quantized ones — the resulting model is
suitable for inference / export.

The mutation is **in place** on ``model.net``: after training,
``model.net`` IS the converted model. The callback exposes the
quantizer instance as ``self.quantizer`` for callers who want to
pickle quantizer-specific state alongside their checkpoint, and
tracks the prepare/convert phase via ``self.is_prepared`` and
``self.is_converted`` for downstream inspection.

A completed conversion also contributes a versioned checkpoint transform
containing ``qat_config`` and ``groupsize``. The final ``LAST`` checkpoint
persists that recipe, so :meth:`NNModel.from_checkpoint` can rebuild the
converted torchao topology before loading its quantized tensors.

Args:
    qat_config: torchao recipe shortcut. See
        :func:`qat_train_step_factory`.
    groupsize: group size for the int4 weight quantizer. 32 is the
        default — small enough to apply to toy nets in tests
        (where hidden_dim=64) while being a real-world setting.
        Larger groupsizes (128, 256) give better compression at
        the cost of accuracy.
```

##### `nnx.quantize.qat.QATLifecycleCallback.on_train_begin`

```python
nnx.quantize.qat.QATLifecycleCallback.on_train_begin(self, ctx: '_CallbackContext') -> 'None'
```

Insert fake-quant ops into ``ctx.model.net`` in place.

##### `nnx.quantize.qat.QATLifecycleCallback.on_train_end`

```python
nnx.quantize.qat.QATLifecycleCallback.on_train_end(self, ctx: '_CallbackContext') -> 'None'
```

Convert fake-quant ops in ``ctx.model.net`` to true int4/int8 modules.

**Details**

```text
After this returns, ``ctx.model.net`` produces real quantized
outputs and is suitable for inference / ONNX export. The model
is no longer trainable through the usual FP32 optimizer path —
a fresh training session on the same NNModel would need a new
QATLifecycleCallback.
```

##### `nnx.quantize.qat.QATLifecycleCallback.checkpoint_transforms`

```python
nnx.quantize.qat.QATLifecycleCallback.checkpoint_transforms(self) -> 'tuple[NNCheckpointTransform, ...]'
```

Describe the completed conversion so checkpoint loaders can replay it.


## 13. Diffusion (`nnx.diffusion`)

#### `nnx.diffusion.schedules.NoiseSchedulers`

```python
class nnx.diffusion.schedules.NoiseSchedulers(Enum)
```

Diffusion noise-schedule factory. Enum-as-factory pattern (like :class:`nnx.Nets`, :class:`nnx.Optims`): each enum variant's ``__call__`` constructs the underlying :class:`NoiseSchedule`.

##### `nnx.diffusion.schedules.NoiseSchedulers.LINEAR`

```python
nnx.diffusion.schedules.NoiseSchedulers.LINEAR = 'linear'
```

Enum value `linear`.

##### `nnx.diffusion.schedules.NoiseSchedulers.COSINE`

```python
nnx.diffusion.schedules.NoiseSchedulers.COSINE = 'cosine'
```

Enum value `cosine`.


#### `nnx.diffusion.schedules.NoiseSchedule`

```python
class nnx.diffusion.schedules.NoiseSchedule(kind: 'NoiseSchedulers', T: 'int', betas: 'torch.Tensor', alphas: 'torch.Tensor', alphas_cumprod: 'torch.Tensor', sqrt_alphas_cumprod: 'torch.Tensor', sqrt_one_minus_alphas_cumprod: 'torch.Tensor', posterior_variance: 'torch.Tensor') -> 'None'
```

Precomputed DDPM noise schedule.

**Details**

```text
All tensors are 1D of length ``T`` and live on the same device. The
factory constructs them on CPU; :meth:`to` returns a new schedule
with every tensor migrated.

Attributes:
    kind: which enum variant produced this schedule (for introspection).
    T: number of diffusion timesteps.
    betas: per-step variance, ``shape=(T,)``.
    alphas: ``1 - betas``.
    alphas_cumprod: cumulative product of alphas (``ᾱ_t`` in the paper).
    sqrt_alphas_cumprod: ``√ᾱ_t`` — the x_0 coefficient in q(x_t | x_0).
    sqrt_one_minus_alphas_cumprod: ``√(1 - ᾱ_t)`` — the noise coefficient.
    posterior_variance: variance of q(x_{t-1} | x_t, x_0), used by the
        reverse-step sampler.
```

##### `nnx.diffusion.schedules.NoiseSchedule.to`

```python
nnx.diffusion.schedules.NoiseSchedule.to(self, device) -> 'NoiseSchedule'
```

Return a copy with every tensor moved to ``device``. The kind and T fields are unchanged.


#### `nnx.diffusion.nets.DiffusionMLP`

```python
class nnx.diffusion.nets.DiffusionMLP(input_dim: 'int', hidden_dims: 'list[int] | None' = None, time_embed_dim: 'int' = 32)
```

Conditional MLP for low-dim diffusion: ``forward(x_t, t) -> ε_pred``.

**Details**

```text
Architecture: sinusoidal time embed → small projection → concat with
flat x_t → MLP → linear head producing a noise prediction of the same
shape as x_t. Bare ReLU activations, no skip connections — a single
file's worth of code, enough to learn a 2D Gaussian mixture or a
small tabular distribution.

Inputs of any rank are supported by flattening dimensions ≥ 1 before
the MLP and un-flattening at the output. The network is *NOT* a
U-Net — it has no spatial structure. For image-space diffusion, the
same train/sample/schedule machinery works against a user-supplied
U-Net.
```

##### `nnx.diffusion.nets.DiffusionMLP.forward`

```python
nnx.diffusion.nets.DiffusionMLP.forward(self, x: 'torch.Tensor', t: 'torch.Tensor') -> 'torch.Tensor'
```

Predict noise added to ``x`` at timestep ``t``.

**Details**

```text
Args:
    x: ``(B, *)`` clean shape; flattened internally to ``(B, D)``.
    t: ``(B,)`` integer timesteps.

Returns:
    Tensor of the same shape as ``x``.
```

##### `nnx.diffusion.nets.DiffusionMLP.unpack_batch`

```python
nnx.diffusion.nets.DiffusionMLP.unpack_batch(self, batch)
```

Standard ``(X-tuple, Y)`` adapter so this net plays nicely with the NNx dataloader contract. ``Y`` is unused by diffusion — every consumer that calls ``unpack_batch`` discards it.


#### `nnx.diffusion.nets.sinusoidal_time_embed`

```python
nnx.diffusion.nets.sinusoidal_time_embed(t: 'torch.Tensor', dim: 'int') -> 'torch.Tensor'
```

Standard transformer-style sinusoidal positional embedding, applied to scalar timesteps so the denoising network can condition on ``t``.

**Details**

```text
Args:
    t: integer or float tensor of shape ``(B,)`` — per-sample timesteps.
    dim: embedding dimension. Half of it carries sin frequencies,
        half carries cos; ``dim`` must be even.

Returns:
    Tensor of shape ``(B, dim)``.
```


#### `nnx.diffusion.training.diffusion_train_step_factory`

```python
nnx.diffusion.training.diffusion_train_step_factory(schedule: 'NoiseSchedule') -> 'TrainStepFn'
```

Build a DDPM noise-prediction :class:`TrainStepFn`.

**Details**

```text
Each call to the returned step fn:

  1. Samples a random per-sample timestep ``t ~ Uniform[0, T)``.
  2. Samples Gaussian noise ``ε ~ N(0, I)`` matching x_0's shape.
  3. Computes ``x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε`` (forward diffusion).
  4. Calls ``model.net(x_t, t)`` to predict ``ε_pred``.
  5. Backprops the MSE between ``ε_pred`` and ``ε``, steps the optimizer.

Loss is reported as both ``.loss`` and ``.error`` on the returned
EDP so BEST checkpoint tracking and the ReduceLROnPlateau scheduler
have a metric to lock onto. The standard supervised classification
metrics (accuracy/f1/...) are not meaningful for a generative
paradigm and stay zero.

Args:
    schedule: a :class:`NoiseSchedule` from :class:`NoiseSchedulers`.
        Built on any device; the step fn lazily migrates the
        indexed tensors to ``model.device`` per call.

Returns:
    A function suitable for ``NNModel.train(..., train_step_fn=...)``.
```


#### `nnx.diffusion.sampling.sample`

```python
nnx.diffusion.sampling.sample(model: 'NNModel', schedule: 'NoiseSchedule', shape: 'tuple[int, ...]', *, device: 'Optional[torch.device]' = None, generator: 'Optional[torch.Generator]' = None) -> 'torch.Tensor'
```

Run T reverse-diffusion steps and return samples drawn from the distribution the model was trained on.

**Details**

```text
Args:
    model: an :class:`NNModel` whose ``.net`` is the trained
        denoising network (e.g., :class:`DiffusionMLP` or any
        ``forward(x, t) -> ε`` module).
    schedule: the same :class:`NoiseSchedule` used during training.
        Indexed tensors are moved to ``device`` lazily.
    shape: full tensor shape to generate, e.g., ``(256, 2)`` for
        256 2D samples.
    device: target device. Defaults to ``model.device``.
    generator: optional torch.Generator for reproducible sampling
        (pass one built with ``torch.Generator(device).manual_seed(...)``).

Returns:
    A tensor of shape ``shape`` carrying the generated samples.
```


## 14. Training paradigms (`nnx.paradigms`)

Each factory returns a `TrainStepFn` for the `train_step_fn=` hook on `NNModel.train`. The training loop, checkpoint cadence, callbacks, and persistence are unchanged — only the per-batch update is swapped.

### 14.1. Knowledge distillation

#### `nnx.paradigms.distillation.kd_train_step_factory`

```python
nnx.paradigms.distillation.kd_train_step_factory(teacher: 'NNModel', *, alpha: 'float' = 0.5, temperature: 'float' = 4.0) -> 'TrainStepFn'
```

Build a knowledge-distillation :class:`TrainStepFn`.

**Details**

```text
Args:
    teacher: a fully-trained :class:`NNModel` whose net produces
        logits of the same shape as the student's. The teacher's
        parameters are frozen (``requires_grad=False``) and its
        net is set to eval mode on factory call.
    alpha: weight on the distillation (soft) loss. The hard-label
        loss gets ``1 − α``. ``α=1.0`` is pure distillation;
        ``α=0.0`` collapses to standard supervised training (the
        teacher is loaded but unused). 0.5 is the common default.
    temperature: softmax temperature applied to BOTH student and
        teacher logits before the KL. Higher T flattens the
        distribution and exposes more dark knowledge; the
        ``× T²`` factor in front of the KL keeps gradient
        magnitude comparable to the hard-label term across T.
        4.0 is the classical Hinton choice.

Returns:
    A ``TrainStepFn`` suitable for ``NNModel.train(..., train_step_fn=...)``.

Raises:
    ValueError: if ``alpha`` is not in [0, 1], or ``temperature`` ≤ 0.
```


#### `nnx.paradigms.distillation.feature_kd_train_step_factory`

```python
nnx.paradigms.distillation.feature_kd_train_step_factory(teacher: 'NNModel', *, auxiliary_layers: 'dict[str, str]', alpha: 'float' = 0.5, beta: 'float' = 0.5, temperature: 'float' = 4.0) -> 'TrainStepFn'
```

Build a FitNets-style feature-distillation :class:`TrainStepFn`.

**Details**

```text
Extends :func:`kd_train_step_factory` with an additional MSE term
matching named intermediate-layer activations between the (frozen)
teacher and the trainable student. Forward hooks register on the
pairs in ``auxiliary_layers``; collected activations feed an
elementwise MSE that's mixed into the loss via ``beta``::

    L = α · KL_soft · T² + β · MSE(student_act, teacher_act) + (1 − α) · L_hard

Args:
    teacher: a fully-trained :class:`NNModel` whose net produces
        logits of the same shape as the student's. The teacher's
        parameters are frozen (``requires_grad=False``) and its
        net is set to eval mode on factory call — same guarantee
        as :func:`kd_train_step_factory`.
    auxiliary_layers: dict mapping ``teacher_layer_name ->
        student_layer_name`` for each (teacher, student) pair to
        match. Names are dotted paths resolved via
        :meth:`torch.nn.Module.get_submodule` against the teacher
        / student ``net``. Must be non-empty. The teacher and
        student activations at each pair must share shape — if
        they don't, the factory raises ``ValueError`` on the
        first forward (the projector ``FeatureRegressor`` from
        FitNets is intentionally deferred).
    alpha: weight on the soft (logit-KL) term. The hard-label
        loss gets ``1 − α``. 0.5 is the common default.
    beta: weight on the feature-MSE term. 0.5 is the common
        starting point; tune downward if it dominates the logit
        term, upward to bias the student toward matching internal
        representations.
    temperature: softmax temperature for the logit-KL term —
        identical contract to :func:`kd_train_step_factory`.

Returns:
    A ``TrainStepFn`` suitable for ``NNModel.train(...,
    train_step_fn=...)``.

Raises:
    ValueError: if ``alpha`` or ``beta`` is not in [0, 1], if
        ``temperature`` ≤ 0, or if ``auxiliary_layers`` is empty.
        On the first batch, if any paired teacher/student
        activation shapes disagree.
```


#### `nnx.paradigms.born_again.born_again_train`

```python
nnx.paradigms.born_again.born_again_train(model: 'NNModel', *, generations: 'int' = 3, train_params: 'NNTrainParams', **kd_kwargs: 'Any') -> 'list[NNRun]'
```

Iterate G generations of self-distillation on a single model.

**Details**

```text
Generation 0 trains plain (no teacher) — standard supervised loss.
Each subsequent generation uses a deep-copied, frozen, eval-mode
snapshot of the model *after* the prior generation completed as the
teacher for a Hinton-style KD step (via :func:`kd_train_step_factory`).

The same ``NNModel`` wrapper is reused, but its network weights are reset
to the caller-provided initialization before every student generation.
This follows the original Born-Again Networks procedure while keeping
memory usage to the live student, one frozen teacher, and one initial
state dictionary.

Args:
    model: the :class:`NNModel` to train. Its initial weights seed every
        fresh student; its final state corresponds to the LAST generation.
    generations: how many generations to run. ``generations=1`` is
        a plain supervised run (no KD) — kept as a degenerate case
        so callers can sweep generations including the baseline.
        Must be ``≥ 1``.
    train_params: base configuration for every :meth:`NNModel.train`
        call. Generation zero uses it unchanged. Each later generation
        records the preceding run as its parent, producing a distinct
        content-addressed run with independent history and BEST tracking.
    **kd_kwargs: forwarded to :func:`kd_train_step_factory` for
        generations ≥ 1 (``alpha``, ``temperature``). Ignored on
        generation 0 (no teacher).

Returns:
    A list of :class:`NNRun` objects, one per generation, in order.
    ``runs[0]`` is the plain run; ``runs[k]`` for ``k > 0`` is the
    KD run that used generation ``k-1``'s model as teacher.

Raises:
    ValueError: if ``generations < 1``.
```


### 14.2. Contrastive

#### `nnx.paradigms.contrastive.simclr_train_step_factory`

```python
nnx.paradigms.contrastive.simclr_train_step_factory(*, temperature: 'float' = 0.5) -> 'TrainStepFn'
```

Build a SimCLR :class:`TrainStepFn`.

**Details**

```text
Args:
    temperature: temperature in :func:`nt_xent_loss`. 0.5 default.

Returns:
    A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.
    The training loader MUST yield batches of two augmented views
    per source sample — typically ``(view1, view2)`` tensors, or
    ``((view1, view2), y_unused)`` when reusing a labelled dataset.
    ``model.net`` is invoked once per view (no batch-doubling) so
    BatchNorm statistics see one view at a time; users who want
    all-at-once normalization can stack the views and forward once.

    **Sharp edge:** a labeled ``(X, Y)`` batch from a standard
    ``TensorDataset`` will silently be interpreted as
    ``(view1=X, view2=Y)`` and produce a shape-mismatch in
    :func:`nt_xent_loss`. Use a paired-view dataset whose
    ``__getitem__`` returns ``(view1, view2)`` instead.

Raises:
    ValueError: if ``temperature`` <= 0.
```


#### `nnx.paradigms.contrastive.nt_xent_loss`

```python
nnx.paradigms.contrastive.nt_xent_loss(z1: 'torch.Tensor', z2: 'torch.Tensor', *, temperature: 'float' = 0.5) -> 'torch.Tensor'
```

SimCLR's Normalized Temperature-scaled cross-entropy loss.

**Details**

```text
Args:
    z1: ``(B, D)`` embeddings of the first view of each sample.
    z2: ``(B, D)`` embeddings of the second view.
    temperature: divisor on the cosine similarity. Lower T sharpens
        the distribution; 0.5 is the SimCLR default. Must be > 0.

Returns:
    Scalar loss tensor (mean across the 2B positions in the batch).

Raises:
    ValueError: if shapes mismatch, ``temperature`` ≤ 0, or the
        batch has fewer than 2 pairs (no negatives to contrast).
```


### 14.3. Augmentation

#### `nnx.paradigms.augmentation.mixup_train_step_factory`

```python
nnx.paradigms.augmentation.mixup_train_step_factory(*, alpha: 'float' = 0.4) -> 'TrainStepFn'
```

Build a Mixup :class:`TrainStepFn`.

**Details**

```text
Args:
    alpha: Beta-distribution shape parameter. ``λ ~ Beta(α, α)``;
        α=1.0 yields a uniform mix, lower values concentrate λ
        near 0 or 1 (closer to no-mixing). 0.4 is the
        classification default; image-task papers often use 0.2-1.0.
        Must be positive.

Returns:
    A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.
    Reports a Mixup-weighted ``error`` and the mixed loss. The
    loss honors the model's ``loss_fn`` (so this works for any
    classification loss, not just CrossEntropy).

Raises:
    ValueError: if ``alpha`` <= 0.
```


#### `nnx.paradigms.augmentation.cutmix_train_step_factory`

```python
nnx.paradigms.augmentation.cutmix_train_step_factory(*, alpha: 'float' = 1.0) -> 'TrainStepFn'
```

Build a CutMix :class:`TrainStepFn` for 4D image batches.

**Details**

```text
Args:
    alpha: Beta-distribution shape parameter for the area ratio.
        ``λ ~ Beta(α, α)``; controls the size of the swapped
        rectangle. 1.0 is the original paper default.
        Must be positive.

Returns:
    A ``TrainStepFn`` for image classification (4D ``(B, C, H, W)``
    inputs). Raises at step time on lower-rank input — CutMix's
    spatial cut isn't well-defined without H and W.

Raises:
    ValueError: if ``alpha`` <= 0.
```


### 14.4. Mixture-of-Experts

`MoELinear` is the drop-in layer (documented in §4); `moe_train_step_factory` adds the Switch-style load-balancing aux loss to the supervised step.

#### `nnx.paradigms.moe.moe_train_step_factory`

```python
nnx.paradigms.moe.moe_train_step_factory(*, aux_loss_weight: 'float' = 0.01) -> 'TrainStepFn'
```

Build an MoE-aware supervised :class:`TrainStepFn`.

**Details**

```text
The returned step performs the standard supervised forward
(``loss = m.loss_fn(net(X), Y)``) and then *adds* the
Switch-style load-balancing penalty summed across every
:class:`MoELinear` layer in ``model.net``, weighted by
``aux_loss_weight``. Backward, grad-clip, and optimizer step go
through :func:`nnx._step_helpers.finalize_step` for the same
NaN-guard + grad-clip tail as the other paradigm factories.

Args:
    aux_loss_weight: weight on the aux loss term (``α`` in the
        Switch formulation). Must be non-negative. ``0.0`` turns
        the factory into a plain supervised step (the aux loss is
        still computed by each MoE forward but contributes 0 to
        backward). Defaults to ``0.01`` — the Switch paper's
        tutorial value; small enough not to dominate the main
        loss, large enough to prevent expert collapse.

Returns:
    A ``TrainStepFn`` for :meth:`NNModel.train`. Works on any
    single-input supervised net that contains ≥ 0
    :class:`MoELinear` layers; if there are no MoE layers, the
    aux loss is 0 and the step is exactly supervised.

Raises:
    ValueError: if ``aux_loss_weight < 0``.
```


### 14.5. I-JEPA

Walkthrough at [I-JEPA](jepa.md). The `ViTNN` encoder is documented in §4.

#### `nnx.paradigms.jepa.jepa_train_step_factory`

```python
nnx.paradigms.jepa.jepa_train_step_factory(target_encoder: 'nn.Module', predictor: 'nn.Module', mask_fn: 'Callable[[int, torch.device], tuple[torch.Tensor, torch.Tensor]]', *, ema_momentum: 'float' = 0.996) -> 'TrainStepFn'
```

Build an I-JEPA :class:`TrainStepFn`.

**Details**

```text
Per step:

  1. Sample ``(context_mask, target_mask)`` for the batch via
     ``mask_fn(n_patches, device)``. Both are 1-D
     ``BoolTensor[n_patches]`` and **complementary** — every
     patch is either context or target.
  2. Forward each input image through ``model.net`` with the
     context mask, producing ``(B, T_ctx + 1, d_model)`` context
     embeddings (CLS at index 0).
  3. Forward the full image (no mask) through ``target_encoder``
     under ``no_grad`` to produce target embeddings. Slice out
     the positions in ``target_mask`` only.
  4. Predict ``(B, T_tgt, d_model)`` from context via
     ``predictor``.
  5. MSE loss against the target embeddings.
  6. :func:`finalize_step` — NaN guard, optimizer step, grad clip.
  7. :func:`update_ema` — EMA-update the target encoder from
     ``model.net``.

Args:
    target_encoder: an EMA copy of ``model.net``. Build via
        :func:`build_target_encoder`. The factory **freezes** it
        again on call and pins to ``eval()`` mode.
    predictor: a :class:`JEPAPredictor` (or any module with the
        same ``forward(context_embeds, context_positions,
        target_positions)`` contract). The predictor's parameters
        are *not* frozen — the optimizer's ``param_groups`` need
        to include them; the simplest path is to register the
        predictor as a submodule of ``model.net`` (the ViTNN)
        before constructing the optimizer.
    mask_fn: callable ``(n_patches, device) -> (context_mask,
        target_mask)`` where both are 1-D ``BoolTensor[n_patches]``.
        Sampled freshly **once per step** and shared across the
        batch. The bundled :func:`random_block_mask` helper is the
        common choice; passing a fixed mask is fine for tests.
    ema_momentum: EMA decay used by :func:`update_ema`. Default
        0.996 (reference I-JEPA).

Returns:
    A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.

Raises:
    ValueError: when ``ema_momentum`` is outside ``[0, 1)``.
```


#### `nnx.paradigms.jepa.JEPAPredictor`

```python
class nnx.paradigms.jepa.JEPAPredictor(*, embed_dim: 'int', n_patches: 'int', predictor_dim: 'Optional[int]' = None, n_layers: 'int' = 2, n_heads: 'int' = 2, ffn_mult: 'int' = 4)
```

Tiny ViT-like predictor: ``(context_embeds, target_positions) -> predicted_target_embeds``.

**Details**

```text
Architecture: project context_embeds to ``predictor_dim``,
concatenate learnable mask tokens (one per target position) plus
that position's positional embedding, run a few ViT blocks, project
back to ``embed_dim``, return the predictions at the target
positions only.

Kept deliberately small — the reference I-JEPA predictor is also
much narrower than the encoder. For our CIFAR-shape demo, two
blocks at ``predictor_dim = embed_dim // 2`` is enough plumbing
to verify the loss decreases without dominating wall-clock time.
```

##### `nnx.paradigms.jepa.JEPAPredictor.forward`

```python
nnx.paradigms.jepa.JEPAPredictor.forward(self, context_embeds: 'torch.Tensor', context_positions: 'torch.Tensor', target_positions: 'torch.Tensor') -> 'torch.Tensor'
```

Predict embeddings at ``target_positions`` from ``context_embeds``.

**Details**

```text
Args:
    context_embeds: ``(B, T_ctx, embed_dim)``. The CLS token
        produced by the encoder is included as the first entry
        (position 0).
    context_positions: ``LongTensor[T_ctx]`` — positions of
        the kept context tokens *including* CLS at index 0.
    target_positions: ``LongTensor[T_tgt]`` — positions of the
        target patches to predict (1..n_patches).

Returns:
    ``(B, T_tgt, embed_dim)`` predicted target embeddings.
```


#### `nnx.paradigms.jepa.build_target_encoder`

```python
nnx.paradigms.jepa.build_target_encoder(source: 'nn.Module') -> 'nn.Module'
```

Deep-copy ``source``, freeze every parameter, return the copy.

**Details**

```text
The target encoder is updated **only** via :func:`update_ema` after
each optimizer step. Freezing here is belt-and-braces — even if a
user accidentally hands the target into an optimizer that scans
``parameters()``, ``requires_grad=False`` keeps the gradients off
and the optimizer's state empty for those tensors.
```


#### `nnx.paradigms.jepa.update_ema`

```python
nnx.paradigms.jepa.update_ema(source: 'nn.Module', target: 'nn.Module', momentum: 'float') -> 'None'
```

In-place EMA update: ``target ← momentum * target + (1 - momentum) * source``.

**Details**

```text
Called once per training step from inside the JEPA train_step_fn.
Runs under ``torch.no_grad`` so the EMA tensors do not become part
of the autograd graph — the target encoder is supposed to be a
detached snapshot.

Args:
    source: the trainable module (i.e., ``model.net``).
    target: the EMA copy returned by :func:`build_target_encoder`.
        Mutated in place.
    momentum: EMA decay in ``[0, 1)``. Higher = slower target
        tracking. I-JEPA's reference recipe uses 0.996 with a
        cosine schedule up to 1.0 over training; the factory's
        default matches.

Raises:
    ValueError: when ``momentum`` is outside ``[0, 1)``.
    KeyError: when a target parameter has no same-named source
        parameter (the name-keyed update contract).
```


#### `nnx.paradigms.jepa.random_block_mask`

```python
nnx.paradigms.jepa.random_block_mask(*, n_patches: 'int', grid_size: 'int', block_scale: 'tuple[float, float]' = (0.15, 0.2), block_aspect: 'tuple[float, float]' = (0.75, 1.5), generator: 'Optional[torch.Generator]' = None, device: 'Optional[torch.device]' = None) -> 'tuple[torch.Tensor, torch.Tensor]'
```

Sample one I-JEPA-style rectangular block mask on a patch grid.

**Details**

```text
Returns ``(context_mask, target_mask)`` where:

  * ``context_mask: BoolTensor[n_patches]`` — True at positions
    kept by the context encoder (i.e., NOT in the target block).
  * ``target_mask: BoolTensor[n_patches]`` — True at positions
    the predictor is asked to predict (i.e., inside the target
    block, exactly the complement of context_mask).

The block is a single rectangle of randomly-sampled width/height
drawn from ``block_scale`` × n_patches with an aspect ratio in
``block_aspect``. Reference I-JEPA samples 4 target blocks per
image; this helper samples 1 — enough for the verify-the-plumbing
example we ship. Users can compose multiple calls if they want
the 4-block recipe.

Args:
    n_patches: total number of patch tokens. Must equal
        ``grid_size**2``.
    grid_size: width (= height) of the patch grid. The
        rectangular block is sampled in this coordinate system.
    block_scale: ``(min, max)`` fraction of ``n_patches`` covered
        by the block. Default ``(0.15, 0.2)`` mirrors I-JEPA.
    block_aspect: ``(min, max)`` width/height ratio.
    generator: optional ``torch.Generator`` for reproducibility.
    device: device on which the masks are placed. ``None`` →
        default tensor device (CPU).

Returns:
    A pair of ``BoolTensor``s, both 1-D length ``n_patches``.

Raises:
    ValueError: when ``grid_size**2 != n_patches``, or when the
        sampled block would be empty / larger than the grid.
```


### 14.6. DPO

Walkthrough at [DPO](dpo.md).

#### `nnx.paradigms.dpo.dpo_train_step_factory`

```python
nnx.paradigms.dpo.dpo_train_step_factory(ref_model: 'NNModel', *, beta: 'float' = 0.1, pad_token_id: 'Optional[int]' = None) -> 'TrainStepFn'
```

Build a Direct Preference Optimization :class:`TrainStepFn`.

**Details**

```text
Args:
    ref_model: a frozen reference policy — typically a copy of the
        SFT checkpoint that the trainable policy was initialized
        from. Its ``net`` is set to eval mode and every parameter
        has ``requires_grad`` cleared on factory call. Must share
        ``vocab_size`` and tokenization with the policy.
    beta: temperature on the implicit reward. Larger ``beta`` makes
        the loss sharper (closer to a hard preference); smaller
        ``beta`` keeps the policy closer to the reference. The
        original DPO paper uses 0.1 as the default; values in
        ``[0.01, 0.5]`` are common. Must be > 0.
    pad_token_id: the id the dataset used to right-pad chosen /
        rejected responses (``NNPreferenceDataset.pad_token_id``).
        When set, padded positions are excluded from the response
        log-prob sums. Without it, pad tokens are scored too — the
        pad terms don't cancel between policy/reference or
        chosen/rejected (different contexts), biasing the objective
        and training the policy to emit pads after short responses.
        ``None`` is only appropriate when every response genuinely
        fills ``max_response_len``. Two caveats: masking is by
        token-id equality, so a genuine occurrence of the pad id
        inside a response is dropped too (pick a dedicated pad id);
        and prompt-side padding remains visible to the model (no
        attention mask) — a pre-existing modeling bias this knob
        doesn't address.

Returns:
    A ``TrainStepFn`` for ``NNModel.train(..., train_step_fn=...)``.
    The training loader MUST yield batches of three
    ``torch.LongTensor`` of shape ``(B, T_*)``::

        (prompt_ids, chosen_ids, rejected_ids)

    — typically from :class:`nnx.NNPreferenceDataset`. All three
    tensors must already be padded / right-aligned by the dataset.

Raises:
    ValueError: if ``beta`` ≤ 0.
```


## 15. Embeddings (`nnx.embeddings`)

End-to-end walkthrough at [Embeddings](embeddings.md). Opt-in via `pip install "thekaveh-nnx[embeddings]"`.

#### `nnx.embeddings.contrastive_trainer.ContrastiveTextDataset`

```python
class nnx.embeddings.contrastive_trainer.ContrastiveTextDataset(pairs: 'list[tuple[str, str]]')
```

Wraps ``(anchor, positive)`` string pairs as a torch ``Dataset``.

**Details**

```text
Each ``__getitem__`` returns a 2-tuple of strings (``anchor``,
``positive``). The default collate from :class:`torch.utils.data.DataLoader`
would attempt to stack these into tensors and crash; pair this
dataset with :func:`pair_collate` (or pass it directly to
:func:`train_contrastive` which wires the collate for you).

Args:
    pairs: list of ``(anchor, positive)`` string tuples. Empty
        input raises :class:`ValueError`. Note that
        :func:`train_contrastive` additionally requires >= 2 pairs
        (NT-Xent needs a negative); a 1-pair dataset is accepted
        here only for embedding/inference-style uses.

Raises:
    ValueError: if ``pairs`` is empty or any entry isn't a 2-tuple
        of strings.
```


#### `nnx.embeddings.contrastive_trainer.train_contrastive`

```python
nnx.embeddings.contrastive_trainer.train_contrastive(backbone: 'Any', dataset: 'Union[ContrastiveTextDataset, list[tuple[str, str]]]', *, n_epochs: 'int' = 3, batch_size: 'int' = 16, lr: 'float' = 2e-05, temperature: 'float' = 0.05, device: 'Optional[Union[str, torch.device]]' = None, shuffle: 'bool' = True, grad_clip_norm: 'Optional[float]' = 1.0, weight_decay: 'float' = 0.0, optimizer_cls: 'type' = torch.optim.adamw.AdamW, verbose: 'bool' = False) -> 'Any'
```

Train ``backbone`` on ``(anchor, positive)`` pairs via NT-Xent.

**Details**

```text
High-level wrapper around :func:`nt_xent_loss`. Builds a
:class:`DataLoader` with :func:`pair_collate`, instantiates an
optimizer over the backbone's trainable parameters, and runs
``n_epochs`` of contrastive updates. The backbone is updated
in-place AND returned for chaining (e.g., directly into
:func:`nnx.embeddings.export_to_faiss`).

For more elaborate setups — callbacks, custom schedulers, multi-
optimizer training, run.id persistence under ``runs/<id>/`` — use
:func:`text_contrastive_train_step_factory` with the standard
:meth:`NNModel.train` driver instead.

Args:
    backbone: text encoder. Either a
        :class:`sentence_transformers.SentenceTransformer` or any
        ``nn.Module`` whose ``forward(list[str]) -> Tensor[B, D]``.
        Parameters with ``requires_grad=False`` are excluded from
        the optimizer (so :func:`nnx.freeze` composes cleanly).
    dataset: a :class:`ContrastiveTextDataset` or a plain list of
        ``(anchor, positive)`` string tuples (we'll wrap it).
    n_epochs: number of full passes. Default 3 — contrastive
        fine-tuning of a pretrained encoder typically needs few.
    batch_size: pairs per batch. NT-Xent's in-batch-negatives
        scaling means bigger is usually better; 16-64 is typical
        for CPU sanity runs, hundreds for GPU.
    lr: optimizer learning rate. Default 2e-5 (the canonical SBERT
        fine-tune LR).
    temperature: NT-Xent temperature. Default 0.05 (sharper than
        SimCLR's image default — text embedders work in a much
        higher-dim cosine space where small temperature helps).
    device: target device. ``None`` infers from the backbone (its
        ``.device`` if present, else its first parameter's device,
        else CPU).
    shuffle: shuffle the dataset each epoch. Default True.
    grad_clip_norm: global L2 grad-clip norm. ``None`` to disable;
        must be positive otherwise (a non-positive norm zeros every
        gradient). Default 1.0 — text encoders are sensitive to
        gradient spikes early in fine-tuning.
    weight_decay: AdamW weight decay. Default 0.0.
    optimizer_cls: optimizer constructor. Default
        :class:`torch.optim.AdamW`. Receives
        ``(trainable_params, lr=lr, weight_decay=weight_decay)``.
    verbose: print per-epoch mean loss. Default False.

Returns:
    The (in-place-mutated) ``backbone``.

Raises:
    ValueError: on a dataset of fewer than 2 pairs, batch_size < 2,
        non-positive epochs, non-positive temperature, or a
        non-positive ``grad_clip_norm`` — NT-Xent needs at least one
        negative, so both the dataset and every batch must carry
        >= 2 pairs.
    FloatingPointError: when the contrastive loss goes non-finite
        mid-training (check lr / temperature / input normalization).
```


#### `nnx.embeddings.contrastive_trainer.embed_texts`

```python
nnx.embeddings.contrastive_trainer.embed_texts(backbone: 'Any', texts: 'list[str]', *, batch_size: 'int' = 64, device: 'Optional[Union[str, torch.device]]' = None, normalize: 'bool' = True) -> 'torch.Tensor'
```

Encode ``texts`` with ``backbone`` and return a ``(N, D)`` tensor.

**Details**

```text
Runs in ``torch.no_grad()`` + ``eval()`` mode — this is the
inference helper, not the training one. The trainer drives
:func:`_encode` directly so gradients flow.

Args:
    backbone: text encoder — a sentence-transformers model or any
        ``nn.Module`` whose ``forward(list[str]) -> Tensor[B, D]``.
    texts: input strings. May be empty (returns a ``(0, ?)``
        placeholder — the embedding dim isn't known until the
        first forward).
    batch_size: how many texts per forward pass. Default 64.
    device: target device. ``None`` uses the backbone's device
        (sentence-transformers exposes one; plain Modules don't, in
        which case we fall back to the first parameter's device,
        or CPU when the backbone has no parameters).
    normalize: if True, L2-normalize each row so dot products with
        the result are cosine similarities. Default True because
        FAISS's ``IndexFlatIP`` interprets the inner product as a
        similarity score and the standard cosine-by-IP trick is
        normalize-then-IP.

Returns:
    A ``(N, D)`` ``torch.Tensor`` on ``device``. Detached from
    any autograd graph.
```


#### `nnx.embeddings.contrastive_trainer.text_contrastive_train_step_factory`

```python
nnx.embeddings.contrastive_trainer.text_contrastive_train_step_factory(*, temperature: 'float' = 0.5) -> 'TrainStepFn'
```

Build a :class:`TrainStepFn` for text-pair contrastive training.

**Details**

```text
This is the text-aware sibling of
:func:`nnx.simclr_train_step_factory`. The training loader must
yield ``(anchors: list[str], positives: list[str])`` batches —
typically by pairing :class:`ContrastiveTextDataset` with
:func:`pair_collate`.

The step runs:

  1. Encode anchors through ``model.net`` → ``z1``.
  2. Encode positives through ``model.net`` → ``z2``.
  3. NT-Xent loss across the ``(2B, 2B)`` similarity matrix.
  4. Standard :func:`finalize_step` tail (NaN guard, grad clip,
     optimizer step).

Args:
    temperature: NT-Xent temperature. Lower sharpens; 0.5 is the
        SimCLR default. Must be > 0.

Returns:
    A ``TrainStepFn`` suitable for ``NNModel.train(..., train_step_fn=...)``.

Raises:
    ValueError: at factory-build time if ``temperature`` ≤ 0.
```


#### `nnx.embeddings.faiss_export.export_to_faiss`

```python
nnx.embeddings.faiss_export.export_to_faiss(backbone: 'Any', corpus: 'list[str]', out_path: 'Union[str, Path]', *, batch_size: 'int' = 64, index_type: 'str' = 'IndexFlatIP', normalize: 'Optional[bool]' = None, device: 'Optional[Union[str, torch.device]]' = None) -> 'str'
```

Embed ``corpus`` with ``backbone`` and write a FAISS index file.

**Details**

```text
The default ``IndexFlatIP`` + ``normalize=True`` combination is
cosine similarity: L2-normalize the embeddings, then use inner
product as the score. This is the standard FAISS-cosine recipe
(FAISS itself doesn't ship a cosine index; the normalize-then-IP
pattern is canonical).

The corpus order is preserved in the index — ``index.search``'s
returned ids are positions into ``corpus``. The caller is
responsible for keeping a parallel list / DataFrame of original
document ids or metadata.

Args:
    backbone: text encoder. Either a
        :class:`sentence_transformers.SentenceTransformer` or any
        ``nn.Module`` whose ``forward(list[str]) -> Tensor[B, D]``.
    corpus: list of strings to embed. Order is the index's id space.
        Empty raises :class:`ValueError` — FAISS rejects 0-length
        adds.
    out_path: destination file path. The parent directory must
        exist. The file is written via FAISS's native
        ``write_index`` (atomic depends on the underlying FS).
    batch_size: forward-pass batch size. Default 64.
    index_type: FAISS index family to build. One of
        ``"IndexFlatIP"`` (default), ``"IndexFlatL2"``,
        ``"IndexHNSWFlat"``.
    normalize: whether to L2-normalize each embedding before
        insertion. ``None`` (the default) auto-selects: True for
        ``IndexFlatIP`` (cosine via IP), False for everything else.
        Pass an explicit bool to override.
    device: target device for the encode pass. ``None`` infers
        from the backbone.

Returns:
    The string path written. Same value as ``str(out_path)`` —
    returned for call-chain convenience.

Raises:
    ImportError: if ``faiss`` isn't installed (lazy import; only
        this call requires it).
    ValueError: empty corpus, unknown ``index_type``.
```


#### `nnx.embeddings.faiss_export.export_to_safetensors`

```python
nnx.embeddings.faiss_export.export_to_safetensors(backbone: 'Any', out_path: 'Union[str, Path]') -> 'str'
```

Persist ``backbone.state_dict()`` to disk for downstream reload.

**Details**

```text
Prefers the ``safetensors`` format (canonical for HuggingFace Hub
artifacts and sentence-transformers ≥3) when the
:mod:`safetensors` package is importable. Falls back to plain
:func:`torch.save` when it isn't, so the function still works on
a vanilla ``pip install thekaveh-nnx`` without the embeddings extra. In
the fallback case ``out_path`` is written as a pickle blob; the
caller's reloader needs to use :func:`torch.load`.

Args:
    backbone: anything with a ``state_dict()`` method.
        Sentence-transformers, raw ``nn.Module``, even a plain
        ``OrderedDict`` of tensors.
    out_path: destination file path. Conventionally suffixed
        ``.safetensors`` for the primary path; ``.pt`` for the
        torch.save fallback. We don't enforce the suffix — that's
        cosmetic.

Returns:
    The string path written.
```


## 16. Interop (`nnx.interop`)

### 16.1. Experimental GGUF export

Walkthrough and stock-runtime limitations at [Experimental GGUF export](gguf.md). Opt-in via `pip install "thekaveh-nnx[gguf-write]"`.

#### `nnx.interop.gguf.writer.write_gguf`

```python
nnx.interop.gguf.writer.write_gguf(transformer_nn: 'TransformerNN', tokenizer: 'NNTokenizerParams', out_path: 'str | os.PathLike', *, architecture: 'str' = 'nnx_transformer', quantization: 'str' = 'F16', model_name: 'Optional[str]' = None) -> 'str'
```

Write a TransformerNN + tokenizer to a single ``.gguf`` file.

**Details**

```text
Args:
    transformer_nn: A ``nnx.TransformerNN`` instance. The forward
        path's tensors are exported under llama.cpp's tensor-naming
        convention (see ``tensor_name_map.map_tensors``).
    tokenizer: An ``nnx.NNTokenizerParams`` (or any object with a
        ``.tokenizer`` attribute exposing ``.get_vocab()`` and
        ``.get_vocab_size()``). Tokens + merges are emitted under
        the GGUF tokenizer keys.
    out_path: Destination ``.gguf`` path.
    architecture: ``general.architecture`` metadata value. Defaults
        to ``"nnx_transformer"``. Stock llama.cpp/Ollama do not
        implement this architecture; the artifact is intended for
        GGUF inspection or a reader that explicitly supports NNx.
        Do not relabel it ``"llama"``: NNx uses interleaved RoPE,
        which is not LLaMA's split-half layout.
    quantization: One of ``"F32"``, ``"F16"``, ``"BF16"``. Sub-F16
        quantizations require the C++ ``llama-quantize`` binary —
        see the ``ImportError`` message for the shell-out recipe.
    model_name: ``general.name`` metadata. Defaults to a
        ``"nnx_transformer_LxD"`` shape-derived name.

Returns:
    The absolute path of the written file as a string.

Raises:
    ImportError: when ``gguf`` is not installed, or when a
        quantization is requested that requires ``llama-quantize``.
    ValueError: when an unknown quantization label is passed.
```


#### `nnx.interop.gguf.tensor_name_map.map_tensors`

```python
nnx.interop.gguf.tensor_name_map.map_tensors(net: 'TransformerNN') -> 'dict[str, np.ndarray]'
```

Walk a ``TransformerNN`` and return ``{gguf_name: numpy_array}``.

**Details**

```text
The caller (``write_gguf``) then iterates this dict and calls
``GGUFWriter.add_tensor`` for each entry. Splitting the iteration
here (rather than inlining it into the writer) keeps the naming
convention testable in isolation — see ``test_gguf_writer.py``.

Args:
    net: A ``TransformerNN`` instance.

Returns:
    Dict ``gguf_name -> numpy.ndarray``. Q/K/V are emitted as three
    separate tensors even though the NNx side stores them fused.
    When ``net.params.tie_embeddings`` is True, ``output.weight``
    is omitted (llama.cpp re-uses ``token_embd.weight`` for tied
    models).
```


#### `nnx.interop.ollama.export_ollama_modelfile`

```python
nnx.interop.ollama.export_ollama_modelfile(transformer_nn: 'TransformerNN', tokenizer: 'NNTokenizerParams', out_dir: 'str | os.PathLike', *, system: 'str' = '', parameters: 'Optional[dict]' = None, template: 'Optional[str]' = None, quantization: 'str' = 'F16', model_name: 'Optional[str]' = None) -> 'str'
```

Emit an experimental ``model.gguf`` + ``Modelfile`` bundle.

**Details**

```text
Stock Ollama does not implement the ``nnx_transformer`` GGUF
architecture. Emission verifies bundle structure only; it does not
establish runtime compatibility.

Args:
    transformer_nn: An NNx ``TransformerNN`` instance — the model
        to export.
    tokenizer: Corresponding ``NNTokenizerParams``.
    out_dir: Output directory. Created if it doesn't exist.
    system: Optional system prompt; emitted as a ``SYSTEM ...``
        block (triple-quoted) when non-empty. Must not contain a
        triple-quote or end with a double-quote (Modelfile block
        delimiters — validated, raises ``ValueError``); same
        constraint applies to ``template``.
    parameters: Optional dict of Ollama runtime parameters
        from the documented 0.32.2 set. Each entry becomes a
        ``PARAMETER <key> <value>`` line; only ``stop`` accepts a
        list or tuple, rendered as repeated lines. String values use
        an injection-safe subset without quotes or control characters.
    template: Optional chat template. Emitted as a
        ``TEMPLATE ...`` block (triple-quoted) when set.
    quantization: Forwarded to :func:`write_gguf`. Defaults to F16.
    model_name: Forwarded to :func:`write_gguf` as ``model_name``.

Returns:
    Absolute path to the emitted ``Modelfile``.
```


## 17. HuggingFace Hub + safetensors

Opt-in via `pip install "thekaveh-nnx[hub]"`. Two integration surfaces:

- **safetensors checkpoints** — `NNCheckpoint.to_file(..., format="safetensors")` and `NNCheckpoint.from_file(..., format="safetensors")` (see §3 `NNCheckpoint`) read and write checkpoints in the safetensors format alongside the default pickle path. Loadable by outside-Python tools (ComfyUI, vLLM, AutoGPTQ).
- **Hub publish / load** — `NNModel` mixes in `huggingface_hub.PyTorchModelHubMixin`, so `save_pretrained(local_dir)`, `push_to_hub(repo_id)`, and `NNModel.from_pretrained(repo_id)` work directly on a trained model. The mixin methods are inherited and live on `NNModel` itself — see §2.1.

Walkthrough at [HuggingFace Hub](hub.md).

## 18. Generation (`nnx.generation`)

`LogitsProcessor` chain for autoregressive sampling. Used by `GenerativeNNModel.generate()` (§2.2). Pure-torch — no optional deps.

#### `nnx.generation.LogitsProcessor`

```python
class nnx.generation.LogitsProcessor(*args, **kwargs)
```

Callable protocol: ``logits, token_history -> adjusted_logits``.

**Details**

```text
``token_history`` is a flat list of int token ids generated so far
(across batch dim 0 — we assume a single-sequence batch in
``GenerativeNNModel.generate``). Processors that don't care about
history (temperature, top-k, top-p) simply ignore the arg.
```


#### `nnx.generation.LogitsChain`

```python
class nnx.generation.LogitsChain(*, processors: 'list[LogitsProcessor]' = <factory>) -> 'None'
```

A typed, ordered sequence of LogitsProcessors.

**Details**

```text
Build via `LogitsChain.builder()` for the safe / discoverable
path; or construct directly from a list for advanced cases. The
`.apply()` method runs the processors against a logits tensor in
order, returning the adjusted tensor.
```

##### `nnx.generation.LogitsChain.apply`

```python
nnx.generation.LogitsChain.apply(self, logits: 'torch.Tensor', token_history: 'list[int]') -> 'torch.Tensor'
```

Run every processor in `self.processors` in order. Thin wrapper around `apply_chain`.

##### `nnx.generation.LogitsChain.builder`

```python
nnx.generation.LogitsChain.builder() -> 'LogitsChainBuilder'
```

Return a fluent builder. See `LogitsChainBuilder`.


#### `nnx.generation.LogitsChainBuilder`

```python
class nnx.generation.LogitsChainBuilder() -> 'None'
```

Fluent builder for a `LogitsChain`.

**Details**

```text
Method order at the call site doesn't matter — `.build()` sorts
the standard processors into NNx's canonical order (matching
`generate()`'s inline-kwargs chain; see the module docstring for
why temperature is deliberately last):
`RepetitionPenalty → TopKFilter → TopPFilter → TemperatureScaling`.
Custom processors (added via `.custom(processor)`) are appended in
the order they were added, after the canonical group.
```

##### `nnx.generation.LogitsChainBuilder.repetition_penalty`

```python
nnx.generation.LogitsChainBuilder.repetition_penalty(self, penalty: 'float') -> 'LogitsChainBuilder'
```

Add a RepetitionPenalty processor with the given penalty.

##### `nnx.generation.LogitsChainBuilder.top_k`

```python
nnx.generation.LogitsChainBuilder.top_k(self, k: 'int') -> 'LogitsChainBuilder'
```

Add a TopKFilter with the given k.

##### `nnx.generation.LogitsChainBuilder.top_p`

```python
nnx.generation.LogitsChainBuilder.top_p(self, p: 'float') -> 'LogitsChainBuilder'
```

Add a TopPFilter (nucleus sampling) with the given p.

##### `nnx.generation.LogitsChainBuilder.temperature`

```python
nnx.generation.LogitsChainBuilder.temperature(self, t: 'float') -> 'LogitsChainBuilder'
```

Add a TemperatureScaling processor with the given temperature.

##### `nnx.generation.LogitsChainBuilder.custom`

```python
nnx.generation.LogitsChainBuilder.custom(self, processor: 'LogitsProcessor') -> 'LogitsChainBuilder'
```

Append a user-supplied LogitsProcessor after the canonical group. Useful for logit-bias / forbidden-token / domain-specific adjustments. Multiple `.custom(...)` calls append in order.

##### `nnx.generation.LogitsChainBuilder.build`

```python
nnx.generation.LogitsChainBuilder.build(self) -> 'LogitsChain'
```

Construct the LogitsChain with processors in canonical order.

**Details**

```text
Standard processors that were chained are emitted in the
fixed `_CANONICAL_ORDER`; custom processors come after, in
the order they were added.
```


#### `nnx.generation.TemperatureScaling`

```python
class nnx.generation.TemperatureScaling(temperature: 'float')
```

Divide logits by ``temperature`` before sampling.

**Details**

```text
``temperature == 0`` is a special case: the chain reduces to greedy
decoding (argmax). We map argmax positions to +inf and others to
-inf so the downstream sampler picks deterministically without
branching on the temperature value.
```


#### `nnx.generation.TopKFilter`

```python
class nnx.generation.TopKFilter(top_k: 'int')
```

Keep only the top-k logits per row; set the rest to -inf.

**Details**

```text
-inf survives the temperature divide (still -inf) and gets mapped
to 0 probability mass by softmax, so the order top-k → temperature
or temperature → top-k both work; we don't enforce an ordering.
```


#### `nnx.generation.TopPFilter`

```python
class nnx.generation.TopPFilter(top_p: 'float')
```

Nucleus (top-p) sampling: keep the smallest set of tokens whose cumulative probability exceeds ``top_p``.

**Details**

```text
Edge case: if a single token already has probability >= top_p, only
that token is retained.
```


#### `nnx.generation.RepetitionPenalty`

```python
class nnx.generation.RepetitionPenalty(penalty: 'float')
```

Penalize previously-seen tokens (HF-style).

**Details**

```text
For each token id ``i`` in ``token_history``:
  * if ``logits[..., i] > 0``: divide by penalty (decreases mass).
  * if ``logits[..., i] < 0``: multiply by penalty (increases
    magnitude → further decreases relative mass after softmax).

A penalty of 1.0 is a no-op (the back-compat default).
```


#### `nnx.generation.apply_chain`

```python
nnx.generation.apply_chain(logits: 'torch.Tensor', *, token_history: 'list[int]', processors: 'list[LogitsProcessor]') -> 'torch.Tensor'
```

Apply every processor in order. No-op when ``processors`` is empty.


#### `nnx.generation.sample_next_token`

```python
nnx.generation.sample_next_token(logits: 'torch.Tensor', *, generator: 'Optional[torch.Generator]' = None) -> 'int'
```

Draw one token id from ``softmax(logits)``.

**Details**

```text
Args:
    logits: shape (1, vocab) — single-sequence sample (the LM
        path's batch-1 generate scope).
    generator: optional torch.Generator for reproducible seeded
        sampling. When None, sampling uses the default RNG (still
        affected by torch.manual_seed at the call site).

Returns:
    An int token id.
```


## 19. Visualization

### 19.1. Run-output viz (`nnx.vis_utils`)

#### `nnx.vis_utils`

```python
module nnx.vis_utils
```

Run-output visualization helpers.

**Details**

```text
`VisUtils` collects the Plotly-based visualizations for *run outputs* —
the artifacts produced after `NNModel.train()` has completed: training
curves, confusion matrices, classification reports, t-SNE projections
of held-out logits, etc. It is the sibling of `nnx.viz` (model-internals
visualization — weight histograms, activation maps, gradient flow,
Netron export); the two subpackages are deliberately independent and
do not share code.

Every method is a `@staticmethod` returning either a `plotly.graph_objects.Figure`
(for plots) or a `pandas.DataFrame` (for tables). The class itself
carries layout constants (`TITLE_SIZE`, `LABEL_SIZE`, `FIG_SIZE`,
`MARGIN_SIZE`) shared across methods, plus an opt-in `RENDERER`
override for environments where Plotly's default renderer doesn't
work (e.g., when serving from a headless container).

Convenience module-level aliases re-export the most common methods at
the bottom of this file so callers can write `nnx.vis_utils.confusion_matrix(...)`
instead of `nnx.vis_utils.VisUtils.confusion_matrix(...)`.
```

##### `nnx.vis_utils.confusion_matrix`

```python
nnx.vis_utils.confusion_matrix(Y_true, Y_pred, class_names=None, title: 'str' = 'Confusion matrix', normalize: 'bool' = False)
```

Render a confusion matrix heatmap. Y_true and Y_pred are 1-D arrays of integer class labels. If `class_names` is provided, axis labels use the named classes; otherwise integer indices.

##### `nnx.vis_utils.classification_report`

```python
nnx.vis_utils.classification_report(Y_true, Y_pred, class_names=None) -> 'pd.DataFrame'
```

Per-class precision / recall / f1 / support as a DataFrame. Use the return value for tabular display (`print(df.to_string())` or notebook auto-display) or to feed back into downstream analysis.

##### `nnx.vis_utils.multi_line_plot`

```python
nnx.vis_utils.multi_line_plot(x, yss, title, yss_legend, x_axis_label, y_axis_label, x_ticks_inc=20, label_size=12, title_size=14, fig_size: 'tuple' = (1000, 600), margin_size={'l': 15, 'r': 15, 't': 30, 'b': 15, 'pad': 0}, renderer=None)
```

Render a multi-group line chart and return the Plotly Figure.

**Details**

```text
Each group in `yss` is drawn with a distinct color; each line within
a group uses a distinct dash style. `yss_legend` is a (group_labels,
line_labels) tuple — group_labels name the colored groups (one per
entry in `yss`), line_labels name the dash styles shared across
groups. Both legends are added as no-trace markers so the legend
reads cleanly; data traces carry "group (line)" hover names.

Returns the Figure. If `renderer` is non-None, also calls
`fig.show(renderer=renderer)` so notebook callers see the chart
inline; pass `renderer=None` (the default) for headless usage.
```

##### `nnx.vis_utils.scatter_plot`

```python
nnx.vis_utils.scatter_plot(vm, renderer=None, fig_size: 'tuple' = (1000, 600), label_size: 'int' = 12, title_size: 'int' = 14, margin_size={'l': 15, 'r': 15, 't': 30, 'b': 15, 'pad': 0})
```

Render a colored scatter plot from a view-model dict and return the Plotly Figure.

**Details**

```text
`vm` is the structure produced by `get_scatter_plot_vm`: title, xs/ys
column views, plus a `ts` group axis carrying labels + colors per
category. Honors `renderer` the same way as `multi_line_plot`.
```

##### `nnx.vis_utils.two_dim_tsne_checkpoint_logits`

```python
nnx.vis_utils.two_dim_tsne_checkpoint_logits(checkpoint: 'NNCheckpoint', ds: 'NNDataset', n_samples: 'int', random_state: 'int | None' = 0, renderer: 'str | None' = None, fig_size: 'tuple' = (1000, 600), title_size: 'int' = 14, label_size: 'int' = 12, margin_size={'l': 15, 'r': 15, 't': 30, 'b': 15, 'pad': 0})
```

Project the first `n_samples` test logits of `checkpoint` to 2D via t-SNE and render them colored by ground-truth class.

**Details**

```text
Useful for eyeballing class separability of an intermediate
checkpoint — pass the BEST checkpoint to see how well-trained the
decision space ended up. Returns the Plotly Figure.
```


### 19.2. Model-internals viz (`nnx.viz`)

Opt-in via `pip install "thekaveh-nnx[viz]"` (pulls `torchinfo` + `captum`) and `pip install "thekaveh-nnx[viz-interactive]"` (adds the `netron` browser viewer for `nnx.viz.netron_export(..., launch=True)`).

#### `nnx.viz.activation.activation_map`

```python
nnx.viz.activation.activation_map(model: 'Union[nn.Module, NNModel]', x: 'torch.Tensor', layer_name: 'str', *, max_channels: 'int' = 16, cols: 'int' = 4, fig_width: 'int' = 900, cell_size: 'int' = 180) -> 'go.Figure'
```

Capture the activation of `layer_name` for input `x` and render it.

**Details**

```text
Registers a forward hook on the named submodule, runs `model(x)` under
`torch.no_grad()`, then removes the hook and turns the captured tensor
into a Plotly heatmap layout:

- 4D ``(N, C, H, W)`` activations: grid of up to `max_channels` per-channel
  heatmaps from the first sample (``N=0``).
- 2D ``(N, F)`` activations: single ``(N, F)`` heatmap.
- Other ranks: flattened single-row heatmap (best-effort fallback).

Args:
    model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
    x: Input tensor (or any object) accepted by `model.__call__`. Moved to
        the same device as the model's first parameter when possible.
    layer_name: Dotted name from `model.named_modules()` — e.g. `"layers.2"`
        for a Sequential, `"conv1"` for a class attribute. Pass an empty
        string `""` to hook the top-level module itself.
    max_channels: Cap on conv-channel subplots (4D case). Defaults to 16 —
        enough to spot patterns without crushing the layout for 256-channel
        feature maps.
    cols: Subplot columns in the 4D grid.
    fig_width: Total figure width in pixels.
    cell_size: Per-subplot square cell size (px). Total height scales
        with the row count.

Returns:
    A Plotly `Figure` containing one or more `Heatmap` traces.

Raises:
    ValueError: If `layer_name` doesn't resolve to a submodule of `model`.
    RuntimeError: If the forward hook on `layer_name` never fires
        (the layer is not reached by this input's forward path).
```


#### `nnx.viz.attribute.attribute`

```python
nnx.viz.attribute.attribute(model: 'Union[nn.Module, NNModel]', x: 'torch.Tensor', *, method: 'str' = 'integrated_gradients', target: 'Any' = None, **method_kwargs: 'Any') -> 'tuple[torch.Tensor, go.Figure]'
```

Compute input attributions via Captum and render a Plotly heatmap.

**Details**

```text
Args:
    model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
        The model is set to `eval()` for the duration of the attribution
        call; the original mode is restored on return.
    x: Input tensor to attribute. Shape `(B, ...)`. Gradient-based
        methods will set `requires_grad_(True)` internally as needed.
    method: One of `"integrated_gradients"`, `"gradient_shap"`,
        `"deep_lift"`, `"saliency"`, `"input_x_gradient"`, `"occlusion"`.
    target: Target class index (or per-batch indices) for classification
        attributors. Forwarded verbatim to Captum's `.attribute(target=)`.
    **method_kwargs: Extra kwargs forwarded to the per-method
        `.attribute(...)` call. Overrides any defaults supplied for
        `gradient_shap` (`baselines`) or `occlusion` (`sliding_window_shapes`).

Returns:
    A tuple `(attribution_tensor, figure)` where `attribution_tensor` is a
        `torch.Tensor` with the same shape as `x` (per Captum's standard
        return contract for these six methods) and `figure` is a Plotly
        `Heatmap` visualizing the attribution. Image-shaped inputs (3-D /
        4-D) are mean-pooled over channels before rendering.

Raises:
    ImportError: If `captum` is not installed. Install via
        `pip install thekaveh-nnx[viz]` or `pip install captum>=0.7.0`.
    ValueError: If `method` is not one of the supported keys.
```


#### `nnx.viz.gradient_flow.gradient_flow`

```python
nnx.viz.gradient_flow.gradient_flow(model: 'Union[nn.Module, NNModel]') -> 'go.Figure'
```

Return a Plotly bar chart of per-parameter L2 gradient norms.

**Details**

```text
Call AFTER ``loss.backward()`` and BEFORE ``optimizer.zero_grad()``.
Each bar is one trainable ``nn.Parameter`` of the model whose
``.grad`` has been populated by the backward pass; bar height is
the L2 norm of that gradient.

Frozen parameters (``requires_grad=False``) are skipped. Parameters
whose gradient is ``None`` (typically because they weren't reached
during the forward pass) are also skipped.

Args:
    model: an ``NNModel`` (unwrapped to its ``.net``) or any
        ``nn.Module`` whose gradients have just been populated by
        ``loss.backward()``.

Returns:
    A Plotly ``Figure`` with one bar per trainable parameter,
    labeled by ``named_parameters()`` dotted name.

Raises:
    ValueError: if no parameter has a populated gradient — most
        often because ``loss.backward()`` wasn't called before
        this function.
```


#### `nnx.viz.netron.netron_export`

```python
nnx.viz.netron.netron_export(model: 'Union[nn.Module, NNModel]', path: 'str', example_input: 'Union[torch.Tensor, tuple, np.ndarray]', *, launch: 'bool' = False, opset_version: 'int' = 17, dynamic_batch: 'bool' = True) -> 'str'
```

Export `model` to an ONNX file at `path` (optionally open Netron).

**Details**

```text
Args:
    model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
    path: Output filename, e.g. ``"model.onnx"``.
    example_input: A tensor (or tuple of tensors) with realistic
        shape / dtype used to trace the network.
    launch: When True, call `netron.start(path)` to open the model
        in Netron's browser viewer. Requires `pip install thekaveh-nnx[viz-interactive]`
        (or `pip install netron`). Defaults to False so CI / tests
        can exercise export without spawning a long-lived process.
    opset_version: ONNX opset to target. 17 is broadly supported
        by current runtimes.
    dynamic_batch: When True (default), marks dim 0 as dynamic so
        the exported graph accepts any batch size at inference.

Returns:
    The path written (matches `path` — handy when chaining).

Raises:
    ImportError: When `launch=True` and the `netron` package isn't
        installed. The ONNX export itself uses `torch.onnx`, which
        is part of core PyTorch.
```


#### `nnx.viz.summary.summary`

```python
nnx.viz.summary.summary(model: 'Union[nn.Module, NNModel]', *, input_size: 'tuple[int, ...] | None' = None, input_data: 'Union[torch.Tensor, tuple, list, None]' = None, depth: 'int' = 4, col_names: 'tuple[str, ...]' = ('output_size', 'num_params', 'mult_adds')) -> 'ModelStatistics'
```

Return a `torchinfo.ModelStatistics` summary for `model`.

**Details**

```text
Args:
    model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
    input_size: Shape tuple for a synthetic dummy input, e.g. `(1, 3, 224, 224)`.
        Mutually exclusive with `input_data`.
    input_data: An actual tensor / tuple / list to forward through the model.
        Useful when the model takes multiple positional arguments or a non-tensor
        input (graphs, dicts) that `input_size` can't describe.
    depth: Maximum module-nesting depth to expand in the table.
    col_names: Which torchinfo columns to include. Defaults to the three most
        useful ones for spotting parameter / FLOP regressions across runs.

Returns:
    The `torchinfo.ModelStatistics` instance — print it for the Keras-style
    table, or access `.total_params` / `.trainable_params` / `.total_mult_adds`
    for programmatic regression assertions.

Raises:
    ImportError: If `torchinfo` isn't installed. Install with `pip install thekaveh-nnx[viz]`.
```


#### `nnx.viz.weight_histogram.weight_histogram`

```python
nnx.viz.weight_histogram.weight_histogram(model: 'Union[nn.Module, NNModel]', *, bins: 'int' = 64, cols: 'int' = 3, fig_width: 'int' = 1000, row_height: 'int' = 200) -> 'go.Figure'
```

Return a Plotly grid of per-parameter weight histograms.

**Details**

```text
Args:
    model: An `NNModel` (unwrapped to `.net`) or any `torch.nn.Module`.
    bins: Number of histogram bins per parameter tensor.
    cols: Number of columns in the subplot grid. Rows are computed from the
        parameter count.
    fig_width: Figure width in pixels.
    row_height: Per-row height in pixels; total height = `row_height * rows`.

Returns:
    A Plotly `Figure` with one `Histogram` trace per named parameter tensor.
    Each subplot title is the dotted parameter name (e.g. `layers.0.weight`).
    Empty parameter tensors are skipped from the grid.

Raises:
    ValueError: If `model` has no named parameters (nothing to plot).
```


## 20. Utilities

#### `nnx.utils`

```python
module nnx.utils
```

Pretty-printing helpers used throughout nnx.

**Details**

```text
Both module-level functions (``print_tree``, ``print_table``, ``flatten_dict``)
and the legacy ``Utils`` class API are exported. New code should prefer the
module functions; ``Utils.method(...)`` is kept as a thin back-compat shim so
existing notebooks keep working.
```

##### `nnx.utils.print_tree`

```python
nnx.utils.print_tree(tree, level: 'int' = 0, *, file=None) -> 'None'
```

Pretty-print a nested dict as an indented tree.

**Details**

```text
Pass ``file=`` (any object with ``.write``) to redirect output away
from stdout — useful for capturing in tests or writing to a log.
Defaults to ``sys.stdout``.
```

##### `nnx.utils.print_table`

```python
nnx.utils.print_table(data: 'dict', header: 'bool' = True, title: 'Optional[str]' = None, *, file=None) -> 'None'
```

Print ``data`` as a 2-column key/value table.

**Details**

```text
Pass ``file=`` to redirect output. Defaults to ``sys.stdout``.
```

##### `nnx.utils.flatten_dict`

```python
nnx.utils.flatten_dict(data: 'dict', parent_key: 'str' = '', sep: 'str' = '.') -> 'dict'
```

Flatten a nested dict so nested keys become ``parent.child`` style.

**Details**

```text
>>> flatten_dict({"a": 1, "b": {"c": 2}})
{'a': 1, 'b.c': 2}
```


### 20.1. `Utils` back-compat facade

`nnx.Utils` is a thin staticmethod facade over the module-level functions above, kept so existing notebook code that calls `Utils.print_tree(...)` / `Utils.print_table(...)` / `Utils.flatten_dict(...)` continues to work. New code should prefer the module-level functions directly.
