# 5. HuggingFace Hub integration

NNx ships first-class interop with the HuggingFace ecosystem:

1. **safetensors** as an opt-in checkpoint format on `NNCheckpoint` —
   safe (no arbitrary-code unpickling), mmap-friendly, and readable by
   ComfyUI / vLLM / AutoGPTQ / `transformers` tools.
2. **`PyTorchModelHubMixin`** on `NNModel` — free `save_pretrained` /
   `push_to_hub` / `from_pretrained` for distributing models via the
   Hub.

A third, Hub-independent format — the **run bundle** (`nnx.bundles`, §3) —
carries a run checkpoint's training state and calibrators as data too.

Both paths require the `hub` extra:

```bash
pip install "thekaveh-nnx[hub]"
```

Without it, the rest of NNx keeps working — the integration is gated
behind import-time guards. Calling any Hub method without the extra
raises a clear `ImportError` pointing back at this install line.

## 1. safetensors checkpoints

### 1.1. When to use it

Use safetensors when **any** of the following is true:

- You plan to publish weights to the Hub or share them with anyone
  outside your machine. Pickle checkpoints can execute arbitrary code
  on load (see the security note on `NNCheckpoint.from_file`);
  safetensors cannot.
- You need to load weights into a non-Python tool (ComfyUI, vLLM,
  AutoGPTQ, `transformers`-aware loaders).
- You care about mmap-based zero-copy loads — safetensors files are
  laid out so the underlying tensors can be mapped directly from disk.

Pickle remains the default and is the right choice for local-only
training runs where the convenience of `torch.save`-ing the full
dataclass (preserving the `OrderedDict` key order and the
`NNCheckpoint` identity) outweighs the security trade-off.

### 1.2. Writing a safetensors checkpoint

`NNCheckpoint.to_file` takes a `format` kwarg:

```python
from nnx import NNCheckpoint

# Build a checkpoint as usual…
ckpt = NNCheckpoint(
    idp=...,                # NNIterationDataPoint
    model_params=model.params,
    net_params=model.net_params,
    net_state=model.net.state_dict(),
)

# …then write either format. Pickle is the default.
ckpt.to_file("checkpoint.pt")                            # legacy default
ckpt.to_file("checkpoint.safetensors", format="safetensors")
```

`NNParams`, `NNModelParams`, and `NNIterationDataPoint` are
JSON-serialized into the safetensors `metadata` dict (the format spec
limits metadata to `str -> str`, so a JSON wrapper is the cleanest
fit). The net's tensors are detached and made contiguous, then written
through safetensors' standard `save_file`.

Writes are atomic — the file is staged at `<path>.tmp` and `os.replace`-d
into place — matching the same KeyboardInterrupt-safety guarantee that
the pickle path provides.

### 1.3. Reading a checkpoint of either format

`NNCheckpoint.from_file` auto-detects which format the file was written
in by sniffing the first few bytes:

- Modern `torch.save` produces a ZIP container that starts with
  `b"PK\x03\x04"`.
- Legacy `torch.save` (with `_use_new_zipfile_serialization=False`)
  and bare pickle files start with the `\x80` PROTO opcode.
- safetensors files start with a little-endian u64 header length
  followed by a JSON object — byte 8 is always `{`. (The u64's low
  byte can legitimately be `0x80`, colliding with the pickle PROTO
  opcode, so the loader positively identifies safetensors via byte 8
  before the pickle sniff.)

The same call works for both:

```python
ckpt = NNCheckpoint.from_file("checkpoint.safetensors")  # or .pt
model = NNModel.from_checkpoint(ckpt)
```

## 2. Publishing an NNModel to the Hub

### 2.1. When to use the Hub mixin

Use `save_pretrained` / `push_to_hub` / `from_pretrained` for
**distribution**: shipping a trained NNModel so others can
`from_pretrained("you/your-model")` and run it. The flat on-disk layout
this writes (`model.safetensors` + `config.json` + `README.md`) is what
the Hub expects, and it's what downstream tools probe for.

Keep using `NNCheckpoint` for **local training state**: the
`runs/<id>/checkpoints/` layout that NNx writes during training carries
per-epoch IDPs, optimizer state sidecars, and run.id-keyed metadata
that the Hub layout deliberately strips.

### 2.2. Save a model locally

```python
from nnx import NNModel, NNParams, NNModelParams, Activations, Devices, Losses, Nets

model = NNModel(
    net_params=NNParams(
        input_dim=4, output_dim=2, hidden_dims=[8],
        dropout_prob=0.0, activation=Activations.RELU,
    ),
    params=NNModelParams(
        net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
    ),
)
# …train…
model.save_pretrained("./my-model")
```

This writes three files into `./my-model/`:

- `model.safetensors` — `self.net.state_dict()` as safetensors.
- `config.json` — `{"net_params": <state>, "params": <state>}`, using
  the same public `state()` form NNRun hashes for `run.id` grouping. A
  model built from a registered factory (§2.6) has no `net_params`; its
  `params.net` is the `ModelSpec` descriptor.
- `README.md` — auto-generated model card from the mixin.

### 2.3. Load from a local directory

```python
from nnx import NNModel
model = NNModel.from_pretrained("./my-model")
```

`from_pretrained` reads `config.json`, rebuilds `NNParams` and
`NNModelParams` via their public `from_state` constructors, then loads
the safetensors weights into the freshly-built `self.net`. Bit-exact
round-trip on tensors; `state()` form identical on the params.

### 2.4. Publish to the Hub

```python
# One-time login (writes a token to ~/.cache/huggingface/token):
#   hf auth login

model.push_to_hub("your-user/your-model")
```

The mixin handles repo creation, file upload, and commit. Everything
that `save_pretrained` writes locally is pushed.

### 2.5. Load from the Hub

```python
model = NNModel.from_pretrained("your-user/your-model")
```

Hugging Face's cache directory is used transparently — repeat loads
hit the local cache, not the network.

### 2.6. Registered and runtime-only modules

A model built from a registered factory (`ModelSpec`, see
[Concepts §3.2](concepts.md#32-arbitrary-modules-and-registered-model-factories))
round-trips through the Hub like a built-in one: `config.json` stores the
spec (`id`, `version`, `config`, `seed`) instead of `net_params`, and
`from_pretrained` rebuilds the module through the factory registry — so
the same `register_model_factory(...)` must have run in the loading
process. An unregistered factory raises `MissingModelFactoryError` before
anything is built, and a factory whose topology no longer matches the
weights raises before they load. Pass `batch_adapter=` to
`from_pretrained` for a module that needs one.

A runtime-only module (`NNModel(module=...)`, `reconstructible=False`) has
no factory to rebuild it, so `save_pretrained` / `push_to_hub` reject it
with `MissingModelFactoryError` before any directory or file is written.
Register a factory and train from a `ModelSpec` to publish it.

## 3. Three artifact formats and their trust boundaries

NNx writes three kinds of artifact. Each has one reader, and no reader opens
another's files:

| Format | Written by | Holds | Read by | Trust boundary |
|---|---|---|---|---|
| Pickle checkpoint `runs/<id>/checkpoints/<tag>.pt` (+ `.opt.<generation>.pt` sidecar) | `NNModel.train`, `NNCheckpoint.save` / `to_file()` | weights, params, epoch record, and the training state that resumes the run | `NNCheckpoint.from_file` / `load`, `train(resume_from_run_id=...)` | **Unpickles** (`torch.load(weights_only=False)`): only files you produced. |
| safetensors checkpoint / Hub distribution | `to_file(format="safetensors")`, `save_pretrained` / `push_to_hub` | weights and params (no optimizer or RNG state) | `NNCheckpoint.from_file`, `NNModel.from_pretrained` | Data only; `from_pretrained` downloads from the Hub when given a repo id. |
| Run bundle `<dir>/bundle.json` + `g-<generation>/` | `nnx.bundles.export_bundle` | weights, params, epoch record, training state (a `"resume"` bundle) or none (`"inference"`), calibrator records | `inspect_bundle`, `validate_bundle`, `reconstruct_bundle` | Data only: safetensors plus schema-validated JSON, checked against SHA-256 sums before any tensor is read; never unpickles, imports code or downloads. |

- **Export reads your own run.** `export_bundle(run_id, "bundle")` opens the
  run's pickle checkpoint — the legacy trust boundary, files NNx wrote
  locally — and publishes it as data. From then on the bundle can travel:
  `validate_bundle` checks every payload's size and SHA-256, the generation
  id, and that nothing is missing, unlisted, symlinked or outside the
  bundle, before any tensor is read.
- **Reconstruction takes caller-supplied registries.** A registered module
  (`ModelSpec`) is rebuilt only from the factories you pass
  (`reconstruct_bundle(path, factories={(id, version): factory})`, or the
  process registry by default); the bundle names none to import. A missing
  factory or component is reported before any model is allocated.
- **The formats stay distinguishable.** `NNCheckpoint.from_file` refuses a
  bundle directory and a bundle's payload files; `from_pretrained` refuses a
  bundle directory; the bundle readers refuse a pickle checkpoint, a run
  directory and a Hub distribution (`config.json`), and never fall back to
  unpickling or downloading.
- **What does not fit is refused.** Module extra state that is not a tensor,
  a custom object in optimizer or component state, a runtime-only module
  and an unknown bundle version fail with a message, with no pickle
  fallback.

See [Concepts §21](concepts.md#21-run-bundles-nnxbundles) and
[`examples/run_bundle.py`](../examples/run_bundle.py).

## 4. What this does NOT do

- **`NNRun` is not Hub-published.** The Hub layout is per-model, not
  per-training-run. If you want to publish a full training run
  (idps.csv + run.yaml + every per-phase checkpoint), upload the
  `runs/<id>/` directory directly via `huggingface_hub.upload_folder`.
- **Optimizer state is not in the Hub config.** `save_pretrained`
  writes only the network weights; resuming optimizer state from a
  Hub-loaded model isn't supported. Use `NNCheckpoint` — or a run bundle
  (§3), which carries the training state as data — for warm-resume
  workflows.
- **The Hub mixin doesn't rewrite `NNModel`'s constructor.** It still
  takes `(net_params, params)` keyword args at `__init__` (plus the
  keyword-only `module=` / `batch_adapter=` for your own modules) — the
  mixin is purely additive. Existing code keeps working unchanged.
