# HuggingFace Hub integration

NNx ships first-class interop with the HuggingFace ecosystem:

1. **safetensors** as an opt-in checkpoint format on `NNCheckpoint` —
   safe (no arbitrary-code unpickling), mmap-friendly, and readable by
   ComfyUI / vLLM / AutoGPTQ / `transformers` tools.
2. **`PyTorchModelHubMixin`** on `NNModel` — free `save_pretrained` /
   `push_to_hub` / `from_pretrained` for distributing models via the
   Hub.

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
  followed by a JSON object — neither pickle prefix can appear there.

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
  the same public `state()` form NNRun hashes for `run.id` grouping.
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

## 3. What this does NOT do

- **`NNRun` is not Hub-published.** The Hub layout is per-model, not
  per-training-run. If you want to publish a full training run
  (idps.csv + run.yaml + every per-phase checkpoint), upload the
  `runs/<id>/` directory directly via `huggingface_hub.upload_folder`.
- **Optimizer state is not in the Hub config.** `save_pretrained`
  writes only the network weights; resuming optimizer state from a
  Hub-loaded model isn't supported. Use `NNCheckpoint` for warm-resume
  workflows.
- **The Hub mixin doesn't rewrite `NNModel`'s constructor.** It still
  takes `(net_params, params)` keyword args at `__init__` — the mixin
  is purely additive. Existing code keeps working unchanged.
