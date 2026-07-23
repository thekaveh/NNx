"""save_pretrained / from_pretrained round-trip via local directory.

Hub upload is exercised manually against the real Hub — tests only cover
the local serialization path so they run offline in CI. The on-disk
layout matches what PyTorchModelHubMixin produces (a flat directory
with `model.safetensors`, `config.json`, and a README), so a
``HfApi.upload_folder(local_dir)`` would push the same bits the test
asserts on.
"""

from __future__ import annotations

import json

import pytest
import torch

# Same optional-extra convention every other gated test file follows:
# skip gracefully when the required extras aren't installed (the
# shipped sdist's suite must not hard-fail without them).
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")

from nnx import (  # noqa: E402
    Activations,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNParams,
)


def _tiny_model() -> NNModel:
    return NNModel(
        net_params=NNParams(
            input_dim=4,
            output_dim=2,
            hidden_dims=[8],
            dropout_prob=0.0,
            activation=Activations.RELU,
        ),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
        ),
    )


def test_hub_save_pretrained_emits_safetensors_and_config(tmp_path):
    """save_pretrained writes the canonical PyTorchModelHubMixin layout —
    `model.safetensors` (weights) and `config.json` (NNParams + NNModelParams).
    A `README.md` is also emitted by the mixin's model-card generator;
    we only assert on the load-critical files.
    """
    m = _tiny_model()
    m.save_pretrained(str(tmp_path))
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()


def test_hub_save_pretrained_config_contains_params(tmp_path):
    """config.json must carry the round-trippable form of both params
    dataclasses so `from_pretrained` can rebuild the model identically.
    Use the `state()` form (not asdict) — that's the public, stable,
    hash-grouping serialization NNRun uses on disk.
    """
    m = _tiny_model()
    m.save_pretrained(str(tmp_path))

    with open(tmp_path / "config.json") as f:
        cfg = json.load(f)
    assert "net_params" in cfg
    assert "params" in cfg
    assert cfg["net_params"] == m.net_params.state()
    assert cfg["params"] == m.params.state()


def test_hub_from_pretrained_round_trip(tmp_path):
    """A model loaded from a save_pretrained'd directory must have:
    - bit-exact weights in `.net`,
    - identical params + net_params state() dicts,
    - a working forward pass.
    """
    m = _tiny_model()
    m.save_pretrained(str(tmp_path))

    rt = NNModel.from_pretrained(str(tmp_path))

    for k in m.net.state_dict():
        assert torch.equal(m.net.state_dict()[k], rt.net.state_dict()[k])
    assert rt.params.state() == m.params.state()
    assert rt.net_params.state() == m.net_params.state()

    # Forward pass must work — exercises the rebuilt net + device + loss_fn.
    x = torch.randn(2, 4)
    with torch.no_grad():
        out_m = m.net(x)
        out_rt = rt.net(x)
    assert torch.equal(out_m, out_rt)


def test_hub_from_pretrained_map_location_overrides_serialized_device(tmp_path):
    model = _tiny_model()
    model.save_pretrained(str(tmp_path))
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["params"]["device"] = Devices.CUDA.value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    restored = NNModel.from_pretrained(str(tmp_path), map_location="cpu")

    assert restored.device.type == "cpu"


def test_hub_from_pretrained_rejects_indexed_map_location(tmp_path):
    model = _tiny_model()
    model.save_pretrained(str(tmp_path))

    with pytest.raises(ValueError, match="indexed Hub map_location"):
        NNModel.from_pretrained(str(tmp_path), map_location="cuda:1")


def test_hub_from_pretrained_accepts_torch_device_map_location(tmp_path):
    model = _tiny_model()
    model.save_pretrained(str(tmp_path))

    restored = NNModel.from_pretrained(str(tmp_path), map_location=torch.device("cpu"))

    assert restored.device.type == "cpu"


def test_hub_save_pretrained_overwrite(tmp_path):
    """A second save_pretrained into the same directory must rewrite,
    not error. Round-trip after overwrite still works.
    """
    m1 = _tiny_model()
    m1.save_pretrained(str(tmp_path))
    m2 = _tiny_model()  # fresh init → different weights
    m2.save_pretrained(str(tmp_path))

    rt = NNModel.from_pretrained(str(tmp_path))
    # Round-trip recovers the *second* save's weights, not the first.
    for k in m2.net.state_dict():
        assert torch.equal(m2.net.state_dict()[k], rt.net.state_dict()[k])


def test_hub_export_rejects_non_tensor_extra_state_clearly(tmp_path):
    class ExtraStateNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

        def get_extra_state(self):
            return {"labels": ["class-a"]}

        def set_extra_state(self, state):
            self.extra_state = state

    model = _tiny_model()
    model.net = ExtraStateNet()

    with pytest.raises(TypeError, match="Hugging Face Hub export.*non-tensor state_dict"):
        model.save_pretrained(str(tmp_path))


def test_hub_mixin_inheritance_is_visible():
    """NNModel must surface the PyTorchModelHubMixin API publicly — the
    three methods that distribution workflows depend on.
    """
    assert hasattr(NNModel, "save_pretrained")
    assert hasattr(NNModel, "from_pretrained")
    assert hasattr(NNModel, "push_to_hub")


def test_hub_from_pretrained_rejects_unexpected_model_kwargs(tmp_path):
    """Unknown kwargs forwarded by the mixin must raise instead of being
    silently dropped — NNModel rebuilds entirely from config.json, so a
    silently-ignored kwarg (e.g. a typo'd knob) would lie to the caller."""
    import pytest

    m = _tiny_model()
    m.save_pretrained(str(tmp_path))
    with pytest.raises(TypeError, match="unexpected model kwargs"):
        NNModel.from_pretrained(str(tmp_path), nonexistent_knob=1)


def test_hub_from_pretrained_strict_is_honored(tmp_path):
    """`strict` must actually reach load_state_dict. Pre-fix the code
    read `strict=strict if strict else True` — always True — so
    strict=False was impossible. Default stays strict (a key mismatch
    means a corrupted artifact); strict=False opts into partial loads."""
    import pytest
    from safetensors.torch import load_file, save_file

    m = _tiny_model()
    m.save_pretrained(str(tmp_path))

    weights_path = tmp_path / "model.safetensors"
    sd = load_file(str(weights_path))
    dropped = sorted(sd)[0]
    del sd[dropped]
    save_file(sd, str(weights_path))

    with pytest.raises(RuntimeError, match="[Mm]issing key"):
        NNModel.from_pretrained(str(tmp_path))  # default strict=True

    rt = NNModel.from_pretrained(str(tmp_path), strict=False)
    assert isinstance(rt, NNModel)


def test_hub_round_trip_tied_transformer(tmp_path):
    """save_pretrained on a DEFAULT transformer (tie_embeddings=True)
    crashed with safetensors' 'Some tensors share memory' — the Hub
    writer missed the .clone() its NNCheckpoint sibling carries. The
    round-trip must restore weights bit-exactly with the tie intact and
    greedy generation parity."""
    import pytest

    pytest.importorskip("tokenizers")
    from nnx import GenerativeNNModel, NNTransformerParams
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe

    torch.manual_seed(0)
    params = NNTransformerParams(
        input_dim=64,
        output_dim=64,
        dropout_prob=0.0,
        vocab_size=64,
        n_layers=1,
        n_heads=2,
        d_model=16,
        ffn_mult=2,
        max_seq_len=16,
    )
    tk = train_bpe(
        files=None,
        texts=["the cat sat on the mat"],
        vocab_size=64,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tokenizer = NNTokenizerParams.of(tokenizer=tk, path=str(tmp_path / "tok.json"))
    m = GenerativeNNModel(
        net_params=params,
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        tokenizer=tokenizer,
    )
    out_dir = tmp_path / "hub"
    m.save_pretrained(str(out_dir))  # pre-fix: RuntimeError (shared memory)

    rt = GenerativeNNModel.from_pretrained(str(out_dir))
    assert rt.tokenizer is not None
    for k in m.net.state_dict():
        assert torch.equal(m.net.state_dict()[k], rt.net.state_dict()[k])
    assert rt.net.lm_head.weight is rt.net.tok_embed.weight, "tie not reassembled"
    a = m.generate(prompt="the", max_new_tokens=6, temperature=0.0)
    b = rt.generate(prompt="the", max_new_tokens=6, temperature=0.0)
    assert a == b


def test_hub_config_persists_and_replays_topology_transforms(tmp_path, monkeypatch):
    from nnx.nn import nn_model as nn_model_module
    from nnx.nn.params.nn_checkpoint import NNCheckpointTransform

    model = _tiny_model()
    transform = NNCheckpointTransform(name="test-transform", options={"value": 1})
    model._topology_transforms = (transform,)
    model.save_pretrained(str(tmp_path))
    applied = []
    monkeypatch.setattr(nn_model_module, "_apply_checkpoint_transform", lambda restored, item: applied.append(item))

    NNModel.from_pretrained(str(tmp_path))

    assert applied == [transform]


def test_remote_hub_load_uses_one_atomic_snapshot(tmp_path, monkeypatch):
    model = _tiny_model()
    model.save_pretrained(str(tmp_path))
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    restored = NNModel._from_pretrained(model_id="owner/repo")

    assert isinstance(restored, NNModel)
    assert len(calls) == 1
