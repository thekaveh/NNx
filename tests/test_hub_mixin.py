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
import os

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
    model = _tiny_model()

    class ExtraStateNet(type(model.net)):  # same topology, plus non-tensor extra state
        def get_extra_state(self):
            return {"labels": ["class-a"]}

        def set_extra_state(self, state):
            self.extra_state = state

    model.net.__class__ = ExtraStateNet

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


def test_hub_config_persists_and_replays_topology_transforms(tmp_path):
    """config.json records the model's transforms, and from_pretrained
    replays them for real — a LoRA wrapper is rebuilt — before loading."""
    from nnx.peft import LoRALinear
    from nnx.transforms import TransformRecipe, lora

    model = _tiny_model()
    first = next(name for name, module in model.net.named_modules() if type(module) is torch.nn.Linear)
    TransformRecipe([lora(first, r=2, alpha=4.0)]).materialize(model)
    model.save_pretrained(str(tmp_path))
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["transforms"] == [transform.state() for transform in model._topology_transforms]

    rebuilt = NNModel.from_pretrained(str(tmp_path))

    assert rebuilt._topology_transforms == model._topology_transforms
    assert isinstance(rebuilt.net.get_submodule(first), LoRALinear)
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    model.net.eval(), rebuilt.net.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt.net(x), model.net(x))


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


# ---------- FEAT-006: registered and runtime-only modules ----------


class _HubEncoder(torch.nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(4, width), torch.nn.Tanh(), torch.nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def test_hub_round_trips_registered_modules_through_the_factory_registry(tmp_path):
    from nnx import MissingModelFactoryError, ModelSpec, register_model_factory, unregister_model_factory

    register_model_factory("tests.hub_encoder", 1, lambda config: _HubEncoder(**config))
    try:
        spec = ModelSpec("tests.hub_encoder", 1, {"width": 6}, seed=2)
        model = NNModel(params=NNModelParams(net=spec, loss=Losses.CROSS_ENTROPY))
        with torch.no_grad():
            model.net.body[0].weight.add_(1.0)  # trained-looking weights, not the seeded init
        model.save_pretrained(str(tmp_path))
        with open(tmp_path / "config.json") as f:
            cfg = json.load(f)
        assert "net_params" not in cfg and cfg["params"]["net"] == spec.state()
        loaded = NNModel.from_pretrained(str(tmp_path))
        assert loaded.params.net == spec and loaded.net_params is None and isinstance(loaded.net, _HubEncoder)
        for key, value in model.net.state_dict().items():
            assert torch.equal(loaded.net.state_dict()[key], value), key
    finally:
        unregister_model_factory("tests.hub_encoder", 1)
    with pytest.raises(MissingModelFactoryError, match="not registered"):
        NNModel.from_pretrained(str(tmp_path))  # unknown factory: rejected before anything is built


def test_hub_rejects_runtime_modules_before_writing_files(tmp_path):
    from nnx import MissingModelFactoryError

    model = NNModel(module=_HubEncoder(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    target = tmp_path / "artifact"
    with pytest.raises(MissingModelFactoryError, match="runtime-only"):
        model.save_pretrained(str(target))
    assert not target.exists()  # no directory, config or weights were written
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "config.json").write_text("{}")
    with pytest.raises(MissingModelFactoryError):
        model.save_pretrained(str(existing))
    assert (existing / "config.json").read_text() == "{}"  # an existing artifact is untouched


def test_hub_save_and_load_replays_a_transformation_recipe(tmp_path):
    """FEAT-016: a LoRA-then-low-rank recipe rebuilds from config.json —
    same ordered keys, trainability and predictions, no SVD rerun."""
    from unittest import mock

    from nnx.transforms import TransformRecipe, lora, low_rank

    model = _tiny_model()
    linears = [name for name, module in model.net.named_modules() if type(module) is torch.nn.Linear]
    recipe = TransformRecipe(
        [lora(linears[0], r=2, alpha=4.0), low_rank(linears[1], rank=2)], materialization="in_place"
    )
    recipe.materialize(model)
    with torch.no_grad():
        model.net.get_submodule(linears[0]).lora_B.normal_(0, 0.1, generator=torch.Generator().manual_seed(1))
    model.save_pretrained(str(tmp_path))
    config = json.loads((tmp_path / "config.json").read_text())
    assert [item["name"] for item in config["transforms"]] == ["lora", "low_rank"]
    with mock.patch("torch.linalg.svd", side_effect=AssertionError("SVD rerun during reload")):
        rebuilt = NNModel.from_pretrained(str(tmp_path))
    assert rebuilt._topology_transforms == model._topology_transforms
    saved, loaded = model.net.state_dict(), rebuilt.net.state_dict()
    assert list(saved) == list(loaded)
    assert [(n, p.requires_grad) for n, p in model.net.named_parameters()] == [
        (n, p.requires_grad) for n, p in rebuilt.net.named_parameters()
    ]
    model.net.eval(), rebuilt.net.eval()
    x = torch.randn(4, model.net_params.input_dim)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt.net(x), model.net(x), rtol=1e-5, atol=1e-6)


def test_hub_load_refuses_an_unknown_recipe_operation_before_loading_tensors(tmp_path):
    """FEAT-016: local ``from_pretrained`` rejects an unknown recorded
    operation or version, naming its index, before any tensor is read; a
    transformed artifact whose config records no recipe says so."""
    from unittest import mock

    from nnx.transforms import TransformRecipe, lora, low_rank

    model = _tiny_model()
    linears = [name for name, module in model.net.named_modules() if type(module) is torch.nn.Linear]
    TransformRecipe([lora(linears[0], r=2, alpha=4.0), low_rank(linears[1], rank=2)]).materialize(model)
    model.save_pretrained(str(tmp_path))
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    cases = (
        (
            {**config["transforms"][1], "version": 7},
            r"topology transform 1 \('low_rank' version 7\).*unknown version 7",
        ),
        ({"name": "prune", "version": 1, "options": {}}, r"topology transform 1 \('prune' version 1\).*unsupported"),
    )
    for bad, pattern in cases:
        config_path.write_text(json.dumps({**config, "transforms": [config["transforms"][0], bad]}))
        with mock.patch("safetensors.torch.load_file", side_effect=AssertionError("tensors loaded")):
            with pytest.raises(ValueError, match=pattern):
                NNModel.from_pretrained(str(tmp_path))
    config_path.write_text(json.dumps({key: value for key, value in config.items() if key != "transforms"}))
    with pytest.raises(ValueError, match="record no transformation recipe"):
        NNModel.from_pretrained(str(tmp_path))


def test_a_non_strict_hub_load_of_unrecorded_recipe_weights_still_loads_partially(tmp_path):
    """FEAT-016: ``strict=False`` keeps its documented partial load — the
    no-recipe refusal applies to strict loads only."""
    from nnx.transforms import TransformRecipe, lora

    model = _tiny_model()
    first = next(name for name, module in model.net.named_modules() if type(module) is torch.nn.Linear)
    TransformRecipe([lora(first, r=2, alpha=4.0)]).materialize(model)
    model.save_pretrained(str(tmp_path))
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({key: value for key, value in config.items() if key != "transforms"}))
    loaded = NNModel.from_pretrained(str(tmp_path), strict=False)
    assert loaded._topology_transforms == ()
    for name, tensor in loaded.net.state_dict().items():
        if name in model.net.state_dict():
            assert torch.equal(tensor, model.net.state_dict()[name])


def test_save_pretrained_refuses_a_recipe_model_its_recipe_cannot_rebuild(tmp_path):
    """FEAT-016: unrecorded surgery after a recipe is refused before any
    file is written, instead of failing when the artifact is loaded."""
    from nnx.surgery import low_rank_factorize
    from nnx.transforms import TransformRecipe, lora

    model = _tiny_model()
    linears = [name for name, module in model.net.named_modules() if type(module) is torch.nn.Linear]
    TransformRecipe([lora(linears[0], r=2, alpha=4.0)]).materialize(model)
    parent, _, attr = linears[1].rpartition(".")
    owner = model.net.get_submodule(parent) if parent else model.net
    setattr(owner, attr, low_rank_factorize(model.net.get_submodule(linears[1]), rank=2))  # unrecorded
    target = tmp_path / "artifact"
    with pytest.raises(ValueError, match="save_pretrained refused before writing anything"):
        model.save_pretrained(str(target))
    assert not target.exists()


@pytest.mark.parametrize("surgery", ["lora", "ia3", "widen"])
def test_save_pretrained_refuses_unrecorded_surgery_without_a_recipe(tmp_path, surgery):
    """A model with no recipe whose topology no longer matches its params (a
    LoRA / IA3 wrap, a widen) is refused before anything is written —
    the artifact would only fail when loaded."""
    from nnx.peft import apply_ia3_to, apply_lora_to
    from nnx.surgery import widen

    model = _tiny_model()
    if surgery == "lora":
        apply_lora_to(model.net, "*")
    elif surgery == "ia3":
        apply_ia3_to(model.net, "*")
    else:
        first = next(name for name, module in model.net.named_modules() if type(module) is torch.nn.Linear)
        model.net = widen(model.net, layer_name=first, new_width=32)
    target = tmp_path / "artifact"
    with pytest.raises(ValueError, match="save_pretrained refused before writing anything"):
        model.save_pretrained(str(target))
    assert not target.exists()


def test_hub_loading_re_resolves_the_precision_policy_on_the_destination_device(tmp_path):
    """FEAT-028: the saved policy is re-resolved on ``map_location`` — never
    taken from saved metadata — and ``precision=`` replaces it."""
    import json as _json

    import numpy as np

    from nnx import PrecisionPolicy, PrecisionUnsupportedError

    bf16 = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, precision=PrecisionPolicy("bf16")
        ),
    )
    bf16.save_pretrained(str(tmp_path))
    restored = NNModel.from_pretrained(str(tmp_path), map_location="cpu")
    assert restored.resolved_precision.effective == "bf16"
    assert restored.predict(torch.randn(2, 4)).logits.dtype == np.float32  # bf16 converts to a safe CPU dtype
    # A config saved for CUDA fp16 cannot run on the CPU destination ...
    config_path = tmp_path / "config.json"
    config = _json.loads(config_path.read_text())
    config["params"]["precision"] = {"mode": "fp16"}
    config_path.write_text(_json.dumps(config))
    with pytest.raises(PrecisionUnsupportedError, match="CUDA only"):
        NNModel.from_pretrained(str(tmp_path), map_location="cpu")
    # ... unless the caller replaces the policy.
    replaced = NNModel.from_pretrained(str(tmp_path), map_location="cpu", precision=PrecisionPolicy("fp32"))
    assert replaced.resolved_precision.effective == "fp32"


@pytest.mark.parametrize(
    "filename", ["../tokenizer.json", "/abs/tokenizer.json", "sub/tokenizer.json", "a\\\\b.json", "..", ""]
)
def test_a_downloaded_config_cannot_point_the_tokenizer_outside_the_artifact(filename, tmp_path):
    from nnx.nn.generative_nn_model import GenerativeNNModel

    with pytest.raises(ValueError, match="plain file name"):
        GenerativeNNModel._hub_reconstruction_kwargs({"tokenizer": {"filename": filename}}, tmp_path)


@pytest.mark.parametrize(
    "key, override",
    [
        ("params", "object"),
        ("net_params", "object"),
        ("tokenizer", "object"),
        ("params", {"loss": "mean_squared_error"}),
        ("net_params", {"input_dim": 99}),
        ("transforms", [{"name": "lora"}]),
    ],
)
def test_hub_from_pretrained_refuses_an_explicit_override_instead_of_dropping_it(tmp_path, key, override):
    m = _tiny_model()
    m.save_pretrained(str(tmp_path))
    if override == "object":
        override = {"params": m.params, "net_params": m.net_params, "tokenizer": object()}[key]
    with pytest.raises(TypeError, match=f"from_pretrained got {key}"):
        NNModel.from_pretrained(str(tmp_path), **{key: override})


# ---------- #509: unreplayable saves are refused; top-level tensors are excludable ----------


def _trained_with_own_transform(tmp_path, monkeypatch, loop: str = "model") -> NNModel:
    """The ticket's Example 1: a callback declares its own train-end
    transform through the public hook, so no NNx loader can replay it."""
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import Callback, NNCheckpointTransform, NNOptimParams, NNTrainerParams, NNTrainParams, Trainer
    from nnx.objectives import supervised_objective

    class Convert(Callback):
        def checkpoint_transforms(self):
            return (NNCheckpointTransform(name="my-convert"),)

        def on_train_end(self, ctx):
            old = ctx.model.net.layers[0]
            ctx.model.net.layers[0] = torch.nn.Linear(old.in_features, old.out_features)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = _tiny_model()
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 4, generator=generator), torch.randint(0, 2, (8,), generator=generator)),
        batch_size=4,
    )
    optim = NNOptimParams.builder().sgd(max_lr=0.1).build()
    if loop == "model":
        model.train(NNTrainParams(n_epochs=1, train_loader=loader, optim=optim), callbacks=[Convert()])
    else:
        params = NNTrainerParams.builder().n_epochs(1).train_loader(loader).optimizer("default", optim).build()
        Trainer(model).train(params=params, objective=supervised_objective(), callbacks=[Convert()])
    assert [transform.name for transform in model._topology_transforms] == ["my-convert"]
    return model


@pytest.mark.parametrize("loop", ["model", "trainer"])
def test_save_pretrained_refuses_a_callbacks_own_transform_before_writing(tmp_path, monkeypatch, loop):
    """#509: from_pretrained could not replay a callback's own transform, so
    save_pretrained refuses it before the directory exists, as export_bundle
    does."""
    model = _trained_with_own_transform(tmp_path, monkeypatch, loop)
    target = tmp_path / "artifact"
    with pytest.raises(
        ValueError,
        match=r"save_pretrained refused before writing anything: topology transform 0 \('my-convert' version 1\)",
    ):
        model.save_pretrained(str(target))
    assert not target.exists()


def _mocked_hf_api(monkeypatch):
    from unittest import mock

    api = mock.MagicMock()
    api.create_repo.return_value.repo_id = "owner/repo"
    api.upload_folder.return_value = "https://huggingface.co/owner/repo/commit/0"
    monkeypatch.setattr("huggingface_hub.hub_mixin.HfApi", mock.MagicMock(return_value=api))
    return api


@pytest.mark.parametrize("case", ["own-transform", "unrecorded-lora", "runtime-module"])
def test_push_to_hub_refuses_before_creating_the_repository(tmp_path, monkeypatch, case):
    """#509: the mixin creates the repository before it saves; every
    save-side refusal comes first, so nothing is created or uploaded."""
    from nnx import MissingModelFactoryError
    from nnx.peft import apply_lora_to

    if case == "own-transform":
        model = _trained_with_own_transform(tmp_path, monkeypatch)
        error, pattern = ValueError, r"push_to_hub refused before creating the repository.*'my-convert' version 1"
    elif case == "unrecorded-lora":
        model = _tiny_model()
        apply_lora_to(model.net, "*")
        error, pattern = ValueError, "push_to_hub refused before creating the repository.*unrecorded surgery"
    else:
        model = NNModel(module=_HubEncoder(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
        error, pattern = MissingModelFactoryError, "push_to_hub needs a rebuildable module"
    api = _mocked_hf_api(monkeypatch)
    with pytest.raises(error, match=pattern):
        model.push_to_hub("owner/repo")
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()


def test_push_to_hub_still_uploads_a_portable_model(monkeypatch):
    api = _mocked_hf_api(monkeypatch)
    uploaded = []
    api.upload_folder.side_effect = lambda **kwargs: uploaded.append(sorted(os.listdir(kwargs["folder_path"])))
    _tiny_model().push_to_hub("owner/repo")
    api.create_repo.assert_called_once()
    assert uploaded == [["README.md", "config.json", "model.safetensors"]]


def test_a_qat_converted_model_still_saves_and_round_trips(tmp_path, monkeypatch):
    """#509: a torchao QAT conversion replays from data, so it is not refused."""
    pytest.importorskip("torchao")
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import NNOptimParams, NNTrainParams, QATLifecycleCallback, qat_train_step_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    torch.manual_seed(0)
    model = NNModel(  # hidden widths divide the default int4 groupsize (32)
        net_params=NNParams(
            input_dim=32, output_dim=3, hidden_dims=[64], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    loader = DataLoader(TensorDataset(torch.randn(16, 32), torch.randint(0, 3, (16,))), batch_size=8)
    model.train(
        NNTrainParams(n_epochs=1, train_loader=loader, optim=NNOptimParams.builder().sgd(max_lr=0.1).build()),
        train_step_fn=qat_train_step_factory(),
        callbacks=[QATLifecycleCallback()],
    )
    assert [transform.name for transform in model._topology_transforms] == ["torchao_qat"]
    model.save_pretrained(str(tmp_path / "hub"))
    rebuilt = NNModel.from_pretrained(str(tmp_path / "hub"))
    assert rebuilt._topology_transforms == model._topology_transforms
    probe = torch.randn(4, 32, generator=torch.Generator().manual_seed(1))
    model.net.eval(), rebuilt.net.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt.net(probe), model.net(probe))


def test_an_unknown_transform_names_both_causes(tmp_path):
    """#509: an unknown transform is either a newer NNx's operation or a
    callback's own, which no NNx version replays."""
    model = _tiny_model()
    model.save_pretrained(str(tmp_path))
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, "transforms": [{"name": "my-convert", "version": 1, "options": {}}]}))
    with pytest.raises(ValueError) as caught:
        NNModel.from_pretrained(str(tmp_path))
    message = str(caught.value)
    assert "topology transform 0 ('my-convert' version 1)" in message
    assert "newer NNx" in message and "callback's own transform" in message


def _with_top_level(kind: str) -> tuple[NNModel, str]:
    model = _tiny_model()
    if kind == "parameter":
        model.net.register_parameter("logit_scale", torch.nn.Parameter(torch.full((1,), 2.0)))
        return model, "logit_scale"
    if kind == "buffer":
        model.net.register_buffer("class_prior", torch.tensor([0.25, 0.75]))
        return model, "class_prior"
    model.net.add_module("aux_head", torch.nn.Linear(2, 2))  # a top-level submodule, as before #509
    return model, "aux_head"


@pytest.mark.parametrize("kind", ["parameter", "buffer", "submodule"])
def test_a_top_level_tensor_or_submodule_is_excluded_strictly(tmp_path, kind):
    """#509: exclude_submodules names a top-level tensor as it names a
    submodule; every other key still loads strictly."""
    from safetensors.torch import load_file, save_file

    model, name = _with_top_level(kind)
    saved = tmp_path / "hub"
    model.save_pretrained(str(saved))
    with pytest.raises(RuntimeError) as caught:
        NNModel.from_pretrained(str(saved))
    message = str(caught.value)
    assert f"({name}): NNModel.from_pretrained(model_id, exclude_submodules=('{name}',))" in message
    rebuilt = NNModel.from_pretrained(str(saved), exclude_submodules=(name,))
    base = {key: value for key, value in model.net.state_dict().items() if key.split(".", 1)[0] != name}
    assert rebuilt.net.state_dict().keys() == base.keys()
    for key, value in base.items():
        assert torch.equal(rebuilt.net.state_dict()[key], value), key

    weights = saved / "model.safetensors"
    state = load_file(str(weights))
    del state["layers.0.weight"]
    save_file(state, str(weights))
    with pytest.raises(RuntimeError, match=r'Missing key\(s\) in state_dict: "layers.0.weight"'):
        NNModel.from_pretrained(str(saved), exclude_submodules=(name,))
