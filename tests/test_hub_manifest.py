"""The hub manifest: a save lists its files with their digests, a load verifies them, and a save removes only the tokenizer NNx wrote."""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch

from nnx import NNModel
from tests.test_hub_mixin import _resign, _tiny_model  # skips this module when the hub extra is absent

MANIFEST = "nnx_hub_manifest.json"


def _lm_with_tokenizer(tmp_path):
    pytest.importorskip("tokenizers")
    from nnx import Devices, GenerativeNNModel, Losses, Nets, NNModelParams, NNTransformerParams
    from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe

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
    return GenerativeNNModel(
        net_params=params,
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        tokenizer=tokenizer,
    )


def _listed(directory) -> dict:
    return json.loads((directory / MANIFEST).read_text(encoding="utf-8"))["files"]


def test_a_saved_artifact_lists_its_weights_and_config_with_digests_but_not_the_readme(tmp_path):
    _tiny_model().save_pretrained(str(tmp_path))
    listed = _listed(tmp_path)
    assert set(listed) == {"model.safetensors", "config.json"}
    for name, entry in listed.items():
        assert entry["size"] == (tmp_path / name).stat().st_size
        assert len(entry["sha256"]) == 64


def test_a_saved_tokenizer_is_listed_beside_the_weights(tmp_path):
    _lm_with_tokenizer(tmp_path).save_pretrained(str(tmp_path / "hub"))
    assert set(_listed(tmp_path / "hub")) == {"model.safetensors", "config.json", "tokenizer.json"}


def test_a_normal_round_trip_verifies_the_listed_files_without_a_warning(tmp_path):
    model = _tiny_model()
    model.save_pretrained(str(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = NNModel.from_pretrained(str(tmp_path))
    assert not [w for w in caught if MANIFEST in str(w.message)]
    for key, value in model.net.state_dict().items():
        assert torch.equal(restored.net.state_dict()[key], value), key


def test_a_corrupted_weights_file_is_refused_on_load(tmp_path):
    _tiny_model().save_pretrained(str(tmp_path))
    weights = tmp_path / "model.safetensors"
    corrupted = bytearray(weights.read_bytes())
    corrupted[-1] ^= 0xFF  # same size, still a parseable safetensors file: only the digest sees it
    weights.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match=r"model\.safetensors.*does not match its digest"):
        NNModel.from_pretrained(str(tmp_path))


@pytest.mark.parametrize("name", ["model.safetensors", "config.json"])
def test_a_listed_file_that_is_missing_is_refused(tmp_path, name):
    _tiny_model().save_pretrained(str(tmp_path))
    (tmp_path / name).unlink()
    with pytest.raises(ValueError, match=rf"{name} is listed in nnx_hub_manifest\.json but is missing"):
        NNModel.from_pretrained(str(tmp_path))


def test_a_manifest_that_does_not_list_the_weights_is_refused(tmp_path):
    _tiny_model().save_pretrained(str(tmp_path))
    manifest = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    del manifest["files"]["model.safetensors"]
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not list model.safetensors"):
        NNModel.from_pretrained(str(tmp_path))


@pytest.mark.parametrize("name", ["../config.json", "sub/model.safetensors", ".."])
def test_a_manifest_entry_naming_a_path_outside_the_artifact_is_refused(tmp_path, name):
    _tiny_model().save_pretrained(str(tmp_path))
    manifest = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    manifest["files"][name] = manifest["files"]["config.json"]
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a file of the artifact"):
        NNModel.from_pretrained(str(tmp_path))


def test_a_directory_saved_before_manifests_loads_with_a_warning(tmp_path):
    _tiny_model().save_pretrained(str(tmp_path))
    (tmp_path / MANIFEST).unlink()
    with pytest.warns(UserWarning, match="nnx_hub_manifest.json"):
        restored = NNModel.from_pretrained(str(tmp_path))
    assert isinstance(restored, NNModel)


def test_a_config_naming_a_tokenizer_the_manifest_does_not_list_is_refused(tmp_path):
    _lm_with_tokenizer(tmp_path).save_pretrained(str(tmp_path / "hub"))
    manifest = json.loads((tmp_path / "hub" / MANIFEST).read_text(encoding="utf-8"))
    del manifest["files"]["tokenizer.json"]
    (tmp_path / "hub" / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    from nnx import GenerativeNNModel

    with pytest.raises(ValueError, match="declares the tokenizer 'tokenizer.json', which nnx_hub_manifest.json"):
        GenerativeNNModel.from_pretrained(str(tmp_path / "hub"))


def test_resaving_without_a_tokenizer_removes_the_tokenizer_nnx_wrote(tmp_path):
    out = tmp_path / "hub"
    _lm_with_tokenizer(tmp_path).save_pretrained(str(out))
    assert (out / "tokenizer.json").exists()
    _tiny_model().save_pretrained(str(out))
    assert not (out / "tokenizer.json").exists()
    assert set(_listed(out)) == {"model.safetensors", "config.json"}


def test_resaving_without_a_tokenizer_leaves_a_user_written_tokenizer_alone(tmp_path):
    out = tmp_path / "hub"
    _tiny_model().save_pretrained(str(out))
    (out / "tokenizer.json").write_text('{"user": true}', encoding="utf-8")
    _tiny_model().save_pretrained(str(out))
    assert (out / "tokenizer.json").read_text(encoding="utf-8") == '{"user": true}'


def test_a_directory_without_a_manifest_keeps_its_tokenizer(tmp_path):
    out = tmp_path / "hub"
    _tiny_model().save_pretrained(str(out))
    (out / MANIFEST).unlink()
    (out / "tokenizer.json").write_text("{}", encoding="utf-8")
    _tiny_model().save_pretrained(str(out))
    assert os.path.exists(out / "tokenizer.json")


def test_a_re_listed_artifact_still_verifies(tmp_path):
    """The test helper re-lists on purpose; a re-listed artifact is an artifact a load accepts."""
    _tiny_model().save_pretrained(str(tmp_path))
    _resign(tmp_path)
    assert isinstance(NNModel.from_pretrained(str(tmp_path)), NNModel)
