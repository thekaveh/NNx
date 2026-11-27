"""An interrupted save_pretrained never leaves a truncated weights file under its final name."""

from __future__ import annotations

import os
import stat

import pytest

from tests.test_hub_mixin import _tiny_model  # skips this module when safetensors is absent


def test_failed_weights_write_leaves_no_partial_or_final_file(tmp_path, monkeypatch):
    import safetensors.torch

    model = _tiny_model()

    def interrupted_save(tensors, path):
        with open(path, "wb") as handle:  # a partial write, then the failure
            handle.write(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(safetensors.torch, "save_file", interrupted_save)
    with pytest.raises(OSError, match="disk full"):
        model.save_pretrained(str(tmp_path))
    names = set(os.listdir(tmp_path))
    assert "model.safetensors" not in names
    assert "model.safetensors.partial" not in names


def test_successful_save_leaves_only_the_final_files(tmp_path):
    _tiny_model().save_pretrained(str(tmp_path))
    names = set(os.listdir(tmp_path))
    assert "model.safetensors" in names
    assert not any(name.endswith(".partial") for name in names)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_exported_files_keep_the_mode_a_plain_write_gives(tmp_path):
    _tiny_model().save_pretrained(str(tmp_path))
    reference = tmp_path / "reference.txt"
    reference.write_text("x", encoding="utf-8")
    expected = stat.S_IMODE(reference.stat().st_mode)
    # config.json is written through the atomic writer, which must not leave mkstemp's 0600.
    # model.safetensors is created by safetensors itself, with its own mode, as before this change.
    assert stat.S_IMODE((tmp_path / "config.json").stat().st_mode) == expected


def test_a_failed_export_state_dict_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    """export_state_dict writes beside its target and renames only a finished file, so a failed export
    keeps the earlier one at that path and leaves no temporary file behind."""
    import torch

    model = _tiny_model()
    target = tmp_path / "weights.pt"
    model.export_state_dict(str(target))
    previous = target.read_bytes()

    def interrupted_save(obj, path, *args, **kwargs):
        with open(path, "wb") as handle:  # a partial write, then the failure
            handle.write(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="disk full"):
        model.export_state_dict(str(target))
    assert target.read_bytes() == previous
    assert os.listdir(tmp_path) == ["weights.pt"]


def test_an_export_state_dict_round_trips_after_the_atomic_write(tmp_path):
    import torch

    model = _tiny_model()
    target = model.export_state_dict(str(tmp_path / "weights.pt"))
    loaded = torch.load(target, weights_only=True)
    for key, value in model.net.state_dict().items():
        assert torch.equal(loaded[key], value), key
