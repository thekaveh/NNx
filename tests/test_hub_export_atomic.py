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
