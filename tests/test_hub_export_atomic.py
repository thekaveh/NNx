"""An interrupted save_pretrained never leaves a truncated weights file under its final name."""

from __future__ import annotations

import os

import pytest
import safetensors.torch

from tests.test_hub_mixin import _tiny_model


def test_failed_weights_write_leaves_no_partial_or_final_file(tmp_path, monkeypatch):
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
