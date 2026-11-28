"""A failed torch.save never leaves a truncated file under its final name (the PEFT, net and FAISS saves)."""

from __future__ import annotations

import os
import stat

import pytest
import torch
from torch import nn

from nnx import apply_lora_to, save_lora_weights
from nnx.nn.enum.activations import Activations
from nnx.nn.net.feed_fwd_nn import FeedFwdNN
from nnx.nn.params.nn_params import NNParams


def _truncating_save(monkeypatch):
    """torch.save that writes a truncated payload and then fails, as a full disk would."""

    def interrupted(obj, target, *args, **kwargs):
        if isinstance(target, (str, os.PathLike)):
            with open(target, "wb") as handle:
                handle.write(b"truncated")
        else:
            target.write(b"truncated")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", interrupted)


def _feed_forward() -> FeedFwdNN:
    params = NNParams(input_dim=6, output_dim=3, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)
    return FeedFwdNN(params)


def test_a_failed_net_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "net.pt"
    net = _feed_forward()
    net.to_file(str(path))
    previous = path.read_bytes()
    _truncating_save(monkeypatch)
    with pytest.raises(OSError, match="disk full"):
        net.to_file(str(path))
    assert path.read_bytes() == previous
    assert sorted(os.listdir(tmp_path)) == ["net.pt"]  # no temporary file left beside it


def test_a_failed_lora_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    model = nn.Sequential(nn.Linear(4, 4))
    apply_lora_to(model, "*", r=2, alpha=4.0)
    path = tmp_path / "lora.pt"
    save_lora_weights(model, path)
    previous = path.read_bytes()
    _truncating_save(monkeypatch)
    with pytest.raises(OSError, match="disk full"):
        save_lora_weights(model, path)
    assert path.read_bytes() == previous
    assert sorted(os.listdir(tmp_path)) == ["lora.pt"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_a_fresh_net_file_gets_the_mode_a_plain_write_gives(tmp_path):
    reference = tmp_path / "reference.txt"
    reference.write_text("x", encoding="utf-8")
    expected = stat.S_IMODE(os.stat(reference).st_mode)
    path = tmp_path / "net.pt"
    _feed_forward().to_file(str(path))
    assert stat.S_IMODE(os.stat(path).st_mode) == expected
