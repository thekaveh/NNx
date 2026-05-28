"""Tests for NNTokenizerParams — the HF-tokenizer-wrapping params class.

Covers:
  * train_bpe helper produces a Tokenizer with the expected vocab size.
  * NNTokenizerParams serializes via tokenizer.save / from_file and
    round-trips a known encoding.
  * state() shape (single `path` key) per the SP-4 spec.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402

from nnx.nn.params.nn_tokenizer_params import NNTokenizerParams, train_bpe  # noqa: E402


def _tiny_corpus() -> list[str]:
    return [
        "hello world",
        "the quick brown fox",
        "world peace and love",
        "fox jumps over the lazy dog",
        "the cat sat on the mat",
    ]


def test_train_bpe_produces_tokenizer_with_target_vocab(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("\n".join(_tiny_corpus()))
    tk = train_bpe(files=[str(corpus_file)], vocab_size=64, special_tokens=["<pad>", "<bos>", "<eos>"])
    assert isinstance(tk, Tokenizer)
    # With such a small corpus we may not hit the full 64; just check bounds.
    assert tk.get_vocab_size() <= 64
    assert tk.get_vocab_size() >= 3  # at least the 3 special tokens


def test_tokenizer_params_state_returns_path(tmp_path):
    tokenizer_path = tmp_path / "tok.json"
    tk = train_bpe(files=None, vocab_size=64, texts=_tiny_corpus(), special_tokens=["<pad>", "<bos>", "<eos>"])
    params = NNTokenizerParams.of(tokenizer=tk, path=str(tokenizer_path))
    s = params.state()
    assert "path" in s
    assert s["path"] == str(tokenizer_path)
    assert os.path.isfile(tokenizer_path), "save side-effect must persist the tokenizer to disk"


def test_tokenizer_params_round_trips_encoding(tmp_path):
    """State → from_state → encode must produce the same ids as the
    original tokenizer. This is the load path that NNRun.load takes."""
    tokenizer_path = tmp_path / "tok.json"
    tk = train_bpe(files=None, vocab_size=64, texts=_tiny_corpus(), special_tokens=["<pad>", "<bos>", "<eos>"])
    obj = NNTokenizerParams.of(tokenizer=tk, path=str(tokenizer_path))
    rt = NNTokenizerParams.from_state(obj.state())
    text = "the quick fox"
    ids_orig = obj.encode(text)
    ids_rt = rt.encode(text)
    assert ids_orig == ids_rt
    # And decode round-trips too.
    assert obj.decode(ids_orig) == rt.decode(ids_rt)


def test_tokenizer_params_vocab_size_property(tmp_path):
    tokenizer_path = tmp_path / "tok.json"
    tk = train_bpe(files=None, vocab_size=128, texts=_tiny_corpus(), special_tokens=["<pad>", "<bos>", "<eos>"])
    obj = NNTokenizerParams.of(tokenizer=tk, path=str(tokenizer_path))
    assert obj.vocab_size == tk.get_vocab_size()
    assert obj.vocab_size > 0


def test_tokenizer_params_encode_decode_roundtrip(tmp_path):
    tokenizer_path = tmp_path / "tok.json"
    tk = train_bpe(files=None, vocab_size=128, texts=_tiny_corpus(), special_tokens=["<pad>", "<bos>", "<eos>"])
    obj = NNTokenizerParams.of(tokenizer=tk, path=str(tokenizer_path))
    # Use words that appear multiple times in the corpus so BPE actually
    # learns merges for them. "the" and "world" both repeat.
    text = "the world"
    ids = obj.encode(text)
    assert len(ids) > 0
    decoded = obj.decode(ids)
    # BPE decoding may not preserve whitespace exactly; check the
    # well-merged words round-trip.
    assert "the" in decoded
    assert "world" in decoded


def test_tokenizer_params_from_state_loads_from_disk(tmp_path):
    tokenizer_path = tmp_path / "tok.json"
    tk = train_bpe(files=None, vocab_size=64, texts=_tiny_corpus())
    # Write directly via the underlying tokenizer.save and load via state-only.
    tk.save(str(tokenizer_path))
    loaded = NNTokenizerParams.from_state({"path": str(tokenizer_path)})
    assert loaded.vocab_size == tk.get_vocab_size()
