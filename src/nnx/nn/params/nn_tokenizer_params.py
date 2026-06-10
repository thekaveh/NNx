"""HF tokenizer wrapper for NNx.

`NNTokenizerParams` wraps a `tokenizers.Tokenizer` (HF's Rust BPE/WPM
tokenizer) and exposes the standard NNx state()/from_state contract:

  * ``state()`` returns ``{"path": "<tokenizer.json>"}`` — the actual
    tokenizer is persisted to disk by the constructor (or by an explicit
    ``save()`` call) and the path is what travels in the run.yaml.
  * ``from_state(state)`` loads the tokenizer from that path via
    ``Tokenizer.from_file``.

Including the tokenizer bytes inline in the yaml would balloon the
run.yaml and break the "one MD5 hash for the config" invariant — the
file-on-disk pointer is the right trade-off for a TinyStories-scale
demo. Production paths can hash the tokenizer file content alongside
the path; out of scope for the TinyStories-scale LM path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    _HAS_TOKENIZERS = True
except ImportError:  # pragma: no cover — exercised in CI without the lm extra
    _HAS_TOKENIZERS = False
    Tokenizer = None  # type: ignore[assignment,misc]


@dataclass(frozen=True, kw_only=True, slots=True)
class NNTokenizerParams:
    """Frozen dataclass holding a tokenizer + its on-disk pointer.

    The dataclass is frozen so it can sit alongside NNTransformerParams /
    NNModelParams in an NNRun without inviting in-place mutation. The
    actual ``tokenizers.Tokenizer`` object is held in a repr=False field
    so it doesn't bloat the str() output.
    """

    path: str
    tokenizer: object = field(repr=False)

    # ---------- factories ----------

    @staticmethod
    def of(tokenizer: object, path: str) -> NNTokenizerParams:
        """Construct from a live Tokenizer instance and persist it to ``path``.

        This is the train-time entry point: train a tokenizer, then call
        ``NNTokenizerParams.of(tk, path="runs/tok.json")`` to wrap it
        with a paired on-disk artifact.
        """
        _require_tokenizers()
        # save() is the official HF Rust path — produces the same JSON
        # blob that ``Tokenizer.from_file`` consumes. Write to a temp
        # name and os.replace into place so an interrupt mid-save can't
        # leave a truncated tokenizer.json that from_state can never
        # load (matches the atomic-write convention of NNRun.save /
        # NNCheckpoint.to_file).
        # Parent dir first: the Rust-side save otherwise fails from a
        # fresh cwd with a cryptic "No such file or directory (os error
        # 2)" for paths like "artifacts/tok.json" (parity with
        # write_gguf / export_ollama_modelfile).
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        tokenizer.save(tmp)
        os.replace(tmp, path)
        return NNTokenizerParams(path=path, tokenizer=tokenizer)

    @staticmethod
    def from_state(state: dict) -> NNTokenizerParams:
        """Load from a state dict produced by :meth:`state`. The single
        required key is ``path``; the tokenizer is reconstructed from
        the file the path points to.

        The path is stored exactly as the caller gave it to :meth:`of` —
        typically cwd-relative — so loading from a different working
        directory requires the same relative layout. That's deliberate:
        storing an absolute path would break run portability across
        machines, which is the more common need."""
        _require_tokenizers()
        path = state["path"]
        tk = Tokenizer.from_file(path)  # type: ignore[union-attr]
        return NNTokenizerParams(path=path, tokenizer=tk)

    # ---------- (de)serialization ----------

    def state(self) -> dict:
        """Return the serializable view — only the path goes into run.yaml."""
        return {"path": self.path}

    # ---------- ergonomic helpers ----------

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())  # type: ignore[union-attr]

    def encode(self, text: str) -> list[int]:
        enc = self.tokenizer.encode(text)  # type: ignore[union-attr]
        return list(enc.ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)  # type: ignore[union-attr]


def _require_tokenizers() -> None:
    if not _HAS_TOKENIZERS:
        raise ImportError(
            "The `tokenizers` package is required for NNTokenizerParams. "
            "Install with `pip install 'tokenizers>=0.20'` or "
            "`pip install -e '.[lm]'` from the NNx checkout."
        )


def train_bpe(
    files: Optional[list[str]] = None,
    *,
    vocab_size: int = 8192,
    texts: Optional[list[str]] = None,
    special_tokens: Optional[list[str]] = None,
    min_frequency: int = 2,
) -> Tokenizer:  # bound to `None` when the optional `lm` extra isn't installed; annotation only evaluated lazily under `from __future__ import annotations`
    """Train a BPE tokenizer on either a list of files or a list of texts.

    Mirrors the HF "quick BPE" recipe — Whitespace pre-tokenizer + BPE
    model + BpeTrainer. Returns the trained Tokenizer instance; the
    caller is responsible for persisting via
    ``NNTokenizerParams.of(tk, path=...)``.

    Args:
        files: paths to plaintext files (one corpus line per file row).
            If None, ``texts`` is consulted instead.
        vocab_size: target vocab. Actual size may be smaller for tiny
            corpora.
        texts: in-memory list of training strings — useful for unit
            tests and the examples without writing a temp file.
        special_tokens: e.g. ``["<pad>", "<bos>", "<eos>"]``. Included
            in the vocab and not split during tokenization.
        min_frequency: minimum pair frequency to merge — higher values
            give smaller, more conservative vocabs.

    Returns:
        Tokenizer: a trained ``tokenizers.Tokenizer`` ready for encode/decode + save.
    """
    _require_tokenizers()
    if files is None and texts is None:
        raise ValueError("Must provide either `files` or `texts`.")

    tk = Tokenizer(BPE(unk_token="<unk>"))  # type: ignore[union-attr]
    tk.pre_tokenizer = Whitespace()  # type: ignore[union-attr]
    trainer = BpeTrainer(  # type: ignore[union-attr]
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>"]),
    )

    if files is not None:
        tk.train(files=list(files), trainer=trainer)
    else:
        tk.train_from_iterator(iter(texts), trainer=trainer, length=len(texts))

    return tk
