"""Tests for nnx.embeddings.faiss_export.

The headline TDD assertion is :func:`test_export_to_faiss_creates_searchable_index`:
embed a 100-text corpus, save, reload, and verify that searching for
each item's own embedding returns its own index at top-1.

Optional-deps gating: FAISS tests are skipped (not failed) when
``faiss-cpu`` isn't installed; safetensors tests likewise skip when
the package is missing.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

# Note: ``tests/conftest.py`` sets ``KMP_DUPLICATE_LIB_OK=TRUE`` and
# ``OMP_NUM_THREADS=1`` before any nnx/torch import to dodge a
# macOS-specific libomp double-init bug in faiss-cpu's parallel
# search kernel.
from nnx import set_seed
from nnx.embeddings import embed_texts, export_to_faiss, export_to_safetensors
from nnx.embeddings.faiss_export import _build_index

HAS_FAISS = importlib.util.find_spec("faiss") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


# Same tiny encoder as the contrastive trainer tests — no network, no
# HF Hub download. Kept inline (rather than imported across files) so
# the two test modules stay independent.
class _HashEmbedder(nn.Module):
    def __init__(self, vocab_size: int = 4096, dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.embed.weight.device
        out: list[torch.Tensor] = []
        for t in texts:
            ids = [hash(w) % self.vocab_size for w in t.split()] or [0]
            v = self.embed(torch.tensor(ids, dtype=torch.long, device=device)).mean(dim=0)
            out.append(v)
        return torch.stack(out, dim=0)


# -------------------------------------------------------------------------
# Validation / error-path tests — run without faiss installed.
# -------------------------------------------------------------------------


def test_export_rejects_empty_corpus():
    with pytest.raises(ValueError, match="corpus is empty"):
        export_to_faiss(_HashEmbedder(), [], "/dev/null")


# -------------------------------------------------------------------------
# FAISS-required tests.
# -------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
def test_build_index_rejects_unknown_type():
    import faiss

    with pytest.raises(ValueError, match="unsupported index_type"):
        _build_index(faiss, 16, "IndexBogusGarbage")


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
def test_build_index_constructs_known_types():
    import faiss

    for kind in ["IndexFlatIP", "IndexFlatL2", "IndexHNSWFlat"]:
        idx = _build_index(faiss, 16, kind)
        # FAISS indices all expose `d` (dim) and `ntotal`.
        assert idx.d == 16
        assert idx.ntotal == 0


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
def test_export_to_faiss_creates_searchable_index(tmp_path):
    """Headline assertion: a corpus written to a FAISS index, reloaded,
    and queried with each item's own embedding returns that item's
    own corpus index at top-1.

    Implemented with ``IndexFlatIP`` + auto-normalize (= cosine sim),
    which makes self-similarity exactly 1.0 by construction — the
    top-1 result MUST be the query's own row.
    """
    set_seed(0)
    backbone = _HashEmbedder(vocab_size=8192, dim=32)
    corpus = [f"document number {i} with unique words {i * 7 + 13} foo bar" for i in range(100)]

    out = tmp_path / "index.faiss"
    path = export_to_faiss(backbone, corpus, str(out))
    assert path == str(out)
    assert out.exists()

    import faiss

    idx = faiss.read_index(str(out))
    assert idx.ntotal == 100

    # Query: encode each corpus item the SAME way the export pipeline
    # did (normalize=True for IndexFlatIP) and check top-1 is itself.
    queries = embed_texts(backbone, corpus, normalize=True).cpu().numpy().astype("float32")
    _, ids = idx.search(queries, 1)
    for i in range(100):
        assert ids[i][0] == i, f"top-1 for corpus[{i}] was {ids[i][0]}, expected {i}"


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
def test_export_to_faiss_query_one_item(tmp_path):
    """Smoke test for the README-style "search for text 5" pattern from
    the plan."""
    set_seed(0)
    backbone = _HashEmbedder(vocab_size=8192, dim=32)
    # Each text gets a deliberately distinctive token so the hash
    # collisions don't mask the top-1.
    corpus = [f"document text content alpha_{i} beta_{i} gamma_{i}" for i in range(100)]

    out = tmp_path / "index.faiss"
    export_to_faiss(backbone, corpus, str(out))

    import faiss

    idx = faiss.read_index(str(out))
    q_emb = embed_texts(backbone, [corpus[5]], normalize=True).cpu().numpy().astype("float32")
    _, ids = idx.search(q_emb, 1)
    assert ids[0][0] == 5


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
def test_export_to_faiss_l2_index_preserves_corpus_size(tmp_path):
    """The IndexFlatL2 path doesn't auto-normalize (would distort
    distances); confirm it still round-trips."""
    backbone = _HashEmbedder(vocab_size=4096, dim=16)
    corpus = [f"text {i}" for i in range(20)]
    out = tmp_path / "l2.faiss"
    export_to_faiss(backbone, corpus, str(out), index_type="IndexFlatL2")

    import faiss

    idx = faiss.read_index(str(out))
    assert idx.ntotal == 20


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
def test_export_to_faiss_explicit_normalize_override(tmp_path):
    """Default normalize=None auto-selects True for IndexFlatIP and
    False for the others. Passing an explicit bool MUST take precedence.

    We verify by computing the IndexFlatIP search scores for two
    exports of the same corpus — one with normalize=True (the default
    for IP) and one with normalize=False (the explicit override). The
    score of v vs v itself is 1.0 under cosine but ||v||² under the
    raw inner product, so the diagonal scores MUST differ between the
    two builds. If the override were ignored, both would be ≈1.0."""
    backbone = _HashEmbedder(vocab_size=4096, dim=16)
    corpus = [f"text alpha_{i} beta_{i}" for i in range(10)]

    out_norm = tmp_path / "ip-norm.faiss"
    out_raw = tmp_path / "ip-raw.faiss"
    export_to_faiss(backbone, corpus, str(out_norm), index_type="IndexFlatIP", normalize=True)
    export_to_faiss(backbone, corpus, str(out_raw), index_type="IndexFlatIP", normalize=False)

    import faiss

    norm_idx = faiss.read_index(str(out_norm))
    raw_idx = faiss.read_index(str(out_raw))

    # Query with each embedding form on its matching index — diagonals
    # should look like ones (normalized) vs the squared L2 norms (raw).
    q_norm = embed_texts(backbone, corpus, normalize=True).cpu().numpy().astype("float32")
    q_raw = embed_texts(backbone, corpus, normalize=False).cpu().numpy().astype("float32")

    norm_diag, _ = norm_idx.search(q_norm, 1)
    raw_diag, _ = raw_idx.search(q_raw, 1)

    # Normalized self-similarity ≈ 1.0; raw self-IP ≈ ||v||² and the
    # hash embedder's outputs are NOT unit norm so the raw diagonal
    # must differ from the normalized one by more than a numerical
    # rounding margin.
    import numpy as np

    assert np.allclose(norm_diag.ravel(), 1.0, atol=1e-4)
    assert not np.allclose(raw_diag.ravel(), 1.0, atol=1e-2), (
        "normalize=False override appears to have been ignored — raw IP diagonal is suspiciously close to 1.0"
    )


# -------------------------------------------------------------------------
# export_to_safetensors
# -------------------------------------------------------------------------


def test_export_to_safetensors_writes_a_file(tmp_path):
    """Whether safetensors or torch.save backs the write, the file
    must exist and be non-empty."""
    backbone = _HashEmbedder(vocab_size=128, dim=8)
    out = tmp_path / "weights.bin"
    path = export_to_safetensors(backbone, str(out))
    assert path == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_to_safetensors_roundtrip_torch_save_fallback(tmp_path):
    """The torch.save fallback (used when safetensors isn't installed)
    must round-trip the state_dict. We verify by loading with
    torch.load(weights_only=True) — the safe loader, since the dict
    is pure tensors."""
    backbone = _HashEmbedder(vocab_size=128, dim=8)
    out = tmp_path / "weights.pt"

    if HAS_SAFETENSORS:
        # We can't reach the fallback path without mocking the import;
        # patch the import to force the fallback for this test.
        import nnx.embeddings.faiss_export as fe

        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def _raise_import(name, *a, **k):
            if name.startswith("safetensors"):
                raise ImportError("forced for test")
            return original_import(name, *a, **k)

        import builtins as _builtins

        _builtins.__import__ = _raise_import
        try:
            fe.export_to_safetensors(backbone, str(out))
        finally:
            _builtins.__import__ = original_import
    else:
        export_to_safetensors(backbone, str(out))

    assert out.exists()
    loaded = torch.load(str(out), weights_only=True)
    # Compare against the live state_dict.
    expected = backbone.state_dict()
    assert set(loaded.keys()) == set(expected.keys())
    for k in expected:
        assert torch.equal(loaded[k], expected[k])


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
def test_export_to_safetensors_uses_safetensors_when_available(tmp_path):
    """When safetensors IS installed, the writer must use it — verify
    by reading the file back via safetensors.torch.load_file and
    confirming the tensors match."""
    backbone = _HashEmbedder(vocab_size=64, dim=8)
    out = tmp_path / "weights.safetensors"
    export_to_safetensors(backbone, str(out))

    from safetensors.torch import load_file

    loaded = load_file(str(out))
    expected = backbone.state_dict()
    assert set(loaded.keys()) == set(expected.keys())
    for k in expected:
        assert torch.equal(loaded[k], expected[k])
