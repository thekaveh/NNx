"""Tests for nnx.embeddings.contrastive_trainer.

Two layers of coverage:

  - Unit tests on :class:`ContrastiveTextDataset`, :func:`pair_collate`,
    and the validation surface of :func:`train_contrastive` /
    :func:`text_contrastive_train_step_factory`. These run with no
    optional deps — only torch.

  - End-to-end "training reduces anchor-positive distance" check using
    a tiny hash-embedding encoder built inline (no sentence-transformers
    download, fully hermetic). The skip-gracefully guard on the
    optional ``sentence-transformers`` integration is in
    :func:`test_train_contrastive_accepts_sentence_transformers_when_installed`.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from nnx import set_seed
from nnx.embeddings import (
    ContrastiveTextDataset,
    embed_texts,
    pair_collate,
    text_contrastive_train_step_factory,
    train_contrastive,
)

# `_is_sentence_transformer` is a private helper intentionally tested
# directly (it's the routing decision for embed_texts' backbone dispatch);
# pulling it from the submodule path is the explicit "I know this is
# internal" affordance.
from nnx.embeddings.contrastive_trainer import _is_sentence_transformer

# -------------------------------------------------------------------------
# ContrastiveTextDataset
# -------------------------------------------------------------------------


def test_dataset_rejects_empty_pairs():
    with pytest.raises(ValueError, match="at least one pair"):
        ContrastiveTextDataset([])


def test_dataset_rejects_non_tuple_entries():
    with pytest.raises(ValueError, match="not a 2-tuple"):
        ContrastiveTextDataset(["a string, not a pair"])  # type: ignore[list-item]


def test_dataset_rejects_non_string_pair_entries():
    with pytest.raises(ValueError, match="non-string"):
        ContrastiveTextDataset([("hi", 42)])  # type: ignore[list-item]


def test_dataset_len_and_getitem():
    pairs = [("cat", "feline"), ("dog", "canine")]
    ds = ContrastiveTextDataset(pairs)
    assert len(ds) == 2
    assert ds[0] == ("cat", "feline")
    assert ds[1] == ("dog", "canine")


def test_pair_collate_splits_lists():
    batch = [("a", "x"), ("b", "y"), ("c", "z")]
    anchors, positives = pair_collate(batch)
    assert anchors == ["a", "b", "c"]
    assert positives == ["x", "y", "z"]


# -------------------------------------------------------------------------
# Tiny hermetic text encoder — no network, no HF Hub dep.
# -------------------------------------------------------------------------


class _HashEmbedder(nn.Module):
    """Minimal text encoder for tests.

    Hashes each (word) token into a fixed-size vocab via ``hash``,
    looks it up in an :class:`nn.Embedding`, and mean-pools per text.
    Trainable: the embedding table itself. Just enough capacity for
    the test to demonstrate that NT-Xent actually pulls paired vectors
    together in cosine space.
    """

    def __init__(self, vocab_size: int = 1024, dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        # Init weights at small magnitude — large random init gives
        # post-norm vectors that look almost uniformly distributed on
        # the sphere, drowning the contrastive signal in two epochs.
        with torch.no_grad():
            self.embed.weight.mul_(0.1)

    def _tokenize(self, text: str) -> list[int]:
        return [hash(w) % self.vocab_size for w in text.split()] or [0]

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.embed.weight.device
        out: list[torch.Tensor] = []
        for t in texts:
            ids = torch.tensor(self._tokenize(t), dtype=torch.long, device=device)
            v = self.embed(ids).mean(dim=0)
            out.append(v)
        return torch.stack(out, dim=0)


def _mean_pair_cosine(backbone: nn.Module, pairs: list[tuple[str, str]]) -> float:
    """Average cosine similarity between anchor and its positive across
    all pairs. Higher = pairs are more "aligned" in embedding space."""
    anchors = [a for a, _ in pairs]
    positives = [p for _, p in pairs]
    a_emb = embed_texts(backbone, anchors, normalize=True)
    p_emb = embed_texts(backbone, positives, normalize=True)
    return float((a_emb * p_emb).sum(dim=-1).mean())


# -------------------------------------------------------------------------
# train_contrastive — high-level training loop
# -------------------------------------------------------------------------


def test_train_contrastive_rejects_bad_inputs():
    backbone = _HashEmbedder()
    pairs = [("a", "b")]
    with pytest.raises(ValueError, match="n_epochs"):
        train_contrastive(backbone, pairs, n_epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        train_contrastive(backbone, pairs, batch_size=0)
    with pytest.raises(ValueError, match="temperature"):
        train_contrastive(backbone, pairs, temperature=0.0)


def test_train_contrastive_rejects_all_frozen_backbone():
    """If the user froze everything, the optimizer would have nothing to
    do — surface that loudly instead of running a no-op loop."""
    backbone = _HashEmbedder()
    for p in backbone.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError, match="no trainable parameters"):
        train_contrastive(backbone, [("a", "b")] * 4, n_epochs=1, batch_size=2)


def test_train_contrastive_returns_backbone():
    """Confirms the in-place mutation + return contract."""
    set_seed(0)
    backbone = _HashEmbedder()
    pairs = [("cat sat on mat", "feline rested on rug")] * 8
    returned = train_contrastive(backbone, pairs, n_epochs=1, batch_size=4)
    assert returned is backbone


def test_contrastive_training_reduces_pair_distance():
    """The headline TDD assertion: train a tiny embedder on synthetic
    (sentence, paraphrase) pairs for 3 epochs and confirm the mean
    anchor-positive cosine similarity rises by ≥0.05 over baseline."""
    set_seed(0)

    # 16 distinct (anchor, positive) pairs — paraphrase-style. Each
    # anchor's "paraphrase" shares enough vocabulary with the original
    # that a hash-bag encoder CAN drive them together with NT-Xent —
    # this is not a measure of paraphrase quality, only of whether
    # the contrastive objective is wired correctly.
    pairs = [
        ("the cat sat on the mat", "a feline rested on a rug"),
        ("the dog barked loudly", "the canine howled loudly"),
        ("she opened the book", "she opened a novel"),
        ("rain falls in spring", "rain drops in spring"),
        ("the sun is bright", "sunlight is bright"),
        ("he runs fast", "he sprints quickly"),
        ("the river flows east", "the stream flows eastward"),
        ("she sang a song", "she sang a tune"),
        ("the mountain is tall", "the peak is tall"),
        ("birds fly south", "birds migrate south"),
        ("trees grow tall", "trees grow upward"),
        ("the lake is calm", "the pond is calm"),
        ("she reads quietly", "she reads silently"),
        ("the fire crackles", "the flame crackles"),
        ("snow falls in winter", "snow drifts in winter"),
        ("waves crash on shore", "waves break on shore"),
    ]
    # Repeat to give NT-Xent a 32-sample epoch.
    pairs = pairs * 2

    backbone = _HashEmbedder(vocab_size=2048, dim=32)
    initial = _mean_pair_cosine(backbone, pairs)

    train_contrastive(
        backbone,
        pairs,
        n_epochs=3,
        batch_size=8,
        lr=5e-2,  # ↑ vs the default 2e-5 — the test backbone is from-scratch
        temperature=0.1,
    )

    final = _mean_pair_cosine(backbone, pairs)
    assert final > initial + 0.05, (
        f"contrastive training did not pull pairs together: "
        f"initial cosine = {initial:.4f}, final = {final:.4f} "
        f"(expected delta ≥ 0.05)"
    )


# -------------------------------------------------------------------------
# text_contrastive_train_step_factory — the low-level NNModel-driven path
# -------------------------------------------------------------------------


def test_factory_rejects_bad_temperature():
    with pytest.raises(ValueError, match="temperature"):
        text_contrastive_train_step_factory(temperature=0.0)


def test_factory_step_rejects_bad_batch_shape():
    """The step expects (anchors: list[str], positives: list[str]).
    A plain tensor batch shouldn't silently train on garbage."""
    set_seed(0)
    backbone = _HashEmbedder()
    step = text_contrastive_train_step_factory(temperature=0.1)

    # Build a minimal NNModel-style mock for the step.
    class _Mock:
        def __init__(self, net):
            self.net = net
            self.device = torch.device("cpu")

    from nnx.nn.nn_model import TrainStepContext

    optimizer = torch.optim.SGD(backbone.parameters(), lr=1e-3)
    ctx = TrainStepContext(
        model=_Mock(backbone),  # type: ignore[arg-type]
        batch=torch.randn(4, 8),
        optimizer=optimizer,
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
    )
    with pytest.raises(ValueError, match="anchors"):
        step(ctx)


def test_factory_step_runs_and_moves_weights():
    """End-to-end: build a context with the right batch shape, run the
    step, confirm weights moved and a loss came back."""
    set_seed(0)
    backbone = _HashEmbedder()
    step = text_contrastive_train_step_factory(temperature=0.1)

    class _Mock:
        def __init__(self, net):
            self.net = net
            self.device = torch.device("cpu")

    from nnx.nn.nn_model import TrainStepContext

    optimizer = torch.optim.SGD(backbone.parameters(), lr=1e-1)
    pre = backbone.embed.weight.detach().clone()
    anchors = ["the cat sat", "the dog ran", "she sang", "rain fell"]
    positives = ["a feline rested", "a canine ran", "she sang again", "rain dropped"]
    ctx = TrainStepContext(
        model=_Mock(backbone),  # type: ignore[arg-type]
        batch=(anchors, positives),
        optimizer=optimizer,
        scaler=None,
        grad_clip_norm=None,
        extra_metrics=None,
        accumulate_grad_batches=1,
        batch_idx=0,
        epoch_idx=0,
    )
    edp = step(ctx)
    assert edp.loss is not None and edp.loss > 0
    assert edp.error == edp.loss
    assert not torch.equal(pre, backbone.embed.weight.detach()), (
        "text contrastive step ran but embedding weights did not change"
    )


# -------------------------------------------------------------------------
# embed_texts
# -------------------------------------------------------------------------


def test_embed_texts_empty_returns_2d():
    backbone = _HashEmbedder(dim=16)
    out = embed_texts(backbone, [])
    assert out.dim() == 2
    assert out.shape[0] == 0


def test_embed_texts_normalize_true_unit_norm():
    backbone = _HashEmbedder(dim=16)
    emb = embed_texts(backbone, ["a", "b", "c"], normalize=True)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_embed_texts_normalize_false_unnormalized():
    """Without normalization, norms should NOT all be 1.0 (extremely
    unlikely for random inputs)."""
    backbone = _HashEmbedder(dim=16)
    emb = embed_texts(backbone, ["hello world", "different words"], normalize=False)
    norms = emb.norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_embed_texts_batches_match_unbatched():
    """The result must not depend on batch_size — same texts give the
    same embeddings whether processed in one chunk or many."""
    set_seed(0)
    backbone = _HashEmbedder(dim=16)
    texts = [f"text number {i}" for i in range(10)]
    e_one = embed_texts(backbone, texts, batch_size=10)
    e_many = embed_texts(backbone, texts, batch_size=3)
    assert torch.allclose(e_one, e_many, atol=1e-6)


# -------------------------------------------------------------------------
# Sentence-transformers integration — skip-gracefully when unavailable.
# -------------------------------------------------------------------------


HAS_SBERT = importlib.util.find_spec("sentence_transformers") is not None


def test_is_sentence_transformer_negatives():
    """Plain modules and non-modules MUST NOT be flagged as SBERT
    so the routing in :func:`_encode` doesn't accidentally invoke
    ``.preprocess`` on a custom encoder."""
    assert _is_sentence_transformer(_HashEmbedder()) is False
    assert _is_sentence_transformer("not a module") is False
    assert _is_sentence_transformer(None) is False


@pytest.mark.skipif(not HAS_SBERT, reason="sentence-transformers not installed")
def test_is_sentence_transformer_detects_sbert_subclass():
    """A torch.nn.Module that quacks like SBERT (has preprocess) MUST be
    flagged — this is the routing trigger in :func:`_encode`."""

    class _FakeSBERT(nn.Module):
        def preprocess(self, texts):
            return {}

    assert _is_sentence_transformer(_FakeSBERT()) is True
