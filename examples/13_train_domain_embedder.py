"""Domain-specific text embedder — contrastive fine-tune + FAISS export.

End-to-end demonstration of ``nnx.embeddings``: build synthetic
``(sentence, paraphrase)`` training pairs, train a tiny text encoder
from scratch via NT-Xent, embed a small corpus, write a FAISS index,
and run a self-similarity query to confirm the index serves correctly.

The encoder here is a deliberately minimal bag-of-words hash embedder —
no HuggingFace Hub download, no network. In production you'd plug in a
``sentence_transformers.SentenceTransformer`` and the same flow holds:

    from sentence_transformers import SentenceTransformer

    backbone = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    nnx.embeddings.train_contrastive(backbone, pairs, n_epochs=3)
    nnx.embeddings.export_to_faiss(backbone, corpus, "index.faiss")

After ``export_to_faiss`` returns, NNx's job is done. Pick up the file
from LangChain / LlamaIndex / Haystack / raw FAISS — chunking,
reranking, and prompt orchestration are framework choices, not NNx
concerns.

Run:
    python examples/13_train_domain_embedder.py
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch import nn

import nnx
from nnx.embeddings import (
    ContrastiveTextDataset,
    embed_texts,
    export_to_faiss,
    train_contrastive,
)


class HashEmbedder(nn.Module):
    """Bag-of-words hash embedder — minimal text encoder for demos.

    Hash each whitespace-split token into a vocab slot, look it up in an
    :class:`nn.Embedding`, mean-pool per text. The embedding table is
    the only trainable parameter — small enough that contrastive
    training visibly moves it in a few epochs.
    """

    def __init__(self, vocab_size: int = 4096, dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        # Small initial magnitudes so the post-normalize vectors aren't
        # uniformly distributed on the unit sphere from the start —
        # contrastive learning needs some signal in the raw cosines.
        with torch.no_grad():
            self.embed.weight.mul_(0.1)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.embed.weight.device
        rows: list[torch.Tensor] = []
        for t in texts:
            ids = [hash(w) % self.vocab_size for w in t.split()] or [0]
            v = self.embed(torch.tensor(ids, dtype=torch.long, device=device)).mean(dim=0)
            rows.append(v)
        return torch.stack(rows, dim=0)


def main():
    nnx.set_seed(0)

    # ---- 1. Synthetic (sentence, paraphrase) pairs.
    # Each pair shares enough vocabulary that a bag-of-words encoder can
    # plausibly drive them together; the demo isn't about paraphrase
    # quality, it's about the pipeline.
    print("=" * 60)
    print("Step 1: build synthetic (sentence, paraphrase) pairs")
    print("=" * 60)
    pairs = [
        ("the cat sat on the mat", "a feline rested on a rug"),
        ("the dog barked loudly", "the canine howled loudly"),
        ("she opened the book", "she opened the novel"),
        ("rain falls in spring", "rain drops in spring"),
        ("the sun is bright", "sunlight is bright"),
        ("he runs fast", "he sprints quickly"),
        ("the river flows east", "the stream flows eastward"),
        ("she sang a song", "she sang a tune"),
        ("birds fly south", "birds migrate south"),
        ("snow falls in winter", "snow drifts in winter"),
    ] * 4  # ×4 → 40 training pairs
    dataset = ContrastiveTextDataset(pairs)
    print(f"{len(dataset)} training pairs\n")

    # ---- 2. Train a domain embedder.
    print("=" * 60)
    print("Step 2: train domain embedder via NT-Xent contrastive loss")
    print("=" * 60)
    backbone = HashEmbedder(vocab_size=4096, dim=64)

    def _mean_pair_cos() -> float:
        anchors = [a for a, _ in pairs]
        positives = [p for _, p in pairs]
        a_emb = embed_texts(backbone, anchors, normalize=True)
        p_emb = embed_texts(backbone, positives, normalize=True)
        return float((a_emb * p_emb).sum(dim=-1).mean())

    pre_sim = _mean_pair_cos()
    train_contrastive(
        backbone,
        dataset,
        n_epochs=5,
        batch_size=8,
        lr=5e-2,  # much higher than the SBERT default — backbone is from-scratch
        temperature=0.1,
        verbose=True,
    )
    post_sim = _mean_pair_cos()
    print(f"\nmean anchor-positive cosine: pre={pre_sim:.4f}, post={post_sim:.4f}")
    print(f"delta: {post_sim - pre_sim:+.4f}  (training pulled the pairs together)\n")

    # ---- 3. Embed a corpus and export to FAISS.
    print("=" * 60)
    print("Step 3: embed corpus, export to FAISS")
    print("=" * 60)
    # Concatenate every sentence + its paraphrase as the searchable
    # corpus. In a real workflow this would be your document set.
    corpus = [s for pair in pairs for s in pair]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    corpus = [c for c in corpus if not (c in seen or seen.add(c))]
    print(f"corpus size: {len(corpus)} unique sentences")

    with tempfile.TemporaryDirectory() as tmp:
        index_path = os.path.join(tmp, "domain.faiss")
        export_to_faiss(backbone, corpus, index_path, index_type="IndexFlatIP")
        print(f"wrote {index_path} ({os.path.getsize(index_path)} bytes)")

        # ---- 4. Reload from disk and query.
        print("\n" + "=" * 60)
        print("Step 4: reload + query from FAISS")
        print("=" * 60)
        import faiss  # type: ignore[import-not-found]

        idx = faiss.read_index(index_path)
        query = "the cat sat on the mat"
        q_emb = embed_texts(backbone, [query], normalize=True).cpu().numpy().astype("float32")
        sims, ids = idx.search(q_emb, 3)
        print(f"query: {query!r}")
        print("top-3 results:")
        for rank, (i, s) in enumerate(zip(ids[0], sims[0], strict=True)):
            print(f"  #{rank + 1}  (cosine={s:.4f}) {corpus[i]!r}")

    print("\n" + "=" * 60)
    print("Done. NNx's job ends at the FAISS index on disk.")
    print("Plug into LangChain / LlamaIndex / Haystack / raw FAISS for retrieval.")
    print("=" * 60)


if __name__ == "__main__":
    main()
