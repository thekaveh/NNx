# Embeddings — Contrastive Trainer + FAISS Export

`nnx.embeddings` is NNx's one RAG-adjacent surface: train a domain-specific text embedder via the existing SimCLR / NT-Xent contrastive machinery, then export the trained model to a FAISS index for downstream retrieval.

NNx does NOT host the RAG stack. Chunking, reranking, prompt orchestration, vector-DB clients — those are inference-time concerns and live in your retriever-framework of choice (LangChain, LlamaIndex, Haystack, raw FAISS). NNx's responsibility ends at the FAISS index on disk.

## 1. When to use this

Use `nnx.embeddings` when:

- You have domain-specific text and the off-the-shelf SBERT / OpenAI embedding API is mediocre at retrieving in-domain results — e.g., legal contracts, medical notes, code search, narrow product catalogs.
- You can produce `(anchor, positive)` pairs cheaply — query/document mining, in-batch positives, paraphrase generation, click logs.
- You want a portable artifact (a FAISS index file) that any RAG framework can load. NNx doesn't lock you into LangChain / LlamaIndex / Haystack.

If you don't have pairs, or your off-the-shelf embedding is already good enough, skip this — it's not worth the training overhead.

## 2. Install

```bash
pip install "thekaveh-nnx[embeddings]"
```

Pulls in `faiss-cpu` and `sentence-transformers`. Both are optional at import time: `import nnx.embeddings` works even without them — the `ImportError` is deferred to the call site that actually needs each one.

## 3. Quickstart

```python
from sentence_transformers import SentenceTransformer

import nnx
from nnx.embeddings import (
    ContrastiveTextDataset,
    train_contrastive,
    export_to_faiss,
)

# 1. Domain pairs — (anchor, positive). In-batch negatives are free
#    via NT-Xent; you don't need to mine hard negatives explicitly.
pairs = [
    ("how do I reset my password", "I forgot my password — how to reset"),
    ("invoice past due notice", "your bill is overdue"),
    # ... thousands more, mined from your domain
]

# 2. Start from a pretrained SBERT backbone (recommended) — or any
#    nn.Module whose forward(list[str]) -> Tensor[B, D] matches.
backbone = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 3. Contrastive fine-tune — 3 epochs is usually plenty.
train_contrastive(
    backbone,
    pairs,
    n_epochs=3,
    batch_size=32,
    lr=2e-5,          # SBERT-canonical fine-tune LR
    temperature=0.05, # text encoders work in high-dim cosine space;
                      # sharper than SimCLR's image default (0.5)
)

# 4. Embed your corpus and write a FAISS index. The default
#    IndexFlatIP + normalize-then-IP gives you cosine similarity.
corpus = ["doc 1 text", "doc 2 text", ...]  # your retrieval corpus
export_to_faiss(backbone, corpus, "domain.faiss")
```

`domain.faiss` is now a standard FAISS index file. Hand it off to any framework:

```python
# LangChain — wrap the raw index; LCFAISS.load_local expects a *folder*
# with index.faiss + index.pkl (docstore), not a bare index file.
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS as LCFAISS
idx = faiss.read_index("domain.faiss")
vs = LCFAISS(embedding_function=..., index=idx,
             docstore=InMemoryDocstore({}), index_to_docstore_id={})

# LlamaIndex
from llama_index.vector_stores.faiss import FaissVectorStore
vs = FaissVectorStore.from_persist_path("domain.faiss")

# Raw FAISS
import faiss
idx = faiss.read_index("domain.faiss")
```

NNx's job ends here. Pick the retrieval framework that fits your stack — they all consume the same FAISS index file.

## 4. API

### 4.1. `ContrastiveTextDataset`

```python
ContrastiveTextDataset(pairs: list[tuple[str, str]])
```

Wraps `(anchor, positive)` string pairs as a `torch.utils.data.Dataset`. Each `__getitem__` returns a 2-tuple of strings; pair with `pair_collate` (see §4.2) if you build your own `DataLoader` instead of going through `train_contrastive`.

Raises `ValueError` on empty input or non-string entries — NT-Xent needs at least one pair to form a batch.

### 4.2. `pair_collate`

```python
pair_collate(batch: Iterable[tuple[str, str]]) -> tuple[list[str], list[str]]
```

`DataLoader` `collate_fn` for `ContrastiveTextDataset`. Splits a batch of `(anchor, positive)` 2-tuples into two parallel lists: `(anchors: list[str], positives: list[str])`. `train_contrastive` (§4.3) uses it as the default `collate_fn`; pass it explicitly when you wire your own `DataLoader`:

```python
from nnx.embeddings import ContrastiveTextDataset, pair_collate
from torch.utils.data import DataLoader

loader = DataLoader(
    ContrastiveTextDataset(pairs),
    batch_size=16,
    shuffle=True,
    collate_fn=pair_collate,
)
```

### 4.3. `train_contrastive`

```python
train_contrastive(
    backbone,                       # SentenceTransformer or any nn.Module(list[str]) -> Tensor[B, D]
    dataset,                        # ContrastiveTextDataset or raw list of (anchor, positive)
    *,
    n_epochs=3,
    batch_size=16,
    lr=2e-5,
    temperature=0.05,
    device=None,                    # None infers from backbone
    shuffle=True,
    grad_clip_norm=1.0,             # global L2 grad-clip; None to disable
    weight_decay=0.0,
    optimizer_cls=torch.optim.AdamW,
    verbose=False,
) -> backbone                       # in-place + returned for chaining
```

Runs `n_epochs` of NT-Xent contrastive updates over the dataset and returns the (in-place-mutated) backbone. Parameters with `requires_grad=False` are excluded from the optimizer, so freezing layers via `nnx.freeze(backbone, ...)` composes cleanly.

For the full callback / checkpoint / `runs/<id>/` machinery, drop down to `text_contrastive_train_step_factory` (see §4.5).

### 4.4. `embed_texts`

```python
embed_texts(
    backbone,
    texts: list[str],
    *,
    batch_size=64,
    device=None,
    normalize=True,
) -> torch.Tensor[N, D]
```

Inference helper. Runs in `torch.no_grad()` + `eval()` mode. The same call that `export_to_faiss` uses internally — useful for ad-hoc similarity probes or building your own non-FAISS retrieval.

### 4.5. `text_contrastive_train_step_factory`

```python
text_contrastive_train_step_factory(*, temperature=0.5) -> TrainStepFn
```

Lower-level: returns an `nnx.TrainStepFn` suitable for `NNModel.train(train_step_fn=...)`. The training loader must yield `(anchors: list[str], positives: list[str])` batches — pair `ContrastiveTextDataset` with `pair_collate` when building the `DataLoader`. Use this when you want NNx's standard run-tracking, callbacks, `runs/<id>/` persistence, etc., wrapped around the contrastive step.

### 4.6. `export_to_faiss`

```python
export_to_faiss(
    backbone,
    corpus: list[str],
    out_path,
    *,
    batch_size=64,
    index_type="IndexFlatIP",       # or "IndexFlatL2" / "IndexHNSWFlat"
    normalize=None,                 # None → auto: True for IP, False for L2/HNSW
    device=None,
) -> str                            # the path written
```

Embeds `corpus` and writes a FAISS index file. The corpus order becomes the index's id space — search results return positions into `corpus`.

Default `IndexFlatIP` + auto-normalize = cosine similarity. FAISS doesn't ship a native cosine index; the standard pattern is to normalize embeddings to unit norm and use inner product as the score, which equals cosine on the unit sphere.

For approximate search at scale, switch `index_type="IndexHNSWFlat"`. The wrapper builds with FAISS's standard `M=32` recall/memory trade-off; for finer-grained tuning, build the index manually with `faiss.IndexHNSWFlat(dim, M)` and `index.hnsw.efConstruction = ...`.

### 4.7. `export_to_safetensors`

```python
export_to_safetensors(backbone, out_path) -> str
```

Persists the backbone's `state_dict()` to disk for HuggingFace Hub / sentence-transformers reload. Uses the `safetensors` format when the `safetensors` package is importable (it's a transitive dep of `sentence-transformers≥3`); falls back to plain `torch.save` otherwise.

## 5. How it composes with the rest of NNx

`nnx.embeddings` is built entirely on top of the existing public surface:

- **NT-Xent loss** — `nnx.nt_xent_loss` (the SimCLR objective). The trainer reuses this for in-batch contrastive learning.
- **`finalize_step` helper** — the `text_contrastive_train_step_factory` path routes through `nnx._step_helpers.finalize_step` for NaN-guard, grad-clip, and the standard paradigm-factory contract (no AMP, no gradient accumulation).
- **Freezing / LoRA** — both compose cleanly. `nnx.freeze(backbone, "encoder.layer.0.*")` excludes those parameters from the optimizer; `nnx.apply_lora_to(backbone, "*.dense")` then adds trainable LoRA residuals on the still-trainable subset. The optimizer in `train_contrastive` collects only `requires_grad=True` parameters, so both layered composition works.

## 6. What this is NOT

- Not a LangChain / LlamaIndex / Haystack wrapper. Those are downstream retrieval frameworks; pick the one your stack uses.
- Not a chunker. Document chunking is a preprocessing concern — solve it in the retrieval framework (every major one ships their own).
- Not a reranker. Cross-encoder rerankers are a separate training problem with their own loss (pointwise / pairwise). The contrastive trainer here is biencoder-only.
- Not a vector-database client. The FAISS index is a file on disk; if you need Pinecone / Weaviate / Milvus / pgvector, embed via `embed_texts` and `INSERT` the vectors yourself.
- Not a chunker, query expander, prompt orchestrator, or evaluation harness. NNx's responsibility ends at the FAISS index file.

That deliberate scope-cut is the point. NNx ships the one training-time piece (a domain embedder you fully control) and stays out of the inference-orchestration arms race.
