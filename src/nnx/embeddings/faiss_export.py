"""FAISS index export for trained text embedders.

The handoff from NNx training to downstream RAG: take a trained
backbone, embed a corpus, write a FAISS index file. Any FAISS-aware
retriever (LangChain ``FAISS`` vector store, LlamaIndex ``FaissVectorStore``,
Haystack ``FAISSDocumentStore``, raw ``faiss.read_index``) can then
load the file and serve queries.

NNx's job ends here. Chunking, reranking, prompt orchestration, query
expansion — all out of scope. Those are framework-of-the-month and we
explicitly don't pin them.

Public surface:

  - :func:`export_to_faiss` — embed a corpus, build a FAISS index of the
    requested type, save to disk.
  - :func:`export_to_safetensors` — persist the backbone's state_dict
    for HuggingFace Hub / sentence-transformers reload. Falls back to
    ``torch.save`` when the ``safetensors`` package isn't installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from .contrastive_trainer import embed_texts


def _import_faiss():
    """Import faiss lazily with a helpful error message.

    Keeping the import inside the call site means ``import
    nnx.embeddings`` doesn't blow up when the optional FAISS extra
    isn't installed — only the actual export call does.
    """
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "export_to_faiss requires the optional `faiss-cpu` (or `faiss-gpu`) "
            "package. Install via:\n"
            '    pip install "thekaveh-nnx[embeddings]"\n'
            "or directly:\n"
            "    pip install faiss-cpu"
        ) from e
    return faiss


def _build_index(faiss_mod: Any, dim: int, index_type: str) -> Any:
    """Construct a FAISS index of the requested type and dimension.

    Supported strings:

      - ``IndexFlatIP``   — exact inner-product search. Cosine sim
        when the inputs are L2-normalized (the
        :func:`embed_texts` default).
      - ``IndexFlatL2``   — exact L2-distance search.
      - ``IndexHNSWFlat`` — approximate, graph-based. Built with
        ``M=32`` (FAISS's standard default).

    Anything else raises :class:`ValueError`. We don't auto-route to
    every FAISS index family — the call site is explicit and rejecting
    misspellings beats silently building the wrong index.
    """
    if index_type == "IndexFlatIP":
        return faiss_mod.IndexFlatIP(dim)
    if index_type == "IndexFlatL2":
        return faiss_mod.IndexFlatL2(dim)
    if index_type == "IndexHNSWFlat":
        # M=32 is the FAISS docs' standard recall/memory trade-off.
        return faiss_mod.IndexHNSWFlat(dim, 32)
    raise ValueError(
        f"unsupported index_type {index_type!r}. Pass one of 'IndexFlatIP', 'IndexFlatL2', 'IndexHNSWFlat'."
    )


def export_to_faiss(
    backbone: Any,
    corpus: list[str],
    out_path: Union[str, Path],
    *,
    batch_size: int = 64,
    index_type: str = "IndexFlatIP",
    normalize: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> str:
    """Embed ``corpus`` with ``backbone`` and write a FAISS index file.

    The default ``IndexFlatIP`` + ``normalize=True`` combination is
    cosine similarity: L2-normalize the embeddings, then use inner
    product as the score. This is the standard FAISS-cosine recipe
    (FAISS itself doesn't ship a cosine index; the normalize-then-IP
    pattern is canonical).

    The corpus order is preserved in the index — ``index.search``'s
    returned ids are positions into ``corpus``. The caller is
    responsible for keeping a parallel list / DataFrame of original
    document ids or metadata.

    Args:
        backbone: text encoder. Either a
            :class:`sentence_transformers.SentenceTransformer` or any
            ``nn.Module`` whose ``forward(list[str]) -> Tensor[B, D]``.
        corpus: list of strings to embed. Order is the index's id space.
            Empty raises :class:`ValueError` — FAISS rejects 0-length
            adds.
        out_path: destination file path. The parent directory must
            exist. The file is written via FAISS's native
            ``write_index`` (atomic depends on the underlying FS).
        batch_size: forward-pass batch size. Default 64.
        index_type: FAISS index family to build. One of
            ``"IndexFlatIP"`` (default), ``"IndexFlatL2"``,
            ``"IndexHNSWFlat"``.
        normalize: whether to L2-normalize each embedding before
            insertion. ``None`` (the default) auto-selects: True for
            ``IndexFlatIP`` (cosine via IP), False for everything else.
            Pass an explicit bool to override.
        device: target device for the encode pass. ``None`` infers
            from the backbone.

    Returns:
        The string path written. Same value as ``str(out_path)`` —
        returned for call-chain convenience.

    Raises:
        ImportError: if ``faiss`` isn't installed (lazy import; only
            this call requires it).
        ValueError: empty corpus, unknown ``index_type``.
    """
    if not corpus:
        raise ValueError("corpus is empty — FAISS rejects zero-row adds")

    faiss = _import_faiss()

    if normalize is None:
        # Cosine-via-IP is the default reason callers pick IndexFlatIP,
        # so auto-normalize there. L2 / HNSW shouldn't be normalized
        # unless the caller specifically wants unit-sphere geometry.
        normalize = index_type == "IndexFlatIP"

    emb = embed_texts(
        backbone,
        corpus,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
    )
    # FAISS expects float32 contiguous arrays on CPU.
    emb_np = emb.detach().cpu().to(torch.float32).contiguous().numpy()
    dim = emb_np.shape[1]

    index = _build_index(faiss, dim, index_type)
    index.add(emb_np)

    out_path = str(out_path)
    faiss.write_index(index, out_path)
    return out_path


def export_to_safetensors(backbone: Any, out_path: Union[str, Path]) -> str:
    """Persist ``backbone.state_dict()`` to disk for downstream reload.

    Prefers the ``safetensors`` format (canonical for HuggingFace Hub
    artifacts and sentence-transformers ≥3) when the
    :mod:`safetensors` package is importable. Falls back to plain
    :func:`torch.save` when it isn't, so the function still works on
    a vanilla ``pip install thekaveh-nnx`` without the embeddings extra. In
    the fallback case ``out_path`` is written as a pickle blob; the
    caller's reloader needs to use :func:`torch.load`.

    Args:
        backbone: anything with a ``state_dict()`` method.
            Sentence-transformers, raw ``nn.Module``, even a plain
            ``OrderedDict`` of tensors.
        out_path: destination file path. Conventionally suffixed
            ``.safetensors`` for the primary path; ``.pt`` for the
            torch.save fallback. We don't enforce the suffix — that's
            cosmetic.

    Returns:
        The string path written.
    """
    out_path = str(out_path)
    sd = backbone.state_dict() if hasattr(backbone, "state_dict") else backbone

    try:
        from safetensors.torch import save_file  # type: ignore[import-not-found]
    except ImportError:
        # No safetensors — plain torch.save fallback. Pickle blob,
        # so reload with torch.load(weights_only=True) on trusted files.
        torch.save(sd, out_path)
        return out_path

    # safetensors only accepts contiguous tensors. Some state_dict
    # entries (e.g. non-leaf tensors from shared embeddings) might be
    # non-contiguous; force contiguous + detach + CPU to keep the
    # writer happy and the file device-portable.
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            # Skip non-tensor state (rare — schedulers / metadata).
            # The user can save those separately if they need them.
            continue
        cleaned[k] = v.detach().cpu().contiguous()

    save_file(cleaned, out_path)
    return out_path
