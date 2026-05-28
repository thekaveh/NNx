"""Map NNx TransformerNN parameter names to llama.cpp / GGUF naming.

llama.cpp expects a very specific set of tensor names, e.g.::

    token_embd.weight
    blk.{i}.attn_norm.weight
    blk.{i}.attn_q.weight
    blk.{i}.attn_k.weight
    blk.{i}.attn_v.weight
    blk.{i}.attn_output.weight
    blk.{i}.ffn_norm.weight
    blk.{i}.ffn_gate.weight     # = SwiGLU w1
    blk.{i}.ffn_up.weight       # = SwiGLU w3
    blk.{i}.ffn_down.weight     # = SwiGLU w2
    output_norm.weight
    output.weight               # LM head; omitted when embeddings are tied

NNx's ``TransformerNN`` exposes these as a fused QKV projection
(``blocks.{i}.attn.w_qkv.weight`` of shape ``(3 * d_model, d_model)``)
and a SwiGLU triple ``w1`` (gate), ``w3`` (up), ``w2`` (down). The
splitter below unpacks the fused QKV into three tensors so llama.cpp's
expected naming holds without changing the NNx forward path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    # Type-only import — same rationale as in writer.py.
    from nnx.nn.net.transformer_nn import TransformerNN


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach + cpu + contiguous + numpy. GGUFWriter accepts numpy
    arrays; tensors with grad / non-CPU storage / non-contiguous strides
    silently break that path."""
    return t.detach().cpu().contiguous().numpy()


def _weight(module) -> torch.Tensor:
    """Grab ``module.weight`` as a ``torch.Tensor``. The cast helper
    exists purely so pyright doesn't widen ``nn.Module`` attribute
    access to ``Tensor | Module`` at every call site."""
    w = module.weight
    assert isinstance(w, torch.Tensor)
    return w


def map_tensors(net: TransformerNN) -> dict[str, np.ndarray]:
    """Walk a ``TransformerNN`` and return ``{gguf_name: numpy_array}``.

    The caller (``write_gguf``) then iterates this dict and calls
    ``GGUFWriter.add_tensor`` for each entry. Splitting the iteration
    here (rather than inlining it into the writer) keeps the naming
    convention testable in isolation — see ``test_gguf_writer.py``.

    Args:
        net: A ``TransformerNN`` instance.

    Returns:
        Dict ``gguf_name -> numpy.ndarray``. Q/K/V are emitted as three
        separate tensors even though the NNx side stores them fused.
        When ``net.params.tie_embeddings`` is True, ``output.weight``
        is omitted (llama.cpp re-uses ``token_embd.weight`` for tied
        models).
    """
    out: dict[str, np.ndarray] = {}

    # Token embedding — always present, always named ``token_embd.weight``.
    out["token_embd.weight"] = _to_numpy(_weight(net.tok_embed))

    d_model = net.params.d_model

    for i, block in enumerate(net.blocks):
        # Pre-attention norm (RMSNorm).
        out[f"blk.{i}.attn_norm.weight"] = _to_numpy(_weight(block.norm1))

        # Split the fused QKV projection. Layout in NNx:
        #   w_qkv.weight has shape (3 * d_model, d_model) and the
        #   forward path does w_qkv(x).chunk(3, dim=-1) on the OUTPUT,
        #   i.e. the leading dim is the concatenated [Q | K | V] block.
        # That maps cleanly to a row-wise slice of the weight tensor.
        w_qkv = _weight(block.attn.w_qkv)  # (3*d, d)
        out[f"blk.{i}.attn_q.weight"] = _to_numpy(w_qkv[0:d_model, :])
        out[f"blk.{i}.attn_k.weight"] = _to_numpy(w_qkv[d_model : 2 * d_model, :])
        out[f"blk.{i}.attn_v.weight"] = _to_numpy(w_qkv[2 * d_model : 3 * d_model, :])

        # Attention output projection.
        out[f"blk.{i}.attn_output.weight"] = _to_numpy(_weight(block.attn.w_o))

        # Pre-FFN norm.
        out[f"blk.{i}.ffn_norm.weight"] = _to_numpy(_weight(block.norm2))

        # SwiGLU: NNx names them w1 (gate) / w2 (down) / w3 (up);
        # llama.cpp calls the same matrices ffn_gate / ffn_down / ffn_up.
        out[f"blk.{i}.ffn_gate.weight"] = _to_numpy(_weight(block.ffn.w1))
        out[f"blk.{i}.ffn_down.weight"] = _to_numpy(_weight(block.ffn.w2))
        out[f"blk.{i}.ffn_up.weight"] = _to_numpy(_weight(block.ffn.w3))

    # Final output norm — llama.cpp calls this ``output_norm.weight``.
    out["output_norm.weight"] = _to_numpy(_weight(net.norm_out))

    # LM head. When embeddings are tied (NNx default), llama.cpp re-uses
    # ``token_embd.weight`` and we deliberately omit ``output.weight`` so
    # the file size halves and the reader doesn't see two copies of the
    # same matrix.
    if not net.params.tie_embeddings:
        out["output.weight"] = _to_numpy(_weight(net.lm_head))

    return out
