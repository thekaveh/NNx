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

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ..._reparameterization import UNNAMED, legacy_reparameterizations

if TYPE_CHECKING:
    # Type-only import — same rationale as in writer.py.
    from nnx.nn.net.transformer_nn import TransformerNN


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach + cpu + contiguous + float32 numpy. GGUFWriter accepts numpy
    arrays; tensors with grad / non-CPU storage / non-contiguous strides
    silently break that path, and numpy has no bfloat16. Upcasting to
    float32 is exact for fp16 / bf16 weights."""
    return t.detach().cpu().float().contiguous().numpy()


def _weight(module) -> torch.Tensor:
    """Grab ``module.weight`` as a ``torch.Tensor``. The cast helper
    exists purely so pyright doesn't widen ``nn.Module`` attribute
    access to ``Tensor | Module`` at every call site."""
    w = module.weight
    assert isinstance(w, torch.Tensor)
    return w


def _refuse_adapters(net: torch.nn.Module) -> None:
    """GGUF stores plain weights: a PEFT-wrapped projection has no weight of
    its own, and a prefix tuner's learned prefix has no GGUF slot, so
    exporting either would fail or silently drop the tuned part."""
    from nnx.peft import IA3Linear, LoRALinear  # DoRALinear is a LoRALinear

    adapted = [
        name
        for name, module in net.named_modules()
        if isinstance(module, (LoRALinear, IA3Linear)) or hasattr(module, "_nnx_prefix_tuner")
    ]
    if adapted:
        raise TypeError(
            f"GGUF export needs plain weights, but {', '.join(adapted[:5])}"
            f"{' …' if len(adapted) > 5 else ''} carry a PEFT adapter (LoRA / DoRA / IA3 / prefix tuning). "
            "Export the base model, or fold the adapter into plain nn.Linear weights first."
        )
    # A factorized projection (low_rank: a Sequential of two Linears) has no
    # single weight for its GGUF slot either.
    projections = ("w_qkv", "w_o", "w1", "w2", "w3", "lm_head")
    factorized = [
        name
        for name, module in net.named_modules()
        if name.rpartition(".")[2] in projections and not isinstance(module, torch.nn.Linear)
    ]
    if factorized:
        raise TypeError(
            f"GGUF export needs plain nn.Linear projections, but {', '.join(factorized[:5])} "
            "are factorized or wrapped. Export a model with plain projections."
        )


def _listed(names: list[str], what: str) -> str:
    """``names`` (the first five, and how many in all) and the verb ``carry``."""
    if len(names) == 1:
        return f"{names[0]} carries {what}"
    if len(names) > 5:
        return f"{', '.join(names[:5])} … ({len(names)} in all) carry {what}"
    return f"{', '.join(names[:-1])} and {names[-1]} carry {what}"


def _refuse_reparameterized(net: torch.nn.Module) -> None:
    """An unbaked prune mask and a legacy ``weight_norm`` / ``spectral_norm``
    recompute ``.weight`` in a forward pre-hook, so the cached tensor export
    reads is stale after a ``load_state_dict`` or an optimizer step, and a
    masked tied ``lm_head`` has no GGUF slot of its own (``token_embd``
    holds the unmasked embedding). Every module is checked, by hook type
    (``nnx._reparameterization``), never by parameter names. The
    ``torch.nn.utils.parametrize`` forms compute ``.weight`` on every read,
    so they export."""
    masked: list[str] = []
    normed: list[str] = []
    for path, module in net.named_modules():
        masks, norms = legacy_reparameterizations(module)
        for names, found in ((masks, masked), (norms, normed)):
            for name in names:
                # The tensor's qualified name, or the module's for a hook without one.
                found.append((path or "<root>") if name == UNNAMED else f"{path}.{name}" if path else name)
    if not masked and not normed:
        return
    problems: list[str] = []
    fixes: list[str] = []
    if masked:
        problems.append(_listed(masked, "an unbaked prune mask"))
        fixes.append("bake the masks (torch.nn.utils.prune.remove, or magnitude_prune(bake=True))")
    if normed:
        problems.append(_listed(normed, "a legacy weight_norm / spectral_norm hook"))
        fixes.append(
            "remove the legacy norms (torch.nn.utils.remove_weight_norm / remove_spectral_norm) or use their "
            "torch.nn.utils.parametrizations forms, which export reads"
        )
    tied_head = ""
    if "lm_head.weight" in masked and getattr(getattr(net, "params", None), "tie_embeddings", False):
        # Named even past the first five: GGUF has no slot for a tied head's mask.
        tied_head = (
            " The masked lm_head.weight is tied: its one GGUF slot, token_embd.weight, holds the unmasked embedding."
        )
    raise TypeError(
        f"GGUF export reads each module's stored weight, but {' and '.join(problems)}. A prune mask or legacy "
        f"norm recomputes the weight only when its module is called, so the file would not hold the weights the "
        f"forward uses.{tied_head} Before exporting, {'; and '.join(fixes)}."
    )


def map_tensors(net: TransformerNN) -> dict[str, np.ndarray]:
    """Walk a ``TransformerNN`` and return ``{gguf_name: numpy_array}``.

    The caller (``write_gguf``) then iterates this dict and calls
    ``GGUFWriter.add_tensor`` for each entry. Splitting the iteration
    here (rather than inlining it into the writer) keeps the naming
    convention testable in isolation.

    Args:
        net: A ``TransformerNN`` instance.

    Returns:
        Dict ``gguf_name -> numpy.ndarray``. Q/K/V are emitted as three
        separate tensors even though the NNx side stores them fused.
        When ``net.params.tie_embeddings`` is True, ``output.weight``
        is omitted (llama.cpp re-uses ``token_embd.weight`` for tied
        models).

    Raises:
        TypeError: when the network carries a PEFT adapter, a projection
            is not a plain ``nn.Linear``, or any module carries an unbaked
            prune mask or a legacy ``weight_norm`` / ``spectral_norm``
            hook (see :func:`nnx.interop.gguf.write_gguf`).
    """
    _refuse_adapters(net)
    # Nor does a weight that a forward pre-hook recomputes.
    _refuse_reparameterized(net)
    out: dict[str, np.ndarray] = {}

    # Token embedding — always present, always named ``token_embd.weight``.
    out["token_embd.weight"] = _to_numpy(_weight(net.tok_embed))

    d_model = net.params.d_model

    for i, raw_block in enumerate(net.blocks):
        block = cast(Any, raw_block)
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
