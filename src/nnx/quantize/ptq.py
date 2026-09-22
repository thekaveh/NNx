"""Post-training INT8 weight-only quantization via ``torchao``.

``torchao`` replaced the deprecated ``torch.ao.quantization`` (removed
in PyTorch 2.10). :class:`Int8WeightOnlyConfig` rewrites
:class:`torch.nn.Linear` weights in place to symmetric int8 per-channel;
activations stay FP32. Compatible with the vision (``FeedFwdNN``) and
GNN (``GraphConvNN`` / ``GraphSageNN`` / ``GraphAttNN``) networks NNx
ships — any module exposing ``nn.Linear`` submodules is a valid target.

The transform is **post-training** — no retraining, no calibration set,
no labeled data needed. The single-call API:

    m_q = quantize_int8(model)

deep-copies ``model.net``, replaces the FP32 weights on every ``nn.Linear``
with int8-backed :class:`AffineQuantizedTensor` parameters, and returns
a fresh sibling of ``model``'s own class (an :class:`NNModel`, or a
subclass such as :class:`GenerativeNNModel` with its tokenizer and
``generate`` intact) whose ``net`` is the quantized copy. The original
``model`` is left untouched so callers can keep both around for an
accuracy comparison.

Compared to FP32, int8 weight-only typically achieves ~4x smaller weights
at production scale (dim >= 64 — see ``examples/12_quantize_int8.py``).
Activations are computed in FP32 then cast to int8 only inside the
weight matmul, so accuracy loss is small for most networks.

This module is PTQ INT8 weight-only only. QAT (the standard 8da4w
recipe — int8 dynamic per-token activations, int4 grouped per-channel
weights) lives in the sibling :mod:`nnx.quantize.qat`. PTQ INT4
weight-only and per-tensor activation quantization are not yet shipped.
"""

from __future__ import annotations

from copy import copy, deepcopy
from typing import TypeVar

from ..nn.nn_model import NNModel

_M = TypeVar("_M", bound=NNModel)


def quantize_int8(model: _M) -> _M:
    """Return a new instance of ``type(model)`` with int8 weight-only quantized ``net``.

    Deep-copies ``model.net`` and applies
    ``torchao.quantization.quantize_(net, Int8WeightOnlyConfig())`` to
    the copy. Every ``nn.Linear`` submodule of the copy has its weight
    parameter replaced with an :class:`AffineQuantizedTensor` (int8
    per-channel, symmetric). Activations stay FP32 — only the weights
    are stored in int8.

    The original ``model`` is untouched. The returned wrapper is a shallow
    copy of ``model`` — the same class (a :class:`GenerativeNNModel` keeps
    its tokenizer and ``generate``; a user subclass keeps its overrides and
    custom attributes) sharing every other attribute (``params``,
    ``net_params``, ``device``, ``loss_fn``, ``tokenizer``) with the
    original — only ``net`` is the independent quantized copy. No
    constructor is re-run, so no weights are re-initialized and no RNG
    is consumed; subclasses that define their own copy hooks are copied
    through them.

    Args:
        model: a trained :class:`NNModel` (or subclass). PTQ has no
            training step; this function is a pure post-process.

    Returns:
        a new instance of ``type(model)`` whose ``net`` is the quantized
        deep-copy of ``model.net``. The new model can be used for
        ``predict`` / ``evaluate`` / ``generate`` / ``to_onnx`` exactly
        like the original; ``train`` on the quantized model is not
        supported (QAT lands in a separate module).

    Raises:
        ImportError: if ``torchao`` is not installed. Install with
            ``pip install thekaveh-nnx[quantize]``.
    """
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as e:  # pragma: no cover — opt-in extra
        raise ImportError(
            "quantize_int8 requires torchao. Install with `pip install thekaveh-nnx[quantize]` "
            "(or `pip install 'torchao>=0.17'` directly)."
        ) from e

    # Operate on a copy so the caller's original NNModel stays untouched —
    # lets users keep both around for an accuracy delta comparison.
    quantized_net = deepcopy(model.net)
    # version=2 silences the v1-deprecation UserWarning torchao emits. The
    # default granularity is PerRow (per-output-channel), which is the
    # standard weight-only INT8 layout.
    quantize_(quantized_net, Int8WeightOnlyConfig(version=2))

    # Build a sibling of the caller's OWN class without re-running
    # __init__ (which would build a fresh FP32 net via self.params.net(...)
    # and, for subclasses, may need extra arguments). copy.copy keeps the
    # exact subtype — GenerativeNNModel's generate / tokenizer, user
    # overrides and custom attributes — and shares every other attribute
    # so loss_fn / device / params line up with the original (FIX-017).
    m = copy(model)
    m.net = quantized_net
    return m
