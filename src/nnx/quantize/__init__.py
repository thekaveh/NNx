"""Post-training quantization for NNx models.

Today this subpackage exposes a single entry point:

- :func:`quantize_int8` — one-call PTQ INT8 weight-only quantization
  via ``torchao``. Operates on the underlying ``model.net``; returns a
  new :class:`NNModel` whose linear weights are stored in int8 (FP32
  activations). The original ``NNModel`` is untouched.

INT4 weight-only, activation quantization, and quantization-aware
training (QAT) are tracked separately.

Install the ``torchao`` runtime dependency with::

    pip install nnx[quantize]
"""

from __future__ import annotations

from .ptq import quantize_int8

__all__ = ["quantize_int8"]
