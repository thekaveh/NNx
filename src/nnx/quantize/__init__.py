"""Quantization for NNx models — PTQ and QAT.

Two complementary entry points, both backed by ``torchao``:

- :func:`quantize_int8` — one-call PTQ INT8 weight-only. Operates on
  the underlying ``model.net``; returns a new :class:`NNModel` whose
  linear weights are stored in int8 (FP32 activations). The original
  :class:`NNModel` is untouched. No retraining, no calibration data.
- :func:`qat_train_step_factory` + :class:`QATLifecycleCallback` —
  quantization-aware training. The model trains with fake-quant ops
  in the forward pass and is converted to real int4/int8 modules at
  the end. Recovers accuracy that aggressive low-bit PTQ would lose,
  at the cost of one training run.

INT4 weight-only PTQ and per-tensor activation quantization land here
as future PRs.

Install the ``torchao`` runtime dependency with::

    pip install thekaveh-nnx[quantize]
"""

from __future__ import annotations

from .ptq import quantize_int8
from .qat import QATLifecycleCallback, qat_train_step_factory

__all__ = ["quantize_int8", "qat_train_step_factory", "QATLifecycleCallback"]
