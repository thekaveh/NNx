"""Quantization-aware training (QAT) via ``torchao``.

QAT inserts *fake-quant* ops during the forward pass so the network
sees rounding noise while it trains, then converts to a fully-quantized
model at the end. Compared to PTQ (:func:`nnx.quantize_int8`), QAT
typically recovers most of the accuracy lost to aggressive low-bit
schemes (e.g., int4 weights, int8 dynamic activations) at the cost of
one additional training run.

The integration is intentionally thin:

- :func:`qat_train_step_factory` returns a :class:`TrainStepFn`. The
  per-batch step is unchanged from the user's base step (or the default
  supervised step) — the fake-quant ops live inside the module
  hierarchy, so the standard forward/backward already exercises them.
- :class:`QATLifecycleCallback` does the heavy lifting on the training
  boundaries:

  * ``on_train_begin``: ``quantizer.prepare(model.net)`` — swap every
    eligible ``nn.Linear`` for its fake-quantized counterpart, in place.
  * ``on_train_end``: ``quantizer.convert(model.net)`` — replace the
    fake-quantized linears with truly-quantized ones (int4 weights
    packed, int8 activations) for inference.

The two pieces are companions: dropping the callback while keeping the
factory yields a normal FP32 training run; the factory is only useful
*with* the callback in :meth:`NNModel.train`'s callbacks list.

Currently the only supported config is the standard "8da4w" recipe
(int8 dynamic per-token activations, int4 grouped per-channel weights)
backed by ``torchao.quantization.qat.Int8DynActInt4WeightQATQuantizer``.
Other configs will land here as future PRs.

Install the ``torchao`` runtime dependency with::

    pip install nnx-pytorch[quantize]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..nn.callbacks import Callback
from ..nn.nn_model import TrainStepFn, default_train_step

if TYPE_CHECKING:  # pragma: no cover — type-checking-only imports.
    from ..nn.nn_model import _CallbackContext


# Supported QAT recipe shortcuts. Keeping this as a small dict (rather
# than an enum) lets future configs land as one-line additions without
# touching call sites that already pass the string literal.
_SUPPORTED_CONFIGS: tuple[str, ...] = ("8da4w",)


def _build_quantizer(qat_config: str, *, groupsize: int = 32):
    """Resolve a config shortcut to a torchao quantizer instance.

    Centralized so the factory and the callback share one source of
    truth for *which* recipe ``qat_config`` maps to. Raises ImportError
    with a pointer to the pip extra when torchao isn't installed and
    ValueError when the recipe name isn't recognized.
    """
    try:
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
    except ImportError as e:  # pragma: no cover — opt-in extra
        raise ImportError(
            "QAT requires torchao. Install with `pip install nnx-pytorch[quantize]` "
            "(or `pip install 'torchao>=0.17'` directly)."
        ) from e

    if qat_config == "8da4w":
        # int8 dynamic per-token activations + int4 grouped per-channel
        # weights — the standard torchao recipe. groupsize=32 is small
        # enough to fit toy nets in the test suite (width 64 is divisible
        # by 32) while still being a real-world setting for production
        # models.
        return Int8DynActInt4WeightQATQuantizer(groupsize=groupsize)

    raise ValueError(f"unknown qat_config {qat_config!r}; supported: {_SUPPORTED_CONFIGS}")


def qat_train_step_factory(
    base_step: Optional[TrainStepFn] = None,
    qat_config: str = "8da4w",
) -> TrainStepFn:
    """Return a :class:`TrainStepFn` that runs ``base_step`` against a
    fake-quantized model.

    The returned step is the *same* as ``base_step`` (or
    :func:`default_train_step` when ``base_step`` is None) — fake-quant
    insertion happens once, via :class:`QATLifecycleCallback`, on
    ``on_train_begin``. The per-batch forward/backward then exercises
    those fake-quant ops automatically through the standard module
    forward.

    Why split the work between a factory and a callback?

    - The factory validates ``qat_config`` early (at construction time)
      so misconfigurations surface before the data loader spins up.
    - The callback owns the lifecycle: ``prepare`` at start, ``convert``
      at end. Bundling that into the per-batch step would re-check the
      module state every iteration and complicate gradient flow.

    Both pieces are needed in :meth:`NNModel.train`::

        callback = QATLifecycleCallback(qat_config="8da4w")
        step_fn  = qat_train_step_factory(qat_config="8da4w")
        model.train(params=..., callbacks=[callback], train_step_fn=step_fn)

    Args:
        base_step: optional underlying training step to wrap. ``None``
            (the default) uses :func:`default_train_step` — the standard
            supervised forward/backward. Pass a custom step here to
            combine QAT with e.g. knowledge distillation or mixup; the
            fake-quant ops live in the model graph, so any standard
            step picks them up transparently.
        qat_config: shortcut for the torchao QAT recipe. Currently only
            ``"8da4w"`` is supported (int8 dynamic activations + int4
            grouped weights). Validated eagerly so a typo doesn't
            propagate to the callback.

    Returns:
        a :class:`TrainStepFn` ready to pass to
        ``NNModel.train(..., train_step_fn=...)``.

    Raises:
        ValueError: if ``qat_config`` is not in
            :data:`_SUPPORTED_CONFIGS`.
        ImportError: if ``torchao`` is not installed.
    """
    # Validate eagerly — fail at factory construction time so the
    # caller's `model.train(...)` invocation doesn't blow up two minutes
    # into a long-running data load.
    _build_quantizer(qat_config)

    return base_step if base_step is not None else default_train_step


class QATLifecycleCallback(Callback):
    """Manage the torchao ``prepare`` / ``convert`` lifecycle around training.

    Add to ``callbacks=[...]`` in :meth:`NNModel.train`. On train begin,
    swaps every eligible :class:`torch.nn.Linear` in ``model.net`` for
    its fake-quantized counterpart (the model now learns to be robust
    to int4/int8 rounding). On train end, the fake-quantized linears
    are converted to actually-quantized ones — the resulting model is
    suitable for inference / export.

    The mutation is **in place** on ``model.net``: after training,
    ``model.net`` IS the converted model. The callback exposes the
    quantizer instance as ``self.quantizer`` for callers who want to
    pickle quantizer-specific state alongside their checkpoint, and
    tracks the prepare/convert phase via ``self.is_prepared`` and
    ``self.is_converted`` for downstream inspection.

    Args:
        qat_config: torchao recipe shortcut. See
            :func:`qat_train_step_factory`.
        groupsize: group size for the int4 weight quantizer. 32 is the
            default — small enough to apply to toy nets in tests
            (where hidden_dim=64) while being a real-world setting.
            Larger groupsizes (128, 256) give better compression at
            the cost of accuracy.
    """

    def __init__(self, qat_config: str = "8da4w", *, groupsize: int = 32):
        self.qat_config = qat_config
        self.groupsize = groupsize
        # Resolve the quantizer eagerly so a torchao ImportError or a
        # bad config name surfaces at callback construction, not on
        # ``on_train_begin`` (which would otherwise crash mid-train).
        self.quantizer = _build_quantizer(qat_config, groupsize=groupsize)
        self.is_prepared: bool = False
        self.is_converted: bool = False

    def on_train_begin(self, ctx: _CallbackContext) -> None:
        """Insert fake-quant ops into ``ctx.model.net`` in place."""
        # Idempotency guard: re-running prepare on an already-prepared
        # net would corrupt the linear-wrapping. Real-world training
        # loops don't hit this, but a callback shared across two
        # back-to-back ``model.train()`` calls would.
        if self.is_prepared:
            return
        self.quantizer.prepare(ctx.model.net)
        self.is_prepared = True

    def on_train_end(self, ctx: _CallbackContext) -> None:
        """Convert fake-quant ops in ``ctx.model.net`` to true int4/int8 modules.

        After this returns, ``ctx.model.net`` produces real quantized
        outputs and is suitable for inference / ONNX export. The model
        is no longer trainable through the usual FP32 optimizer path —
        a fresh training session on the same NNModel would need a new
        QATLifecycleCallback.
        """
        # Symmetric idempotency guard, paired with ``on_train_begin``.
        if self.is_converted:
            return
        if not self.is_prepared:
            # No prepare ever ran — converting an FP32 model is a no-op
            # that would still mutate state-flags misleadingly. Bail.
            return
        self.quantizer.convert(ctx.model.net)
        self.is_converted = True
