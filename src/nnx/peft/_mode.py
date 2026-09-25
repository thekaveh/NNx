"""Train/eval mode inheritance for PEFT wrappers (FIX-013).

A freshly constructed ``nn.Module`` — and every child it builds, such as
LoRA's ``nn.Dropout`` — defaults to train mode. Wrapping a layer or model
that the caller already switched to ``eval()`` therefore used to produce a
train-mode wrapper, and a nonzero-dropout adapter made inference
nondeterministic. Each wrapper calls :func:`inherit_training_mode` at the
end of construction so it (and the children it created) report the mode
of the module they wrap.

Internal; not part of the public API.
"""

from __future__ import annotations

from torch import nn


def inherit_training_mode(wrapper: nn.Module, wrapped: nn.Module) -> None:
    """Give ``wrapper`` and the children it created the mode of ``wrapped``.

    ``wrapped`` and everything inside it keep their own flags, so a
    mixed-mode model is never flattened, and the surrounding model is never
    switched. The mode is runtime state only — nothing is registered or
    serialized — and a later ``.train()`` / ``.eval()`` behaves as usual.
    """
    mode = wrapped.training
    wrapper.training = mode
    for child in wrapper.children():
        if child is not wrapped:
            child.train(mode)
