"""Shared source-resolution helper for PEFT adapter ``load_*_weights``.

Every PEFT ``load_<adapter>_weights(module, source)`` function accepts
either a filesystem path or a state-dict and resolves it to a plain
dict before applying adapter-specific key filtering. The resolution
step was duplicated verbatim across LoRA / IA3 / Prompt / Prefix and
carried a security-critical invariant — ``weights_only=True`` on the
``torch.load`` call — that needed to stay in sync across all four
sites. Centralized here so a future tightening (or relaxation under
documented context) happens in exactly one place.

The companion ``load_pretrained`` in ``nnx.finetune.loading`` accepts
an additional ``nn.Module`` source and uses ``map_location="cpu"``, so
it deliberately does not consume this helper — its surface is wider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

PeftStateDictSource = Union[str, Path, dict]


def _resolve_source_to_state_dict(source: PeftStateDictSource, fn_name: str) -> dict:
    """Resolve a PEFT adapter checkpoint source to a plain state-dict.

    Path inputs go through ``torch.load(..., weights_only=True)`` so a
    user-supplied file cannot trigger arbitrary-code execution at
    unpickle time. Dict inputs pass through unchanged. Anything else
    surfaces a ``TypeError`` naming the calling function for an
    actionable error message at the API boundary.

    Args:
        source: either a path to a ``torch.save``-d state-dict or the
            state-dict itself.
        fn_name: the calling function's name (e.g.
            ``"load_lora_weights"``); used in the ``TypeError``
            message so the error names the entry point, not this
            helper.

    Returns:
        A plain dict whose values are tensors / scalars; ready for
        downstream key-filtering and ``load_state_dict(..., strict=False)``.
    """
    if isinstance(source, (str, Path)):
        # weights_only=True: the adapter checkpoint is a plain dict of
        # tensors with no Python objects, so the strict loader works
        # AND removes the arbitrary-code-execution risk on
        # user-supplied paths.
        return torch.load(str(source), weights_only=True)
    if isinstance(source, dict):
        return source
    raise TypeError(f"{fn_name} source must be a path or dict, got {type(source).__name__}")
