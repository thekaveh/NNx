"""Load pretrained weights from sources other than NNCheckpoint.

NNCheckpoint is the library's native format — pickled object carrying
net_state plus the params used to construct the net. For fine-tuning
work the user often starts from weights that DIDN'T come from nnx:
torchvision checkpoints, HuggingFace `pytorch_model.bin`, raw
state-dicts saved by a colleague.

This module handles those cases. Key remapping (foreign-layer-name →
local-layer-name) lets you adapt naming conventions without manually
rewriting state-dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class LoadPretrainedResult:
    """Outcome of a :func:`load_pretrained` call.

    Compared with :meth:`torch.nn.Module.load_state_dict`, this gives
    you back not just the missing/unexpected keys but also the list
    of keys actually applied (after any remapping) — useful for
    confirming the load did what you intended.
    """
    loaded_keys:     list[str]      # keys present in source AND module (after remap)
    missing_keys:    list[str]      # in module's state_dict, not in source
    unexpected_keys: list[str]      # in source, no match in module


def load_pretrained(
    module: nn.Module,
    source: Union[str, Path, dict, nn.Module],
    *,
    key_map: Optional[dict[str, str]] = None,
    strict: bool = False,
    prefix: Optional[str] = None,
) -> LoadPretrainedResult:
    """Load weights into ``module`` from an external source.

    The source can be:
      - a path (str or Path) to a ``.pt`` / ``.pth`` file holding a
        state-dict (loaded with ``weights_only=True`` for safety);
      - a state-dict (``dict``) already in memory;
      - another ``nn.Module``, in which case its state-dict is used.

    Args:
        module: target module to load into. Mutated in place.
        source: see above.
        key_map: optional remapping from source keys to target keys.
            Applied before matching. E.g., ``{"backbone.": "net."}``
            rewrites every key that starts with ``backbone.``.
            Substring replacement, not regex — for clarity.
        strict: when True, raise if any source key has no target match
            OR any target key has no source. Default False (fine-tuning
            commonly partial-loads).
        prefix: optional prefix to strip from source keys before
            matching. E.g., ``prefix="model."`` turns ``model.layer.0``
            into ``layer.0``. Applied BEFORE ``key_map``.

    Returns:
        :class:`LoadPretrainedResult` with the loaded / missing /
        unexpected key sets.
    """
    # 1. Resolve `source` to a state-dict.
    if isinstance(source, nn.Module):
        src: dict = dict(source.state_dict())
    elif isinstance(source, (str, Path)):
        # weights_only=True is safe — state-dicts are tensors + scalars
        # only. Avoids the ACE risk that NNCheckpoint.from_file warns
        # about. Caller passing a malicious file gets a clean
        # UnpicklingError, not arbitrary code execution.
        src = torch.load(str(source), weights_only=True, map_location="cpu")
    elif isinstance(source, dict):
        src = dict(source)
    else:
        raise TypeError(
            f"load_pretrained: source must be a path, dict, or nn.Module; "
            f"got {type(source).__name__}"
        )

    # 2. Apply prefix-stripping (if requested) then key_map remapping.
    remapped: dict = {}
    for key, val in src.items():
        new_key = key
        if prefix is not None and new_key.startswith(prefix):
            new_key = new_key[len(prefix):]
        if key_map:
            for foreign, local in key_map.items():
                if new_key.startswith(foreign):
                    new_key = local + new_key[len(foreign):]
                    break
        remapped[new_key] = val

    # 3. Match against the target module's keys and pick the overlap.
    target_keys = set(module.state_dict().keys())
    source_keys = set(remapped.keys())

    loaded_keys = sorted(target_keys & source_keys)
    missing_keys = sorted(target_keys - source_keys)
    unexpected_keys = sorted(source_keys - target_keys)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "load_pretrained(strict=True) found mismatches:\n"
            f"  missing in source:    {missing_keys}\n"
            f"  unexpected in source: {unexpected_keys}"
        )

    # 4. Apply only the overlapping keys. PyTorch's load_state_dict
    # with strict=False does this, but we've already computed the
    # intersection so we can pass just those entries.
    overlap = {k: remapped[k] for k in loaded_keys}
    module.load_state_dict(overlap, strict=False)

    return LoadPretrainedResult(
        loaded_keys=loaded_keys,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
