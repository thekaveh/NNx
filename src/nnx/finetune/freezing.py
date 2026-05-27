"""Layer freezing utilities for fine-tuning.

The standard transfer-learning idiom — `requires_grad = False` on
parameters you don't want trained — is mechanically simple but fiddly
when you have dozens of submodules. These helpers let you mark
parameters by `fnmatch`-style glob pattern against their dotted name:

    freeze(model.net, "encoder.*")         # freeze the whole encoder
    freeze(model.net, "*.bias")            # freeze every bias term
    unfreeze(model.net, "encoder.layer.5") # un-freeze one specific block

`NNModel.freeze` / `NNModel.unfreeze` delegate here; using the free
functions directly is useful when you're freezing parameters of a
module that isn't an `NNModel` (e.g., a HuggingFace transformer you
loaded externally).
"""

from __future__ import annotations

import fnmatch

from torch import nn


def freeze(module: nn.Module, *patterns: str) -> int:
    """Set ``requires_grad=False`` on every parameter under ``module``
    whose dotted name matches any of ``patterns``.

    Patterns use ``fnmatch`` shell-glob semantics: ``*`` matches any
    sequence of characters **including dots** (not just one path segment),
    ``?`` matches a single character, ``[seq]`` matches one character
    from the set. Match is against the parameter's full dotted name,
    e.g., ``encoder.layer.5.weight``. So ``"encoder.*"`` matches every
    parameter under the encoder subtree, including deeply nested ones
    like ``encoder.layer.5.weight``.

    Args:
        module: any ``nn.Module``.
        *patterns: one or more fnmatch globs. If no patterns are
            given, raises ``ValueError`` (freeze-all-by-default is too
            dangerous to be the no-arg behavior).

    Returns:
        The number of parameters newly frozen (i.e., previously had
        ``requires_grad=True``). Useful for assertion / logging.
    """
    if not patterns:
        raise ValueError("freeze() requires at least one pattern; pass '*' to freeze every parameter")
    n = 0
    for name, param in module.named_parameters():
        if any(fnmatch.fnmatch(name, p) for p in patterns):
            if param.requires_grad:
                n += 1
            param.requires_grad = False
    return n


def unfreeze(module: nn.Module, *patterns: str) -> int:
    """Mirror of :func:`freeze` — set ``requires_grad=True`` on matching
    parameters. Returns the count newly unfrozen."""
    if not patterns:
        raise ValueError("unfreeze() requires at least one pattern; pass '*' to unfreeze every parameter")
    n = 0
    for name, param in module.named_parameters():
        if any(fnmatch.fnmatch(name, p) for p in patterns):
            if not param.requires_grad:
                n += 1
            param.requires_grad = True
    return n


def frozen(module: nn.Module) -> list[str]:
    """List the dotted parameter names currently frozen under ``module``.

    Returned list is sorted by name for stable test assertions. Useful
    for logging at ``train()`` entry so users can see exactly which
    parameters are excluded from training.
    """
    return sorted(name for name, param in module.named_parameters() if not param.requires_grad)
