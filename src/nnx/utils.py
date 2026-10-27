"""Pretty-printing helpers used throughout nnx.

Both module-level functions (``print_tree``, ``print_table``, ``flatten_dict``)
and the legacy ``Utils`` class API are exported. New code should prefer the
module functions; ``Utils.method(...)`` is kept as a thin back-compat shim so
existing notebooks keep working.
"""

from __future__ import annotations

import sys
from types import MethodType
from typing import Optional

import numpy as np
import torch
from prettytable import PrettyTable


def _writable(array: np.ndarray) -> np.ndarray:
    """A writable array for ``torch.from_numpy``: pandas' read-only
    ``.to_numpy()`` would otherwise raise torch's undefined-behaviour
    warning, though NNx never writes to the tensor. Negative strides (a
    reversed view), which torch cannot view, are copied too."""
    if array.flags.writeable and all(stride >= 0 for stride in array.strides):
        return array
    return array.copy()


def _capture_training_modes(module: torch.nn.Module) -> list[tuple[torch.nn.Module, bool]]:
    """Snapshot every module flag so mixed train/eval trees round-trip."""
    return [(child, child.training) for child in module.modules()]


def _overrides_train(module: torch.nn.Module) -> bool:
    """Whether ``module.train`` is anything but the stock ``Module.train``."""
    return type(module).train is not torch.nn.Module.train or "train" in module.__dict__


def _restore_training_modes(modes: list[tuple[torch.nn.Module, bool]]) -> None:
    """Restore exact flags while honoring custom ``Module.train`` hooks.

    Each custom hook runs once, bottom-up, with its descendants' ``train``
    suppressed. With those suppressed, the stock ``Module.train`` only sets
    its own flag, so it is replayed as that flag write in the same bottom-up
    order, which also undoes a hook's writes to a module outside its subtree
    that comes earlier in pre-order. The cost is O(modules) plus the
    subtrees of the modules that override ``train``, however deep the net.
    """
    for module, training in reversed(modes):
        module.training = training

    def preserve_mode(child: torch.nn.Module, _mode: bool = True):
        return child

    for module, training in reversed(modes):
        if not _overrides_train(module):
            module.training = training
            continue
        child_train_overrides = []
        for child in list(module.modules())[1:]:
            previous = child.__dict__.get("train")
            child.__dict__["train"] = MethodType(preserve_mode, child)
            child_train_overrides.append((child, previous))
        try:
            module.train(training)
        finally:
            for child, previous in child_train_overrides:
                if previous is None:
                    child.__dict__.pop("train", None)
                else:
                    child.__dict__["train"] = previous


def print_tree(tree, level: int = 0, *, file=None) -> None:
    """Pretty-print a nested dict as an indented tree.

    Pass ``file=`` (any object with ``.write``) to redirect output away
    from stdout — useful for capturing in tests or writing to a log.
    Defaults to ``sys.stdout``.
    """
    out = file if file is not None else sys.stdout
    if not isinstance(tree, dict) or not tree:
        # Empty dict (top-level or nested value): nothing to print, and
        # the max() below would raise on an empty key sequence.
        return

    # str(...): keys may be non-string (int epochs, enums).
    max_key_len = max(len(str(key)) for key in tree.keys())

    for key, val in tree.items():
        if isinstance(val, dict):
            print(" " * level * 4 + f"[-] {key}: ", file=out)
            print_tree(val, level + 1, file=out)
        else:
            print(" " * level * 4 + f"[+] {str(key).ljust(max_key_len)} : {val}", file=out)


def print_table(data: dict, header: bool = True, title: Optional[str] = None, *, file=None) -> None:
    """Print ``data`` as a 2-column key/value table.

    Pass ``file=`` to redirect output. Defaults to ``sys.stdout``.
    """
    out = file if file is not None else sys.stdout
    table = PrettyTable(["Key", "Value"])
    table.header = header
    if title is not None:
        table.title = title

    for key, val in data.items():
        table.add_row([key, val])

    print(table, file=out)


def flatten_dict(data: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict so nested keys become ``parent.child`` style.

    >>> flatten_dict({"a": 1, "b": {"c": 2}})
    {'a': 1, 'b.c': 2}
    """
    flattened = []
    for key, val in data.items():
        flattened_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(val, dict):
            flattened.extend(flatten_dict(val, flattened_key, sep=sep).items())
        else:
            flattened.append((flattened_key, val))
    return dict(flattened)


class Utils:
    """Back-compat static-method facade for the module functions above.

    Prefer the module-level functions (``from nnx.utils import print_tree``)
    in new code. ``Utils.method(...)`` continues to work for existing
    callers — each is a thin delegation to the corresponding module function.
    """

    # Bound as staticmethods so `Utils.print_tree(...)` works without `self`,
    # and `Utils.print_tree` returns the underlying function (callable from
    # tests, etc.).
    print_tree = staticmethod(print_tree)
    print_table = staticmethod(print_table)
    flatten_dict = staticmethod(flatten_dict)
