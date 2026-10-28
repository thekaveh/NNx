"""Nesting depth of JSON-like values, measured without recursion. Internal.

The JSON parser's own nesting limit varies by Python version and stack size
(about 1,000 levels on 3.10 / 3.11, 10,000 on 3.12 / 3.13, more on 3.14),
and the recursive freezes and thaws behind NNx's immutable configs overflow
the recursion limit far sooner. So readers check a parsed value, and writers
check a value before freezing it, against an explicit limit here: the same
refusal on every Python version, never a raw ``RecursionError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# The nesting bound of NNx's own JSON files — provenance manifests, the
# calibration and abstention artifacts and the history journal — counted from
# the file's top level. A writer refuses a value nested deeper than this minus
# the levels its file wraps around it, so every file NNx writes reads back.
# Far deeper than any real config, and well below the ~495 levels at which the
# recursive freeze overflows the default recursion limit on 3.10 / 3.11 (where
# each comprehension adds a frame per level, before PEP 709). Run bundles keep
# their own, tighter bound (``nnx.bundles._MAX_DEPTH``).
MAX_JSON_DEPTH = 256

_LEAVES = (str, int, float, type(None))  # never pushed: a large state is mostly scalars


def too_deep(value: Any, limit: int) -> bool:
    """Whether ``value`` nests more than ``limit`` levels of mappings and
    lists / tuples (a scalar nests 0 levels, ``{}`` and ``[]`` one). It stops
    at the first branch past ``limit``, so a self-containing value ends too."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list, tuple)) or isinstance(item, Mapping):
            if depth > limit:
                return True
            children = item.values() if isinstance(item, Mapping) else item
            stack.extend((child, depth + 1) for child in children if not isinstance(child, _LEAVES))
    return False
