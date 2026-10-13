"""set_seed validates its seed before seeding anything."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("bad", [2**32, -1, 1.5, True])
def test_set_seed_refuses_a_seed_before_touching_any_stream(bad, monkeypatch):
    """Checked first: PYTHONHASHSEED (inherited by spawned workers) and the
    Python stream are not set from a seed NumPy would refuse."""
    import os
    import random

    from nnx import set_seed

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    state = random.getstate()
    with pytest.raises(ValueError, match="seed"):
        set_seed(bad)
    assert "PYTHONHASHSEED" not in os.environ and random.getstate() == state
