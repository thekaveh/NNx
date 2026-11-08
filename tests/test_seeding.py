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


def test_a_scoped_seed_seeds_and_the_scope_restores_the_mps_generator(monkeypatch):
    """FEAT-050: a CPU pin of the MPS branches, over a fake MPS generator.
    A probe's scoped seed seeds MPS; the scope restores what it found."""
    import torch

    from nnx.seeding import _seed_scope, _set_seed
    from tests._mps import fake_mps_generator

    mps = fake_mps_generator(monkeypatch)
    mps.draw()
    before = mps.state.clone()
    with _seed_scope():
        _set_seed(7, scoped=True)
        assert mps.seeds == [7]
        seeded = mps.state.clone()
        mps.draw()
    assert torch.equal(mps.state, before) and not torch.equal(seeded, before)
