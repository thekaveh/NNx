"""Tests for NNTrainerParams.builder() — composite multi-optim Builder.

The composite Builder accepts pre-built NNOptimParams / NNSchedulerParams
instances via `.optimizer(name, params)` / `.scheduler(name, params)`,
keeps them in name-keyed dicts, and enforces
`schedulers.keys() ⊆ optims.keys()` at .build() time (not at
__post_init__, which is where the dataclass enforces it today).
"""

from __future__ import annotations

from nnx import NNOptimParams, NNSchedulerParams, NNTrainerParams
from nnx.nn.enum.optims import Optims


def _make_adam(max_lr: float = 1e-3) -> NNOptimParams:
    return NNOptimParams.builder().adam(max_lr=max_lr, betas=(0.9, 0.999), weight_decay=0.0).build()


def _make_plateau() -> NNSchedulerParams:
    return (
        NNSchedulerParams.builder()
        .reduce_on_plateau(min_lr=1e-7, factor=0.5, patience=3, cooldown=1, threshold=1e-3)
        .build()
    )


def test_builder_minimal_single_optim():
    """Build a trainer config with one optimizer and no schedulers.
    The default-empty schedulers dict is the back-compat shape."""
    tp = (
        NNTrainerParams.builder()
        .n_epochs(10)
        .optimizer("main", _make_adam())
        .build()
    )
    assert tp.n_epochs == 10
    assert set(tp.optims.keys()) == {"main"}
    assert tp.optims["main"].name == Optims.ADAM
    assert tp.schedulers == {}


def test_builder_gan_recipe_two_optims_two_schedulers():
    """The §3.4 spec's canonical user — Example 09's GAN. Two
    optimizers ("g", "d"), two matching schedulers."""
    tp = (
        NNTrainerParams.builder()
        .n_epochs(50)
        .optimizer("g", _make_adam(max_lr=2e-4))
        .optimizer("d", _make_adam(max_lr=2e-4))
        .scheduler("g", _make_plateau())
        .scheduler("d", _make_plateau())
        .build()
    )
    assert set(tp.optims.keys()) == {"g", "d"}
    assert set(tp.schedulers.keys()) == {"g", "d"}


def test_builder_preserves_omit_when_default_invariant():
    """A trainer params with the defaults must round-trip identically
    to a direct-ctor one. `schedulers={}`, `seed=None`,
    `save_phase_checkpoints=True` all omit from state() at default."""
    built = (
        NNTrainerParams.builder()
        .n_epochs(10)
        .optimizer("main", _make_adam())
        .build()
    )
    direct = NNTrainerParams(
        n_epochs=10,
        optims={"main": _make_adam()},
    )
    assert built.state() == direct.state()
    assert "schedulers" not in built.state()
    assert "seed" not in built.state()
    assert "save_phase_checkpoints" not in built.state()
