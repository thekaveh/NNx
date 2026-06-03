"""Tests for NNTrainerParams.builder() — composite multi-optim Builder.

The composite Builder accepts pre-built NNOptimParams / NNSchedulerParams
instances via `.optimizer(name, params)` / `.scheduler(name, params)`,
keeps them in name-keyed dicts, and enforces
`schedulers.keys() ⊆ optims.keys()` at .build() time (not at
__post_init__, which is where the dataclass enforces it today).
"""

from __future__ import annotations

import pytest

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
    tp = NNTrainerParams.builder().n_epochs(10).optimizer("main", _make_adam()).build()
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
    built = NNTrainerParams.builder().n_epochs(10).optimizer("main", _make_adam()).build()
    direct = NNTrainerParams(
        n_epochs=10,
        optims={"main": _make_adam()},
    )
    assert built.state() == direct.state()
    assert "schedulers" not in built.state()
    assert "seed" not in built.state()
    assert "save_phase_checkpoints" not in built.state()


def test_builder_rejects_scheduler_for_unknown_optim():
    """Spec §3.4 key win: the Builder catches `.scheduler("d", ...)`
    without a prior `.optimizer("d", ...)` at the Builder boundary,
    NOT at __post_init__. Error message names the missing optim
    name + lists known optims to help the user."""
    with pytest.raises(ValueError, match=r"scheduler\(\) called with names not present"):
        (
            NNTrainerParams.builder()
            .n_epochs(10)
            .optimizer("g", _make_adam())
            .scheduler("d", _make_plateau())  # "d" not yet registered as optim
            .build()
        )


def test_builder_rejects_build_without_any_optim():
    """The dataclass's __post_init__ raises if optims is empty. The
    Builder surfaces the same error — we don't pre-empt the dataclass
    check, but we do hit it cleanly."""
    with pytest.raises(ValueError, match=r"optims must have at least one entry"):
        NNTrainerParams.builder().n_epochs(10).build()


def test_builder_error_message_names_missing_optim():
    """The error message should be actionable — it names the
    unknown scheduler key AND lists the known optim names."""
    with pytest.raises(ValueError) as exc_info:
        (
            NNTrainerParams.builder()
            .n_epochs(10)
            .optimizer("g", _make_adam())
            .optimizer("d", _make_adam())
            .scheduler("typo", _make_plateau())
            .build()
        )
    msg = str(exc_info.value)
    assert "typo" in msg
    assert "'g'" in msg and "'d'" in msg


def test_builder_save_phase_checkpoints_false_appears_in_state():
    """save_phase_checkpoints(False) must store on the dataclass and
    appear in state() (it's non-default). Confirms the setter writes."""
    tp = (
        NNTrainerParams.builder()
        .n_epochs(10)
        .optimizer("main", _make_adam())
        .save_phase_checkpoints(False)
        .build()
    )
    assert tp.save_phase_checkpoints is False
    assert tp.state().get("save_phase_checkpoints") is False


def test_builder_save_phase_checkpoints_true_after_false_overrides_to_true():
    """Regression: a prior `.save_phase_checkpoints(False)` followed
    by `.save_phase_checkpoints(True)` must reach the dataclass default
    (True). Pre-fix the True call was a silent no-op because the body
    skipped storing when `value is True`. state() must omit the field
    at the default."""
    tp = (
        NNTrainerParams.builder()
        .n_epochs(10)
        .optimizer("main", _make_adam())
        .save_phase_checkpoints(False)
        .save_phase_checkpoints(True)
        .build()
    )
    assert tp.save_phase_checkpoints is True
    assert "save_phase_checkpoints" not in tp.state()


def test_builder_chains_train_loader_and_seed():
    """Builder composes the optional chainable methods cleanly."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)), batch_size=2)

    tp = NNTrainerParams.builder().n_epochs(10).optimizer("main", _make_adam()).seed(42).train_loader(loader).build()
    assert tp.seed == 42
    assert tp.train_loader is loader


def test_builder_extra_metrics_chains():
    tp = (
        NNTrainerParams.builder()
        .n_epochs(5)
        .optimizer("main", _make_adam())
        .extra_metrics({"my_metric": lambda y, y_hat: float((y == y_hat).mean())})
        .build()
    )
    assert tp.extra_metrics is not None
    assert "my_metric" in tp.extra_metrics


def test_builder_round_trips_through_state():
    """Composition with the inner Plan-1/2 Builders must round-trip
    through state() / from_state() identically to direct-ctor."""
    built = (
        NNTrainerParams.builder()
        .n_epochs(10)
        .optimizer("g", _make_adam(max_lr=2e-4))
        .optimizer("d", _make_adam(max_lr=2e-4))
        .scheduler("g", _make_plateau())
        .scheduler("d", _make_plateau())
        .seed(42)
        .build()
    )
    assert NNTrainerParams.from_state(built.state()) == built
