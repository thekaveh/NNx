"""Boundary-validation tests for the base params dataclasses.

These cover the [[params-boundary-validation]] contract for `NNParams`
(the base class every net's params subclass derives from) and
`NNTrainParams`: numeric fields must fail-fast in `__post_init__` with a
`ValueError` when out of range, rather than building a malformed
nn.Linear / nn.Dropout or making training a silent no-op far from the
field's origin. Validation must never touch `state()` — it only rejects
invalid configs, so a valid config still hashes to the same run.id.

The transformer-specific dimensions live in
`test_nn_transformer_params_builder.py`; the scheduler-field guards live
in `test_nn_scheduler_params_builder.py`; the NNTrainerParams `n_epochs`
guard lives in `test_trainer_params.py`.
"""

from __future__ import annotations

import pytest

from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_train_params import NNTrainParams


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dropout_prob": -0.1}, "0.0 <= dropout_prob <= 1.0"),
        ({"dropout_prob": 1.5}, "0.0 <= dropout_prob <= 1.0"),
        ({"input_dim": 0}, "input_dim > 0"),
        ({"input_dim": -3}, "input_dim > 0"),
        ({"output_dim": 0}, "output_dim > 0"),
        ({"hidden_dims": [16, 0, 8]}, "all hidden_dims > 0"),
        ({"hidden_dims": [-1]}, "all hidden_dims > 0"),
    ],
)
def test_nn_params_rejects_out_of_range_fields(overrides, match):
    kwargs = dict(dropout_prob=0.0, input_dim=4, output_dim=2)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        NNParams(**kwargs)


def test_nn_params_accepts_valid_boundary_values():
    """The guards must not reject legitimate edge values: dropout_prob at
    both ends of [0, 1] and an empty hidden_dims (no hidden layers)."""
    NNParams(dropout_prob=0.0, input_dim=4, output_dim=2)
    NNParams(dropout_prob=1.0, input_dim=1, output_dim=1, hidden_dims=[])
    NNParams(dropout_prob=0.5, input_dim=8, output_dim=3, hidden_dims=[16, 8])


@pytest.mark.parametrize("bad_n_epochs", [0, -1])
def test_nn_train_params_rejects_non_positive_n_epochs(bad_n_epochs):
    """`n_epochs` drives `range(params.n_epochs)` in NNModel.train, so a
    0/negative value would otherwise make training a silent no-op."""
    with pytest.raises(ValueError, match="n_epochs >= 1"):
        NNTrainParams(n_epochs=bad_n_epochs)


@pytest.mark.parametrize("owner", ["NNTrainParams", "NNTrainerParams"])
def test_resume_checkpoint_normalizes_to_its_tag_and_rejects_unknown_tags(owner):
    """A ``Checkpoints`` member serializes as its tag string — the enum and
    the string are one run identity and both dump as YAML — and an unknown
    tag fails at construction, not after an epoch has trained."""
    import yaml

    from nnx import Checkpoints, NNTrainParams
    from nnx.trainer import NNTrainerParams

    cls = NNTrainParams if owner == "NNTrainParams" else _trainer_params(NNTrainerParams)
    as_enum = cls(n_epochs=1, resume_from_run_id="a" * 32, resume_from_checkpoint=Checkpoints.BEST)
    as_text = cls(n_epochs=1, resume_from_run_id="a" * 32, resume_from_checkpoint="best")
    assert as_enum.resume_from_checkpoint == "best"
    assert as_enum.state() == as_text.state()
    assert "best" in yaml.safe_dump(as_enum.state())
    assert cls(n_epochs=1, resume_from_checkpoint="custom_e3").resume_from_checkpoint == "custom_e3"
    for bad in ("LAST", "../best", "custom", 3):
        with pytest.raises(ValueError, match="resume_from_checkpoint"):
            cls(n_epochs=1, resume_from_checkpoint=bad)


@pytest.mark.parametrize("owner", ["NNTrainParams", "NNTrainerParams"])
def test_seed_and_data_id_fail_at_construction(owner):
    import numpy as np

    from nnx import NNTrainParams
    from nnx.trainer import NNTrainerParams

    cls = NNTrainParams if owner == "NNTrainParams" else _trainer_params(NNTrainerParams)
    for bad in (-1, 1.5, True, 2**32):
        with pytest.raises(ValueError, match="seed"):
            cls(n_epochs=1, seed=bad)
    seeded = cls(n_epochs=1, seed=np.int64(7))
    assert type(seeded.seed) is int and seeded.state() == cls(n_epochs=1, seed=7).state()
    for bad in ("", "   ", 3):
        with pytest.raises(ValueError, match="data_id"):
            cls(n_epochs=1, data_id=bad)


def _trainer_params(cls):
    """``NNTrainerParams`` with its one required field filled in."""
    import functools

    from nnx import NNOptimParams, NNParamGroupSpec, Optims

    optim = NNOptimParams(
        name=Optims.ADAM,
        max_lr=1e-3,
        momentum=(0.9, 0.999),
        weight_decay=0.0,
        param_groups=[NNParamGroupSpec(name_pattern="*", lr=1e-3)],
    )
    return functools.partial(cls, optims={"all": optim})
