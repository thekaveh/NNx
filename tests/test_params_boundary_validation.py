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
    resume = "a" * 32
    assert cls(n_epochs=1, resume_from_run_id=resume, resume_from_checkpoint="custom_e3").resume_from_checkpoint == (
        "custom_e3"
    )
    for bad in ("LAST", "../best", "custom", 3, None):
        with pytest.raises(ValueError, match="resume_from_checkpoint"):
            cls(n_epochs=1, resume_from_run_id=resume, resume_from_checkpoint=bad)
    # Lineage only (no resume reads it): kept as given, so saved runs still load.
    assert cls(n_epochs=1, resume_from_checkpoint="custom").resume_from_checkpoint == "custom"


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


@pytest.mark.parametrize("bad", [True, "0.1", float("nan")])
def test_dropout_probabilities_must_be_real_numbers(bad):
    with pytest.raises(ValueError, match="dropout_prob"):
        NNParams(input_dim=4, output_dim=2, hidden_dims=[3], dropout_prob=bad)
    with pytest.raises(ValueError, match="dropout_probs"):
        NNParams(input_dim=4, output_dim=2, hidden_dims=[3], dropout_prob=0.0, dropout_probs=[bad])
    assert NNParams(input_dim=4, output_dim=2, hidden_dims=[3], dropout_prob=0).dropout_prob == 0  # ints still pass


@pytest.mark.parametrize("field", ["attn_dropout", "resid_dropout"])
@pytest.mark.parametrize("bad", [True, "0.1", float("nan")])
def test_transformer_dropout_probabilities_must_be_real_numbers(field, bad):
    from nnx import NNTransformerParams

    with pytest.raises(ValueError, match=field):
        NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            max_seq_len=8,
            **{field: bad},
        )


def test_a_lineage_tag_must_be_yaml_storable():
    import pathlib

    from nnx import NNTrainParams

    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        NNTrainParams(n_epochs=1, parent_run_id="a" * 32, resume_from_checkpoint=pathlib.Path("x"))


def test_numpy_scalars_become_plain_floats_in_state():
    """A NumPy scalar passes validation; it must reach state() as a plain
    float, or run.yaml (yaml.safe_dump) fails after the first epoch. Python
    ints stay ints, so existing run ids never shift."""
    import numpy as np
    import yaml

    from nnx import NNOptimParams, NNParams, NNSchedulerParams, NNTransformerParams, Optims

    params = [
        NNParams(
            input_dim=2, output_dim=2, hidden_dims=[2], dropout_prob=np.float64(0.1), dropout_probs=[np.float32(0.5)]
        ),
        NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            max_seq_len=8,
            attn_dropout=np.float64(0.1),
            resid_dropout=np.float32(0.0),
            rope_base=np.float64(1e4),
        ),
        NNSchedulerParams(
            min_lr=np.float64(1e-6), factor=np.float32(0.5), patience=1, cooldown=0, threshold=np.float64(0.0)
        ),
        NNOptimParams(
            name=Optims.ADAM, max_lr=np.float32(1e-3), momentum=(np.float64(0.9), 0.999), weight_decay=np.float64(0.0)
        ),
    ]
    for p in params:
        yaml.safe_dump(p.state())  # raised RepresenterError for NumPy scalars
    assert NNParams(input_dim=2, output_dim=2, hidden_dims=[2], dropout_prob=0).state()["dropout_prob"] == 0
    assert type(NNParams(input_dim=2, output_dim=2, hidden_dims=[2], dropout_prob=0).dropout_prob) is int
