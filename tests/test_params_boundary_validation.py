"""Boundary-validation tests for the base params dataclasses.

These cover the [[params-boundary-validation]] contract for `NNParams`
(the base class every net's params subclass derives from) and
`NNTrainParams`: numeric fields must fail-fast in `__post_init__` with a
`ValueError` when out of range, rather than building a malformed
nn.Linear / nn.Dropout or making training a silent no-op far from the
field's origin. Validation must never touch `state()` — it only rejects
invalid configs, so a valid config still hashes to the same run.id.

The file also holds later boundary regressions found by maintenance
reviews (optimizer momentum shapes, a warm-up past the default horizon,
overflowing integers, predict input dtypes, library warnings from a
loader-less `__main__`). The transformer-specific dimensions live in
`test_nn_transformer_params_builder.py`; most scheduler-field guards live
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


def test_an_integer_momentum_accepted_at_construction_is_valid_at_train_time():
    """momentum=0 (plain SGD) and betas=(0, 0) pass construction, so
    is_valid() (which train() checks) must accept them too."""
    from nnx import NNOptimParams, Optims

    assert NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0, weight_decay=0).is_valid()
    assert NNOptimParams(name=Optims.ADAM, max_lr=0.1, momentum=(0, 0), weight_decay=0).is_valid()


def _trainer_optims():
    from nnx import NNOptimParams, Optims

    return {"main": NNOptimParams(name=Optims.SGD, max_lr=0.1, momentum=0.0, weight_decay=0.0)}


_TRAINER_OPTIMS = _trainer_optims()


@pytest.mark.parametrize("flag", ["save_phase_checkpoints", "overwrite_existing"])
@pytest.mark.parametrize("bad", [0, 1, "false"])
def test_train_params_flags_must_be_bools(flag, bad):
    from nnx import NNTrainParams
    from nnx.trainer import NNTrainerParams

    with pytest.raises(ValueError, match=flag):
        NNTrainParams(n_epochs=1, **{flag: bad})
    with pytest.raises(ValueError, match=flag):
        NNTrainerParams(n_epochs=1, optims=_TRAINER_OPTIMS, **{flag: bad})


def test_trainer_auto_step_schedulers_must_be_a_bool():
    from nnx.trainer import NNTrainerParams

    with pytest.raises(ValueError, match="auto_step_schedulers"):
        NNTrainerParams(n_epochs=1, optims=_TRAINER_OPTIMS, auto_step_schedulers="false")


def test_a_numpy_integer_setting_keeps_the_python_int_run_identity():
    import numpy as np

    from nnx import NNOptimParams, Optims

    plain = NNOptimParams(name=Optims.SGD, max_lr=1, momentum=0.9, weight_decay=0)
    numpy = NNOptimParams(name=Optims.SGD, max_lr=np.int64(1), momentum=0.9, weight_decay=np.int64(0))
    assert numpy.state() == plain.state()


@pytest.mark.parametrize(
    ("name", "momentum"),
    [("SGD", 0), ("SGD_NESTEROV", 1), ("ADAM", (0, 0)), ("ADAMW", (0, 0))],
)
def test_an_integer_momentum_builds_its_optimizer(name, momentum):
    """What construction and is_valid() accept, train() builds."""
    import torch

    from nnx import NNOptimParams, Optims
    from nnx.optimizers import build_optimizer

    params = NNOptimParams(name=getattr(Optims, name), max_lr=0.1, momentum=momentum, weight_decay=0)
    build_optimizer(torch.nn.Linear(2, 2), params)


def test_a_validation_record_with_only_extra_values_survives_idps_csv(tmp_path):
    """A custom eval step may report only extra values: the reloaded
    history keeps that record (concepts §6.2: val_edp, extra included)."""
    import pandas as pd

    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
    from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint
    from nnx.nn.params.nn_run import _read_idps_csv

    idp = NNIterationDataPoint(
        lr=0.1,
        iter_idx=0,
        epoch_idx=0,
        batch_idx=0,
        train_edp=NNEvaluationDataPoint(loss=1.0),
        val_edp=NNEvaluationDataPoint(extra={"bleu": 0.5}),
    )
    path = tmp_path / "idps.csv"
    pd.json_normalize(data=[idp.state()]).to_csv(path)
    (loaded,) = _read_idps_csv(str(path))
    assert loaded.val_edp is not None and loaded.val_edp.extra == {"bleu": 0.5}


def test_tie_embeddings_must_be_a_bool():
    from nnx import NNTransformerParams

    with pytest.raises(ValueError, match="tie_embeddings"):
        NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            max_seq_len=8,
            tie_embeddings="false",
        )


@pytest.mark.parametrize(
    "bad",
    [{"schedulers": None}, {"optims": {"a": "not-params"}}, {"schedulers": {"main": "cosine"}}],
    ids=["schedulers-none", "optims-entry", "schedulers-entry"],
)
def test_trainer_params_containers_are_checked_at_construction(bad):
    from nnx.trainer import NNTrainerParams

    fields = {"optims": _TRAINER_OPTIMS, **bad}
    with pytest.raises(ValueError, match="NNTrainerParams"):
        NNTrainerParams(n_epochs=1, **fields)


def test_enum_fields_refuse_plain_strings():
    """A string entered state() and the run id, then failed far away as
    "'str' object is not callable"."""
    from nnx import NNOptimParams, NNParams

    with pytest.raises(TypeError, match="activation"):
        NNParams(input_dim=2, output_dim=2, hidden_dims=[2], dropout_prob=0.0, activation="relu")
    with pytest.raises(TypeError, match="activations"):
        NNParams(input_dim=2, output_dim=2, hidden_dims=[2], dropout_prob=0.0, activations=["relu"])
    with pytest.raises(TypeError, match="NNOptimParams.name"):
        NNOptimParams(name="adam", max_lr=0.1, momentum=(0.9, 0.999), weight_decay=0.0)


def test_a_warmup_longer_than_its_schedule_is_refused():
    from nnx import NNSchedulerParams, Schedulers

    with pytest.raises(ValueError, match="warmup_steps <= total_steps"):
        NNSchedulerParams(
            kind=Schedulers.LINEAR_WARMUP_DECAY,
            min_lr=0.0,
            factor=0.5,
            patience=0,
            cooldown=0,
            threshold=0.0,
            warmup_steps=10,
            total_steps=5,
        )


@pytest.mark.parametrize(("name", "momentum"), [("ADAM", 0.9), ("SGD", (0.9, 0.999))])
def test_a_momentum_of_the_wrong_shape_fails_with_a_reason_when_built(name, momentum):
    """Constructible by design (ExperimentPlan.validate reports it); a
    direct build_optimizer names the problem instead of a bare assert."""
    import torch

    from nnx import NNOptimParams, Optims
    from nnx.optimizers import build_optimizer

    params = NNOptimParams(name=getattr(Optims, name), max_lr=0.1, momentum=momentum, weight_decay=0.0)
    assert not params.is_valid()
    with pytest.raises(ValueError, match="pair|one real momentum"):
        build_optimizer(torch.nn.Linear(2, 2), params)


def test_a_warmup_longer_than_the_default_horizon_is_refused_when_built():
    import torch

    from nnx import NNSchedulerParams, Schedulers

    params = NNSchedulerParams(
        kind=Schedulers.LINEAR_WARMUP_DECAY,
        min_lr=0.0,
        factor=0.5,
        patience=0,
        cooldown=0,
        threshold=0.0,
        warmup_steps=10,
    )
    optimizer = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="warmup_steps"):
        Schedulers.LINEAR_WARMUP_DECAY(optimizer, params, n_epochs=3)


def test_predict_takes_numpy_float64_input():
    """NumPy's default float64 failed in the first Linear (Double vs Float)."""
    import numpy as np

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams

    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    result = model.predict(np.random.default_rng(0).standard_normal((3, 4)))
    assert result.logits.shape == (3, 3)


def test_evaluation_metrics_print_their_values():
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    text = repr(NNEvaluationDataPoint(loss=1.0, extra={"bleu": 0.5}))
    assert "{'bleu': 0.5}" in text and "object at 0x" not in text


def test_an_int_too_large_for_a_float_is_a_field_naming_value_error():
    from nnx import NNOptimParams

    with pytest.raises(ValueError, match="max_lr"):
        NNOptimParams.builder().adam(max_lr=10**400).build()


def test_multi_line_plot_refuses_more_lines_than_dash_styles():
    pytest.importorskip("plotly")
    from nnx.vis_utils import VisUtils

    xs = [0, 1]
    with pytest.raises(ValueError, match="at most 4 lines"):
        VisUtils.multi_line_plot(xs, [[[0, 1]] * 5], "t", (["g"], [str(i) for i in range(5)]), "x", "y")


@pytest.mark.parametrize("filters", ["default", "ignore"])
def test_a_library_warning_from_a_main_without_a_loader_warns_and_never_crashes(filters):
    """`python -c` / a REPL / a notebook `__main__` has no source loader:
    looking its line up raised ImportError (ValueError in IPython) before any
    filter ran, aborting train() / predict() instead of warning."""
    import subprocess
    import sys

    code = (
        f"import warnings; warnings.simplefilter({filters!r})\n"
        "from nnx.nn.callbacks import _warn_at_user_frame\n"
        "_warn_at_user_frame('probe warning', UserWarning)\n"
        "print('survived')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "survived" in done.stdout
    assert ("probe warning" in done.stderr) == (filters == "default")
