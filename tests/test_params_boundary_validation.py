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


@pytest.mark.parametrize("factor", [1.0, 1.5])
def test_a_plateau_factor_of_one_or_more_is_refused_at_construction(factor):
    from nnx import NNSchedulerParams

    with pytest.raises(ValueError, match="factor < 1 for a plateau"):
        NNSchedulerParams(min_lr=1e-7, factor=factor, patience=1, cooldown=0, threshold=1e-3)


@pytest.mark.parametrize("script", ["check_feature_composition.py", "check_export_conformance.py"])
def test_gate_scripts_parse_their_arguments_under_python_oo(script):
    """-OO strips docstrings; the argparse description must not need one."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / script
    if not path.exists():
        pytest.skip("repository scripts are not shipped in the sdist")
    done = subprocess.run([sys.executable, "-OO", str(path), "--help"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("epochs", [["3"], [-1], [2.5], 3, "3"])
def test_model_checkpoint_epochs_that_never_match_are_refused(epochs):
    from nnx.nn.callbacks import ModelCheckpoint

    with pytest.raises(ValueError, match="epochs"):
        ModelCheckpoint(epochs=epochs)
    assert ModelCheckpoint(epochs=[0, 2]).epochs == {0, 2}


def test_public_moe_and_vit_constructors_refuse_float_and_bool_sizes():
    from nnx import MoELinear, ViTNN

    with pytest.raises(ValueError, match="top_k"):
        MoELinear(3, 2, num_experts=4, top_k=2.0)
    with pytest.raises(ValueError, match="top_k"):
        MoELinear(3, 2, num_experts=4, top_k=True)
    with pytest.raises(ValueError, match="ffn_mult"):
        ViTNN(image_size=8, patch_size=4, in_channels=1, d_model=8, n_layers=1, n_heads=2, ffn_mult=1.5)
    with pytest.raises(ValueError, match="attn_dropout"):
        ViTNN(image_size=8, patch_size=4, in_channels=1, d_model=8, n_layers=1, n_heads=2, attn_dropout=True)


def test_a_refused_task_accumulator_leaves_the_training_modes(monkeypatch):
    """The accumulator was built after eval(), outside the restoring try."""
    import torch

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, TaskSpec
    from nnx.nn import nn_model as nn_model_module

    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(2)
        ),
    )
    model.net.train()

    def refuse(adapter, who):
        raise ValueError("no bounded accumulator")

    monkeypatch.setattr(nn_model_module, "_bounded_task_accumulator", refuse)
    with pytest.raises(ValueError, match="no bounded accumulator"):
        nn_model_module._evaluate_task(
            model, [(torch.randn(2, 2), torch.tensor([0, 1]))], None, None, bounded=True, who="t"
        )
    assert model.net.training


def test_numpy_string_labels_are_stored_yaml_safe():
    """np.str_ passes isinstance(str) but yaml.safe_dump refuses it: a run
    with DataFrame-derived labels trained, then failed to save run.yaml."""
    import numpy as np
    import yaml

    from nnx import TaskSpec
    from nnx.prediction import ProbabilitySpec

    labels = list(np.array(["cat", "dog"]))
    for spec in (TaskSpec.categorical(2, labels=labels), ProbabilitySpec(kind="categorical", labels=labels)):
        assert all(type(label) is str for label in spec.labels)
        yaml.safe_dump(spec.state())


def test_bootstrap_accepts_numpy_scalars_as_plain_numbers():
    import numpy as np

    from nnx.comparison import Bootstrap

    boot = Bootstrap(seed=np.int64(3), resamples=np.int32(10), level=np.float32(0.9))
    assert (type(boot.seed), type(boot.resamples), type(boot.level)) == (int, int, float)


def test_decision_results_validate_ids_and_vendor_scores_at_construction():
    from nnx.decisions import ChoiceResult, InvalidDecisionResponse, ScoreResult

    with pytest.raises(InvalidDecisionResponse, match="non-empty strings"):
        ChoiceResult("d", ((1, 0.5), (2, 0.5)))
    with pytest.raises(InvalidDecisionResponse, match="vendor_score"):
        ScoreResult("d", (("a", 0.5), ("b", 0.5)), vendor_score=float("nan"))


@pytest.mark.parametrize(
    "build",
    [
        lambda: __import__("nnx").ViTBlock(d_model=8, n_heads=2, ffn_mult=1.5),
        lambda: __import__("nnx").ViTBlock(d_model=True, n_heads=1),
        lambda: __import__("nnx").ViTBlock(d_model=8, n_heads=2, resid_dropout=True),
        lambda: __import__("nnx").MoELinear(3, 2, num_experts=4.0),
        lambda: __import__("nnx").MoELinear(True, 2, num_experts=4),
    ],
    ids=["vit-block-ffn-mult", "vit-block-bool-width", "vit-block-bool-dropout", "moe-float-experts", "moe-bool-in"],
)
def test_every_public_block_constructor_refuses_float_and_bool_counts(build):
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize("count", [-1, 2.5, True])
def test_generate_refuses_a_max_new_tokens_that_is_not_a_count(count):
    from nnx import GenerativeNNModel

    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerativeNNModel.generate(object(), "hi", max_new_tokens=count)  # type: ignore[arg-type]


@pytest.mark.parametrize("k", [2.5, True, float("nan"), 0])
def test_top_k_filter_refuses_what_its_pipeline_stage_refuses(k):
    from nnx.generation.logits_processors import TopKFilter

    with pytest.raises(ValueError, match="top_k"):
        TopKFilter(k)


def test_numpy_strings_and_bools_in_configs_are_stored_as_plain_values():
    import numpy as np
    import yaml

    from nnx.models import ModelSpec
    from nnx.monitors import MetricSpec
    from nnx.provenance import ExperimentManifest

    spec = ModelSpec("my.enc", 1, {"act": np.str_("relu"), "on": np.True_})
    yaml.safe_dump(spec.state())
    yaml.safe_dump(MetricSpec("acc", config={"k": np.str_("x")}).state())
    plain = ExperimentManifest(config={"on": True, "s": "x"})
    assert ExperimentManifest(config={"on": np.True_, "s": np.str_("x")}).fingerprint() == plain.fingerprint()
    # NumPy-string keys and decision ids too.
    yaml.safe_dump(ModelSpec("my.enc", 1, {np.str_("k"): 1}).state())
    assert (
        ExperimentManifest(config={np.str_("k"): 1}).fingerprint() == ExperimentManifest(config={"k": 1}).fingerprint()
    )
    from nnx.decisions import ChoiceResult

    result = ChoiceResult("d", ((np.str_("a"), 0.5), (np.str_("b"), 0.5)))
    assert all(type(option_id) is str for option_id, _ in result.distribution)


def test_link_categories_refuse_a_bare_string():
    torch = pytest.importorskip("torch")
    from nnx.link_tasks import LinkTaskError, split_links

    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    with pytest.raises(LinkTaskError, match="not the string"):
        split_links(edges, 4, val=1, test=1, seed=0, edge_labels=[0, 1, 1, 0], categories="ab")


@pytest.mark.parametrize("field,value", [("max_prompt_len", True), ("max_response_len", 2.5), ("pad_token_id", -1)])
def test_preference_dataset_validates_lengths_and_pad_id_before_encoding(field, value):
    from nnx import NNPreferenceDataset

    class Refuse:
        def encode(self, text):
            raise AssertionError("encoded before validating")

    with pytest.raises(ValueError, match=field):
        NNPreferenceDataset(prompts=["p"], chosen=["c"], rejected=["r"], tokenizer=Refuse(), **{field: value})


def test_to_onnx_takes_a_float64_numpy_example(tmp_path):
    pytest.importorskip("onnx")
    import numpy as np

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams

    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[3], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.to_onnx(str(tmp_path / "m.onnx"), np.zeros((2, 4)))
    assert (tmp_path / "m.onnx").exists()


def test_checkpoint_to_file_refuses_an_unknown_format_before_creating_a_directory(tmp_path):
    from nnx import NNCheckpoint

    target = tmp_path / "new" / "sub" / "c.pt"
    with pytest.raises(ValueError, match="unknown checkpoint format"):
        NNCheckpoint.to_file(object(), str(target), format="pt")  # type: ignore[arg-type]
    assert not (tmp_path / "new").exists()


def test_validate_response_refuses_a_non_question_as_a_request_error():
    from nnx.decisions import InvalidDecisionRequest, validate_response

    with pytest.raises(InvalidDecisionRequest, match="not a decision question"):
        validate_response("x", 0.5)  # type: ignore[arg-type]


def test_temperature_scaling_does_not_overflow_half_precision_logits():
    """fp16 logits / a small T overflowed to several +inf, which the sampler
    reads as the greedy marker and resolves to the first one, not the argmax."""
    import torch

    from nnx.generation.logits_processors import TemperatureScaling
    from nnx.generation.sampling import sample_next_token

    logits = torch.tensor([[8.0, 10.0, -3.0]], dtype=torch.float16)
    scaled = TemperatureScaling(1e-4)(logits, [])
    assert torch.isfinite(scaled).all()
    assert int(sample_next_token(scaled, generator=torch.Generator().manual_seed(0))) == 1


@pytest.mark.parametrize("kwargs", [{"stop": "END"}, {"stop": [""]}, {"seed": 0.7}, {"seed": True}, {"seed": -1}])
def test_generate_validates_stop_and_seed(kwargs):
    from nnx import GenerativeNNModel

    with pytest.raises(ValueError, match="stop|seed"):
        GenerativeNNModel.generate(object(), "hi", **kwargs)  # type: ignore[arg-type]


def test_softened_kl_refuses_single_logit_and_mismatched_shapes():
    """A softmax over one logit is always 1 (a zero soft loss that trains
    nothing); a mismatched teacher would broadcast."""
    import torch

    from nnx._step_helpers import softened_kl

    with pytest.raises(ValueError, match="at least 2 class logits"):
        softened_kl(torch.randn(4, 1), torch.randn(4, 1), 4.0)
    with pytest.raises(ValueError, match="student's shape"):
        softened_kl(torch.randn(4, 10), torch.randn(4, 1), 4.0)


def test_paradigm_predictions_threshold_a_bce_head():
    import torch

    from nnx._step_helpers import predicted_labels

    logits = torch.tensor([[3.0], [2.0], [1.0], [-2.0]])
    assert predicted_labels(torch.nn.BCEWithLogitsLoss(), logits).flatten().tolist() == [1, 1, 1, 0]
    assert predicted_labels(torch.nn.CrossEntropyLoss(), torch.tensor([[0.0, 1.0]])).tolist() == [1]


def test_experiment_manifest_keeps_one_shot_labels():
    import numpy as np

    from nnx.provenance import ExperimentManifest

    once = ExperimentManifest(labels=(x for x in ["cat", "dog"]))
    assert once.labels == ("cat", "dog")
    assert all(type(label) is str for label in ExperimentManifest(labels=np.array(["cat", "dog"])).labels)
    assert once.fingerprint() == ExperimentManifest(labels=["cat", "dog"]).fingerprint()


def test_wildcard_peft_refuses_a_multihead_attention_projection():
    import torch

    from nnx import apply_lora_to

    class WithAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(8, 2, batch_first=True)
            self.head = torch.nn.Linear(8, 2)

    with pytest.raises(ValueError, match="MultiheadAttention"):
        apply_lora_to(WithAttention(), "*", r=2)


def test_deepen_refuses_a_feed_fwd_subclass_with_its_own_forward():
    from nnx import Activations, deepen
    from nnx.nn.net.feed_fwd_nn import FeedFwdNN

    class Residual(FeedFwdNN):
        def forward(self, x):
            return super().forward(x) + 0.0

    net = Residual(NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU))
    with pytest.raises(ValueError, match="stock forward"):
        deepen(net, after_layer_name="layers.0")


def test_a_registered_run_renders_its_factory():
    from nnx.models import ModelSpec
    from nnx.nn.params.nn_run import _descriptor_fields

    a = dict(_descriptor_fields(ModelSpec("acme.encoder", 1, {"width": 4})))
    b = dict(_descriptor_fields(ModelSpec("other.decoder", 3, {"width": 4})))
    assert a["factory"] == "acme.encoder@v1" and b["factory"] == "other.decoder@v3"


def _binary_ffn(seed: int = 0):
    import torch

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams

    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(input_dim=3, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.BINARY_CROSS_ENTROPY),
    )


def test_alpha_zero_distillation_trains_a_binary_head(tmp_path, monkeypatch):
    """alpha=0 is plain supervised training: the single-logit refusal of the
    soft term must not block it (a cycle-15 regression)."""
    import torch

    from nnx import NNOptimParams, NNTrainParams
    from nnx.paradigms.distillation import feature_kd_train_step_factory, kd_train_step_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    loader = [(torch.randn(4, 3), torch.randint(0, 2, (4, 1)).float())]
    teacher = _binary_ffn(1)
    from nnx import Trainer
    from nnx.objectives import kd_objective
    from nnx.trainer import NNTrainerParams

    run = Trainer(_binary_ffn()).train(
        params=NNTrainerParams.builder()
        .n_epochs(1)
        .train_loader(loader)
        .optimizer("all", NNOptimParams.builder().sgd(max_lr=0.1).build())
        .build(),
        objective=kd_objective(teacher, alpha=0.0),
    )
    assert run.idps and run.idps[-1].train_edp.loss is not None
    for i, step in enumerate(
        (
            kd_train_step_factory(teacher, alpha=0.0),
            feature_kd_train_step_factory(teacher, auxiliary_layers={"layers.0": "layers.0"}, alpha=0.0, beta=0.5),
        )
    ):
        params = NNTrainParams(
            n_epochs=1, train_loader=loader, optim=NNOptimParams.builder().sgd(max_lr=0.1).build(), data_id=f"kd{i}"
        )
        _binary_ffn().train(params=params, train_step_fn=step)


def test_born_again_refuses_a_binary_head_before_generation_zero(tmp_path, monkeypatch):
    import os

    import torch

    from nnx import NNOptimParams, NNTrainParams
    from nnx.paradigms.born_again import born_again_train

    monkeypatch.chdir(tmp_path)
    params = NNTrainParams(
        n_epochs=1,
        train_loader=[(torch.randn(4, 3), torch.randint(0, 2, (4, 1)).float())],
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
    )
    with pytest.raises(ValueError, match="single output"):
        born_again_train(_binary_ffn(), train_params=params, generations=2)
    assert not os.path.exists("runs")


@pytest.mark.parametrize("layer", ["TransformerEncoderLayer", "TransformerDecoderLayer"])
def test_wildcard_peft_refuses_a_transformer_encoder_layer(layer):
    import torch

    from nnx import apply_lora_to

    with pytest.raises(ValueError, match=layer):
        apply_lora_to(getattr(torch.nn, layer)(8, 2, 16, batch_first=True), "linear*", r=2)


@pytest.mark.parametrize("paradigm", ["mixup", "cutmix", "moe"])
def test_paradigm_steps_record_bce_accuracy_by_threshold(paradigm, tmp_path, monkeypatch):
    """A single-logit BCE head with positive logits and all-ones targets is
    right everywhere; an argmax over one logit would score 0."""
    import torch

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNTrainParams
    from nnx.paradigms.augmentation import cutmix_train_step_factory, mixup_train_step_factory
    from nnx.paradigms.moe import moe_train_step_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = NNModel(
        net_params=NNParams(input_dim=12, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.BINARY_CROSS_ENTROPY),
    )
    with torch.no_grad():
        for layer in model.net.layers:
            layer.weight.zero_()
            layer.bias.fill_(5.0)  # every logit positive whatever the input
    X = torch.randn(4, 3, 2, 2) if paradigm == "cutmix" else torch.randn(4, 12)
    step = {
        "mixup": lambda: mixup_train_step_factory(alpha=0.4),
        "cutmix": lambda: cutmix_train_step_factory(alpha=1.0),
        "moe": lambda: moe_train_step_factory(),
    }[paradigm]()
    run = model.train(
        params=NNTrainParams(
            n_epochs=1, train_loader=[(X, torch.ones(4, 1))], optim=NNOptimParams.builder().sgd(max_lr=1e-6).build()
        ),
        train_step_fn=step,
    )
    assert run.idps[0].train_edp.accuracy == pytest.approx(1.0)


def test_forward_benchmarks_digest_modules_with_non_tensor_state():
    import torch

    from nnx.benchmarking import _weights_digest

    class WithExtra(torch.nn.Linear):
        def get_extra_state(self):
            return {"note": "x"}

        def set_extra_state(self, state):
            pass

    assert len(_weights_digest(WithExtra(2, 2))) == 64


def test_export_bundle_refuses_the_best_pointer(tmp_path):
    from nnx.bundles import BundleError, export_bundle

    with pytest.raises(BundleError, match="own id"):
        export_bundle("best", str(tmp_path / "out"))


@pytest.mark.parametrize("execution", [{"meta": {1: 2}}, {"t": (1, 2)}, {1: "a"}])
def test_benchmark_record_execution_must_round_trip(execution):
    from nnx.decisions.benchmark import BenchmarkError, Record

    with pytest.raises(BenchmarkError, match="execution"):
        Record("s1", "d", "p", "answered", distribution=(("a", 0.5), ("b", 0.5)), execution=execution)


def test_sgd_nesterov_needs_momentum_and_scheduler_kind_needs_the_enum():
    from nnx import NNOptimParams, NNSchedulerParams, Optims

    with pytest.raises(ValueError, match="SGD_NESTEROV requires momentum"):
        NNOptimParams(name=Optims.SGD_NESTEROV, max_lr=0.1, weight_decay=0.0, momentum=0.0)
    with pytest.raises(TypeError, match="Schedulers member"):
        NNSchedulerParams(min_lr=0.0, factor=0.5, patience=1, cooldown=0, threshold=0.0, kind="step")  # type: ignore[arg-type]


def test_a_model_checkpoint_tag_shaped_like_a_sidecar_is_not_deleted(tmp_path):
    from nnx.nn.params.nn_checkpoint import _generation_sidecar_paths

    ckpt = tmp_path / "last.pt"
    keep = tmp_path / "last.pt.opt.keep_e1.pt"
    sidecar = tmp_path / ("last.pt.opt." + "a" * 32 + ".pt")
    for path in (ckpt, keep, sidecar):
        path.write_bytes(b"")
    assert _generation_sidecar_paths(str(ckpt)) == [str(sidecar)]


def test_a_recipe_refuses_a_linear_its_parent_reads_directly():
    """The apply_*_to helpers refuse it; a recorded recipe trained, then
    failed in eval (the encoder layer's fast path reads the weight)."""
    import torch

    from nnx import Losses, NNModel, NNModelParams
    from nnx.models import ModelSpec, register_model_factory
    from nnx.transforms import RecipeError, TransformRecipe, lora

    register_model_factory(
        "tests.encoder_layer",
        1,
        lambda config: torch.nn.Sequential(torch.nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)),
        replace=True,
    )
    model = NNModel(params=NNModelParams(net=ModelSpec("tests.encoder_layer", 1), loss=Losses.MEAN_SQUARED_ERROR))
    with pytest.raises(RecipeError, match="reads its weight directly"):
        TransformRecipe((lora("0.linear1", r=2, alpha=4.0),)).validate(model)


@pytest.mark.parametrize("paradigm", ["moe", "distillation", "feature distillation"])
def test_moe_refuses_regression_targets_before_updating(paradigm, tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(tmp_path)  # the refusal comes after the run is admitted

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNTrainParams
    from nnx.paradigms.distillation import feature_kd_train_step_factory, kd_train_step_factory
    from nnx.paradigms.moe import moe_train_step_factory

    def build():
        return NNModel(
            net_params=NNParams(
                input_dim=3, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR),
        )

    model = build()
    step = {
        "moe": lambda: moe_train_step_factory(),
        "distillation": lambda: kd_train_step_factory(build(), alpha=0.5),
        "feature distillation": lambda: feature_kd_train_step_factory(
            build(), auxiliary_layers={"layers.0": "layers.0"}, alpha=0.5, beta=0.5
        ),
    }[paradigm]()
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    with pytest.raises(ValueError, match=f"{paradigm} step scores classification only"):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=[(torch.randn(4, 3), torch.randn(4, 2))],
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            ),
            train_step_fn=step,
        )
    assert all(torch.equal(v, before[k]) for k, v in model.net.state_dict().items())


def test_weighted_accuracy_reads_ignored_and_probability_targets():
    import torch

    from nnx.paradigms.augmentation import _weighted_acc

    y_hat = torch.tensor([0, 1, 1, 0])
    ignored = torch.tensor([0, 1, -100, -100])
    assert _weighted_acc(torch.nn.CrossEntropyLoss(), y_hat, ignored, ignored, 1.0) == pytest.approx(1.0)
    probabilities = torch.nn.functional.one_hot(torch.tensor([0, 1, 1, 0]), 2).float()
    assert _weighted_acc(torch.nn.CrossEntropyLoss(), y_hat, probabilities, probabilities, 0.5) == pytest.approx(1.0)


def test_kd_and_moe_records_skip_ignored_rows(tmp_path, monkeypatch):
    import torch

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNTrainParams
    from nnx.paradigms.distillation import kd_train_step_factory
    from nnx.paradigms.moe import moe_train_step_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    def model():
        net = NNModel(
            net_params=NNParams(
                input_dim=3, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )
        with torch.no_grad():
            for layer in net.net.layers:
                layer.weight.zero_()
                layer.bias.zero_()
            net.net.layers[-1].bias[1] = 5.0  # always predicts class 1
        return net

    loader = [(torch.randn(4, 3), torch.tensor([1, 1, -100, -100]))]
    for i, step in enumerate((kd_train_step_factory(model(), alpha=0.5), moe_train_step_factory())):
        run = model().train(
            params=NNTrainParams(
                n_epochs=1, train_loader=loader, optim=NNOptimParams.builder().sgd(max_lr=1e-9).build(), data_id=f"r{i}"
            ),
            train_step_fn=step,
        )
        assert run.idps[0].train_edp.accuracy == pytest.approx(1.0)  # the two scored rows, not 2 of 4


def test_the_progress_bar_names_a_loss_fallback():
    import torch

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams
    from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint

    class Bar:
        text = ""

        def set_postfix_str(self, text):
            Bar.text = text

    model = NNModel(
        net_params=NNParams(input_dim=2, output_dim=2, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.1)
    model._update_tqdm_postfix(Bar(), optimizer, None, NNEvaluationDataPoint(loss=1.0), None)
    assert Bar.text.startswith("train.loss=")
    model._update_tqdm_postfix(Bar(), optimizer, None, NNEvaluationDataPoint(loss=1.0, error=0.25), None)
    assert Bar.text.startswith("train.error=")


def test_a_parameter_unfrozen_outside_the_optimizer_is_warned_about(tmp_path, monkeypatch):
    """param_groups leaves out parameters frozen at build time: unfreezing
    one later trains nothing, so the loop warns instead of staying silent."""
    import torch

    from nnx import (
        Activations,
        Callback,
        Devices,
        Losses,
        Nets,
        NNModel,
        NNModelParams,
        NNOptimParams,
        NNParamGroupSpec,
        NNTrainParams,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.freeze("layers.0.*")

    class Unfreeze(Callback):
        def on_epoch_end(self, ctx) -> None:
            ctx.model.unfreeze("*")

    optim = NNOptimParams.builder().sgd(max_lr=0.1).param_groups([NNParamGroupSpec(name_pattern="*")]).build()
    with pytest.warns(RuntimeWarning, match="no optimizer owns them"):
        model.train(
            params=NNTrainParams(
                n_epochs=2, train_loader=[(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))], optim=optim
            ),
            callbacks=[Unfreeze()],
        )


@pytest.mark.parametrize("paradigm", ["mixup", "cutmix"])
def test_mixup_and_cutmix_refuse_regression_targets_before_updating(paradigm, tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(tmp_path)  # the refusal comes after the run is admitted

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNTrainParams
    from nnx.paradigms.augmentation import cutmix_train_step_factory, mixup_train_step_factory

    model = NNModel(
        net_params=NNParams(input_dim=12, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR),
    )
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    X = torch.randn(4, 3, 2, 2) if paradigm == "cutmix" else torch.randn(4, 12)
    step = mixup_train_step_factory() if paradigm == "mixup" else cutmix_train_step_factory()
    with pytest.raises(ValueError, match=f"{paradigm} step scores classification only"):
        model.train(
            params=NNTrainParams(
                n_epochs=1, train_loader=[(X, torch.randn(4, 1))], optim=NNOptimParams.builder().sgd(max_lr=0.1).build()
            ),
            train_step_fn=step,
        )
    assert all(torch.equal(v, before[k]) for k, v in model.net.state_dict().items())


def test_int32_categorical_targets_are_scored(tmp_path, monkeypatch):
    import torch

    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, TaskSpec

    model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=3, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(3)
        ),
    )
    record = model.evaluate([(torch.randn(4, 3), torch.tensor([0, 1, 2, 1], dtype=torch.int32))])
    assert record.loss is not None


def test_integral_bounds_are_shown_exactly():
    from nnx.models import ModelSpec

    with pytest.raises(ValueError, match="<= 4294967295"):
        ModelSpec("a", 1, seed=2**32)


def test_benchmarks_digest_torchao_quantized_models():
    import torch

    pytest.importorskip("torchao")
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams
    from nnx.benchmarking import _weights_digest, benchmark_forward
    from nnx.quantize import quantize_int8

    def build(seed):
        torch.manual_seed(seed)
        return NNModel(
            net_params=NNParams(
                input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU
            ),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )

    qa, qb = quantize_int8(build(0)), quantize_int8(build(1))
    assert _weights_digest(qa.net) != _weights_digest(qb.net)
    assert benchmark_forward(qa.net, torch.randn(2, 4), warmup=0, repeats=1).weights_digest == _weights_digest(qa.net)


def test_the_unowned_trainables_warning_reaches_trainer_runs_at_the_callers_line(tmp_path, monkeypatch):
    import warnings

    import torch

    from nnx import (
        Activations,
        Callback,
        Devices,
        Losses,
        Nets,
        NNModel,
        NNModelParams,
        NNOptimParams,
        NNParamGroupSpec,
        Trainer,
    )
    from nnx.objectives import supervised_objective
    from nnx.trainer import NNTrainerParams

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    model.freeze("layers.0.*")

    class Unfreeze(Callback):
        def on_epoch_end(self, ctx) -> None:
            ctx.model.unfreeze("*")

    params = (
        NNTrainerParams.builder()
        .n_epochs(2)
        .train_loader([(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))])
        .optimizer(
            "all",
            NNOptimParams.builder().sgd(max_lr=0.1).param_groups([NNParamGroupSpec(name_pattern="*")]).build(),
        )
        .build()
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Trainer(model).train(params=params, objective=supervised_objective(), callbacks=[Unfreeze()])
    hits = [w for w in caught if "no optimizer owns them" in str(w.message)]
    assert hits and hits[0].filename == __file__


def test_benchmark_digests_sparse_state():
    import torch

    from nnx.benchmarking import _weights_digest

    net = torch.nn.Linear(3, 3)
    net.register_buffer("mask", torch.eye(3).to_sparse())
    assert len(_weights_digest(net)) == 64


def test_a_tiny_fp32_temperature_still_picks_the_argmax():
    import torch

    from nnx.generation.logits_processors import TemperatureScaling
    from nnx.generation.sampling import sample_next_token

    scaled = TemperatureScaling(1e-39)(torch.tensor([[1.0, 2.0, 0.5]]), [])
    assert int(sample_next_token(scaled, generator=torch.Generator().manual_seed(0))) == 1


@pytest.mark.parametrize("groupsize", [True, 16.0, 0])
def test_qat_groupsize_must_be_a_positive_int(groupsize):
    pytest.importorskip("torchao")
    from nnx.quantize.qat import QATLifecycleCallback

    with pytest.raises(ValueError, match="groupsize"):
        QATLifecycleCallback(groupsize=groupsize)


def test_paradigm_steps_refuse_sequence_output():
    import torch

    from nnx._step_helpers import require_row_logits

    with pytest.raises(ValueError, match="mixup step scores"):
        require_row_logits(torch.zeros(2, 3, 5), "mixup")
    require_row_logits(torch.zeros(2, 5), "mixup")


def test_a_refused_prefix_tuner_leaves_the_model_trainable():
    import torch

    from nnx import NNTransformerParams, PrefixTuner
    from nnx.nn.net.transformer_nn import TransformerNN

    net = TransformerNN(
        NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            max_seq_len=8,
        )
    )
    converted = torch.nn.Module()  # a converted layer holding only an integer weight buffer
    converted.register_buffer("weight", torch.zeros(24, 8, dtype=torch.int8))
    net.blocks[0].attn.w_qkv = converted
    with pytest.raises(TypeError, match="no floating weight"):
        PrefixTuner(net, n_prefix=2)
    assert all(p.requires_grad for p in net.parameters() if p.is_floating_point())


def test_embed_texts_returns_on_the_requested_device_without_moving_the_backbone():
    import torch

    from nnx.embeddings import embed_texts

    class Bag(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 4)

        def forward(self, texts):
            return self.lin(torch.ones(len(texts), 3))

    backbone = Bag()
    out = embed_texts(backbone, ["a", "b"], device="meta")
    assert out.device.type == "meta" and next(backbone.parameters()).device.type == "cpu"


def test_qat_refuses_a_groupsize_that_prepares_nothing(tmp_path, monkeypatch):
    import torch

    pytest.importorskip("torchao")
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNTrainParams
    from nnx.quantize.qat import QATLifecycleCallback

    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=2, hidden_dims=[5], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    with pytest.raises(ValueError, match="prepared nothing"):
        model.train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=[(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))],
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            ),
            callbacks=[QATLifecycleCallback(groupsize=32)],
        )


def test_tsne_refuses_a_regression_checkpoint_before_inference(tmp_path, monkeypatch):
    import torch

    pytest.importorskip("plotly")
    from nnx import (
        Activations,
        Checkpoints,
        Devices,
        Losses,
        Nets,
        NNCheckpoint,
        NNModel,
        NNModelParams,
        NNOptimParams,
        NNTrainParams,
        TaskSpec,
    )
    from nnx.vis_utils import VisUtils

    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNParams(input_dim=3, output_dim=1, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.regression(1)
        ),
    )
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=[(torch.randn(4, 3), torch.randn(4, 1))],
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        )
    )

    class NeverIterated:
        output_dim = 1

        @property
        def test_loader(self):
            raise AssertionError("inference ran before the refusal")

    with pytest.raises(ValueError, match="colours points by class"):
        VisUtils.two_dim_tsne_checkpoint_logits(
            NNCheckpoint.load(run=run.id, type=Checkpoints.LAST), NeverIterated(), n_samples=4
        )


@pytest.mark.parametrize("paradigm", ["mixup", "cutmix", "distillation", "feature distillation", "moe"])
def test_every_classification_paradigm_step_refuses_sequence_output(paradigm, tmp_path, monkeypatch):
    import torch

    from nnx import Losses, NNModel, NNModelParams, NNOptimParams, NNTrainParams
    from nnx.paradigms.augmentation import cutmix_train_step_factory, mixup_train_step_factory
    from nnx.paradigms.distillation import feature_kd_train_step_factory, kd_train_step_factory
    from nnx.paradigms.moe import moe_train_step_factory

    monkeypatch.chdir(tmp_path)

    class Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(12, 5)])

        def forward(self, x):
            return self.layers[0](x.reshape(x.shape[0], -1)).unsqueeze(1).expand(-1, 3, -1)  # (B, T, V)

    def build():
        return NNModel(module=Seq(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))

    step = {
        "mixup": lambda: mixup_train_step_factory(),
        "cutmix": lambda: cutmix_train_step_factory(),
        "distillation": lambda: kd_train_step_factory(build(), alpha=0.5),
        "feature distillation": lambda: feature_kd_train_step_factory(
            build(), auxiliary_layers={"layers.0": "layers.0"}, alpha=0.5, beta=0.5
        ),
        "moe": lambda: moe_train_step_factory(),
    }[paradigm]()
    X = torch.randn(4, 3, 2, 2) if paradigm == "cutmix" else torch.randn(4, 12)
    with pytest.raises(ValueError, match=r"scores \(rows, classes\) logits"):
        build().train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=[(X, torch.tensor([0, 1, 0, 1]))],
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            ),
            train_step_fn=step,
        )


def test_a_prefix_tuner_places_prefixes_on_a_factorized_projection():
    import torch

    from nnx import NNTransformerParams, PrefixTuner
    from nnx.nn.net.transformer_nn import TransformerNN
    from nnx.surgery import low_rank_factorize

    net = TransformerNN(
        NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            max_seq_len=8,
        )
    )
    net.blocks[0].attn.w_qkv = low_rank_factorize(net.blocks[0].attn.w_qkv, rank=4)
    tuner = PrefixTuner(net, n_prefix=2)
    assert tuner.prefix_keys[0].dtype == torch.float32
    tuner(torch.randint(0, 16, (1, 4)))


def test_sparse_digests_tell_values_apart():
    import torch

    from nnx.benchmarking import _weights_digest

    def net(scale):
        module = torch.nn.Linear(3, 3)
        torch.nn.init.zeros_(module.weight)
        torch.nn.init.zeros_(module.bias)
        module.register_buffer("mask", (torch.eye(3) * scale).to_sparse())
        return module

    assert _weights_digest(net(1.0)) == _weights_digest(net(1.0))
    assert _weights_digest(net(1.0)) != _weights_digest(net(2.0))


def _tiny_classifier(output_dim=2):
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams

    return NNModel(
        net_params=NNParams(
            input_dim=3, output_dim=output_dim, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _one_epoch(**extra):
    import torch

    from nnx import NNOptimParams, NNTrainParams

    return NNTrainParams(
        n_epochs=1,
        train_loader=[(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))],
        optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        **extra,
    )


def test_kd_objective_refuses_sequence_output_even_when_time_equals_vocab(tmp_path, monkeypatch):
    import torch

    from nnx import Losses, NNModel, NNModelParams, NNOptimParams, NNTrainParams, kd_objective

    monkeypatch.chdir(tmp_path)

    class Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 8)

        def forward(self, x):
            return self.lin(x).unsqueeze(1).expand(-1, 8, -1)  # (B, T=8, V=8)

    def build():
        return NNModel(module=Seq(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))

    with pytest.raises(ValueError, match=r"kd_objective step scores \(rows, classes\) logits"):
        build().train(
            params=NNTrainParams(
                n_epochs=1,
                train_loader=[(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))],
                optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            ),
            objective=kd_objective(build(), alpha=0.5),
        )


def test_training_refuses_a_pruning_reparameterization_its_checkpoints_cannot_rebuild(tmp_path, monkeypatch):
    from nnx.prune import magnitude_prune

    monkeypatch.chdir(tmp_path)
    model = _tiny_classifier()
    magnitude_prune(model.net, 0.5, bake=False)
    with pytest.raises(ValueError, match="pruning reparameterization"):
        model.train(params=_one_epoch())
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_unbaked_pruning_refuses_a_linear_its_parent_reads_directly():
    import torch

    from nnx.prune import magnitude_prune

    net = torch.nn.Sequential(torch.nn.MultiheadAttention(8, 2), torch.nn.Linear(8, 2))
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with pytest.raises(ValueError, match=r"0\.out_proj are read by their parent"):
        magnitude_prune(net, 0.5, bake=False)
    assert net.state_dict().keys() == before.keys()
    assert magnitude_prune(net, 0.5, bake=False, layer_pattern="1") == 1


def test_qat_names_the_root_linear_case_when_nothing_was_prepared(tmp_path, monkeypatch):
    import torch

    pytest.importorskip("torchao")
    from nnx import Losses, NNModel, NNModelParams
    from nnx.quantize.qat import QATLifecycleCallback

    monkeypatch.chdir(tmp_path)
    model = NNModel(module=torch.nn.Linear(64, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    import nnx

    params = nnx.NNTrainParams(
        n_epochs=1,
        train_loader=[(torch.randn(4, 64), torch.tensor([0, 1, 0, 1]))],
        optim=nnx.NNOptimParams.builder().sgd(max_lr=0.1).build(),
    )
    with pytest.raises(ValueError, match="itself one nn.Linear"):
        model.train(params=params, callbacks=[QATLifecycleCallback(groupsize=32)])


def test_a_tiny_temperature_on_negative_logits_still_picks_the_argmax():
    import torch

    from nnx.generation.logits_processors import TemperatureScaling
    from nnx.generation.sampling import sample_next_token

    scaled = TemperatureScaling(1e-38)(torch.tensor([[-6.0, -5.0, -7.0]]), [])
    assert int(sample_next_token(scaled, generator=torch.Generator().manual_seed(0))) == 1


@pytest.mark.parametrize("feature", [False, True], ids=["kd", "feature_kd"])
def test_pure_distillation_ignores_a_fully_masked_label_batch(feature, tmp_path, monkeypatch):
    import torch

    from nnx import NNOptimParams, NNTrainParams
    from nnx.paradigms.distillation import feature_kd_train_step_factory, kd_train_step_factory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    teacher = _tiny_classifier()
    step = (
        feature_kd_train_step_factory(teacher, auxiliary_layers={"layers.0": "layers.0"}, alpha=1.0, beta=0.5)
        if feature
        else kd_train_step_factory(teacher, alpha=1.0)
    )
    run = _tiny_classifier().train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=[(torch.randn(4, 3), torch.full((4,), -100))],
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
        ),
        train_step_fn=step,
    )
    assert run is not None


def test_weight_digests_tell_shapes_apart():
    import torch

    from nnx.benchmarking import _weights_digest

    def net(shape):
        module = torch.nn.Module()
        module.register_buffer("table", torch.arange(6).reshape(shape))
        return module

    assert _weights_digest(net((2, 3))) != _weights_digest(net((3, 2)))


@pytest.mark.parametrize("kind", ["legacy_weight_norm", "weight_norm", "spectral_norm"])
def test_training_refuses_a_parametrized_net_its_checkpoints_cannot_reload(kind, tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(tmp_path)
    model = _tiny_classifier()
    layer = model.net.layers[0]
    if kind == "legacy_weight_norm":
        torch.nn.utils.weight_norm(layer)
    else:
        getattr(torch.nn.utils.parametrizations, kind)(layer)
    with pytest.raises(ValueError, match="layers.0.weight are reparameterized"):
        model.train(params=_one_epoch())


def test_a_mask_value_below_a_normal_temperature_keeps_sampling():
    import torch

    from nnx.generation.logits_processors import TemperatureScaling

    floor = torch.finfo(torch.float32).min
    scaled = TemperatureScaling(0.5)(torch.tensor([[5.0, 4.9, floor]]), [])
    assert torch.isfinite(scaled[0, :2]).all() and scaled[0, 2] == float("-inf")  # not greedy


@pytest.mark.parametrize("lr", [0.0, -1e-3, float("nan"), True])
def test_train_contrastive_refuses_a_learning_rate_that_cannot_train(lr):
    import torch

    from nnx.embeddings import train_contrastive

    with pytest.raises(ValueError, match="lr must be positive"):
        train_contrastive(torch.nn.Linear(2, 2), [("a", "b"), ("c", "d")], lr=lr)


def test_distillation_refuses_flat_single_logit_heads():
    import torch

    from nnx._step_helpers import softened_kl

    with pytest.raises(ValueError, match=r"\(rows, classes\) logits"):
        softened_kl(torch.randn(8), torch.randn(8), 4.0)


def test_weight_digests_tell_per_channel_axes_apart():
    import torch

    from nnx.benchmarking import _weights_digest

    def net(axis):
        module = torch.nn.Module()
        values = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        scales, zero_points = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.int64)
        module.register_buffer("q", torch.quantize_per_channel(values, scales, zero_points, axis, torch.qint8))
        return module

    assert _weights_digest(net(0)) != _weights_digest(net(1))


def test_born_again_refuses_a_single_output_runtime_module_before_generation_zero(tmp_path, monkeypatch):
    import torch

    from nnx import Losses, NNModel, NNModelParams, born_again_train

    monkeypatch.chdir(tmp_path)
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    model = NNModel(module=module, params=NNModelParams(loss=Losses.BINARY_CROSS_ENTROPY))
    with pytest.raises(ValueError, match="single output"):
        born_again_train(model, train_params=_one_epoch(), generations=2)
    assert not (tmp_path / "runs").exists()


def test_train_contrastive_accepts_a_numpy_learning_rate():
    import numpy as np

    from nnx.embeddings import train_contrastive

    with pytest.raises(Exception) as caught:  # object() is no backbone: fails after the lr check
        train_contrastive(object(), [("a", "b"), ("c", "d")], lr=np.float32(1e-3))
    assert "lr must be positive" not in str(caught.value)


def test_loggers_name_a_declared_metric_that_shadows_a_record_field_apart():
    from nnx import NNEvaluationDataPoint
    from nnx.nn.callbacks import _edp_metric_iter

    edp = NNEvaluationDataPoint(loss=1.0, accuracy=0.0, metrics={"accuracy": 0.25, "nll": 0.5})
    pairs = list(_edp_metric_iter(edp))
    assert ("accuracy", 0.0) in pairs and ("metrics/accuracy", 0.25) in pairs and ("nll", 0.5) in pairs
    assert [name for name, _ in pairs].count("accuracy") == 1


def test_a_failed_attempt_record_never_replaces_the_training_error(monkeypatch):
    import warnings

    from nnx.provenance import _AttemptRecorder

    def broken(self, status, error):
        raise OSError("disk gone")

    monkeypatch.setattr(_AttemptRecorder, "_finish", broken)
    recorder = object.__new__(_AttemptRecorder)
    recorder.attempt = type("A", (), {"attempt_id": "a1"})()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recorder.fail(IndexError("the training error"))  # neither raises nor replaces it


def test_an_explicit_plateau_scheduler_prints_its_plateau_settings():
    from nnx import NNSchedulerParams, Schedulers

    text = str(
        NNSchedulerParams(
            kind=Schedulers.REDUCE_LR_ON_PLATEAU, min_lr=1e-5, factor=0.5, patience=7, cooldown=0, threshold=1e-4
        )
    )
    assert "patience=7" in text and "cooldown=" in text and "threshold=" in text


def test_multi_line_plot_ticks_follow_the_x_values():
    pytest.importorskip("plotly")
    from nnx.vis_utils import VisUtils

    fig = VisUtils.multi_line_plot(
        [10, 20, 30, 40],
        [[[1, 2, 3, 4], [2, 3, 4, 5]]],
        "t",
        (["g"], ["train", "val"]),
        "epoch",
        "loss",
        x_ticks_inc=2,
        renderer=None,
    )
    assert tuple(fig.layout.xaxis.tickvals) == (10, 30)


def test_born_again_does_not_guess_a_head_from_registration_order(tmp_path, monkeypatch):
    import torch

    from nnx import Losses, NNModel, NNModelParams, born_again_train

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    class Gated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(3, 8)
            self.head = torch.nn.Linear(8, 2)
            self.gate = torch.nn.Linear(8, 1)  # registered last, never the output

        def forward(self, x):
            return self.head(torch.relu(self.body(x)))

    model = NNModel(module=Gated(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    runs = born_again_train(model, train_params=_one_epoch(), generations=2)
    assert len(runs) == 2


def test_a_legacy_spectral_norm_is_reported_as_a_parametrization_not_a_prune_mask(tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(tmp_path)
    model = _tiny_classifier()
    torch.nn.utils.spectral_norm(model.net.layers[0])
    with pytest.raises(ValueError, match="layers.0.weight are reparameterized") as caught:
        model.train(params=_one_epoch())
    assert "remove_spectral_norm" in str(caught.value) and "pruning" not in str(caught.value)


def test_born_again_refuses_a_sequence_model_before_generation_zero(tmp_path, monkeypatch):
    from nnx import Devices, Losses, Nets, NNModel, NNModelParams, NNTransformerParams, born_again_train

    monkeypatch.chdir(tmp_path)
    model = NNModel(
        net_params=NNTransformerParams(
            input_dim=16,
            output_dim=16,
            dropout_prob=0.0,
            vocab_size=16,
            n_layers=1,
            n_heads=2,
            d_model=8,
            max_seq_len=8,
        ),
        params=NNModelParams(net=Nets.TRANSFORMER, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    with pytest.raises(ValueError, match="outputs a sequence"):
        born_again_train(model, train_params=_one_epoch(), generations=2)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("pad", [True, -1])
def test_dpo_refuses_a_pad_id_that_masks_nothing(pad):
    import torch

    from nnx import Losses, NNModel, NNModelParams
    from nnx.paradigms.dpo import dpo_train_step_factory

    ref = NNModel(module=torch.nn.Linear(2, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    with pytest.raises(ValueError, match="pad_token_id"):
        dpo_train_step_factory(ref, pad_token_id=pad)


@pytest.mark.parametrize(
    "call",
    [
        lambda: __import__("nnx").surgery.low_rank_factorize(__import__("torch").nn.Linear(4, 4), rank=2.0),
        lambda: __import__("nnx").surgery.low_rank_factorize(__import__("torch").nn.Linear(4, 4), rank=True),
        lambda: __import__("nnx").surgery.expand_embedding(
            __import__("torch").nn.Embedding(4, 2), new_num_embeddings=8.0
        ),
        lambda: __import__("nnx").surgery.widen(
            __import__("torch").nn.Sequential(
                __import__("torch").nn.Linear(2, 4), __import__("torch").nn.ReLU(), __import__("torch").nn.Linear(4, 2)
            ),
            layer_name="0",
            new_width=8.0,
        ),
    ],
    ids=["rank_float", "rank_bool", "embeddings_float", "widen_float"],
)
def test_surgery_sizes_must_be_whole_numbers(call):
    with pytest.raises(ValueError, match="rank|new_num_embeddings|new_width"):
        call()


def test_a_user_frame_warning_from_a_script_survives_an_error_filter(tmp_path):
    import subprocess
    import sys

    script = tmp_path / "train_script.py"  # a script's __main__ has no __spec__
    script.write_text(
        "import warnings\n"
        "from nnx.nn.callbacks import _warn_at_user_frame\n"
        "warnings.simplefilter('error')\n"
        "try:\n"
        "    _warn_at_user_frame('real message')\n"
        "except RuntimeWarning as caught:\n"
        "    assert 'real message' in str(caught), caught\n"
        "else:\n"
        "    raise AssertionError('no warning')\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True, timeout=120)


def test_a_moe_forward_reads_resolved_activations_not_the_params():
    import torch

    from nnx import Activations
    from nnx.nn.net.feed_fwd_moe_nn import FeedFwdMoENN
    from nnx.nn.params.nn_moe_params import NNMoEParams

    params = NNMoEParams(
        input_dim=3,
        output_dim=2,
        hidden_dims=[4, 4],
        dropout_prob=0.0,
        activation=Activations.RELU,
        num_experts=2,
        top_k=1,
    )
    net = FeedFwdMoENN(params)
    assert len(net._hidden) == 2
    net.__dict__["_params"] = None  # a forward that touched the params object would fail
    assert net(torch.randn(5, 3)).shape == (5, 2)


def test_a_confusion_matrix_without_names_labels_the_classes_present():
    pytest.importorskip("plotly")
    from nnx.vis_utils import VisUtils

    fig = VisUtils.confusion_matrix([0, 2, 2], [0, 2, 0])
    assert list(fig.data[0].x) == [0, 2] and list(fig.data[0].y) == [0, 2]


def test_a_standardizer_call_on_a_dataframe_checks_the_column_order():
    import pandas as pd

    from nnx.preprocessing import PreprocessingError, Standardizer

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 40.0]})
    standardizer = Standardizer.fit(frame, rows=range(3))
    with pytest.raises(PreprocessingError, match="column order"):
        standardizer(frame[["b", "a"]])


def test_activation_map_takes_the_first_tensor_of_a_mapping_output():
    import torch

    pytest.importorskip("plotly")
    from nnx.viz import activation_map

    class Keyed(torch.nn.Module):
        def forward(self, x):
            return {"out": x * 2}

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.keyed = Keyed()

        def forward(self, x):
            return self.keyed(x)["out"]

    fig = activation_map(Net(), torch.randn(2, 3), "keyed")
    assert fig is not None


def test_an_explicit_default_train_step_records_the_declared_training_metrics(tmp_path, monkeypatch):
    from nnx import MetricSpec, MonitorSpec
    from nnx.nn.nn_model import default_train_step

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    params = _one_epoch(metrics=[MetricSpec("accuracy")], monitor=MonitorSpec(metric="accuracy", split="train"))
    run = _tiny_classifier().train(params=params, train_step_fn=default_train_step)  # as qat_train_step_factory returns
    summary = run.idps[-1].train_summary
    assert summary is not None and "accuracy" in summary.metrics


def test_an_unscorable_batch_on_one_rank_stops_every_rank():
    import torch

    from nnx.nn.nn_model import _agreed_scorable_target

    class _OtherRankRefused:
        def gather(self, value):
            return [value, True]  # rank 1 refused its batch

    with pytest.raises(ValueError, match=r"rank\(s\) \[1\]"):
        _agreed_scorable_target(_OtherRankRefused(), torch.nn.MSELoss(), torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize("beta", [True, "0.1", None])
def test_dpo_refuses_a_beta_that_is_not_a_real_number(beta):
    import torch

    from nnx import Losses, NNModel, NNModelParams
    from nnx.paradigms.dpo import dpo_train_step_factory

    ref = NNModel(module=torch.nn.Linear(2, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    with pytest.raises(ValueError, match="beta"):
        dpo_train_step_factory(ref, beta=beta)
