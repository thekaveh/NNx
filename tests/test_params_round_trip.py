"""Contract test: every params dataclass with `state()` / `from_state()`
serialization must round-trip — `obj == from_state(state())`.

Drift here is silent and catastrophic (saved runs that won't reload),
so this test exists to fail loudly when fields are added/renamed without
keeping both sides of the contract in sync."""
from __future__ import annotations

from nnx.nn.enum.activations import Activations
from nnx.nn.enum.devices import Devices
from nnx.nn.enum.losses import Losses
from nnx.nn.enum.nets import Nets
from nnx.nn.enum.optims import Optims
from nnx.nn.enum.schedulers import Schedulers
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint
from nnx.nn.params.nn_model_params import NNModelParams
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_params import NNParams
from nnx.nn.params.nn_scheduler_params import NNSchedulerParams
from nnx.nn.params.nn_train_params import NNTrainParams


def test_nn_params_round_trip():
    obj = NNParams(
        input_dim=784, output_dim=10, dropout_prob=0.2,
        activation=Activations.RELU, hidden_dims=[128, 64], n_heads=None,
    )
    assert NNParams.from_state(obj.state()) == obj


def test_nn_params_round_trip_with_n_heads():
    obj = NNParams(
        input_dim=8, output_dim=3, dropout_prob=0.0,
        activation=Activations.LEAKY_RELU, hidden_dims=None, n_heads=4,
    )
    assert NNParams.from_state(obj.state()) == obj


def test_nn_model_params_round_trip():
    obj = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    assert NNModelParams.from_state(obj.state()) == obj


def test_nn_model_params_round_trip_with_mixed_precision():
    obj = NNModelParams(
        net=Nets.GRAPH_CONV, device=Devices.CUDA, loss=Losses.NEGATIVE_LOG_LIKELIHOOD,
        mixed_precision=True,
    )
    assert NNModelParams.from_state(obj.state()) == obj


def test_nn_optim_params_round_trip_sgd():
    obj = NNOptimParams(name=Optims.SGD, max_lr=1e-2, momentum=0.9, weight_decay=5e-5)
    assert NNOptimParams.from_state(obj.state()) == obj


def test_nn_optim_params_round_trip_adam():
    obj = NNOptimParams(name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=0.0)
    assert NNOptimParams.from_state(obj.state()) == obj


def test_nn_optim_params_round_trip_with_param_groups():
    from nnx import NNParamGroupSpec

    obj = NNOptimParams(
        name=Optims.ADAM, max_lr=1e-3, momentum=(0.9, 0.999), weight_decay=5e-4,
        param_groups=[
            NNParamGroupSpec(name_pattern="encoder.*", lr_multiplier=0.01),
            NNParamGroupSpec(name_pattern="*.bias", weight_decay=0.0),
        ],
    )
    rt = NNOptimParams.from_state(obj.state())
    assert rt == obj


def test_nn_param_group_spec_round_trip():
    from nnx import NNParamGroupSpec

    cases = [
        NNParamGroupSpec(name_pattern="*"),
        NNParamGroupSpec(name_pattern="encoder.*", lr=1e-5),
        NNParamGroupSpec(name_pattern="*.bias", lr_multiplier=0.1),
        NNParamGroupSpec(name_pattern="head.*", lr=1e-3, weight_decay=0.0),
    ]
    for spec in cases:
        assert NNParamGroupSpec.from_state(spec.state()) == spec


def test_nn_scheduler_params_round_trip_plateau():
    obj = NNSchedulerParams(
        min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3,
    )
    rt = NNSchedulerParams.from_state(obj.state())
    assert rt == obj
    assert rt.kind is None


def test_nn_scheduler_params_round_trip_cosine():
    obj = NNSchedulerParams(
        min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3,
        kind=Schedulers.COSINE_ANNEALING, T_max=100,
    )
    rt = NNSchedulerParams.from_state(obj.state())
    assert rt == obj
    assert rt.kind == Schedulers.COSINE_ANNEALING


def test_nn_train_params_round_trip():
    obj = NNTrainParams(
        n_epochs=10,
        optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=5e-5),
        scheduler=NNSchedulerParams(
            min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3,
        ),
    )
    # train_loader/val_loader live on the dataclass but are repr=False and
    # not serialized into state(), so a from_state() reconstruction won't
    # have them set — equality holds because they default to None on both sides.
    assert NNTrainParams.from_state(obj.state()) == obj


def test_evaluation_data_point_round_trip_full():
    obj = NNEvaluationDataPoint(
        loss=0.42, error=0.15, accuracy=0.85, f1=0.84, recall=0.83, precision=0.86,
    )
    assert NNEvaluationDataPoint.from_state(obj.state()) == obj


def test_evaluation_data_point_round_trip_no_loss_no_error():
    obj = NNEvaluationDataPoint(accuracy=0.7, f1=0.7, recall=0.7, precision=0.7)
    rt = NNEvaluationDataPoint.from_state(obj.state())
    assert rt.loss is None
    assert rt.error is None
    assert rt == obj


def test_nn_trainer_params_round_trip():
    """NNTrainerParams must round-trip — multi-optim dict serializes
    deterministically, schedulers default to empty, seed honors
    omit-when-default."""
    from nnx import NNParamGroupSpec, NNTrainerParams

    obj = NNTrainerParams(
        n_epochs=4,
        optims={
            "G": NNOptimParams(
                name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
                param_groups=[NNParamGroupSpec(name_pattern="G.*", lr=2e-4)],
            ),
            "D": NNOptimParams(
                name=Optims.ADAM, max_lr=2e-4, momentum=(0.5, 0.999), weight_decay=0.0,
                param_groups=[NNParamGroupSpec(name_pattern="D.*", lr=2e-4)],
            ),
        },
        schedulers={
            "G": NNSchedulerParams(min_lr=1e-7, factor=0.5, patience=2, cooldown=1, threshold=1e-3),
        },
        seed=42,
    )
    rt = NNTrainerParams.from_state(obj.state())
    assert rt == obj


def test_nn_model_params_state_omits_mixed_precision_when_false():
    """CRITICAL back-compat invariant: NNModelParams with
    mixed_precision=False (the default) must emit the same state() it
    did before this field existed — otherwise every existing run.id
    shifts. Mirrors the param_groups / trainer omit-when-default pattern."""
    obj = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    state = obj.state()
    assert "mixed_precision" not in state, (
        "mixed_precision=False must be omitted from state() to preserve run.id "
        f"back-compat; got {state!r}"
    )
    assert set(state.keys()) == {"net", "loss", "device"}


def test_nn_model_params_state_emits_mixed_precision_when_true():
    obj = NNModelParams(
        net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY,
        mixed_precision=True,
    )
    state = obj.state()
    assert state.get("mixed_precision") is True
    # Round-trip still works.
    rt = NNModelParams.from_state(state)
    assert rt == obj


def test_nn_scheduler_params_state_omits_kind_when_none():
    """CRITICAL back-compat invariant: a plain ReduceLROnPlateau
    NNSchedulerParams (the only scheduler before the Schedulers enum
    existed) must emit the same state() it did before — otherwise
    every existing run.id shifts. Same omit-when-default pattern."""
    obj = NNSchedulerParams(
        min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3,
    )
    state = obj.state()
    assert "kind" not in state
    assert "step_size" not in state
    assert "T_max" not in state
    assert "max_lr" not in state
    assert "total_steps" not in state
    assert "warmup_steps" not in state
    assert set(state.keys()) == {"min_lr", "factor", "cooldown", "patience", "threshold"}


def test_nn_scheduler_params_state_emits_kind_when_set():
    """When kind is set, both kind and its variant-specific knob round-trip."""
    obj = NNSchedulerParams(
        min_lr=1e-7, factor=0.5, patience=5, cooldown=2, threshold=1e-3,
        kind=Schedulers.COSINE_ANNEALING, T_max=100,
    )
    state = obj.state()
    assert state.get("kind") == "cosine_annealing"
    assert state.get("T_max") == 100
    rt = NNSchedulerParams.from_state(state)
    assert rt == obj


def test_nn_run_state_omits_trainer_when_none():
    """CRITICAL back-compat invariant: NNRun built without a trainer
    field (the NNModel.train path) must emit the same state() — and
    therefore the same run.id — as before this field existed."""
    from nnx.nn.params.nn_run import NNRun

    run = NNRun(
        net=NNParams(
            input_dim=4, output_dim=2, dropout_prob=0.0,
            activation=Activations.RELU, hidden_dims=[8],
        ),
        train=NNTrainParams(n_epochs=1),
        model=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )
    state = run.state()
    assert "trainer" not in state, (
        "NNRun with trainer=None must omit the key to preserve existing run.id hashes; "
        f"got keys {sorted(state.keys())}"
    )


def test_iteration_data_point_round_trip_with_val():
    train_edp = NNEvaluationDataPoint(
        loss=0.5, error=0.2, accuracy=0.8, f1=0.79, recall=0.78, precision=0.81,
    )
    val_edp = NNEvaluationDataPoint(
        loss=0.6, error=0.25, accuracy=0.75, f1=0.74, recall=0.73, precision=0.76,
    )
    obj = NNIterationDataPoint(
        lr=1e-3, iter_idx=10, epoch_idx=1, batch_idx=5,
        train_edp=train_edp, val_edp=val_edp,
    )
    flat = {
        'lr': obj.lr, 'iter_idx': obj.iter_idx,
        'epoch_idx': obj.epoch_idx, 'batch_idx': obj.batch_idx,
    }
    for k, v in obj.train_edp.state().items():
        flat[f'train_edp.{k}'] = v
    for k, v in obj.val_edp.state().items():
        flat[f'val_edp.{k}'] = v
    assert NNIterationDataPoint.from_state(flat) == obj
