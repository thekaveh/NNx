"""FEAT-016: replayable model transformation recipes (``nnx.transforms``)."""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.enum.optims import Optims
from nnx.nn.params.nn_checkpoint import NNCheckpoint, NNCheckpointTransform
from nnx.nn.params.nn_evaluation_data_point import NNEvaluationDataPoint
from nnx.nn.params.nn_iteration_data_point import NNIterationDataPoint
from nnx.nn.params.nn_optim_params import NNOptimParams
from nnx.nn.params.nn_train_params import NNTrainParams
from nnx.optimizers import build_optimizer
from nnx.surgery import low_rank_factorize
from nnx.transforms import RecipeError, TransformOp, TransformRecipe, check_optimizer, lora, low_rank

_OPTIM = NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    monkeypatch.chdir(tmp_path)


def _model(seed: int = 0) -> NNModel:
    torch.manual_seed(seed)
    return NNModel(
        net_params=NNParams(
            input_dim=6, output_dim=3, hidden_dims=[16, 12], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
    )


def _recipe(materialization="in_place") -> TransformRecipe:
    # LoRA on the input layer, then a rank-4 SVD factorization of the hidden one.
    return TransformRecipe(
        [lora("layers.0", r=4, alpha=8.0), low_rank("layers.1", rank=4)], materialization=materialization
    )


def _snapshot(model: NNModel):
    return (
        [(name, id(module)) for name, module in model.net.named_modules()],
        {k: v.clone() for k, v in model.net.state_dict().items()},
        [p.requires_grad for p in model.net.parameters()],
        model._topology_transforms,
    )


def _same(a, b) -> bool:
    return (
        a[0] == b[0]
        and a[2] == b[2]
        and a[3] == b[3]
        and a[1].keys() == b[1].keys()
        and all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    )


def _checkpoint(model: NNModel) -> NNCheckpoint:
    idp = NNIterationDataPoint(lr=0.1, iter_idx=0, epoch_idx=0, batch_idx=0, train_edp=NNEvaluationDataPoint(loss=1.0))
    return NNCheckpoint(
        idp=idp,
        model_params=model.params,
        net_params=model.net_params,
        net_state=model.net.state_dict(),
        transforms=model._topology_transforms,
    )


def _same_model(a: NNModel, b: NNModel) -> None:
    sa, sb = a.net.state_dict(), b.net.state_dict()
    assert list(sa) == list(sb)  # ordered keys
    assert [v.shape for v in sa.values()] == [v.shape for v in sb.values()]
    assert [(n, p.requires_grad) for n, p in a.net.named_parameters()] == [
        (n, p.requires_grad) for n, p in b.net.named_parameters()
    ]
    x = torch.randn(7, 6, generator=torch.Generator().manual_seed(3))
    a.net.eval(), b.net.eval()
    with torch.no_grad():
        torch.testing.assert_close(b.net(x), a.net(x), rtol=1e-5, atol=1e-6)


def _perturb(model: NNModel) -> None:
    with torch.no_grad():  # a "trained" adapter: nonzero B, so LoRA changes the output
        model.net.layers[0].lora_B.normal_(0, 0.1, generator=torch.Generator().manual_seed(1))


# --- AC1: the recipe itself --------------------------------------------------------------------------


def test_a_recipe_records_id_version_config_and_order_immutably():
    source = [lora("layers.0", r=4, alpha=8.0), low_rank("layers.1", rank=4)]
    recipe = TransformRecipe(source)
    source.append(lora("layers.2"))  # the caller's list changes; the recipe does not
    source[0] = low_rank("layers.2", rank=2)
    assert [(op.id, op.version, op.targets) for op in recipe.operations] == [
        ("lora", 1, ("layers.0",)),
        ("low_rank", 1, ("layers.1",)),
    ]
    assert dict(recipe.operations[0].config) == {"r": 4, "alpha": 8.0, "dropout": 0.0}
    with pytest.raises(TypeError):
        recipe.operations[0].config["r"] = 9  # type: ignore[index]
    assert recipe.materialization == "in_place"  # the default: a trained model's own weights are transformed
    assert TransformRecipe(source, materialization="fresh").materialization == "fresh"
    import pickle

    assert pickle.loads(pickle.dumps(recipe)) == recipe and hash(recipe) == hash(TransformRecipe(recipe.operations))
    assert [t.state() for t in recipe.checkpoint_transforms()] == [
        {"name": "lora", "version": 1, "options": {"targets": ["layers.0"], "r": 4, "alpha": 8.0, "dropout": 0.0}},
        {"name": "low_rank", "version": 1, "options": {"targets": ["layers.1"], "rank": 4, "method": "svd"}},
    ]


def test_validating_mutates_nothing():
    model = _model()
    before = _snapshot(model)
    _recipe().validate(model)
    with pytest.raises(RecipeError):
        TransformRecipe([lora("layers.0"), lora("layers.9")]).validate(model)
    assert _same(before, _snapshot(model))


def test_fresh_materialization_builds_a_new_base_and_in_place_mutates():
    model = _model()
    before = _snapshot(model)
    fresh = _recipe("fresh").materialize(model)
    assert fresh is not model and _same(before, _snapshot(model))  # the source is untouched
    assert [t.name for t in fresh._topology_transforms] == ["lora", "low_rank"]
    assert type(fresh.net.layers[1]) is nn.Sequential
    same = _recipe("in_place").materialize(model)
    assert same is model and [t.name for t in model._topology_transforms] == ["lora", "low_rank"]


def test_a_failed_materialization_leaves_the_model_as_it_was(monkeypatch):
    import nnx.transforms as transforms

    model = _model()
    before = _snapshot(model)
    original = transforms._build
    calls = []

    def fail_second(op, linear, *, allocate_only):
        calls.append(op.id)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return original(op, linear, allocate_only=allocate_only)

    monkeypatch.setattr(transforms, "_build", fail_second)
    with pytest.raises(RuntimeError, match="boom"):
        _recipe().materialize(model)
    assert _same(before, _snapshot(model))


# --- AC2 + AC3: reconstruction ---------------------------------------------------------------------------


def test_lora_then_low_rank_rebuilds_a_fresh_base_without_rerunning_svd(tmp_path):
    # (the safetensors format: tests/test_checkpoint_safetensors.py, which needs the `hub` extra)
    model = _recipe().materialize(_model())
    _perturb(model)
    path = str(tmp_path / "recipe.pt")
    _checkpoint(model).to_file(path, format="pickle")
    with mock.patch("torch.linalg.svd", side_effect=AssertionError("SVD rerun during reload")):
        rebuilt = NNModel.from_checkpoint(NNCheckpoint.from_file(path))
    _same_model(model, rebuilt)
    assert rebuilt._topology_transforms == model._topology_transforms
    assert type(rebuilt.net.layers[1][0]) is nn.Linear and rebuilt.net.layers[1][0].out_features == 4


# --- AC4: validation --------------------------------------------------------------------------------------


CASES = {
    "missing target": ([lora("layers.0"), low_rank("layers.7", rank=2)], (1, "low_rank", "layers.7", "no such module")),
    "repeated on one subtree": (
        [lora("layers.0"), low_rank("layers.0", rank=2)],
        (1, "low_rank", "layers.0", "inside the subtree 'layers.0'"),
    ),
    "overlapping targets": ([lora("layers.0", "layers.0")], (0, "lora", None, "names a target twice")),
    "unsupported module type": ([lora("layers")], (0, "lora", "layers", "only nn.Linear is supported")),
    "unknown version": (
        [TransformOp(id="lora", targets=("layers.0",), config={"r": 2, "alpha": 4.0, "dropout": 0.0}, version=2)],
        (0, "lora", None, "unknown version 2"),
    ),
    "unknown operation": ([TransformOp(id="prune", targets=("layers.0",))], (0, "prune", None, "unknown operation")),
    "rank too large": ([low_rank("layers.2", rank=20)], (0, "low_rank", "layers.2", "exceeds min(in, out)")),
    "glob target": ([lora("layers.*")], (0, "lora", "layers.*", "explicit dotted module paths")),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_validation_names_the_operation_index_and_target(case):
    operations, (index, op_id, target, reason) = CASES[case]
    model = _model()
    before = _snapshot(model)
    with pytest.raises(RecipeError) as caught:
        TransformRecipe(operations, materialization="in_place").materialize(model)
    assert any(p[0] == index and p[1] == op_id and p[2] == target and reason in p[3] for p in caught.value.problems)
    assert f"recipe operation {index} ({op_id})" in str(caught.value)
    assert _same(before, _snapshot(model))  # nothing applied


def test_a_recipe_on_a_previously_transformed_subtree_is_refused():
    model = _recipe().materialize(_model())
    with pytest.raises(RecipeError, match=r"inside the subtree 'layers.1'"):
        TransformRecipe([lora("layers.1.0")], materialization="in_place").materialize(model)


def test_an_aliased_target_is_refused():
    model = _model()
    model.net.alias = model.net.layers[0]  # one Linear under two names
    with pytest.raises(RecipeError, match=r"registered under \['layers.0', 'alias'\]"):
        TransformRecipe([lora("layers.0")], materialization="in_place").validate(model)


def test_a_stale_optimizer_is_refused_with_a_rebuild_instruction():
    model = _model()
    stale = build_optimizer(model.net, _OPTIM)  # built before the topology changes
    with pytest.raises(RecipeError, match=r"optimizer 0 holds this target's parameters — build the optimizer after"):
        _recipe().materialize(model, optimizers=[stale])
    assert model._topology_transforms == ()  # refused before anything changed
    _recipe().materialize(model)
    with pytest.raises(
        ValueError, match="does not fit the model's current topology.*build it over the trainable parameters after"
    ):
        check_optimizer(model, stale)
    check_optimizer(model, build_optimizer(model.net, _OPTIM))  # rebuilt after: fine


# --- AC5: trained recipe round trips ------------------------------------------------------------------------


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(24, 6, generator=generator)
    labels = torch.randint(0, 3, (24,), generator=generator)
    return DataLoader(TensorDataset(features, labels), batch_size=8)


def _train(model: NNModel, n_epochs: int = 2, callbacks=(), **train):
    params = NNTrainParams(
        n_epochs=n_epochs, seed=0, train_loader=_loader(), val_loader=_loader(), optim=_OPTIM, **train
    )
    return model.train(params, callbacks=list(callbacks))


@pytest.mark.parametrize("tag", [Checkpoints.LAST, Checkpoints.BEST, Checkpoints.FIRST])
def test_a_trained_recipe_round_trips_through_its_run_checkpoints(tag, tmp_path):
    model = _recipe().materialize(_model())
    run = _train(model)
    checkpoint = NNCheckpoint.load(run=run.id, type=tag)
    assert checkpoint is not None and [t.name for t in checkpoint.transforms] == ["lora", "low_rank"]
    rebuilt = NNModel.from_checkpoint(checkpoint)
    if tag is Checkpoints.LAST:
        _same_model(model, rebuilt)
    path = str(tmp_path / "trained.pt")
    checkpoint.to_file(path, format="pickle")
    _same_model(rebuilt, NNModel.from_checkpoint(NNCheckpoint.from_file(path)))


def test_a_recipe_run_resumes_from_its_checkpoint():
    model = _recipe().materialize(_model())
    parent = _train(model, n_epochs=2)
    child_model = _recipe().materialize(_model(seed=5))  # the same recipe on a fresh model
    child = _train(child_model, n_epochs=1, resume_from_run_id=parent.id)
    assert child.resume_status is not None and child.resume_status.mode == "stateful"
    assert [t.name for t in NNCheckpoint.load(run=child.id, type=Checkpoints.LAST).transforms] == ["lora", "low_rank"]


def test_unrecorded_surgery_on_a_recipe_model_is_still_refused():
    model = _recipe().materialize(_model())
    model.net.layers[2] = low_rank_factorize(model.net.layers[2], rank=2)  # outside the recipe
    with pytest.raises(ValueError, match="differs from its descriptor plus its recorded transformation recipe"):
        _train(model)


# --- AC6: unknown operations / raw artifacts ---------------------------------------------------------------


def test_from_checkpoint_rejects_an_unknown_operation_version_before_loading_tensors():
    model = _recipe().materialize(_model())
    checkpoint = _checkpoint(model)
    future = NNCheckpointTransform(
        name="low_rank", version=7, options={"targets": ["layers.1"], "rank": 4, "method": "svd"}
    )
    unknown = NNCheckpointTransform(name="prune", version=1, options={})
    for bad, text in ((future, "unknown version 7"), (unknown, "unsupported checkpoint transform 'prune'")):
        tampered = NNCheckpoint(
            idp=checkpoint.idp,
            model_params=checkpoint.model_params,
            net_params=checkpoint.net_params,
            net_state=checkpoint.net_state,
            transforms=(checkpoint.transforms[0], bad),
        )
        with mock.patch.object(nn.Module, "load_state_dict", side_effect=AssertionError("tensors loaded")):
            with pytest.raises(
                ValueError, match=rf"topology transform 1 \('{bad.name}' version {bad.version}\).*{text}"
            ):
                NNModel.from_checkpoint(tampered)


def test_a_raw_state_dict_cannot_rebuild_a_recipe_alone(tmp_path):
    model = _recipe().materialize(_model())
    raw = NNCheckpoint(
        idp=_checkpoint(model).idp,
        model_params=model.params,
        net_params=model.net_params,
        net_state=model.net.state_dict(),  # weights of a transformed topology, no recipe
    )
    with pytest.raises(ValueError, match="record no transformation recipe: a raw state dict or an adapter-only export"):
        NNModel.from_checkpoint(raw)
    assert "records no" in (NNModel.export_state_dict.__doc__ or "")
    from nnx.peft import save_lora_weights

    assert "records no" in (save_lora_weights.__doc__ or "")


# --- review regressions ------------------------------------------------------------------------------------


def test_a_resume_refuses_a_different_recipe_even_with_pre_transform_state():
    parent = _train(TransformRecipe([lora("layers.0", r=4, alpha=8.0)]).materialize(_model()))
    other = TransformRecipe([lora("layers.0", r=4, alpha=16.0)]).materialize(_model(seed=5))
    with pytest.raises(ValueError, match="materialize the same nnx.transforms.TransformRecipe"):
        _train(other, n_epochs=1, resume_from_run_id=parent.id)
    with pytest.raises(ValueError, match="materialize the same nnx.transforms.TransformRecipe"):
        _train(_model(seed=6), n_epochs=1, resume_from_run_id=parent.id)  # no recipe at all


def test_a_recipe_run_has_its_own_run_id_and_no_duplicate_weights():
    base_run = _train(_model())
    recipe_run = _train(_recipe().materialize(_model()))
    assert recipe_run.id != base_run.id  # no collision, no overwrite
    assert [t["name"] for t in recipe_run.state()["transforms"]] == ["lora", "low_rank"]
    assert "transforms" not in base_run.state()  # run ids without a recipe are unchanged
    from nnx.nn.params.nn_run import NNRun

    assert NNRun.load(recipe_run.id).id == recipe_run.id
    _, training_state = NNCheckpoint.load_with_training_state(run=recipe_run.id, type=Checkpoints.LAST)
    assert training_state is not None and training_state.get("model") is None


def test_the_provenance_manifest_records_the_recipe():
    from nnx.provenance import ExperimentManifest

    base = ExperimentManifest.for_model(_model())
    assert "transforms" not in base.model  # manifests without a recipe are unchanged
    with_recipe = ExperimentManifest.for_model(_recipe().materialize(_model()))
    assert [t["name"] for t in with_recipe.model["transforms"]] == ["lora", "low_rank"]
    assert with_recipe.fingerprint() != base.fingerprint()


def test_a_fresh_materialization_needs_a_plain_nnmodel():
    class Custom(NNModel):
        pass

    torch.manual_seed(0)
    model = Custom(net_params=_model().net_params, params=_model().params)
    with pytest.raises(RecipeError, match="use 'in_place' for a Custom"):
        _recipe("fresh").materialize(model)
    transformed = _recipe().materialize(_model())
    with pytest.raises(RecipeError, match="starts from an untransformed"):
        TransformRecipe([lora("layers.2")], materialization="fresh").validate(transformed)  # validate mirrors it


def test_validate_mirrors_a_fresh_materialization_for_optimizers():
    model = _model()
    held = build_optimizer(model.net, _OPTIM)
    _recipe("fresh").validate(model, optimizers=[held])  # a fresh base's parameters are not held
    with pytest.raises(RecipeError, match="build the optimizer after"):
        _recipe("in_place").validate(model, optimizers=[held])


def test_the_trainer_refuses_surgery_outside_the_recipe():
    from nnx.trainer import NNTrainerParams, Trainer

    model = _recipe().materialize(_model())
    model.net.layers[2] = low_rank_factorize(model.net.layers[2], rank=2)
    params = NNTrainerParams(n_epochs=1, train_loader=_loader(), optims={"main": _OPTIM})
    with pytest.raises(ValueError, match="differs from its descriptor plus its recorded transformation recipe"):
        Trainer(model).train(params, trainer_step_fn=lambda ctx: NNEvaluationDataPoint(loss=0.5))


def test_a_shape_only_change_outside_the_recipe_is_refused():
    model = _recipe().materialize(_model())
    # Same tensor names, different rank: re-factorize the recipe's own target.
    model.net.layers[1] = low_rank_factorize(model.net.layers[1][1], rank=2)
    with pytest.raises(ValueError, match="layers.1.0.weight has shape"):
        model._assert_reconstructible_topology()
    lora_model = TransformRecipe([lora("layers.0", r=4, alpha=8.0)]).materialize(_model())
    lora_model.net.layers[0].alpha = 16.0  # same shapes, different recorded configuration
    with pytest.raises(ValueError, match="is not the LoRA wrapper its recipe records"):
        lora_model._assert_reconstructible_topology()


def test_a_custom_step_on_a_recipe_model_is_checked_too():
    model = _recipe().materialize(_model())
    model.net.layers[2] = low_rank_factorize(model.net.layers[2], rank=2)
    params = NNTrainParams(n_epochs=1, train_loader=_loader(), optim=_OPTIM)
    with pytest.raises(ValueError, match="differs from its descriptor plus its recorded transformation recipe"):
        model.train(params, train_step_fn=lambda ctx: NNEvaluationDataPoint(loss=0.5))


def test_a_subset_optimizer_built_after_the_recipe_passes():
    model = _recipe().materialize(_model())
    subset = torch.optim.SGD([model.net.layers[0].lora_A, model.net.layers[0].lora_B], lr=0.1)
    check_optimizer(model, subset)
    stale = torch.optim.SGD(model.net.layers[0].base.parameters(), lr=0.1)  # the base a LoRA operation froze
    with pytest.raises(ValueError, match=r"holds the frozen base but none of the adapter of \[.layers.0.\]"):
        check_optimizer(model, stale)


def test_malformed_recorded_options_name_the_transform():
    model = _recipe().materialize(_model())
    checkpoint = _checkpoint(model)
    for options in ([], {"targets": [["layers.1"]], "rank": 4, "method": "svd"}):
        bad = NNCheckpointTransform(name="low_rank", version=1, options=options)  # type: ignore[arg-type]
        tampered = NNCheckpoint(
            idp=checkpoint.idp,
            model_params=checkpoint.model_params,
            net_params=checkpoint.net_params,
            net_state=checkpoint.net_state,
            transforms=(checkpoint.transforms[0], bad),
        )
        with pytest.raises(
            ValueError, match=r"topology transform 1 \('low_rank' version 1\).*malformed recorded options"
        ):
            NNModel.from_checkpoint(tampered)


def test_a_recipe_refuses_a_runtime_only_module():
    module = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))
    model = NNModel(params=NNModelParams(device=Devices.CPU, loss=Losses.CROSS_ENTROPY), module=module)
    for materialization in ("in_place", "fresh"):
        with pytest.raises(RecipeError, match="runtime-only module"):
            TransformRecipe([lora("0", r=2, alpha=4.0)], materialization=materialization).materialize(model)
    assert type(module[0]) is nn.Linear and model._topology_transforms == ()  # left untouched
    recorded = NNCheckpointTransform(
        name="lora", version=1, options={"targets": ["0"], "r": 2, "alpha": 4.0, "dropout": 0.0}
    )
    pristine = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))
    tampered = NNCheckpoint(
        idp=_checkpoint(model).idp,
        model_params=model.params,
        net_params=None,
        net_state=module.state_dict(),
        transforms=(recorded,),
    )
    with pytest.raises(ValueError, match="nothing can replay on a caller-owned module"):
        NNModel.from_checkpoint(tampered, module=pristine)
    assert type(pristine[0]) is nn.Linear  # never transformed behind the caller's back


def test_a_registered_module_raw_state_gets_the_recipe_error():
    from nnx.models import ModelSpec, register_model_factory

    register_model_factory(
        "tests.recipe_mlp", 1, lambda config: nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))
    )
    params = NNModelParams(net=ModelSpec("tests.recipe_mlp", 1), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = TransformRecipe([lora("0", r=2, alpha=4.0)]).materialize(NNModel(params=params))
    raw = NNCheckpoint(
        idp=_checkpoint(model).idp, model_params=model.params, net_params=None, net_state=model.net.state_dict()
    )
    with pytest.raises(ValueError, match="record no transformation recipe"):
        NNModel.from_checkpoint(raw)


# --- review round 2 ----------------------------------------------------------------------------------------


def test_a_fresh_recipe_is_validated_against_the_base_it_builds():
    from nnx.peft import apply_lora_to

    model = _model()
    apply_lora_to(model.net, "layers.0", r=2)  # an unrecorded change to the source only
    fresh = TransformRecipe([lora("layers.0", r=4, alpha=8.0)], materialization="fresh")
    rng = torch.get_rng_state()
    fresh.validate(model)  # the base it builds has an nn.Linear there
    assert torch.equal(torch.get_rng_state(), rng)  # the dry-run base leaves the random streams alone
    assert type(fresh.materialize(model).net.layers[0]).__name__ == "LoRALinear"
    widened = _model()
    widened.net.layers[1] = nn.Linear(16, 40)  # rank 14 fits only the source's layer, not the base's (16→12)
    with pytest.raises(RecipeError, match=r"operation 0 \(low_rank\), target 'layers.1': rank 14 exceeds"):
        TransformRecipe([low_rank("layers.1", rank=14)], materialization="fresh").materialize(widened)


def test_a_live_train_end_transform_does_not_bypass_the_pre_transform_guard():
    from nnx.nn import nn_model as module

    qat = NNCheckpointTransform(name="torchao_qat", version=1, options={"qat_config": "8da4w", "groupsize": 32})
    model = _model()
    converted = NNCheckpoint(
        idp=_checkpoint(model).idp,
        model_params=model.params,
        net_params=model.net_params,
        net_state=model.net.state_dict(),
        transforms=(qat,),
    )
    with mock.patch.object(NNCheckpoint, "load_with_training_state", return_value=(converted, {})):
        with pytest.raises(ValueError, match="no pre-transform training state"):
            module._load_resume_source("a" * 32, "last", "auto", trainer=False, live_transforms=(qat,))


def test_a_lora_rebuild_leaves_the_random_streams_alone():
    from nnx.transforms import _replay

    recorded = TransformRecipe([lora("layers.0", r=4, alpha=8.0)]).checkpoint_transforms()[0]
    model = _model()
    rng = torch.get_rng_state()
    _replay(model, recorded)
    assert type(model.net.layers[0]).__name__ == "LoRALinear"
    assert torch.equal(torch.get_rng_state(), rng)


def test_check_optimizer_reports_a_vanished_target_as_a_value_error():
    model = _recipe().materialize(_model())
    optimizer = build_optimizer(model.net, _OPTIM)
    model.net.layers = nn.Sequential()  # unrecorded surgery removed every recorded target
    with pytest.raises(ValueError, match="lora target 'layers.0' is gone"):
        check_optimizer(model, optimizer)


def test_unrecorded_snapshots_of_a_train_end_transform_are_unchanged():
    from nnx.nn.callbacks import ModelCheckpoint

    model = _model()
    model._topology_transforms = (NNCheckpointTransform(name="preexisting"),)  # not a recipe
    run = _train(model, n_epochs=1, callbacks=[ModelCheckpoint(epochs=[0], tag="snap")])
    snapshot = NNCheckpoint.from_file(f"runs/{run.id}/checkpoints/snap_e0.pt")
    assert snapshot is not None and snapshot.transforms == ()  # snapshots record recipes only, as before
    best = NNCheckpoint.load(run=run.id, type=Checkpoints.BEST)
    assert best is not None and best.transforms == ()  # so do in-loop tags


class _TrainEndTransform:
    """A train-end topology change (a stand-in for a QAT conversion) on top of a recipe."""

    def __init__(self):
        self.completed = False

    def on_train_end(self, ctx) -> None:  # noqa: ANN001 - callback context
        ctx.model.net.register_buffer("converted_marker", torch.ones(1))
        self.completed = True

    def checkpoint_transforms(self) -> tuple[NNCheckpointTransform, ...]:
        return (NNCheckpointTransform(name="test-transform"),) if self.completed else ()


def test_a_recipe_with_a_train_end_transform_keeps_pre_transform_state_and_resumes():
    from nnx import Callback

    callback = type("TrainEnd", (_TrainEndTransform, Callback), {})()
    parent = _train(_recipe().materialize(_model()), n_epochs=2, callbacks=[callback])
    last, state = NNCheckpoint.load_with_training_state(run=parent.id, type=Checkpoints.LAST)
    assert last is not None and state is not None
    assert [t.name for t in last.transforms] == ["lora", "low_rank", "test-transform"]
    assert "converted_marker" in last.net_state and "converted_marker" not in state["model"]
    assert list(state["model"]) == list(_recipe().materialize(_model()).net.state_dict())
    child = _train(_recipe().materialize(_model(seed=5)), n_epochs=1, resume_from_run_id=parent.id)
    assert child.resume_status is not None and child.resume_status.mode == "stateful"


def test_a_recipe_model_trains_and_resumes_through_the_trainer():
    from nnx.trainer import NNTrainerParams, Trainer

    def step(ctx):
        optimizer = ctx.optimizers["main"]
        optimizer.zero_grad()
        x, y = ctx.batch
        loss = ctx.model.loss_fn(ctx.model.net(x), y)
        loss.backward()
        optimizer.step()
        return NNEvaluationDataPoint(loss=float(loss))

    def params(**resume):
        return NNTrainerParams(n_epochs=2, seed=0, train_loader=_loader(), optims={"main": _OPTIM}, **resume)

    model = _recipe().materialize(_model())
    parent = Trainer(model).train(params(), trainer_step_fn=step)
    last, state = NNCheckpoint.load_with_training_state(run=parent.id, type=Checkpoints.LAST)
    assert last is not None and [t.name for t in last.transforms] == ["lora", "low_rank"]
    assert state is not None and state.get("model") is None  # the weights are stored once
    _same_model(model, NNModel.from_checkpoint(last))
    child = Trainer(_recipe().materialize(_model(seed=5))).train(
        params(resume_from_run_id=parent.id), trainer_step_fn=step
    )
    assert child.resume_status is not None and child.resume_status.mode == "stateful"
    with pytest.raises(ValueError, match="materialize the same nnx.transforms.TransformRecipe"):
        Trainer(_model()).train(params(resume_from_run_id=parent.id), trainer_step_fn=step)


# --- review round 3 ----------------------------------------------------------------------------------------


def _register_mlp(name: str, calls: list[int] | None = None) -> NNModelParams:
    from nnx.models import ModelSpec, register_model_factory

    def factory(config):
        if calls is not None:
            calls.append(1)
        return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))

    register_model_factory(name, 1, factory)
    return NNModelParams(net=ModelSpec(name, 1), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)


def test_a_registered_module_recipe_rank_is_checked_before_training():
    model = TransformRecipe([low_rank("0", rank=4)]).materialize(NNModel(params=_register_mlp("tests.rank_mlp")))
    model._assert_reconstructible_topology()  # the recipe's own topology passes
    model.net[0] = low_rank_factorize(nn.Linear(6, 8), rank=2)  # same names, another rank
    with pytest.raises(ValueError, match=r"0\.0\.weight has shape \(2, 6\), its recipe and base give \(4, 6\)"):
        model._assert_reconstructible_topology()


def test_a_fresh_materialization_builds_its_base_once():
    calls: list[int] = []
    model = NNModel(params=_register_mlp("tests.counted_mlp", calls))
    calls.clear()
    TransformRecipe([lora("0", r=2, alpha=4.0)], materialization="fresh").materialize(model)
    assert len(calls) == 1


def test_check_optimizer_ignores_parameters_outside_the_net_and_refuses_a_fresh_source():
    model = _recipe().materialize(_model())
    temperature = nn.Parameter(torch.ones(1))  # e.g. a learnable loss temperature
    check_optimizer(model, torch.optim.SGD([*model.net.parameters(), temperature], lr=0.1))
    source = _model()
    source_optimizer = build_optimizer(source.net, _OPTIM)
    fresh = _recipe("fresh").materialize(source)
    with pytest.raises(
        ValueError, match=r"it holds \d+ parameters its recipe replaced, so it was built before the recipe"
    ):
        check_optimizer(fresh, source_optimizer)
    check_optimizer(fresh, build_optimizer(fresh.net, _OPTIM))


def test_malformed_targets_are_refused_by_name():
    with pytest.raises(RecipeError, match=r"targets must be explicit dotted module paths, got a list"):
        TransformRecipe([lora(["layers.0", "layers.1"], r=4)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="got the string '01'"):
        TransformOp(id="low_rank", targets="01", config={"rank": 1, "method": "svd"})  # type: ignore[arg-type]


def test_rebuilt_low_rank_factors_have_defined_values():
    from nnx.transforms import _replay

    recorded = TransformRecipe([low_rank("layers.1", rank=4)]).checkpoint_transforms()[0]
    model = _model()
    _replay(model, recorded)
    assert all(torch.equal(p, torch.zeros_like(p)) for p in model.net.layers[1].parameters())


def test_a_runtime_module_recipe_checkpoint_is_refused_before_the_module_is_wrapped():
    module = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))
    model = NNModel(params=NNModelParams(device=Devices.CPU, loss=Losses.CROSS_ENTROPY), module=module)
    recorded = TransformRecipe([lora("0", r=2, alpha=4.0)]).checkpoint_transforms()
    checkpoint = NNCheckpoint(
        idp=_checkpoint(model).idp,
        model_params=model.params,
        net_params=None,
        net_state=module.state_dict(),
        transforms=recorded,
    )
    with mock.patch.object(NNModel, "__init__", side_effect=AssertionError("module= wrapped")):
        with pytest.raises(ValueError, match="nothing can replay on a caller-owned module"):
            NNModel.from_checkpoint(checkpoint, module=module)


def test_dry_runs_and_rebuilds_leave_an_unused_cuda_context_alone():
    import nnx.seeding as seeding

    captured = []
    real = seeding._capture_rng_state

    def spy(loader=None, *, cuda=True):
        captured.append(cuda)
        return real(loader, cuda=cuda)

    with mock.patch.object(torch.cuda, "is_initialized", return_value=False):
        with mock.patch.object(seeding, "_capture_rng_state", side_effect=spy):
            _recipe("fresh").validate(_model())
            NNModel.from_checkpoint(_checkpoint(_recipe().materialize(_model())))
    assert captured and not any(captured)


# --- review round 4 ----------------------------------------------------------------------------------------


def test_a_recipe_on_a_layer_the_base_lacks_is_refused():
    from nnx.peft import apply_lora_to

    model = _model()
    model.net.extra = nn.Linear(4, 4, bias=False)  # unrecorded surgery adds a layer
    recipe = TransformRecipe([lora("extra", r=2, alpha=4.0)])
    with pytest.raises(
        RecipeError,
        match=r"differs from its descriptor plus its recorded transformation recipe.*unexpected tensors \['extra.weight'\]",
    ):
        recipe.materialize(model)  # refused before anything is recorded or saved
    assert type(model.net.extra) is nn.Linear and model._topology_transforms == ()
    apply_lora_to(model.net, "extra", r=2, alpha=4.0)  # recorded by hand, bypassing validation
    model._topology_transforms = recipe.checkpoint_transforms()
    with pytest.raises(ValueError, match="lora target 'extra' is not a layer of the base"):
        model._assert_reconstructible_topology()  # the pre-training check refuses it too


@pytest.mark.parametrize("materialization", ["fresh", "in_place"])
def test_a_failed_materialization_leaves_the_random_streams_alone(materialization, monkeypatch):
    model = _model()
    torch.manual_seed(1)
    before = torch.get_rng_state()
    if materialization == "fresh":
        recipe = TransformRecipe([low_rank("layers.1", rank=999)], materialization="fresh")  # the base is built first
    else:
        from nnx import transforms

        real = transforms._build
        calls = []

        def failing(op, linear, *, allocate_only):
            calls.append(op.id)
            if len(calls) == 2:
                raise RuntimeError("interrupted after a LoRA initialization")
            return real(op, linear, allocate_only=allocate_only)

        monkeypatch.setattr(transforms, "_build", failing)
        recipe = _recipe()
    with pytest.raises((RecipeError, RuntimeError)):
        recipe.materialize(model)
    assert torch.equal(torch.get_rng_state(), before)


def test_a_bitfit_optimizer_built_after_the_recipe_passes():
    model = TransformRecipe([lora("layers.0", r=2, alpha=4.0)]).materialize(_model())
    model.net.layers[0].base.bias.requires_grad_(True)  # base bias deliberately unfrozen, adapter left out
    check_optimizer(model, torch.optim.SGD([model.net.layers[0].base.bias, *model.net.layers[2].parameters()], lr=0.1))
    with pytest.raises(ValueError, match="holds the frozen base but none of the adapter"):
        check_optimizer(model, torch.optim.SGD([model.net.layers[0].base.weight], lr=0.1))


def test_the_pre_training_topology_check_leaves_an_unused_cuda_context_alone():
    import nnx.seeding as seeding

    captured = []
    real = seeding._capture_rng_state

    def spy(loader=None, *, cuda=True):
        captured.append(cuda)
        return real(loader, cuda=cuda)

    model = _recipe().materialize(_model())
    with mock.patch.object(torch.cuda, "is_initialized", return_value=False):
        with mock.patch.object(seeding, "_capture_rng_state", side_effect=spy):
            model._assert_reconstructible_topology()
    assert captured == []  # the base was recorded at construction: nothing is rebuilt, no stream is read


# --- review round 5 ----------------------------------------------------------------------------------------


def test_an_in_place_target_resized_by_unrecorded_surgery_is_refused():
    model = _model()
    model.net.layers[1] = nn.Linear(16, 40)  # the base gives Linear(16, 12)
    with pytest.raises(RecipeError, match=r"layers.1.weight has shape \(40, 16\), its recipe and base give \(12, 16\)"):
        TransformRecipe([lora("layers.1", r=2, alpha=4.0)]).materialize(model)


def test_a_failed_fresh_base_build_leaves_the_random_streams_alone():
    from nnx import transforms

    model = _model()
    torch.manual_seed(1)
    before = torch.get_rng_state()

    def failing(source):
        torch.rand(3)  # the build draws from the streams, then fails
        raise RuntimeError("the base could not be built")

    with mock.patch.object(transforms, "_fresh_base", side_effect=failing):
        with pytest.raises(RuntimeError, match="could not be built"):
            _recipe("fresh").materialize(model)
    assert torch.equal(torch.get_rng_state(), before)


def test_integer_and_float_lora_numbers_are_one_recipe_with_one_run_id():
    from nnx.nn.params.nn_run import NNRun

    ints = TransformRecipe([lora("layers.0", r=2, alpha=4, dropout=0)])
    floats = TransformRecipe([lora("layers.0", r=2, alpha=4.0, dropout=0.0)])
    assert ints == floats and ints.checkpoint_transforms() == floats.checkpoint_transforms()
    assert dict(ints.operations[0].config) == {"r": 2, "alpha": 4.0, "dropout": 0.0}
    model = _model()
    params = NNTrainParams(n_epochs=1, seed=0, train_loader=_loader(), optim=_OPTIM)

    def run_id(recipe):
        transforms = recipe.checkpoint_transforms()
        return NNRun(train=params, model=model.params, net=model.net_params, transforms=transforms).id

    assert run_id(ints) == run_id(floats)


# --- review round 6 ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("surgery", ["drop a target's bias", "widen another layer"])
def test_unrecorded_surgery_anywhere_refuses_an_in_place_recipe(surgery):
    model = _model()
    if surgery == "drop a target's bias":
        model.net.layers[0] = nn.Linear(6, 16, bias=False)  # same weight shape, no bias
    else:
        model.net.layers[2] = nn.Linear(12, 5)  # a layer the recipe does not touch
    with pytest.raises(RecipeError, match="differs from its descriptor plus its recorded transformation recipe"):
        TransformRecipe([lora("layers.0", r=2, alpha=4.0)]).materialize(model)
    assert model._topology_transforms == ()


def test_a_registered_module_resized_by_unrecorded_surgery_is_refused():
    model = NNModel(params=_register_mlp("tests.resized_mlp"))
    model.net[0], model.net[2] = nn.Linear(6, 40), nn.Linear(40, 3)
    with pytest.raises(RecipeError, match=r"0\.weight has shape \(40, 6\), its recipe and base give \(8, 6\)"):
        TransformRecipe([lora("0", r=2, alpha=4.0)]).materialize(model)


def test_an_unrepresentable_lora_alpha_is_a_recipe_error():
    with pytest.raises(RecipeError, match="alpha must be a finite positive number"):
        TransformRecipe([lora("layers.0", r=2, alpha=10**400)])
    model = _recipe().materialize(_model())
    checkpoint = _checkpoint(model)
    options = {"targets": ["layers.0"], "r": 4, "alpha": 10**400, "dropout": 0.0}
    tampered = NNCheckpoint(
        idp=checkpoint.idp,
        model_params=checkpoint.model_params,
        net_params=checkpoint.net_params,
        net_state=checkpoint.net_state,
        transforms=(NNCheckpointTransform(name="lora", version=1, options=options), checkpoint.transforms[1]),
    )
    with pytest.raises(ValueError, match=r"topology transform 0 \('lora' version 1\) cannot be replayed.*alpha"):
        NNModel.from_checkpoint(tampered)


def test_a_reloaded_recipe_keeps_the_canonical_form_and_run_id():
    from nnx.nn.params.nn_run import NNRun

    model = TransformRecipe([lora("layers.0", r=4, alpha=8.0)]).materialize(_model())
    checkpoint = _checkpoint(model)
    written_with_ints = NNCheckpoint(
        idp=checkpoint.idp,
        model_params=checkpoint.model_params,
        net_params=checkpoint.net_params,
        net_state=checkpoint.net_state,
        transforms=(
            NNCheckpointTransform(
                name="lora", version=1, options={"targets": ["layers.0"], "r": 4, "alpha": 8, "dropout": 0}
            ),
        ),
    )
    reloaded = NNModel.from_checkpoint(written_with_ints)
    assert [t.state() for t in reloaded._topology_transforms] == [t.state() for t in model._topology_transforms]
    params = NNTrainParams(n_epochs=1, seed=0, train_loader=_loader(), optim=_OPTIM)
    ids = {
        NNRun(train=params, model=m.params, net=m.net_params, transforms=m._topology_transforms).id
        for m in (model, reloaded)
    }
    assert len(ids) == 1


def test_the_optimizer_error_names_only_what_happened():
    model = TransformRecipe([low_rank("layers.1", rank=4)]).materialize(_model(), optimizers=())
    unrelated = torch.optim.SGD([_model().net.layers[0].weight], lr=0.1)  # another model's parameters
    check_optimizer(model, unrelated)
    replaced_model = _model()
    before = build_optimizer(replaced_model.net, _OPTIM)
    TransformRecipe([low_rank("layers.1", rank=4)]).materialize(replaced_model)
    with pytest.raises(ValueError) as caught:
        check_optimizer(replaced_model, before)
    assert "its recipe replaced" in str(caught.value) and "adapter" not in str(caught.value)
    lora_model = TransformRecipe([lora("layers.0", r=2, alpha=4.0)]).materialize(_model())
    with pytest.raises(ValueError) as caught:
        check_optimizer(lora_model, torch.optim.SGD([lora_model.net.layers[0].base.weight], lr=0.1))
    assert "adapter" in str(caught.value) and "its recipe replaced" not in str(caught.value)


def test_an_in_place_validation_rebuilds_nothing():
    model = _model()
    built = []
    real = nn.Linear.__init__

    def counting(self, *args, **kwargs):
        built.append(1)
        real(self, *args, **kwargs)

    with mock.patch.object(nn.Linear, "__init__", counting):
        TransformRecipe([lora("layers.0", r=2, alpha=4.0)]).validate(model)
        model._assert_reconstructible_topology()
    assert built == []  # the base was recorded when the model was built


# --- review round 7 ----------------------------------------------------------------------------------------


def _lazy_model(name: str) -> NNModel:
    from nnx.models import ModelSpec, register_model_factory

    register_model_factory(name, 1, lambda config: nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.Linear(8, 3)))
    params = NNModelParams(net=ModelSpec(name, 1), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    return NNModel(params=params)


def test_a_recipe_on_a_layer_that_was_lazy_at_build_is_refused():
    model = _lazy_model("tests.lazy_target")
    model.net(torch.randn(2, 6))  # the lazy layer is initialized now, but a rebuild starts lazy again
    with pytest.raises(RecipeError, match="'0': was an uninitialized lazy layer when the model was built"):
        TransformRecipe([low_rank("0", rank=2)]).materialize(model)


def test_an_uninitialized_lazy_layer_does_not_break_the_checks():
    model = _lazy_model("tests.lazy_other")  # no forward yet: '0' is still uninitialized
    recipe = TransformRecipe([lora("2", r=2, alpha=4.0)])
    recipe.validate(model)
    recipe.materialize(model)
    model._assert_reconstructible_topology()


def test_a_model_without_a_recorded_base_keeps_the_low_rank_refusal():
    model = _model()
    del model._reference_state  # e.g. an object restored without running __init__
    model.net.layers[1] = low_rank_factorize(model.net.layers[1], rank=4)
    with pytest.raises(ValueError, match="low-rank surgery topology has no reconstruction recipe"):
        model._assert_reconstructible_topology()


def test_a_rollback_restores_flags_after_a_custom_train_hook(monkeypatch):
    from nnx import transforms
    from nnx.models import ModelSpec, register_model_factory

    class Gated(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(6, 8), nn.Linear(8, 3)
            self.gate = nn.Parameter(torch.ones(1))

        def forward(self, x):
            return self.b(torch.relu(self.a(x))) * self.gate

        def train(self, mode=True):
            super().train(mode)
            self.gate.requires_grad_(mode)  # the hook derives trainability from the mode
            return self

    register_model_factory("tests.gated", 1, lambda config: Gated())
    params = NNModelParams(net=ModelSpec("tests.gated", 1), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = NNModel(params=params)
    model.net.eval()
    model.net.gate.requires_grad_(True)  # deliberately trainable while in eval mode
    before = [(n, p.requires_grad) for n, p in model.net.named_parameters()]
    real, calls = transforms._build, []

    def failing(op, linear, *, allocate_only):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return real(op, linear, allocate_only=allocate_only)

    monkeypatch.setattr(transforms, "_build", failing)
    with pytest.raises(RuntimeError, match="interrupted"):
        TransformRecipe([lora("a", r=2, alpha=4.0), lora("b", r=2, alpha=4.0)]).materialize(model)
    assert [(n, p.requires_grad) for n, p in model.net.named_parameters()] == before
    assert model.net.training is False


# --- review round 8 ----------------------------------------------------------------------------------------


def test_an_older_factory_model_keeps_the_low_rank_refusal():
    model = NNModel(params=_register_mlp("tests.legacy_mlp"))
    keys = list(model._reference_state)
    del model._reference_state
    model._reference_state_keys = tuple(keys)  # what a model pickled before this change carries
    model.net[0] = low_rank_factorize(model.net[0], rank=2)
    with pytest.raises(ValueError, match="low-rank surgery topology has no reconstruction recipe"):
        model._assert_reconstructible_topology()


def test_a_fallback_reference_is_rebuilt_once():
    model = _model()
    del model._reference_state
    built = []
    real = nn.Linear.__init__

    def counting(self, *args, **kwargs):
        built.append(1)
        real(self, *args, **kwargs)

    with mock.patch.object(nn.Linear, "__init__", counting):
        model._base_state()
        first = len(built)
        model._base_state()
    assert first > 0 and len(built) == first


def test_a_rollback_survives_a_failing_train_hook(monkeypatch):
    from nnx import transforms

    model = _model()
    before = [(n, p.requires_grad) for n, p in model.net.named_parameters()]
    monkeypatch.setattr(model.net, "train", mock.Mock(side_effect=RuntimeError("hook failed")))
    real, calls = transforms._build, []

    def failing(op, linear, *, allocate_only):
        calls.append(1)
        if len(calls) == 2:
            raise KeyError("the original failure")
        return real(op, linear, allocate_only=allocate_only)

    monkeypatch.setattr(transforms, "_build", failing)
    with pytest.raises(KeyError, match="the original failure"):
        TransformRecipe([lora("layers.0", r=2, alpha=4.0), lora("layers.1", r=2, alpha=4.0)]).materialize(model)
    assert [(n, p.requires_grad) for n, p in model.net.named_parameters()] == before


def test_a_lazy_target_is_reported_once():
    model = _lazy_model("tests.lazy_once")
    with pytest.raises(RecipeError) as caught:
        TransformRecipe([lora("0", r=2, alpha=4.0)]).validate(model)
    assert [problem[2] for problem in caught.value.problems] == ["0"]


def test_the_drift_message_names_its_cause_once():
    model = _recipe().materialize(_model())
    model.net.layers[2] = nn.Linear(12, 5)
    model.net.extra = nn.Linear(2, 2)
    with pytest.raises(ValueError) as caught:
        model._assert_reconstructible_topology()
    assert str(caught.value).count("differs from its descriptor plus its recorded transformation recipe") == 1


# --- review round 9 ----------------------------------------------------------------------------------------


def test_an_older_factory_model_still_takes_an_in_place_recipe():
    model = NNModel(params=_register_mlp("tests.legacy_recipe_mlp"))
    keys = list(model._reference_state)
    del model._reference_state
    model._reference_state_keys = tuple(keys)  # names only, as a model pickled before this change carries
    TransformRecipe([lora("0", r=2, alpha=4.0), low_rank("2", rank=2)]).materialize(model)
    assert [t.name for t in model._topology_transforms] == ["lora", "low_rank"]
    base = model._base_state()
    assert base is not None
    with pytest.raises(TypeError):
        base["0.weight"] = (1, 1)  # type: ignore[index]  # the recorded reference is read-only, never copied


# --- review round 10 ---------------------------------------------------------------------------------------


def test_lazy_layers_are_read_from_the_recorded_shapes():
    model = _lazy_model("tests.lazy_shapes")
    assert model._lazy_base_keys() == {"0.weight", "0.bias"}
    nets_model = _model()
    del nets_model._reference_state  # the built-in fallback rebuilds the reference and keeps its lazy keys
    assert nets_model._lazy_base_keys() == frozenset() and nets_model._base_state() is not None


# --- review round 11 ---------------------------------------------------------------------------------------


def test_a_real_module_name_with_other_characters_is_a_valid_target():
    from nnx.models import ModelSpec, register_model_factory

    class Blocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleDict({"q-proj": nn.Linear(6, 8), "out": nn.Linear(8, 3)})

        def forward(self, x):
            return self.blocks["out"](torch.relu(self.blocks["q-proj"](x)))

    register_model_factory("tests.hyphenated", 1, lambda config: Blocks())
    params = NNModelParams(net=ModelSpec("tests.hyphenated", 1), device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    model = TransformRecipe([lora("blocks.q-proj", r=2, alpha=4.0)]).materialize(NNModel(params=params))
    assert type(model.net.blocks["q-proj"]).__name__ == "LoRALinear"
    for bad in ("blocks.*", "blocks..out", "blocks.[q]"):
        with pytest.raises(RecipeError, match="no globs"):
            TransformRecipe([lora(bad, r=2, alpha=4.0)])


def test_a_recipe_operation_declared_at_train_end_keeps_the_pre_transform_state():
    from nnx import Callback
    from nnx.peft import apply_lora_to

    class LoRAAtTrainEnd(Callback):
        done = False

        def on_train_end(self, ctx):
            apply_lora_to(ctx.model.net, "layers.0", r=2, alpha=4.0)
            self.done = True

        def checkpoint_transforms(self):
            return (lora("layers.0", r=2, alpha=4.0).checkpoint_transform(),) if self.done else ()

    parent = _train(_model(), n_epochs=1, callbacks=[LoRAAtTrainEnd()])
    last, state = NNCheckpoint.load_with_training_state(run=parent.id, type=Checkpoints.LAST)
    assert last is not None and state is not None and [t.name for t in last.transforms] == ["lora"]
    assert state["model"] is not None and state["model_transforms"] == []  # the untransformed state, recorded
    child = _train(_model(seed=5), n_epochs=1, resume_from_run_id=parent.id)  # resumes the pre-transform state
    assert child.resume_status is not None and child.resume_status.mode == "stateful"


def test_a_fresh_dry_run_builds_its_base_on_the_cpu():
    from nnx import transforms

    built = []
    real = transforms._fresh_base

    def spy(model, *, dry_run=False):
        base = real(model, dry_run=dry_run)
        built.append((dry_run, base.params.device))
        return base

    with mock.patch.object(transforms, "_fresh_base", side_effect=spy):
        _recipe("fresh").validate(_model())
    assert built == [(True, Devices.CPU)]
