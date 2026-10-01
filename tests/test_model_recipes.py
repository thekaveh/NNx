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
    assert recipe.materialization == "fresh"
    assert TransformRecipe(source, materialization="in_place").materialization == "in_place"
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


@pytest.mark.parametrize("fmt", ["pickle", "safetensors"])
def test_lora_then_low_rank_rebuilds_a_fresh_base_without_rerunning_svd(fmt, tmp_path):
    model = _recipe().materialize(_model())
    _perturb(model)
    path = str(tmp_path / f"recipe.{fmt}")
    _checkpoint(model).to_file(path, format=fmt)
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
    with pytest.raises(ValueError, match="stale for the model's current topology.*build the optimizer after"):
        check_optimizer(model, stale)
    check_optimizer(model, build_optimizer(model.net, _OPTIM))  # rebuilt after: fine


# --- AC5: trained recipe round trips ------------------------------------------------------------------------


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(24, 6, generator=generator)
    labels = torch.randint(0, 3, (24,), generator=generator)
    return DataLoader(TensorDataset(features, labels), batch_size=8)


def _train(model: NNModel, n_epochs: int = 2, **train):
    params = NNTrainParams(
        n_epochs=n_epochs, seed=0, train_loader=_loader(), val_loader=_loader(), optim=_OPTIM, **train
    )
    return model.train(params)


@pytest.mark.parametrize("tag", [Checkpoints.LAST, Checkpoints.BEST, Checkpoints.FIRST])
def test_a_trained_recipe_round_trips_through_its_run_checkpoints(tag, tmp_path):
    model = _recipe().materialize(_model())
    run = _train(model)
    checkpoint = NNCheckpoint.load(run=run.id, type=tag)
    assert checkpoint is not None and [t.name for t in checkpoint.transforms] == ["lora", "low_rank"]
    rebuilt = NNModel.from_checkpoint(checkpoint)
    if tag is Checkpoints.LAST:
        _same_model(model, rebuilt)
    path = str(tmp_path / "trained.safetensors")
    checkpoint.to_file(path, format="safetensors")
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
    with pytest.raises(ValueError, match="differs from its recorded transformation recipe"):
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
    with pytest.raises(
        ValueError, match="records no transformation recipe: a raw state dict or an adapter-only export"
    ):
        NNModel.from_checkpoint(raw)
    assert "records no" in (NNModel.export_state_dict.__doc__ or "")
    from nnx.peft import save_lora_weights

    assert "records no" in (save_lora_weights.__doc__ or "")
