"""FEAT-006: registered model factories.

A ``ModelSpec(id, version, config, seed)`` names a registered factory; the
module is rebuilt from the registry — never from pickled code — with the
spec's seed and the ambient RNG restored. Rebuilds are checked against the
saved parameter names and shapes before any weight loads; an unknown id
fails first. Runs, checkpoints and their text / HTML views carry the
descriptor instead of built-in ``NNParams`` fields.
"""

from __future__ import annotations

import copy
import pickle

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nnx import (
    Activations,
    Devices,
    GenerativeNNModel,
    Losses,
    MissingModelFactoryError,
    ModelSpec,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNRun,
    NNTrainParams,
    RuntimeModule,
    TaskSpec,
    low_rank_factorize,
    register_model_factory,
    registered_model_factories,
    unregister_model_factory,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint


class Encoder(nn.Module):
    """A caller-defined classifier: plain positional forward, no NNx hooks."""

    def __init__(self, width: int = 8, in_dim: int = 4, classes: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


FACTORY_ID = "tests.encoder"


@pytest.fixture
def encoder_spec():
    register_model_factory(FACTORY_ID, 1, lambda config: Encoder(**config))
    try:
        yield ModelSpec(FACTORY_ID, 1, {"width": 8})
    finally:
        unregister_model_factory(FACTORY_ID, 1)


def _loader(seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(12, 4, generator=generator)
    return DataLoader(TensorDataset(X, (X[:, 0] > 0).long()), batch_size=4)


def _train_params(epochs: int = 2, **kwargs) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=epochs,
        train_loader=_loader(),
        val_loader=_loader(1),
        optim=NNOptimParams.builder().sgd(max_lr=0.05).build(),
        **kwargs,
    )


def _registered(spec: ModelSpec, **kwargs) -> NNModel:
    return NNModel(params=NNModelParams(net=spec, device=Devices.CPU, loss=Losses.CROSS_ENTROPY, **kwargs))


# --- registry and spec ---------------------------------------------------------------


def test_registry_registers_replaces_and_unregisters():
    def factory(config):
        return Encoder()

    register_model_factory("tests.reg", 1, factory)
    try:
        assert ("tests.reg", 1) in registered_model_factories()
        with pytest.raises(ValueError, match="already registered"):
            register_model_factory("tests.reg", 1, factory)
        register_model_factory("tests.reg", 1, factory, replace=True)
        with pytest.raises(ValueError, match="letters, digits"):
            register_model_factory("", 1, factory)
        with pytest.raises(ValueError, match="version"):
            register_model_factory("tests.reg", 0, factory)
        with pytest.raises(TypeError, match="callable"):
            register_model_factory("tests.reg", 2, "not callable")  # type: ignore[arg-type]
    finally:
        assert unregister_model_factory("tests.reg", 1) is True
    assert unregister_model_factory("tests.reg", 1) is False


def test_model_spec_is_immutable_serializable_and_json_like():
    spec = ModelSpec("tests.x", 2, {"b": [1, 2], "a": {"c": 1.5}}, seed=7)
    assert spec.state() == {
        "kind": "registered",
        "id": "tests.x",
        "version": 2,
        "config": {"a": {"c": 1.5}, "b": [1, 2]},
        "seed": 7,
    }
    assert ModelSpec.from_state(spec.state()) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec and copy.deepcopy(spec) == spec
    assert hash(spec) == hash(ModelSpec("tests.x", 2, {"a": {"c": 1.5}, "b": (1, 2)}, seed=7))
    assert str(spec) == "tests.x@v2"
    with pytest.raises(AttributeError, match="immutable"):
        spec.seed = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="JSON-like"):
        ModelSpec("tests.x", 1, {"init": torch.nn.init.zeros_})
    with pytest.raises(ValueError, match="seed"):
        ModelSpec("tests.x", 1, seed=-1)
    with pytest.raises(TypeError, match="Nets member"):
        NNModelParams(net="feed_fwd")  # type: ignore[arg-type]


def test_builtin_descriptors_serialize_exactly_as_before():
    params = NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    assert params.state() == {"net": "feed_fwd", "loss": "cross_entropy", "device": "cpu"}
    net = NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)
    run = NNRun(net=net, train=NNTrainParams(n_epochs=1), model=params)
    assert list(run.state()) == ["id", "model", "net", "train"]


# --- construction ------------------------------------------------------------------------


def test_factory_mode_seeds_construction_and_restores_the_ambient_rng(encoder_spec):
    torch.manual_seed(123)
    before = torch.get_rng_state()
    first = _registered(encoder_spec)
    assert torch.equal(torch.get_rng_state(), before)  # the caller's stream never moved
    torch.manual_seed(999)
    second = _registered(encoder_spec)
    for key, value in first.net.state_dict().items():
        assert torch.equal(value, second.net.state_dict()[key]), key  # same spec, same init
    reseeded = _registered(ModelSpec(FACTORY_ID, 1, {"width": 8}, seed=1))
    assert not torch.equal(reseeded.net.state_dict()["body.0.weight"], first.net.state_dict()["body.0.weight"])
    assert first.net_params is None and isinstance(first.net, Encoder)


def test_unknown_factories_and_bad_factories_fail_before_anything_is_built(encoder_spec):
    with pytest.raises(MissingModelFactoryError, match="registered versions of 'tests.encoder': v1"):
        _registered(ModelSpec(FACTORY_ID, 2))
    with pytest.raises(MissingModelFactoryError, match="register_model_factory"):
        _registered(ModelSpec("tests.unknown", 1))
    register_model_factory("tests.not_a_module", 1, lambda config: "nope")
    try:
        with pytest.raises(TypeError, match="not a torch.nn.Module"):
            _registered(ModelSpec("tests.not_a_module", 1))
    finally:
        unregister_model_factory("tests.not_a_module", 1)
    net = NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU)
    with pytest.raises(ValueError, match="pass net_params=None"):
        NNModel(net_params=net, params=NNModelParams(net=encoder_spec))
    with pytest.raises(ValueError, match="NNModelParams.net is required"):
        NNModel(params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    with pytest.raises(TypeError, match="built-in Nets.TRANSFORMER"):
        GenerativeNNModel(net_params=None, params=NNModelParams(net=encoder_spec))  # type: ignore[arg-type]


def test_a_task_checks_what_it_can_before_the_first_batch(encoder_spec):
    model = _registered(encoder_spec, task=TaskSpec.categorical(2))
    assert model.task_adapter is not None
    with pytest.raises(ValueError, match="needs Losses.CROSS_ENTROPY"):
        NNModel(params=NNModelParams(net=encoder_spec, loss=Losses.MEAN_SQUARED_ERROR, task=TaskSpec.categorical(2)))


# --- rebuild from checkpoints ----------------------------------------------------------------


def test_registered_checkpoints_rebuild_matching_names_shapes_and_predictions(tmp_path, monkeypatch, encoder_spec):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    model = _registered(encoder_spec)
    run = model.train(params=_train_params())
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.net_params is None and checkpoint.reconstructible
    assert checkpoint.model_params.net == encoder_spec
    rebuilt = NNModel.from_checkpoint(checkpoint)
    assert {k: v.shape for k, v in rebuilt.net.state_dict().items()} == {
        k: v.shape for k, v in model.net.state_dict().items()
    }
    X = torch.randn(5, 4)
    assert torch.equal(torch.as_tensor(rebuilt.predict(X).logits), torch.as_tensor(model.predict(X).logits))


def test_unknown_ids_and_changed_topologies_raise_before_weights_load(tmp_path, monkeypatch, encoder_spec):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    run = _registered(encoder_spec).train(params=_train_params(1))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    loads = []
    original = nn.Module.load_state_dict

    def recording_load(self, *args, **kwargs):
        loads.append(type(self).__name__)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "load_state_dict", recording_load)
    unregister_model_factory(FACTORY_ID, 1)
    with pytest.raises(MissingModelFactoryError, match="not registered"):
        NNModel.from_checkpoint(checkpoint)
    # same id and version, but the factory now builds another topology
    register_model_factory(FACTORY_ID, 1, lambda config: Encoder(width=config["width"] * 2))
    with pytest.raises(ValueError, match=r"topology does not match the saved weights \(reshaped: body.0.bias"):
        NNModel.from_checkpoint(checkpoint)
    assert loads == []  # nothing was ever loaded


def test_registered_modules_keep_the_topology_reconstruction_guard(tmp_path, monkeypatch, encoder_spec):
    monkeypatch.chdir(tmp_path)
    model = _registered(encoder_spec)
    model.net.body[0] = low_rank_factorize(model.net.body[0], rank=2)
    with pytest.raises(ValueError, match="no reconstruction recipe"):
        model.train(params=_train_params(1))


# --- runs ----------------------------------------------------------------------------------


def test_registered_runs_render_and_reload_their_descriptor(tmp_path, monkeypatch, encoder_spec):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    run = _registered(encoder_spec).train(params=_train_params(1))
    assert run.net is None and "net" not in run.state()
    assert run.state()["model"]["net"] == encoder_spec.state()
    text = str(run)
    assert "net=tests.encoder@v1" in text and "kind=registered" in text and 'config={"width": 8}' in text
    for builtin_only in ("dims=", "dropout=", "activation=", "n_heads="):
        assert builtin_only not in text
    page = run._repr_html_()
    assert "registered" in page and "hidden_dims" not in page and "input_dim" not in page
    reloaded = NNRun.load(run.id)
    assert reloaded.id == run.id and reloaded.net is None and reloaded.model.net == encoder_spec
    assert reloaded.model.net.kind == "registered"


def test_runtime_runs_render_and_reload_their_descriptor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")

    class Local(Encoder):  # a <locals> qualname must not break the HTML table
        pass

    model = NNModel(module=Local(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_train_params(1))
    descriptor = model.params.net
    assert isinstance(descriptor, RuntimeModule) and descriptor.state()["reconstructible"] is False
    text = str(run)
    assert "kind=runtime" in text and "reconstructible=False" in text and "dims=" not in text
    page = run._repr_html_()
    assert "&lt;locals&gt;" in page and "<locals>" not in page and "hidden_dims" not in page
    reloaded = NNRun.load(run.id)
    assert reloaded.id == run.id and reloaded.model.net == descriptor and reloaded.model.net.kind == "runtime"


def test_run_ids_follow_the_descriptor(encoder_spec):
    def run_id(model: NNModel) -> str:
        return NNRun(net=model.net_params, train=NNTrainParams(n_epochs=1), model=model.params).id

    assert run_id(_registered(encoder_spec)) == run_id(_registered(ModelSpec(FACTORY_ID, 1, {"width": 8})))
    assert run_id(_registered(encoder_spec)) != run_id(_registered(ModelSpec(FACTORY_ID, 1, {"width": 8}, seed=3)))
    wrap = lambda module: NNModel(module=module, params=NNModelParams())  # noqa: E731
    assert run_id(wrap(Encoder())) == run_id(wrap(Encoder()))  # same class and topology
    assert run_id(wrap(Encoder())) != run_id(wrap(Encoder(width=16)))


# --- review hardening -----------------------------------------------------------------------


def test_binary_f1_is_allowed_before_a_registered_modules_width_is_known(tmp_path, monkeypatch, encoder_spec):
    from nnx import MetricSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    run = _registered(encoder_spec).train(params=_train_params(1, metrics=[MetricSpec("f1", 1, {"average": "binary"})]))
    assert "f1" in run.idps[-1].val_edp.metrics


def test_training_never_rebuilds_a_registered_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    builds = []

    def factory(config):
        builds.append(config)
        return Encoder()

    register_model_factory("tests.counted", 1, factory)
    try:
        model = _registered(ModelSpec("tests.counted", 1))
        model.train(params=_train_params(1))
        assert len(builds) == 1  # the reconstruction guard compares the recorded layout
    finally:
        unregister_model_factory("tests.counted", 1)


def test_factories_drawing_from_python_and_numpy_are_seeded_too():
    import random

    import numpy as np

    class Drawn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            init = np.random.standard_normal((2, 4)) + random.random()
            self.linear = nn.Linear(4, 2)
            with torch.no_grad():
                self.linear.weight.copy_(torch.as_tensor(init))

    register_model_factory("tests.drawn", 1, lambda config: Drawn())
    try:
        np.random.seed(1)
        random.seed(1)
        ambient = (np.random.get_state()[1].copy(), random.getstate())
        first = _registered(ModelSpec("tests.drawn", 1))
        assert np.array_equal(np.random.get_state()[1], ambient[0]) and random.getstate() == ambient[1]
        np.random.seed(2)
        random.seed(2)
        second = _registered(ModelSpec("tests.drawn", 1))
        assert torch.equal(first.net.linear.weight, second.net.linear.weight)
    finally:
        unregister_model_factory("tests.drawn", 1)
