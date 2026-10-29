"""FEAT-006: caller-owned modules train, predict and render through batch
adapters, unmodified.

``NNModel(module=...)`` wraps the instance itself (never a clone or a
re-initialization) and describes it with a runtime descriptor; positional
and keyword-input modules see their batches through ``PositionalInputs`` /
``KeywordInputs``. Checkpoint t-SNE and ``nnx.viz.summary`` take the same
adapters — or fail before touching data or the module — and leave a
borrowed module's identity, state keys and training modes unchanged.
"""

from __future__ import annotations

import random
import types

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils.data import DataLoader, Dataset, TensorDataset

from nnx import (
    KeywordInputs,
    Losses,
    MissingModelFactoryError,
    ModelSpec,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNRun,
    NNTrainParams,
    PositionalInputs,
    RuntimeModule,
    register_model_factory,
    unregister_model_factory,
)
from nnx.nn.enum.checkpoints import Checkpoints
from nnx.nn.params.nn_checkpoint import NNCheckpoint


class Classifier(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class TwoInputs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left, self.right = nn.Linear(4, 2), nn.Linear(3, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.left(a) + self.right(b)


class KeywordClassifier(nn.Module):
    """Takes named inputs only, like a Hugging Face-style encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.norm = nn.BatchNorm1d(8)
        self.head = nn.Linear(8, 2)

    def forward(self, *, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.norm(self.embed(features * mask))))


class _Records(Dataset):
    def __init__(self, n: int = 12, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(n, 4, generator=generator)
        self.mask = (torch.rand(n, 4, generator=generator) > 0.2).float()
        self.labels = (self.features[:, 0] > 0).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[i], "mask": self.mask[i], "labels": self.labels[i]}


def _loader(seed: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(12, 4, generator=generator)
    return DataLoader(TensorDataset(X, (X[:, 0] > 0).long()), batch_size=4)


def _params(train_loader, epochs: int = 2, **kwargs) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=epochs,
        train_loader=train_loader,
        optim=NNOptimParams.builder().sgd(max_lr=0.05).build(),
        save_phase_checkpoints=False,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")


# --- batch adapters: train and predict, unmodified ---------------------------------------


def test_positional_modules_train_and_predict_unmodified():
    module = Classifier()
    before = {k: v.clone() for k, v in module.state_dict().items()}
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    assert model.net is module and not hasattr(module, "unpack_batch")
    run = model.train(params=_params(_loader()))
    assert len(run.idps) == 6 and all(np.isfinite(idp.train_edp.loss) for idp in run.idps)
    assert any(not torch.equal(before[k], v) for k, v in module.state_dict().items())  # it trained
    X = torch.randn(5, 4)
    logits, classes = model.predict(X)
    assert logits.shape == (5, 2) and classes.shape == (5,)
    loader_logits = model.predict(DataLoader(TensorDataset(X), batch_size=2)).logits
    np.testing.assert_allclose(loader_logits, logits, rtol=1e-6, atol=1e-7)
    assert model.evaluate(_loader(1)).error is not None

    two = NNModel(module=TwoInputs(), params=NNModelParams(), batch_adapter=PositionalInputs(2))
    batches = [(torch.randn(4, 4), torch.randn(4, 3), torch.tensor([0, 1, 0, 1])) for _ in range(2)]
    two.train(params=_params(batches, epochs=1))
    assert two.predict((torch.randn(3, 4), torch.randn(3, 3))).logits.shape == (3, 2)


def test_keyword_modules_train_and_predict_through_keyword_inputs():
    module = KeywordClassifier()
    adapter = KeywordInputs(("features", "mask"), target="labels")
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY), batch_adapter=adapter)
    run = model.train(params=_params(DataLoader(_Records(), batch_size=4), val_loader=DataLoader(_Records(6, 1), 3)))
    assert run.idps[-1].val_edp is not None and np.isfinite(run.idps[-1].val_edp.loss)
    records = _Records(5, 2)
    inputs = {"features": records.features, "mask": records.mask}
    by_mapping = model.predict(inputs).logits
    by_loader = model.predict(DataLoader(records, batch_size=2)).logits  # labels ignored
    np.testing.assert_allclose(by_loader, by_mapping, rtol=1e-6, atol=1e-7)
    assert module.training  # predict restored the training mode


def test_adapters_reject_batches_they_cannot_split():
    with pytest.raises(ValueError, match="expects batches of 1 input"):
        PositionalInputs().split((torch.zeros(1), torch.zeros(1), torch.zeros(1)))
    with pytest.raises(KeyError, match="missing input"):
        KeywordInputs(("features",)).split({"mask": torch.zeros(1)})
    with pytest.raises(TypeError, match="mapping batches"):
        KeywordInputs(("features",)).split((torch.zeros(1),))
    with pytest.raises(ValueError, match="distinct from the inputs"):
        KeywordInputs(("labels",), target="labels")

    class Dict(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x):
            return {"logits": self.linear(x)}

    model = NNModel(module=Dict(), params=NNModelParams())
    with pytest.raises(TypeError, match="not a tensor"):
        model.predict(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="no target"):
        NNModel(module=Classifier(), params=NNModelParams()).evaluate([(torch.zeros(2, 4),)])


# --- wrapping never clones; factories seed and restore --------------------------------------


def test_wrapping_never_clones_or_reinitializes_and_leaves_the_rng_alone():
    module = Classifier()
    tensors = {k: v.clone() for k, v in module.state_dict().items()}
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    states = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    assert model.net is module
    assert all(torch.equal(tensors[k], v) for k, v in module.state_dict().items())
    assert random.getstate() == states[0] and np.array_equal(np.random.get_state()[1], states[1])
    assert torch.equal(torch.get_rng_state(), states[2])
    descriptor = model.params.net
    assert isinstance(descriptor, RuntimeModule) and descriptor == RuntimeModule.of(module)
    assert model.net_params is None and descriptor.reconstructible is False


def test_wrapping_rejects_mixed_or_mismatched_descriptors():
    with pytest.raises(ValueError, match="net_params .* or module="):
        NNModel(net_params=object(), params=NNModelParams(), module=Classifier())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NNModelParams.net must be None"):
        NNModel(params=NNModelParams(net=Nets.FEED_FWD), module=Classifier())
    with pytest.raises(ValueError, match="does not match the descriptor"):
        NNModel(params=NNModelParams(net=RuntimeModule.of(Classifier(16))), module=Classifier())
    with pytest.raises(MissingModelFactoryError, match="runtime-only"):
        NNModel(params=NNModelParams(net=RuntimeModule.of(Classifier())))
    with pytest.raises(TypeError, match="torch.nn.Module"):
        NNModel(params=NNModelParams(), module="net")  # type: ignore[arg-type]


def test_runtime_checkpoints_reload_only_into_a_supplied_module():
    model = NNModel(module=Classifier(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_params(_loader(), epochs=1))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None and checkpoint.reconstructible is False
    with pytest.raises(MissingModelFactoryError, match="pass module="):
        NNModel.from_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="does not match the descriptor"):
        NNModel.from_checkpoint(checkpoint, module=Classifier(16))
    fresh = Classifier()
    reloaded = NNModel.from_checkpoint(checkpoint, module=fresh)
    assert reloaded.net is fresh
    X = torch.randn(4, 4)
    np.testing.assert_array_equal(reloaded.predict(X).logits, model.predict(X).logits)


# --- visualization: adapters or preflight errors -------------------------------------------------


@pytest.fixture
def keyword_spec():
    register_model_factory("tests.keyword", 1, lambda config: KeywordClassifier())
    try:
        yield ModelSpec("tests.keyword", 1)
    finally:
        unregister_model_factory("tests.keyword", 1)


class _CountingLoader:
    def __init__(self, batches):
        self.batches, self.reads = batches, 0

    def __iter__(self):
        for batch in self.batches:
            self.reads += 1
            yield batch


def _fake_tsne(monkeypatch):
    from nnx import vis_utils

    captured: dict = {}

    class FakeTSNE:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, X):
            captured["rows"] = len(X)
            return np.zeros((len(X), 2))

    monkeypatch.setattr(vis_utils, "TSNE", FakeTSNE)
    return captured


def test_checkpoint_tsne_uses_registries_and_adapters_or_fails_first(monkeypatch, keyword_spec):
    from nnx.vis_utils import VisUtils

    captured = _fake_tsne(monkeypatch)
    adapter = KeywordInputs(("features", "mask"))
    registered = NNModel(params=NNModelParams(net=keyword_spec), batch_adapter=adapter)
    run = registered.train(params=_params(DataLoader(_Records(), batch_size=4), epochs=1))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    loader = DataLoader(_Records(8, 3), batch_size=4)
    ds = types.SimpleNamespace(output_dim=2, test_loader=loader)
    VisUtils.two_dim_tsne_checkpoint_logits(checkpoint, ds, n_samples=6, renderer=None, batch_adapter=adapter)
    assert captured["rows"] == 6

    wrapped = NNModel(module=Classifier(), params=NNModelParams())
    run = wrapped.train(params=_params(_loader(), epochs=1))
    runtime_checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert runtime_checkpoint is not None
    counting = _CountingLoader(list(_loader(4)))
    ds = types.SimpleNamespace(output_dim=2, test_loader=counting)
    with pytest.raises(MissingModelFactoryError):
        VisUtils.two_dim_tsne_checkpoint_logits(runtime_checkpoint, ds, n_samples=6, renderer=None)
    assert counting.reads == 0  # failed before any data was read

    borrowed = Classifier()
    borrowed.train()
    borrowed.body[1].eval()  # mixed modes must survive
    keys = list(borrowed.state_dict())
    VisUtils.two_dim_tsne_checkpoint_logits(runtime_checkpoint, ds, n_samples=6, renderer=None, module=borrowed)
    assert captured["rows"] == 6 and list(borrowed.state_dict()) == keys
    assert borrowed.training and borrowed.body[0].training and not borrowed.body[1].training


def test_summary_takes_adapter_inputs_or_fails_before_touching_the_module():
    pytest.importorskip("torchinfo")
    from nnx.viz import summary

    module = KeywordClassifier()
    module.train()
    module.norm.eval()  # torchinfo alone would flip this back to train
    keys = list(module.state_dict())
    model = NNModel(module=module, params=NNModelParams(), batch_adapter=KeywordInputs(("features", "mask")))
    batch = next(iter(DataLoader(_Records(), batch_size=4)))
    stats = summary(model, batch=batch)
    assert stats.total_params == sum(p.numel() for p in module.parameters())
    assert model.net is module and list(module.state_dict()) == keys
    assert module.training and module.embed.training and not module.norm.training
    with pytest.raises(ValueError, match="keyword inputs"):
        summary(model, input_size=(2, 4))
    with pytest.raises(ValueError, match="not both"):
        summary(model, batch=batch, input_size=(2, 4))
    positional = summary(NNModel(module=Classifier(), params=NNModelParams()), batch=next(iter(_loader())))
    assert positional.total_params == sum(p.numel() for p in Classifier().parameters())


# --- review hardening ------------------------------------------------------------------------


def test_paradigm_steps_and_objectives_split_batches_through_the_adapter(keyword_spec):
    from nnx import kd_objective, kd_train_step_factory, mixup_train_step_factory

    register_model_factory("tests.plain_linear", 1, lambda config: nn.Linear(4, 2))  # no unpack_batch
    try:
        spec = ModelSpec("tests.plain_linear", 1)
        teacher = NNModel(params=NNModelParams(net=spec))
        for step in (mixup_train_step_factory(alpha=0.2), kd_train_step_factory(teacher, alpha=0.5)):
            student = NNModel(params=NNModelParams(net=spec))
            run = student.train(params=_params(_loader(), epochs=1, data_id=f"{id(step)}"), train_step_fn=step)
            assert all(np.isfinite(idp.train_edp.loss) for idp in run.idps)
        student = NNModel(params=NNModelParams(net=spec))
        student.train(params=_params(_loader(), epochs=1, data_id="objective"), objective=kd_objective(teacher))
        keyword = NNModel(params=NNModelParams(net=keyword_spec), batch_adapter=KeywordInputs(("features", "mask")))
        with pytest.raises(ValueError, match="one positional input"):
            keyword.train(
                params=_params(DataLoader(_Records(), batch_size=4), epochs=1), train_step_fn=mixup_train_step_factory()
            )
    finally:
        unregister_model_factory("tests.plain_linear", 1)


def test_lazy_modules_wrap_train_and_reload_into_a_fresh_lazy_module():
    def lazy() -> nn.Module:
        return nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.Linear(8, 2))

    model = NNModel(module=lazy(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_params(_loader(), epochs=1))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    reloaded = NNModel.from_checkpoint(checkpoint, module=lazy())
    X = torch.randn(3, 4)
    np.testing.assert_array_equal(reloaded.predict(X).logits, model.predict(X).logits)


def test_modules_with_their_own_unpack_batch_keep_the_prediction_shortcuts():
    class Unpacking(Classifier):
        def unpack_batch(self, batch):
            x, y = batch
            return x, y  # a bare tensor input, not a tuple

    model = NNModel(module=Unpacking(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    model.train(params=_params(_loader(), epochs=1))  # the bare input is one input, never split into rows
    X = torch.randn(5, 4)
    by_loader = model.predict(DataLoader(TensorDataset(X), batch_size=2)).logits  # 1-element batches
    np.testing.assert_allclose(by_loader, model.predict(X).logits, rtol=1e-6, atol=1e-7)


def test_runtime_run_ids_see_layer_hyperparameters():
    def run_id(dropout: float) -> str:
        module = nn.Sequential(nn.Linear(4, 8), nn.Dropout(dropout), nn.Linear(8, 2))
        model = NNModel(module=module, params=NNModelParams())
        return model.params.net.topology

    assert run_id(0.1) == run_id(0.1) and run_id(0.1) != run_id(0.5)


class _MixedAdapter(PositionalInputs):
    """Positional features plus a keyword mask."""

    def split(self, batch):
        return (batch["features"],), {"mask": batch["mask"]}, batch.get("labels")


class _MaskedClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, features, *, mask):
        return self.linear(features * mask)


def test_checkpoint_tsne_keeps_positional_and_keyword_inputs(monkeypatch):
    from nnx.vis_utils import VisUtils

    captured = _fake_tsne(monkeypatch)
    model = NNModel(module=_MaskedClassifier(), params=NNModelParams(), batch_adapter=_MixedAdapter())
    run = model.train(params=_params(DataLoader(_Records(), batch_size=4), epochs=1))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    ds = types.SimpleNamespace(output_dim=2, test_loader=DataLoader(_Records(8, 5), batch_size=4))
    VisUtils.two_dim_tsne_checkpoint_logits(
        checkpoint, ds, n_samples=7, renderer=None, module=_MaskedClassifier(), batch_adapter=_MixedAdapter()
    )
    assert captured["rows"] == 7


def test_a_materialized_lazy_module_resumes_statefully_from_last():
    # LAST is written after the fit: its optimizer topology has the materialized shapes.
    def lazy() -> nn.Module:
        return nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.Linear(8, 2))

    module = lazy()
    run = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY)).train(params=_params(_loader()))
    resumed = lazy()
    resumed(torch.randn(1, 4))  # materialize before a warm resume, as documented
    model = NNModel(module=resumed, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    warm = model.train(params=_params(_loader(), epochs=3, resume_from_run_id=run.id, resume_mode="stateful"))
    assert warm.resume_status is not None and warm.resume_status.mode == "stateful"


# --- lazy modules: one descriptor rule across materialization (#562) -----------------------------


def _lazy(width: int = 8, dropout: float = 0.0) -> nn.Module:
    return nn.Sequential(nn.LazyLinear(width), nn.Dropout(dropout), nn.ReLU(), nn.Linear(width, 2))


def _dry_run(module: nn.Module, *shape: int) -> nn.Module:
    module(torch.zeros(1, *(shape or (4,))))  # torch's recommended dry run materializes the lazy layers
    return module


def _is_uninitialized(module: nn.Module) -> bool:
    return any(nn.parameter.is_lazy(tensor) for tensor in module.state_dict().values())


def _train_lazy(module: nn.Module, **kwargs) -> tuple[NNModel, NNCheckpoint, str]:
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_params(_loader(), epochs=1, **kwargs))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    return model, checkpoint, run.id


def test_lazy_runs_reload_into_the_trained_module_and_into_fresh_instances():
    model_a, ckpt_a, _ = _train_lazy(_lazy())  # wrapped uninitialized
    model_b, ckpt_b, _ = _train_lazy(_dry_run(_lazy()), data_id="dry")  # wrapped after a dry run
    X = torch.randn(5, 4)
    for model, checkpoint, module in (
        (model_a, ckpt_a, model_a.net),
        (model_a, ckpt_a, _lazy()),
        (model_b, ckpt_b, _lazy()),
        (model_b, ckpt_b, _dry_run(_lazy())),
    ):
        reloaded = NNModel.from_checkpoint(checkpoint, module=module)
        assert reloaded.net is module
        np.testing.assert_array_equal(reloaded.predict(X).logits, model.predict(X).logits)
        # the reloaded module is materialized and matches its descriptor exactly
        assert not _is_uninitialized(module) and RuntimeModule.of(module) == reloaded.params.net


def test_a_trained_lazy_module_rewraps_and_renders_its_checkpoint_tsne(monkeypatch):
    from nnx.vis_utils import VisUtils

    captured = _fake_tsne(monkeypatch)
    model, checkpoint, _ = _train_lazy(_lazy())
    assert model.params.net == RuntimeModule.of(model.net)  # refreshed once the module materialized
    assert checkpoint.model_params.net == model.params.net
    rewrapped = NNModel(module=model.net, params=model.params)
    assert rewrapped.net is model.net
    ds = types.SimpleNamespace(output_dim=2, test_loader=_loader(3))
    VisUtils.two_dim_tsne_checkpoint_logits(checkpoint, ds, n_samples=6, renderer=None, module=model.net)
    assert captured["rows"] == 6


def test_a_model_checkpoint_callback_describes_the_materialized_lazy_module():
    from nnx import ModelCheckpoint

    model = NNModel(module=_lazy(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_params(_loader(), epochs=1), callbacks=[ModelCheckpoint(epochs=[0], tag="snap")])
    snapshot = NNCheckpoint.from_file(f"runs/{run.id}/checkpoints/snap_e0.pt")
    assert snapshot is not None and snapshot.model_params.net == RuntimeModule.of(model.net)
    reloaded = NNModel.from_checkpoint(snapshot, module=model.net)
    assert reloaded.net is model.net


def test_lazy_run_ids_follow_the_documented_materialization_rule():
    def run_id(module: nn.Module, data_id: str) -> str:
        return (
            NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
            .train(params=_params(_loader(), epochs=1, data_id=data_id, overwrite_existing=True))
            .id
        )

    uninitialized = NNModel(module=_lazy(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    expected = NNRun(train=_params(_loader(), epochs=1, data_id="a"), model=uninitialized.params, net=None).id
    assert run_id(_lazy(), "a") == expected  # training keeps the run id reserved at wrap time
    dense = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.0), nn.ReLU(), nn.Linear(8, 2))
    # after a dry run, a lazy net has the run id of the equivalent dense net
    assert run_id(_dry_run(_lazy()), "b") == run_id(dense, "b")
    # wrapped before vs after materialization: two run ids, and a refusal that says so
    assert run_id(_lazy(), "c") != run_id(_dry_run(_lazy()), "c")
    materialized = NNModel(module=_dry_run(_lazy()), params=NNModelParams(loss=Losses.CROSS_ENTROPY)).params
    with pytest.raises(ValueError, match="before vs after materialization gives different run ids") as refused:
        NNModel(module=_lazy(), params=materialized)
    assert "dry-run forward" in str(refused.value)
    # uninitialized lazy nets for different input widths share a descriptor (documented: use data_id / salt)
    assert RuntimeModule.of(_lazy()) == RuntimeModule.of(_lazy())
    assert RuntimeModule.of(_dry_run(_lazy(), 4)) != RuntimeModule.of(_dry_run(_lazy(), 6))


def test_a_descriptor_taken_before_materialization_names_the_lazy_remedy():
    before = NNModel(module=_lazy(), params=NNModelParams(loss=Losses.CROSS_ENTROPY)).params
    with pytest.raises(ValueError, match="does not match the descriptor") as refused:
        NNModel(module=_dry_run(_lazy()), params=before)
    message = str(refused.value)
    assert "lazy" in message and "materializ" in message and "fresh, unmaterialized instance" in message
    assert NNModel(module=_lazy(), params=before).params == before  # an uninitialized instance still matches


def test_lazy_reloads_refuse_other_shapes_classes_and_hyperparameters_before_loading():
    class Other(nn.Sequential):
        pass

    _, checkpoint, _ = _train_lazy(_dry_run(_lazy()))
    shapes = nn.Sequential(nn.LazyLinear(8), nn.Dropout(0.0), nn.ReLU(), nn.Linear(8, 3))
    with pytest.raises(ValueError, match="does not match the saved weights"):
        NNModel.from_checkpoint(checkpoint, module=shapes)
    other = Other(nn.LazyLinear(8), nn.Dropout(0.0), nn.ReLU(), nn.Linear(8, 2))
    with pytest.raises(ValueError, match="does not match the descriptor"):
        NNModel.from_checkpoint(checkpoint, module=other)
    dropout = _lazy(dropout=0.5)
    with pytest.raises(ValueError, match="does not match the descriptor") as refused:
        NNModel.from_checkpoint(checkpoint, module=dropout)
    assert "once its lazy layers materialize" in str(refused.value)
    head = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.0), nn.ReLU(), nn.LazyLinear(3))
    _, dense_checkpoint, _ = _train_lazy(
        nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.0), nn.ReLU(), nn.Linear(8, 2)), data_id="dense"
    )
    with pytest.raises(ValueError, match="does not match the descriptor"):
        NNModel.from_checkpoint(dense_checkpoint, module=head)
    for module in (shapes, other, dropout, head):
        assert _is_uninitialized(module)  # refused before any weight loaded


def test_a_checkpoint_described_before_materialization_reloads_into_a_fresh_instance():
    import dataclasses

    model, checkpoint, _ = _train_lazy(_lazy())
    # what a checkpoint written before #562 holds: the descriptor taken at wrap time
    before = dataclasses.replace(checkpoint.model_params, net=RuntimeModule.of(_lazy()))
    older = dataclasses.replace(checkpoint, model_params=before)
    module = _lazy()
    reloaded = NNModel.from_checkpoint(older, module=module)
    X = torch.randn(5, 4)
    np.testing.assert_array_equal(reloaded.predict(X).logits, model.predict(X).logits)
    assert reloaded.params.net == RuntimeModule.of(module) == model.params.net  # described as materialized
    assert NNModel(module=module, params=reloaded.params).net is module


def test_an_uninitialized_module_cannot_match_a_materialized_descriptor_without_weights():
    materialized = NNModel(module=_dry_run(_lazy()), params=NNModelParams(loss=Losses.CROSS_ENTROPY)).params
    module = _lazy()
    with pytest.raises(ValueError, match="lazy layers that are not materialized") as refused:
        NNModel(module=module, params=materialized)
    assert "dry-run forward" in str(refused.value) and "from_checkpoint" in str(refused.value)
    assert _is_uninitialized(module)
    # loaded weights (not yet a forward) are enough: the wrap completes the materialization
    module.load_state_dict(_dry_run(_lazy()).state_dict())
    assert NNModel(module=module, params=materialized).params == materialized
    assert RuntimeModule.of(module) == materialized.net


class _ImageLoader:
    @staticmethod
    def of(shape: tuple[int, ...]) -> DataLoader:
        generator = torch.Generator().manual_seed(0)
        X = torch.randn(12, *shape, generator=generator)
        return DataLoader(TensorDataset(X, (X.flatten(1)[:, 0] > 0).long()), batch_size=4)


@pytest.mark.parametrize(
    ("build", "shape"),
    [
        (
            lambda: nn.Sequential(nn.LazyConv1d(4, 3, groups=2), nn.LazyBatchNorm1d(), nn.Flatten(), nn.LazyLinear(2)),
            (2, 6),
        ),
        (
            lambda: nn.Sequential(
                nn.LazyConvTranspose2d(4, 2, groups=2),
                nn.LazyInstanceNorm2d(affine=True),
                nn.Flatten(),
                nn.LazyLinear(2),
            ),
            (2, 3, 3),
        ),
    ],
    ids=["conv1d-batchnorm", "convtranspose2d-instancenorm"],
)
def test_lazy_conv_and_norm_layers_reload_into_fresh_instances(build, shape):
    module = build()
    module(torch.zeros(2, *shape))
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_params(_ImageLoader.of(shape), epochs=1))
    checkpoint = NNCheckpoint.load(run=run.id, type=Checkpoints.LAST)
    assert checkpoint is not None
    fresh = build()
    reloaded = NNModel.from_checkpoint(checkpoint, module=fresh)
    assert RuntimeModule.of(fresh) == model.params.net
    X = torch.randn(3, *shape)
    np.testing.assert_allclose(reloaded.predict(X).logits, model.predict(X).logits, rtol=1e-6, atol=1e-6)


def test_a_top_level_lazy_layer_reloads_rewraps_and_is_described_as_materialized():
    # nn.LazyLinear switches its class to nn.Linear when it materializes
    model_a, ckpt_a, _ = _train_lazy(nn.LazyLinear(2))  # wrapped uninitialized
    model_b, ckpt_b, _ = _train_lazy(_dry_run(nn.LazyLinear(2)), data_id="dry")  # wrapped after a dry run
    assert model_a.params.net == RuntimeModule.of(model_a.net) == ckpt_a.model_params.net
    assert model_a.params.net.module == "torch.nn.modules.linear.Linear"
    X = torch.randn(5, 4)
    for model, checkpoint, module in (
        (model_a, ckpt_a, model_a.net),
        (model_a, ckpt_a, nn.LazyLinear(2)),
        (model_b, ckpt_b, nn.LazyLinear(2)),
        (model_b, ckpt_b, _dry_run(nn.LazyLinear(2))),
    ):
        reloaded = NNModel.from_checkpoint(checkpoint, module=module)
        assert reloaded.net is module and type(module) is nn.Linear
        np.testing.assert_array_equal(reloaded.predict(X).logits, model.predict(X).logits)
        assert RuntimeModule.of(module) == reloaded.params.net
    assert NNModel(module=model_a.net, params=model_a.params).net is model_a.net
    # a descriptor taken before materialization names the lazy remedy, not just two hashes
    before = NNModel(module=nn.LazyLinear(2), params=NNModelParams(loss=Losses.CROSS_ENTROPY)).params
    with pytest.raises(ValueError, match="fresh, unmaterialized instance"):
        NNModel(module=_dry_run(nn.LazyLinear(2)), params=before)
    with pytest.raises(ValueError, match="lazy layers that are not materialized"):
        NNModel(module=nn.LazyLinear(2), params=model_a.params)
    with pytest.raises(ValueError, match="does not match the descriptor"):
        NNModel.from_checkpoint(ckpt_a, module=nn.LazyLinear(3))


class _LazyScale(LazyModuleMixin, nn.Module):
    """A lazy layer torch does not ship: NNx cannot tell what its forward infers."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.parameter.UninitializedParameter()

    def initialize_parameters(self, x: torch.Tensor) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize((x.shape[-1],))
                self.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


def test_a_custom_lazy_layer_keeps_its_wrap_time_descriptor():
    def build() -> nn.Module:
        return nn.Sequential(_LazyScale(), nn.Linear(4, 2))

    model, checkpoint, _ = _train_lazy(build())
    # nothing could check a fresh instance against a materialized descriptor, so none is taken
    assert checkpoint.model_params.net == model.params.net == RuntimeModule.of(build())
    fresh = build()
    reloaded = NNModel.from_checkpoint(checkpoint, module=fresh)
    X = torch.randn(5, 4)
    np.testing.assert_array_equal(reloaded.predict(X).logits, model.predict(X).logits)
    # a run wrapped after a dry run reloads into a dry-run instance; the refusal says so
    _, dry_checkpoint, _ = _train_lazy(_dry_run(build()), data_id="dry")
    with pytest.raises(ValueError, match="before passing it as module=") as refused:
        NNModel.from_checkpoint(dry_checkpoint, module=build())
    assert "NNModel.from_checkpoint(" not in str(refused.value)
    assert NNModel.from_checkpoint(dry_checkpoint, module=_dry_run(build())).params == dry_checkpoint.model_params


def test_a_materialized_lazy_model_trains_again_under_its_materialized_run_id():
    params = _params(_loader(), epochs=1, overwrite_existing=True)
    model = NNModel(module=_lazy(), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    first = model.train(params=params).id
    second = model.train(params=params).id
    dry = NNModel(module=_dry_run(_lazy()), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    # documented: once materialized, the model is described (and its later runs keyed) as a dry-run-wrapped one
    assert first != second == NNRun(train=params, model=dry.params, net=None).id


def test_a_refused_lazy_reload_leaves_the_module_unmaterialized():
    _, checkpoint, _ = _train_lazy(_dry_run(_lazy()))
    fresh = _lazy()
    with pytest.raises(TypeError, match="batch_adapter"):
        NNModel.from_checkpoint(checkpoint, module=fresh, batch_adapter=object())  # type: ignore[arg-type]
    assert _is_uninitialized(fresh) and type(fresh[0]) is nn.LazyLinear
    loaded = _lazy()
    loaded.load_state_dict(checkpoint.net_state)  # weights in, not yet a forward
    with pytest.raises(TypeError, match="batch_adapter"):
        NNModel(module=loaded, params=checkpoint.model_params, batch_adapter=object())  # type: ignore[arg-type]
    assert type(loaded[0]) is nn.LazyLinear and hasattr(loaded[0], "_initialize_hook")


def test_to_onnx_refuses_a_mapping_example_before_writing(tmp_path):
    model = NNModel(module=nn.Linear(4, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    path = tmp_path / "model.onnx"
    for example in ({"features": torch.randn(2, 4)}, (torch.randn(2, 4), {"mask": torch.ones(2)})):
        with pytest.raises(TypeError, match="positional tensor inputs only"):
            model.to_onnx(str(path), example)
    assert not path.exists()


def test_an_eval_step_fn_runs_in_eval_mode_and_the_modes_come_back():
    from nnx import NNEvaluationDataPoint

    seen = []

    def evaluate(ctx):
        seen.append(ctx.model.net.training)
        return NNEvaluationDataPoint(loss=1.0)

    module = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    model.train(params=_params(_loader(), val_loader=_loader(1)), eval_step_fn=evaluate)
    assert seen == [False, False]
    assert all(m.training for m in module.modules())  # the training modes came back


def test_an_unbaked_pruned_runtime_module_trains_keeps_its_mask_and_resumes():
    from nnx.prune import magnitude_prune

    module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    magnitude_prune(module, 0.5, bake=False)
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_params(_loader()))
    assert float((module[0].weight == 0).float().mean()) == pytest.approx(0.5)
    warm = model.train(params=_params(_loader(), epochs=3, resume_from_run_id=run.id, resume_mode="stateful"))
    assert warm.resume_status is not None and warm.resume_status.mode == "stateful"
    assert float((module[0].weight == 0).float().mean()) == pytest.approx(0.5)


def _writable_spy(monkeypatch):
    # Torch warns about a read-only array once per process, so assert on the
    # conversion itself rather than on that warning.
    seen = []
    from_numpy = torch.from_numpy

    def spy(array):
        seen.append(array.flags.writeable)
        return from_numpy(array)

    monkeypatch.setattr(torch, "from_numpy", spy)
    X = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    X.flags.writeable = False  # what pandas' .to_numpy() returns
    model = NNModel(module=nn.Linear(4, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    return seen, model, X


def test_predict_hands_torch_a_writable_array(monkeypatch):
    seen, model, X = _writable_spy(monkeypatch)
    model.predict(X)
    assert seen and all(seen)


def test_onnx_exports_hand_torch_a_writable_array(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    from nnx.viz import netron_export

    seen, model, X = _writable_spy(monkeypatch)
    model.to_onnx(str(tmp_path / "model.onnx"), X)
    netron_export(model, str(tmp_path / "netron.onnx"), X)
    assert len(seen) >= 2 and all(seen)


def test_a_parametrized_runtime_module_trains_and_resumes():
    # The advice the train-time refusal gives for a registered net.
    module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    torch.nn.utils.parametrizations.weight_norm(module[0])
    model = NNModel(module=module, params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    run = model.train(params=_params(_loader()))
    warm = model.train(params=_params(_loader(), epochs=3, resume_from_run_id=run.id, resume_mode="stateful"))
    assert warm.resume_status is not None and warm.resume_status.mode == "stateful"
