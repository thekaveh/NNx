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
