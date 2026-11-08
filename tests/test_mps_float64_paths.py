"""MPS has no float64: every float64 sum or copy NNx makes from a tensor that
may live on the model's device widens on the CPU. Each test covers one path
that crashed with "Cannot convert a MPS Tensor to float64 dtype"."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")


def test_low_rank_factorize_keeps_the_factors_on_mps():
    from nnx.surgery.low_rank import low_rank_factorize

    layer = torch.nn.Linear(8, 6).to("mps")
    factors = low_rank_factorize(layer, rank=6)
    x = torch.randn(3, 8, device="mps")
    assert all(p.device.type == "mps" for p in factors.parameters())
    torch.testing.assert_close(factors(x), layer(x), atol=1e-4, rtol=1e-4)


def test_a_regression_task_accumulates_on_mps():
    from nnx import TaskSpec
    from nnx.tasks import task_adapter

    adapter = task_adapter(TaskSpec.regression(1))
    accumulator = adapter.accumulator()
    output = torch.randn(4, 1, device="mps")
    target = torch.randn(4, 1, device="mps")
    accumulator.update(output, target, torch.ones(4, 1, dtype=torch.bool, device="mps"))
    assert accumulator.count == 4


def test_causal_lm_token_sums_on_mps():
    from nnx.lm_tasks import CausalLMTask

    task = CausalLMTask(vocab_size=11, alignment="pre_shifted")
    logits = torch.randn(2, 3, 11, device="mps")
    targets = torch.randint(0, 11, (2, 3), device="mps")
    _, nll, _, n = task.token_sums(
        logits.reshape(-1, 11), targets.reshape(-1), torch.ones(6, dtype=torch.bool, device="mps")
    )
    assert n == 6 and math.isfinite(nll)


def test_standardizer_fits_an_mps_tensor():
    from nnx.preprocessing import Standardizer

    data = torch.randn(10, 3, device="mps")
    fitted = Standardizer.fit(data, rows=range(8))
    reference = Standardizer.fit(data.cpu(), rows=range(8))
    assert list(fitted.mean) == list(reference.mean) and list(fitted.scale) == list(reference.scale)


def test_the_diffusion_schedule_fingerprint_is_device_independent():
    from nnx import NoiseSchedulers
    from nnx.diffusion.objective import _schedule_fingerprint

    schedule = NoiseSchedulers.LINEAR(10)
    assert _schedule_fingerprint(schedule.to("mps")) == _schedule_fingerprint(schedule)


def test_a_summary_by_input_size_leaves_an_mps_model_on_its_device():
    pytest.importorskip("torchinfo")
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams
    from nnx.viz import summary

    model = NNModel(
        net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.MPS, loss=Losses.CROSS_ENTROPY),
    )
    summary(model, input_size=(2, 4))
    assert all(p.device.type == "mps" for p in model.net.parameters())


def test_compare_compile_diffs_an_mps_model_in_float64_on_the_cpu():
    from nnx import CompileSpec
    from nnx.benchmarking import compare_compile

    torch.manual_seed(0)
    net = torch.nn.Linear(4, 2).to("mps")
    comparison = compare_compile(
        net, torch.randn(3, 4, device="mps"), compile=CompileSpec(backend="aot_eager"), warmup=0, repeats=1
    )
    assert comparison.compiled.compile.effective == "compiled" and comparison.max_abs_diff == 0.0


def test_max_abs_diff_widens_every_mps_chunk_on_the_cpu(monkeypatch):
    import nnx.benchmarking as benchmarking

    monkeypatch.setattr(benchmarking, "_DIFF_CHUNK", 4)
    a = torch.arange(10, dtype=torch.float32, device="mps")
    b = a.clone()
    b[-1] += 2
    assert benchmarking._abs_diff(a, b) == 2.0


def test_a_bernoulli_nll_evaluates_on_mps_in_float64():
    """#518: the Bernoulli nll's sigmoid widens on the CPU, so an MPS model's
    confident target-0 terms match BCEWithLogitsLoss, not the 1e-12 floor."""
    import numpy as np

    from nnx import Activations, Devices, Losses, MetricSpec, Nets, NNModel, NNModelParams, NNParams, TaskSpec
    from nnx._probability import nll_terms
    from nnx.monitors import _batch_inputs

    logits = torch.arange(-27600, 27601, dtype=torch.float32, device="mps") / 1000
    for target in (0.0, 1.0):
        targets = torch.full_like(logits, target)
        truth, inputs = _batch_inputs("bernoulli", targets, logits, None, None, frozenset({"probabilities"}))
        exact = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.cpu().double(), targets.cpu().double(), reduction="none"
        )
        np.testing.assert_allclose(nll_terms(truth, inputs["probabilities"]), exact.numpy(), rtol=0, atol=1e-3)

    model = NNModel(
        net_params=NNParams(input_dim=1, output_dim=1, hidden_dims=[], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD, device=Devices.MPS, loss=Losses.BINARY_CROSS_ENTROPY, task=TaskSpec.multilabel(1)
        ),
    )
    with torch.no_grad():
        model.net.layers[0].weight.fill_(1.0)
        model.net.layers[0].bias.zero_()
    record = model.evaluate([(torch.tensor([[17.0]]), torch.tensor([[0.0]]))], metrics=[MetricSpec("nll")])
    assert next(model.net.parameters()).device.type == "mps"
    assert record.metrics["nll"] == pytest.approx(17.0, abs=1e-3)


def _mps_feed_forward():
    from nnx import Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNParams

    return NNModel(
        net_params=NNParams(input_dim=3, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=Activations.RELU),
        params=NNModelParams(net=Nets.FEED_FWD, device=Devices.MPS, loss=Losses.CROSS_ENTROPY),
    )


@pytest.mark.parametrize("kind", ["ndarray", "tensor", "tuple"])
def test_netron_export_casts_a_float64_example_before_the_mps_move(tmp_path, kind):
    """#556: the example was moved to MPS, then cast, and the move raised."""
    pytest.importorskip("onnx")
    import numpy as np

    from nnx.viz import netron_export

    x64 = np.random.default_rng(0).normal(size=(2, 3))
    example = {"ndarray": x64, "tensor": torch.from_numpy(x64), "tuple": (torch.from_numpy(x64),)}[kind]
    model = _mps_feed_forward()
    path = netron_export(model, str(tmp_path / f"{kind}.onnx"), example)
    assert (tmp_path / f"{kind}.onnx").exists() and path.endswith(".onnx")
    assert all(p.device.type == "mps" for p in model.net.parameters())


def test_netron_export_keeps_integer_token_inputs_on_mps(tmp_path):
    pytest.importorskip("onnx")
    from nnx.viz import netron_export

    net = torch.nn.Sequential(torch.nn.Embedding(10, 4), torch.nn.Linear(4, 2)).to("mps")
    assert netron_export(net, str(tmp_path / "tokens.onnx"), torch.tensor([[1, 2, 3]]))
    assert (tmp_path / "tokens.onnx").exists()


def _float64_graphs():
    import numpy as np

    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from nnx.graph_tasks import GraphCollection

    def chain(n):
        src = torch.arange(n - 1)
        return torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])

    rng = np.random.default_rng(0)
    graphs = [Data(x=torch.from_numpy(rng.normal(size=(n, 3))), edge_index=chain(n)) for n in (3, 4, 5)]
    return GraphCollection(graphs, [1, 2, 3], targets=[0, 1, 2])


def test_a_graph_classifier_predicts_evaluates_and_trains_on_float64_graphs_on_mps(tmp_path, monkeypatch):
    """#556: GraphClassifier.forward cast float64 features after NNModel moved them to MPS."""
    from nnx import Devices, Losses, NNModel, NNModelParams, NNOptimParams, NNTrainParams, TaskSpec
    from nnx.graph_tasks import graph_classifier_spec

    collection = _float64_graphs()
    results = {}
    for device in (Devices.CPU, Devices.MPS):
        spec = graph_classifier_spec(input_dim=3, num_classes=3, hidden_dims=[8], pool="mean", seed=0)
        model = NNModel(
            params=NNModelParams(net=spec, device=device, loss=Losses.CROSS_ENTROPY, task=TaskSpec.categorical(3))
        )
        predicted = model.predict(collection.loader(batch_size=3))
        record = model.evaluate(collection.loader(batch_size=3))
        results[device] = (predicted.logits.shape, record.loss)
    assert results[Devices.MPS][0] == results[Devices.CPU][0] == (3, 3)
    assert results[Devices.MPS][1] == pytest.approx(results[Devices.CPU][1], rel=1e-4)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NNX_TQDM_DISABLE", "1")
    run = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=collection.loader(batch_size=3),
            optim=NNOptimParams.builder().adam(max_lr=1e-2).build(),
        )
    )
    assert math.isfinite(run.idps[-1].train_edp.loss) and next(model.net.parameters()).device.type == "mps"


def test_a_link_task_predicts_float64_node_features_on_mps():
    """#556: LinkTask._logits moved float64 features to MPS before LinkPredictor.forward cast them."""
    pytest.importorskip("torch_geometric")
    from nnx import Devices, Losses, NNModel, NNModelParams
    from nnx.link_tasks import LinkTask, link_predictor_spec, split_links

    g = torch.Generator().manual_seed(0)
    n = 30
    community = torch.arange(n) % 2
    edges = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if torch.rand(1, generator=g).item() < (0.4 if community[u] == community[v] else 0.03)
    ]
    x = torch.nn.functional.one_hot(community, 2).double() + 0.1 * torch.randn(n, 2, generator=g).double()
    task = LinkTask(split_links(torch.tensor(edges).t().contiguous(), n, val=0.1, test=0.1, seed=0))
    logits = {}
    for device in (Devices.CPU, Devices.MPS):
        spec = link_predictor_spec(input_dim=2, hidden_dims=[16], seed=0)
        model = NNModel(params=NNModelParams(net=spec, device=device, loss=Losses.MEAN_SQUARED_ERROR))
        logits[device] = task.predict(model, task.loader("test", x, batch_size=10_000)).logits
    assert logits[Devices.MPS].shape == logits[Devices.CPU].shape == (18,)
    torch.testing.assert_close(
        torch.from_numpy(logits[Devices.MPS]), torch.from_numpy(logits[Devices.CPU]), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
def test_a_summary_by_batch_places_a_loader_batch_on_the_mps_model(keyword):
    """#556: summary(batch=) handed torchinfo the loader's CPU batch, which an MPS model refused."""
    pytest.importorskip("torchinfo")
    from torch.utils.data import DataLoader, TensorDataset

    from nnx import Devices, Losses, NNModel, NNModelParams
    from nnx.models import KeywordInputs
    from nnx.viz import summary

    class Keyword(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, *, features):
            return self.linear(features)

    if keyword:
        model = NNModel(
            module=Keyword(),
            params=NNModelParams(device=Devices.MPS, loss=Losses.CROSS_ENTROPY),
            batch_adapter=KeywordInputs(("features",), target="labels"),
        )
        rows = [{"features": torch.randn(3), "labels": torch.tensor(i % 2)} for i in range(8)]
        batch = next(iter(DataLoader(rows, batch_size=4)))
        expected = sum(p.numel() for p in model.net.parameters())
    else:
        model = _mps_feed_forward()
        batch = next(iter(DataLoader(TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,))), batch_size=4)))
        expected = summary(model, input_size=(4, 3)).total_params
    stats = summary(model, batch=batch)
    assert type(stats).__name__ == "ModelStatistics" and stats.total_params == expected
    assert all(p.device.type == "mps" for p in model.net.parameters())
