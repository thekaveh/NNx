"""FEAT-031: torchvision, torch_geometric and plotly are extras, not core.

Each check runs in a fresh interpreter whose import system refuses the three
domain packages, so the core-only behaviour is exercised inside the ordinary
all-extras suite. ``scripts/smoke_core_install.py`` checks the same contract
against an installed wheel in a clean environment (CI's installed-artifact
profiles).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ("torchvision", "torch_geometric", "plotly")

BLOCKER = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {blocked!r}

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{fullname!r}}", name=fullname)
            return None

    sys.meta_path.insert(0, _Block())
    """
)


def _run(code: str, *, blocked=DOMAIN, tmp_path: Path) -> str:
    script = BLOCKER.format(blocked=tuple(blocked)) + textwrap.dedent(code)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


def test_the_core_dependencies_exclude_the_domain_stacks():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = " ".join(project["dependencies"])
    for package in DOMAIN:
        assert package not in core
    extras = project["optional-dependencies"]
    assert extras["vision"] == ["torchvision>=0.19"]
    assert extras["graph"] == ["torch_geometric>=2.4"]
    assert extras["plots"] == ["plotly>=5.18"]
    assert extras["domains"] == ["thekaveh-nnx[vision,graph,plots]"]
    assert any(req.startswith("plotly") for req in extras["viz"])  # the viz figures keep working


def test_a_core_install_trains_predicts_saves_and_reloads_without_the_domains(tmp_path):
    out = _run(
        """
        import sys
        import numpy as np
        import torch
        import nnx
        from nnx import *  # the star import works on core: domain names are exported only when installed
        from nnx import (Activations, Devices, Losses, Nets, NNModel, NNModelParams, NNOptimParams, NNParams,
                         NNTrainParams, Optims)

        torch.manual_seed(0)
        X, Y = torch.randn(32, 4), torch.randint(0, 2, (32,))
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=8)
        model = NNModel(
            net_params=NNParams(input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=Activations.RELU),
            params=NNModelParams(net=Nets.FEED_FWD, device=Devices.CPU, loss=Losses.CROSS_ENTROPY),
        )
        run = model.train(params=NNTrainParams(
            n_epochs=2, train_loader=loader, val_loader=loader,
            optim=NNOptimParams(name=Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
        ))
        before = model.predict(X).logits
        reloaded = NNModel.from_checkpoint(nnx.NNCheckpoint.load(run=run.id, type=nnx.Checkpoints.LAST))
        assert np.allclose(reloaded.predict(X).logits, before)
        assert "NNDataset" not in nnx.__all__ and "GraphAttNN" not in nnx.__all__
        assert "NNTabularDataset" in nnx.__all__ and "GraphNNBase" in nnx.__all__
        html = run._repr_html_()  # degrades to its table without the chart
        assert "<table" in html and "plotly" not in html.lower()
        df = nnx.VisUtils.classification_report([0, 1, 1, 0], [0, 1, 0, 0])
        assert type(df).__name__ == "DataFrame"
        print(sorted(m for m in ("torchvision", "torch_geometric", "plotly") if m in sys.modules))
        """,
        tmp_path=tmp_path,
    )
    assert out.strip().splitlines()[-1] == "[]"


@pytest.mark.parametrize(
    ("code", "extra"),
    [
        ("import nnx; nnx.GraphAttNN", "graph"),
        ("import nnx; nnx.NNGraphDataset", "graph"),
        ("from nnx import GraphSageNN", "graph"),
        ("import nnx.nn.net.graph_conv_nn", "graph"),
        (
            "import nnx; nnx.Nets.GRAPH_CONV(nnx.NNParams(input_dim=4, output_dim=2, hidden_dims=[4], "
            "dropout_prob=0.0, activation=nnx.Activations.RELU))",
            "graph",
        ),
        ("import nnx; nnx.graph_tasks.GraphCollection([], [])", "graph"),
        ("import nnx; nnx.NNDataset", "vision"),
        ("from nnx import NNDataset", "vision"),
        ("import nnx; nnx.VisUtils.multi_line_plot", None),  # attribute access only: no import yet
        ("import nnx, pandas as pd; nnx.VisUtils.confusion_matrix([0, 1], [0, 1], ['a', 'b'])", "plots"),
        ("import nnx, torch; nnx.viz.gradient_flow(torch.nn.Linear(2, 2))", "plots"),
    ],
)
def test_a_domain_request_names_its_extra(code, extra, tmp_path):
    out = _run(
        f"""
        try:
            {code}
        except ImportError as error:
            print("ImportError:", error)
        else:
            print("ok")
        import nnx, torch
        torch.zeros(1)  # unrelated imports and APIs keep working afterwards
        """,
        tmp_path=tmp_path,
    )
    if extra is None:
        assert out.strip() == "ok"
    else:
        assert f'pip install "thekaveh-nnx[{extra}]"' in out, out


def test_lr_finder_refuses_before_touching_data_model_or_rng(tmp_path):
    out = _run(
        """
        import random
        import numpy as np
        import torch
        import nnx

        class Loader:
            def __iter__(self):
                raise AssertionError("data touched")

        torch.manual_seed(1)
        model = nnx.NNModel(
            net_params=nnx.NNParams(input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0,
                                    activation=nnx.Activations.RELU),
            params=nnx.NNModelParams(net=nnx.Nets.FEED_FWD, device=nnx.Devices.CPU, loss=nnx.Losses.CROSS_ENTROPY),
        )
        weights = {k: v.clone() for k, v in model.net.state_dict().items()}
        states = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
        try:
            nnx.lr_finder(model.net, Loader(), loss_fn=torch.nn.functional.cross_entropy)
        except ImportError as error:
            assert 'pip install "thekaveh-nnx[plots]"' in str(error), error
        else:
            raise AssertionError("lr_finder ran without plotly")
        assert all(torch.equal(v, model.net.state_dict()[k]) for k, v in weights.items())
        assert random.getstate() == states[0]
        assert (np.random.get_state()[1] == states[1]).all()
        assert torch.equal(torch.get_rng_state(), states[2])
        print("refused cleanly")
        """,
        tmp_path=tmp_path,
    )
    assert "refused cleanly" in out


def test_an_import_error_from_inside_an_installed_package_keeps_its_cause(tmp_path):
    """Only the domain package itself missing becomes the extra's message; a
    broken dependency of an installed package propagates unchanged."""
    out = _run(
        """
        import importlib.abc, sys

        class _Broken(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch_geometric.nn":
                    raise ModuleNotFoundError("No module named 'torch_scatter_inner'", name="torch_scatter_inner")
                return None

        sys.meta_path.insert(0, _Broken())
        import nnx
        try:
            nnx.GraphAttNN
        except ModuleNotFoundError as error:
            assert error.name == "torch_scatter_inner", error
            assert "thekaveh-nnx" not in str(error)
            print("kept its cause")
        """,
        blocked=(),
        tmp_path=tmp_path,
    )
    assert "kept its cause" in out


def test_a_core_install_lists_only_usable_names_and_pydoc_works(tmp_path):
    out = _run(
        """
        import inspect, pydoc
        import nnx
        assert "GraphAttNN" not in dir(nnx) and "NNDataset" not in dir(nnx)
        inspect.getmembers(nnx)  # getattr on every listed name
        pydoc.render_doc(nnx)
        print("help works")
        """,
        tmp_path=tmp_path,
    )
    assert "help works" in out


def test_a_broken_submodule_of_an_installed_package_is_not_reported_missing(tmp_path):
    out = _run(
        """
        import importlib.abc, sys

        class _NoLoader(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch_geometric.loader":
                    raise ModuleNotFoundError("No module named 'torch_geometric.loader'", name="torch_geometric.loader")
                return None

        sys.meta_path.insert(0, _NoLoader())
        import nnx
        try:
            nnx.NNGraphDataset
        except ModuleNotFoundError as error:
            assert error.name == "torch_geometric.loader" and "thekaveh-nnx" not in str(error), error
            print("kept its cause")
        """,
        blocked=(),
        tmp_path=tmp_path,
    )
    assert "kept its cause" in out


def test_plot_features_refuse_before_running_the_model(tmp_path):
    out = _run(
        """
        import torch, nnx
        calls = []
        net = torch.nn.Sequential(torch.nn.Linear(2, 2))
        net[0].register_forward_hook(lambda *a: calls.append(1))
        for attempt in (
            lambda: nnx.viz.activation_map(net, torch.randn(1, 2), "0"),
            lambda: nnx.viz.attribute(net, torch.randn(1, 2), method="saliency", target=0),
        ):
            try:
                attempt()
            except ImportError as error:
                assert 'pip install "thekaveh-nnx[plots]"' in str(error), error
        assert calls == [], "the model ran before the refusal"
        print("refused first")
        """,
        tmp_path=tmp_path,
    )
    assert "refused first" in out


def test_with_the_extras_installed_the_flat_names_are_the_deep_classes():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torchvision")
    import inspect

    import nnx
    from nnx.nn.dataset.nn_dataset import NNDataset
    from nnx.nn.dataset.nn_graph_dataset import NNGraphDataset
    from nnx.nn.net.graph_att_nn import GraphAttNN
    from nnx.nn.net.graph_conv_nn import GraphConvNN
    from nnx.nn.net.graph_sage_nn import GraphSageNN

    for cls in (NNDataset, NNGraphDataset, GraphAttNN, GraphConvNN, GraphSageNN):
        assert getattr(nnx, cls.__name__) is cls
        assert cls.__name__ in nnx.__all__ and cls.__name__ in dir(nnx)
        assert inspect.signature(getattr(nnx, cls.__name__)) == inspect.signature(cls)
    assert [n.value for n in nnx.Nets] == [
        "conv",
        "feed_fwd",
        "feed_fwd_moe",
        "graph_att",
        "graph_conv",
        "graph_sage",
        "transformer",
        "vit",  # #395
    ]
    with pytest.raises(AttributeError, match="no attribute 'NotAThing'"):
        nnx.NotAThing  # noqa: B018
