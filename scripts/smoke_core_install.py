"""Installed-artifact smoke test for one dependency profile (FEAT-031).

Run it with the interpreter of a clean environment where a built NNx wheel
(or sdist) is installed — never from the checkout, and with ``PYTHONPATH``
unset — naming the profile that environment was installed with::

    python smoke_core_install.py --profile core --output report.json
    python smoke_core_install.py --profile graph  # thekaveh-nnx[graph]

Profiles: ``core`` (no extra), ``vision``, ``graph``, ``plots`` and
``domains`` (all three). For every profile it checks that ``import nnx``
loads none of torchvision / torch_geometric / plotly, that the package is
the installed one and ships ``py.typed``, that the installed distributions
(the wheel's resolved Requires-Dist closure) hold exactly the profile's
domain packages, and that a tiny CPU feed-forward model trains, predicts and
saves / reloads. Each installed extra's own feature then runs, and every
absent one raises an ImportError naming its extra. The report records the
installed closure and three cold ``import nnx`` timings — informational, no
threshold is asserted.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DOMAIN = {"vision": "torchvision", "graph": "torch_geometric", "plots": "plotly"}
PROFILES = {
    "core": set(),
    "vision": {"vision"},
    "graph": {"graph"},
    "plots": {"plots"},
    "domains": {"vision", "graph", "plots"},
}


def _distribution_names() -> set[str]:
    names = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


def _cold_import_seconds(runs: int = 3) -> list[float]:
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        subprocess.run([sys.executable, "-c", "import nnx"], check=True)
        timings.append(round(time.perf_counter() - started, 3))
    return timings


def _core_round_trip(workdir: Path) -> None:
    import numpy as np
    import torch

    import nnx

    torch.manual_seed(0)
    X, Y = torch.randn(32, 4), torch.randint(0, 2, (32,))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=8)
    model = nnx.NNModel(
        net_params=nnx.NNParams(
            input_dim=4, output_dim=2, hidden_dims=[8], dropout_prob=0.0, activation=nnx.Activations.RELU
        ),
        params=nnx.NNModelParams(net=nnx.Nets.FEED_FWD, device=nnx.Devices.CPU, loss=nnx.Losses.CROSS_ENTROPY),
    )
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        run = model.train(
            params=nnx.NNTrainParams(
                n_epochs=2,
                train_loader=loader,
                val_loader=loader,
                optim=nnx.NNOptimParams(name=nnx.Optims.ADAM, max_lr=1e-2, momentum=(0.9, 0.999), weight_decay=0.0),
            )
        )
        before = model.predict(X).logits
        reloaded = nnx.NNModel.from_checkpoint(nnx.NNCheckpoint.load(run=run.id, type=nnx.Checkpoints.LAST))
    finally:
        os.chdir(cwd)
    assert np.allclose(reloaded.predict(X).logits, before), "a reloaded checkpoint predicts differently"


def _extra_feature(extra: str) -> None:
    """Exercise one installed extra through the public surface."""
    import torch

    import nnx

    if extra == "vision":
        from nnx.nn.dataset.nn_dataset import NNDataset

        _same_class(NNDataset)
    elif extra == "graph":
        from nnx.nn.dataset.nn_graph_dataset import NNGraphDataset
        from nnx.nn.net.graph_att_nn import GraphAttNN
        from nnx.nn.net.graph_conv_nn import GraphConvNN
        from nnx.nn.net.graph_sage_nn import GraphSageNN

        for cls in (NNGraphDataset, GraphAttNN, GraphConvNN, GraphSageNN):
            _same_class(cls)
        params = nnx.NNParams(
            input_dim=4, output_dim=2, hidden_dims=[4], dropout_prob=0.0, activation=nnx.Activations.RELU, n_heads=1
        )
        for member, cls in ((nnx.Nets.GRAPH_ATT, GraphAttNN), (nnx.Nets.GRAPH_SAGE, GraphSageNN)):
            assert type(member(params)) is cls
        net = nnx.Nets.GRAPH_CONV(params)
        assert type(net) is GraphConvNN
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        assert net(torch.randn(3, 4), edge_index).shape == (3, 2)
    elif extra == "plots":
        figure = nnx.VisUtils.confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0], class_names=["a", "b"])
        assert type(figure).__name__ == "Figure"


def _same_class(cls: type) -> None:
    """A domain class reached through the flat ``nnx`` name: the same object
    as the deep import, listed in ``__all__`` and ``dir()``, same signature."""
    import inspect

    import nnx

    flat = getattr(nnx, cls.__name__)
    assert flat is cls, f"nnx.{cls.__name__} is not {cls.__module__}.{cls.__name__}"
    assert cls.__name__ in nnx.__all__ and cls.__name__ in dir(nnx)
    assert inspect.signature(flat) == inspect.signature(cls)


def _absent_feature(extra: str) -> None:
    import nnx

    attempts = {
        "vision": lambda: nnx.NNDataset,
        "graph": lambda: nnx.GraphConvNN,
        "plots": lambda: nnx.VisUtils.confusion_matrix([0, 1], [0, 1], class_names=["a", "b"]),
    }
    try:
        attempts[extra]()
    except ImportError as error:
        assert f'pip install "thekaveh-nnx[{extra}]"' in str(error), error
    else:
        raise AssertionError(f"the {extra} feature worked without the {extra} extra")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output", type=Path, default=None, help="where to write the JSON report")
    args = parser.parse_args(argv)
    extras = PROFILES[args.profile]
    if os.environ.get("PYTHONPATH"):
        print("PYTHONPATH must be unset: the smoke checks the installed package", file=sys.stderr)
        return 2

    import nnx

    package_dir = Path(nnx.__file__).resolve().parent
    assert "site-packages" in package_dir.parts, f"nnx was imported from {package_dir}, not an installed artifact"
    assert (package_dir / "py.typed").is_file(), "the installed package ships no py.typed marker"
    loaded = sorted(module for module in DOMAIN.values() if module in sys.modules)
    assert loaded == [], f"`import nnx` loaded domain packages: {loaded}"

    installed = _distribution_names()
    for extra, module in DOMAIN.items():
        distribution = module.replace("_", "-")
        present = distribution in installed
        assert present == (extra in extras), f"{distribution} installed={present} for profile {args.profile!r}"

    nets = [member.value for member in nnx.Nets]
    assert nets == ["conv", "feed_fwd", "feed_fwd_moe", "graph_att", "graph_conv", "graph_sage", "transformer"], nets
    with tempfile.TemporaryDirectory() as workdir:
        _core_round_trip(Path(workdir))
    loaded = sorted(module for module in DOMAIN.values() if module in sys.modules)
    assert loaded == [], f"training a feed-forward model loaded domain packages: {loaded}"
    for extra in DOMAIN:
        (_extra_feature if extra in extras else _absent_feature)(extra)

    report = {
        "profile": args.profile,
        "nnx": nnx.__version__,
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
        "installed_closure": sorted(installed),
        "cold_import_seconds": _cold_import_seconds(),
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
