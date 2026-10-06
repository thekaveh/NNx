"""Opt-in torch.compile, measured: compile cost, warmed speed, a profile (FEAT-029).

Compilation is opt-in and **no speedup is guaranteed** — on a tiny model the
compiled forward is often slower than eager. This example measures instead
of assuming:

  1. **Train compiled.** A small classifier trains one epoch with
     ``model.train(params, compile=CompileSpec())`` (the default ``inductor``
     backend). ``run.compile`` records the request, the effective state and
     the graph capture.
  2. **Reload eagerly.** The run's LAST checkpoint holds the eager module's
     keys (no ``_orig_mod.`` prefix); ``NNModel.from_checkpoint`` rebuilds a
     plain model whose logits equal the trained net's.
  3. **Benchmark.** ``compare_compile`` times eager and compiled copies of
     the same weights: the **first call** (including compilation) separately
     from the **warmed** latency and throughput over ``--repeats`` calls
     after ``--warmup`` untimed ones, with mean / median / stdev.
  4. **Profile.** ``profile_forward`` runs ``torch.profiler`` with a finite
     wait / warmup / active schedule into ``--output-dir`` — outside any
     timed region.

Run:
    python examples/benchmark_compile.py --device cpu --warmup 3 --repeats 10

It prints one JSON document with separate ``compile``, ``warmed`` and
``profile`` sections. The inductor backend needs a working C++ toolchain
(on macOS, the system ``libc++`` must be loadable). Registered in
``tests/test_examples_smoke.py``, which runs exactly that command.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch

from nnx import (
    Activations,
    Checkpoints,
    CompileSpec,
    Devices,
    Losses,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
)
from nnx.benchmarking import compare_compile, profile_forward
from nnx.nn.params.nn_checkpoint import NNCheckpoint

DEVICES = {"cpu": Devices.CPU, "cuda": Devices.CUDA, "mps": Devices.MPS}


def _model(device: Devices) -> NNModel:
    torch.manual_seed(0)
    return NNModel(
        net_params=NNParams(
            input_dim=32, output_dim=4, hidden_dims=[128, 128], dropout_prob=0.0, activation=Activations.RELU
        ),
        params=NNModelParams(net=Nets.FEED_FWD, device=device, loss=Losses.CROSS_ENTROPY),
    )


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(256, 32, generator=generator)
    return X, (X[:, :4].argmax(dim=1))


def run(device: str = "cpu", warmup: int = 3, repeats: int = 10, output_dir: str = "profile") -> dict:
    X, y = _data()
    model = _model(DEVICES[device])
    batches = [(X[i : i + 64], y[i : i + 64]) for i in range(0, len(X), 64)]
    trained = model.train(
        params=NNTrainParams(
            n_epochs=1,
            train_loader=batches,
            optim=NNOptimParams.builder().sgd(max_lr=0.05).build(),
            overwrite_existing=True,
        ),
        compile=CompileSpec(),
    )
    if trained.compile is None or trained.compile.effective != "compiled":
        raise RuntimeError(f"the fit did not compile: {trained.compile}")

    checkpoint = NNCheckpoint.load(trained.id, Checkpoints.LAST)
    if checkpoint is None or any("_orig_mod" in key for key in checkpoint.net_state):
        raise RuntimeError("LAST must hold the eager module's keys")
    reloaded = NNModel.from_checkpoint(checkpoint)
    sample = X[:64].to(model.device)
    with torch.no_grad():
        torch.testing.assert_close(reloaded.net(sample), model.net(sample))

    comparison = compare_compile(model.net, sample, compile=CompileSpec(), warmup=warmup, repeats=repeats)
    profile = profile_forward(model.net, sample, output_dir=output_dir, wait=1, warmup=1, active=3, repeat=1)
    eager, compiled = comparison.eager.state(), comparison.compiled.state()
    return {
        "compile": {
            "training": trained.compile.record(),
            "first_call_seconds": {"eager": eager["first_call_seconds"], "compiled": compiled["first_call_seconds"]},
            "checkpoint_reloaded_eagerly": True,
        },
        "warmed": {
            "eager": {k: eager[k] for k in ("mean_seconds", "median_seconds", "stdev_seconds", "throughput")},
            "compiled": {k: compiled[k] for k in ("mean_seconds", "median_seconds", "stdev_seconds", "throughput")},
            "speedup": comparison.speedup,
            "max_abs_diff": comparison.max_abs_diff,
            "shapes": eager["shapes"],
            "dtypes": eager["dtypes"],
            "device": eager["device"],
            "backend": compiled["backend"],
            "warmup": warmup,
            "repeats": repeats,
            "weights_digest_match": eager["weights_digest"] == compiled["weights_digest"],
        },
        "profile": {
            "output_dir": profile.output_dir,
            "trace_files": [os.path.basename(path) for path in profile.trace_files],
            "schedule": dict(profile.schedule),
            "steps": profile.steps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=sorted(DEVICES), default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", default=None, help="profiler traces (default: a temporary directory)")
    args = parser.parse_args()
    if args.output_dir is None:
        with tempfile.TemporaryDirectory() as directory:
            print(json.dumps(run(args.device, args.warmup, args.repeats, os.path.join(directory, "profile")), indent=2))
    else:
        print(json.dumps(run(args.device, args.warmup, args.repeats, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
