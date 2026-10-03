"""Portable, data-only run bundles (FEAT-015).

A run's checkpoints are pickles, safe to read only when you produced them.
``nnx.bundles`` publishes one run checkpoint — weights, training state and
calibrators — as safetensors plus schema-validated JSON, which anyone can
check and rebuild without unpickling anything. This script runs two
fixtures:

  1. **A built-in classifier** (``Nets.FEED_FWD`` with a labelled
     categorical task) and a temperature calibrator fitted for its weights:
     ``export_bundle`` → ``inspect_bundle`` / ``validate_bundle`` →
     ``reconstruct_bundle``. The rebuilt model predicts bit-for-bit like the
     trained one, the calibrator keeps its labels and the fingerprint of the
     weights it was fitted on, and resuming one epoch from the bundle gives
     exactly the weights of the uninterrupted run.
  2. **A registered module** (a ``ModelSpec`` whose factory this process no
     longer registers): reconstruction names the missing factory before
     building anything, then rebuilds from caller-supplied factories and
     resumes like the continuous run.

It also shows a tampered payload being refused before any tensor is read.
Fully offline, CPU only; every run and bundle is written under a temporary
directory.

Run:
    python examples/run_bundle.py

The bounded ``run_bundle_workflow()`` helper is executed by
``tests/test_examples_smoke.py`` in a temporary working directory, and the
script itself runs end to end there as a subprocess.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from nnx import (
    Activations,
    Devices,
    Losses,
    ModelSpec,
    Nets,
    NNModel,
    NNModelParams,
    NNOptimParams,
    NNParams,
    NNTrainParams,
    TaskSpec,
    register_model_factory,
    unregister_model_factory,
)
from nnx.bundles import (
    BundleIntegrityError,
    BundleReconstructionError,
    export_bundle,
    inspect_bundle,
    reconstruct_bundle,
    validate_bundle,
)
from nnx.calibration import fit_temperature, model_fingerprint

LABELS = ("cat", "dog", "fox")
DATA = torch.Generator().manual_seed(0)
X = torch.randn(48, 4, generator=DATA)
Y = torch.randint(0, 3, (48,), generator=DATA)
BATCHES = [(X[i : i + 16], Y[i : i + 16]) for i in range(0, 48, 16)]


class TinyEncoder(torch.nn.Module):
    """A module NNx does not ship: rebuilt from its registered factory."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(4, width), torch.nn.Tanh(), torch.nn.Linear(width, 3))

    def forward(self, x):
        return self.body(x)


def tiny_encoder(config) -> torch.nn.Module:
    return TinyEncoder(**config)


def _params(n_epochs: int, data_id: str) -> NNTrainParams:
    return NNTrainParams(
        n_epochs=n_epochs,
        data_id=data_id,  # the two schedules are distinct runs
        train_loader=BATCHES,
        optim=NNOptimParams.builder().adam(max_lr=0.01).build(),
        seed=0,
    )


def _classifier() -> NNModel:
    torch.manual_seed(7)
    return NNModel(
        net_params=NNParams(input_dim=4, output_dim=3, hidden_dims=[8], dropout_prob=0.1, activation=Activations.RELU),
        params=NNModelParams(
            net=Nets.FEED_FWD,
            device=Devices.CPU,
            loss=Losses.CROSS_ENTROPY,
            task=TaskSpec.categorical(3, labels=LABELS),
        ),
    )


def _same_weights(left: NNModel, right: NNModel) -> bool:
    right_state = right.net.state_dict()
    return all(torch.equal(tensor, right_state[name]) for name, tensor in left.net.state_dict().items())


def built_in_classifier() -> dict:
    continuous = _classifier()
    continuous.train(params=_params(3, "continuous"))  # the uninterrupted reference
    first = _classifier()
    run = first.train(params=_params(2, "bundled"))
    fit = fit_temperature(
        first.predict(X).logits, Y.numpy(), labels=LABELS, model_id=model_fingerprint(first), split_id="calibration"
    )
    calibrator = fit.calibrator
    assert calibrator is not None, fit.reason

    info = export_bundle(run.id, "classifier-bundle", calibrators=[calibrator])
    summary = inspect_bundle("classifier-bundle")  # manifest and state.json only: no tensor, no pickle
    assert summary.capability == "resume" and summary.epoch == 1 and validate_bundle("classifier-bundle").verified

    rebuilt = reconstruct_bundle("classifier-bundle")
    assert np.array_equal(rebuilt.model.predict(X).logits, first.predict(X).logits)  # bit for bit
    restored = rebuilt.calibrators[0]
    assert restored == calibrator and restored.model_id == model_fingerprint(rebuilt.model)
    rebuilt.resume(_params(1, "bundled"))  # epoch 2, from the bundle's optimizer / scheduler / RNG state
    assert _same_weights(continuous, rebuilt.model)
    return {"payloads": sorted(info.payloads), "temperature": round(restored.temperature, 3)}


def registered_module() -> dict:
    spec = ModelSpec("examples.tiny_encoder", 1, {"width": 6}, seed=3)
    params = NNModelParams(net=spec, device=Devices.CPU, loss=Losses.CROSS_ENTROPY)
    register_model_factory("examples.tiny_encoder", 1, tiny_encoder)
    try:
        torch.manual_seed(7)
        continuous = NNModel(params=params)
        continuous.train(params=_params(3, "continuous-module"))
        torch.manual_seed(7)
        first = NNModel(params=params)
        run = first.train(params=_params(2, "bundled-module"))
        export_bundle(run.id, "module-bundle")
    finally:
        unregister_model_factory("examples.tiny_encoder", 1)

    try:
        reconstruct_bundle("module-bundle")  # this process no longer registers the factory
    except BundleReconstructionError as refused:
        missing = refused.problems
    else:  # pragma: no cover - the factory is not registered
        raise AssertionError("reconstruction built a model without its factory")
    rebuilt = reconstruct_bundle("module-bundle", factories={("examples.tiny_encoder", 1): tiny_encoder})
    assert np.array_equal(rebuilt.model.predict(X).logits, first.predict(X).logits)
    rebuilt.resume(_params(1, "bundled-module"))
    assert _same_weights(continuous, rebuilt.model)
    return {"missing": len(missing), "model": str(spec)}


def tampered_bundle() -> str:
    payload = next(
        os.path.join("classifier-bundle", entry, "model.safetensors")
        for entry in os.listdir("classifier-bundle")
        if entry.startswith("g-")
    )
    with open(payload, "r+b") as handle:  # flip one byte of the weights
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 0xFF]))
    try:
        validate_bundle("classifier-bundle")
    except BundleIntegrityError as refused:
        return str(refused)
    raise AssertionError("a corrupted payload validated")  # pragma: no cover


def run_bundle_workflow() -> dict:
    classifier = built_in_classifier()
    module = registered_module()
    refusal = tampered_bundle()
    print(
        f"classifier bundle {classifier['payloads']}: bit-for-bit predictions, calibrator T={classifier['temperature']}"
    )
    print("resumed one epoch from each bundle == the uninterrupted run (built-in and registered module)")
    print(f"registered module {module['model']}: {module['missing']} missing factory named before any model was built")
    print(f"tampered payload refused before any tensor was read: {refusal}")
    return {"classifier": classifier, "module": module, "refusal": refusal}


def main() -> None:
    home = os.getcwd()
    with tempfile.TemporaryDirectory() as root:
        os.chdir(root)  # runs/ and the bundles land in the temporary directory
        try:
            run_bundle_workflow()
        finally:
            os.chdir(home)


if __name__ == "__main__":
    main()
