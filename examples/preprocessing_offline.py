"""Train-only preprocessing and split transforms, offline (FEAT-018).

Statistics fitted on every row leak held-out values into training. This
example fits on the training membership only and serves the model with the
same frozen statistics:

  1. **Fit on training rows (fixture S).** ``[[0,5],[2,5],[1000,5]]`` with
     rows 0–1 as the training split: mean ``[1,5]``, scale ``[1,1]`` (the
     constant column gets scale 1). The held-out ``1000`` never reaches the
     statistics; it standardizes to ``999``. The source DataFrame is left
     unchanged.
  2. **Train, save, reload, predict.** ``NNTabularDataset(standardize=True,
     split=...)`` trains a tiny model on prepared features. The standardizer
     is saved as JSON and reloaded without refitting. ``NNModel.predict``
     takes **model-ready** input: the reloaded standardizer is applied
     exactly once to the raw rows, which reproduces the training features
     bit for bit (applying it twice would not).
  3. **Split transforms (fixture V).** A tiny in-memory vision dataset whose
     training view adds a sampled offset and whose evaluation view adds 1.
     Views share one untouched base dataset: its ``transform`` stays
     ``None``, labels pass through, and evaluation reads repeat exactly.

Fully offline, CPU only.

Run:
    python examples/preprocessing_offline.py

The bounded ``preprocessing_offline_workflow()`` helper is executed by
``tests/test_examples_smoke.py -k preprocessing_offline`` in a temporary
working directory.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import torch
from torchvision.datasets import VisionDataset

from nnx import Losses, NNModel, NNModelParams, NNOptimParams, NNTabularDataset, NNTrainParams
from nnx.data_splits import SplitManifest
from nnx.nn.dataset.nn_dataset import NNDataset
from nnx.preprocessing import Standardizer


class TinyVision(VisionDataset):
    """Fixture V: 12 train / 6 test in-memory samples of shape (1, 2, 2)."""

    classes = ["a", "b", "c"]

    def __init__(self, root, train=True, download=False, transform=None):
        super().__init__(root, transform=transform)
        n = 12 if train else 6
        self.data = torch.arange(n * 4, dtype=torch.float32).reshape(n, 1, 2, 2)
        self.targets = torch.arange(n) % 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[idx])


class AddSampledOffset:
    """Training augmentation: a random offset in [0, 1)."""

    def __call__(self, x):
        return x + torch.rand(())


class AddOne:
    """Evaluation transform: deterministic."""

    def __call__(self, x):
        return x + 1


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"id": ["r0", "r1", "r2"], "a": [0.0, 2.0, 1000.0], "b": [5.0, 5.0, 5.0], "y": [0, 1, 0]})


def preprocessing_offline_workflow() -> dict:
    torch.manual_seed(0)
    frame = _frame()
    source = frame.copy(deep=True)

    # 1. Fit on the training membership only (S).
    plan = SplitManifest(strategy="explicit", train=("r0", "r1"), validation=(), test=("r2",))
    ds = NNTabularDataset(df=frame, feature_cols=["a", "b"], target_col="y", split=plan, id_col="id", standardize=True)
    fitted = ds.standardizer
    assert fitted is not None and fitted.mean == (1.0, 5.0) and fitted.scale == (1.0, 1.0), fitted
    test_rows = torch.cat([x for x, _ in ds.test_loader])
    assert test_rows.tolist() == [[999.0, 0.0]], test_rows  # held-out value ignored by the fit
    assert frame.equals(source), "the source DataFrame is never modified"

    # 2. Train on prepared features, then save, reload and predict.
    model = NNModel(module=torch.nn.Linear(2, 2), params=NNModelParams(loss=Losses.CROSS_ENTROPY))
    model.train(
        params=NNTrainParams(
            n_epochs=2,
            train_loader=ds.train_loader,
            optim=NNOptimParams.builder().sgd(max_lr=0.1).build(),
            data_id="preprocessing-offline",
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "standardizer.json")
        fitted.save(path)
        reloaded = Standardizer.load(path)  # no refit: the frozen statistics come back exactly
    assert reloaded == fitted and reloaded.digest() == fitted.digest()
    model_ready = reloaded.transform(frame)  # raw rows → prepared features, applied once
    assert torch.equal(model_ready[:2], ds.train_loader.dataset.dataset.tensors[0][:2])
    assert not torch.equal(reloaded.transform(model_ready), model_ready), "a second pass would change them"
    predictions = model.predict(model_ready)

    # 3. Independent training / evaluation views over one base dataset (V).
    with tempfile.TemporaryDirectory() as root:
        vision = NNDataset(
            ds_class=TinyVision,
            root_dir=root,
            download=False,
            val_proportion=0.25,
            seed=0,
            batch_sizes=(4, 4, 4),
            train_transform=AddSampledOffset(),
            eval_transform=AddOne(),
        )
    train_view, val_view = vision.train_loader.dataset, vision.val_loader.dataset
    base = val_view.base
    first = [val_view[i][0] for i in range(len(val_view))]
    _ = [train_view[i] for i in range(len(train_view))]  # augmented training reads in between
    again = [val_view[i][0] for i in range(len(val_view))]
    assert base.transform is None and all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert all(torch.equal(x, base.data[i] + 1) for x, i in zip(first, val_view.indices, strict=True))
    assert [val_view[i][1] for i in range(len(val_view))] == [int(base.targets[i]) for i in val_view.indices]

    summary = {
        "mean": list(fitted.mean),
        "scale": list(fitted.scale),
        "held_out": test_rows.tolist()[0],
        "classes": predictions.classes.tolist(),
        "views": {"train": len(train_view), "validation": len(val_view), "test": len(vision.test_loader.dataset)},
    }
    print(f"fitted on training rows only: mean={summary['mean']} scale={summary['scale']}")
    print(f"held-out row [1000, 5] -> {summary['held_out']} (never part of the statistics)")
    print("NNModel.predict takes model-ready input: model.predict(standardizer.transform(raw_frame))")
    print(f"predicted classes after reload (no refit): {summary['classes']}")
    print(f"split views (train/val/test): {summary['views']}, base transform untouched")
    return summary


def main() -> None:
    preprocessing_offline_workflow()


if __name__ == "__main__":
    main()
