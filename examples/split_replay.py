"""Reproducible splits — plan, persist, replay (FEAT-017).

``NNTabularDataset`` splits with a seeded ``random_split`` by default: rows
of one patient can straddle train and test, and future rows can leak into
training. ``nnx.data_splits.plan_split`` decides membership by sample id
instead, and the resulting ``SplitManifest`` is plain JSON you can persist
and replay:

  1. **Group split (fixture G).** Ids ``s0..s5`` in groups ``a,a,b,b,c,c``
     with proportions ``(1/3, 1/3, 1/3)`` and seed 7: three two-row splits,
     one group each — no group spans two splits.
  2. **Chronological split (fixture T).** Times ``[1,2,2,3,4,5]`` with
     cutoffs ``(3, 5)``: gap 0 gives ``3/2/1`` rows; gap 1 excludes the rows
     just before each cutoff (``s1, s2, s4``) and gives ``1/1/1``.
  3. **Persist and replay after reordering.** The group manifest is saved,
     reloaded and fed to ``NNTabularDataset(split=..., id_col="id")`` on the
     rows in reverse order: every id stays in its split, with no
     ``random_split`` and no global RNG draw.
  4. **One split identity.** The reloaded manifest has the same digest, so
     ``ExperimentManifest(splits={...})`` (FEAT-019) records the same split.

Fully offline, CPU only.

Run:
    python examples/split_replay.py

The bounded ``split_replay_workflow()`` helper is executed by
``tests/test_examples_smoke.py -k split_replay`` in a temporary working
directory.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import torch

from nnx import NNTabularDataset
from nnx.data_splits import SplitManifest, plan_split
from nnx.provenance import ExperimentManifest


def _frame() -> pd.DataFrame:
    """Fixtures G and T in one frame; ``x`` encodes the id so loaders can be read back."""
    return pd.DataFrame(
        {
            "id": [f"s{i}" for i in range(6)],
            "group": ["a", "a", "b", "b", "c", "c"],
            "time": [1, 2, 2, 3, 4, 5],
            "x": [float(i) for i in range(6)],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )


def _ids(loader) -> list[str]:
    if loader is None:
        return []
    return [f"s{int(x)}" for batch_x, _ in loader for x in batch_x[:, 0].tolist()]


def _counts(ds: NNTabularDataset) -> tuple[int, int, int]:
    return len(_ids(ds.train_loader)), len(_ids(ds.val_loader)), len(_ids(ds.test_loader))


def split_replay_workflow() -> dict:
    frame = _frame()
    reordered = frame.iloc[::-1].reset_index(drop=True)  # same rows, reverse order

    # 1. Group split (G): whole groups per split.
    group = plan_split(frame["id"], strategy="group", groups=frame["group"], proportions=(1 / 3, 1 / 3, 1 / 3), seed=7)
    by_id = dict(zip(frame["id"], frame["group"], strict=True))
    for members in (group.train, group.validation, group.test):
        assert len(members) == 2 and len({by_id[i] for i in members}) == 1, members

    # 3. Persist, reload and replay on reordered rows.
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "group_split.json")
        group.save(path)
        reloaded = SplitManifest.load(path)
    assert reloaded == group and reloaded.digest() == group.digest()
    rng = torch.get_rng_state()
    ds = NNTabularDataset(df=reordered, feature_cols=["x"], target_col="y", split=reloaded, id_col="id")
    assert torch.equal(torch.get_rng_state(), rng), "an explicit split draws no randomness"
    assert sorted(_ids(ds.train_loader)) == list(group.train)
    assert _ids(ds.val_loader) == list(group.validation) and _ids(ds.test_loader) == list(group.test)
    counts = {"group": _counts(ds)}

    # 2. Chronological split (T), with and without a gap.
    excluded = {}
    for gap in (0, 1):
        plan = plan_split(frame["id"], strategy="chronological", times=frame["time"], cutoffs=(3, 5), gap=gap)
        replayed = SplitManifest.from_json(plan.to_json())
        ds = NNTabularDataset(df=reordered, feature_cols=["x"], target_col="y", split=replayed, id_col="id")
        counts[f"time_gap_{gap}"] = _counts(ds)
        excluded[f"time_gap_{gap}"] = list(replayed.excluded)
    assert counts == {"group": (2, 2, 2), "time_gap_0": (3, 2, 1), "time_gap_1": (1, 1, 1)}, counts
    assert excluded == {"time_gap_0": [], "time_gap_1": ["s1", "s2", "s4"]}, excluded

    # 4. Provenance records the same split identity for the plan and its replay.
    recorded = ExperimentManifest(splits={"group": reloaded}).splits["group"]
    assert recorded == group.identity() and recorded.value == group.digest()

    summary = {"counts": counts, "excluded": excluded, "group_digest": group.digest()}
    print(f"group split (train/val/test rows): {counts['group']}")
    print(f"time split, gap 0: {counts['time_gap_0']}")
    print(f"time split, gap 1: {counts['time_gap_1']} (excluded {excluded['time_gap_1']})")
    print(f"split identity: {group.digest()[:19]}…")
    return summary


def main() -> None:
    split_replay_workflow()


if __name__ == "__main__":
    main()
