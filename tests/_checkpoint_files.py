"""Helpers for tests that edit or republish a checkpoint's files on disk."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional


def restamp(sidecar: Path) -> None:
    """Record a deliberately edited sidecar in its resume-point manifest (#394),
    as if the run had written that content: the test then exercises the check
    it targets rather than the digest check."""
    manifest_path = sidecar.parent / (sidecar.name.split(".opt.")[0] + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


# Training-state keys and the NNCheckpoint.save keyword each one is written back through.
_SAVE_KEYWORDS = {
    "optimizer": "optimizer_state",
    "scheduler": "scheduler_state",
    "scaler": "scaler_state",
    "rng": "rng_state",
    "model": "resume_net_state",
    "optimizers": "optimizers_state",
    "schedulers": "schedulers_state",
}
_SAVED_AS_IS = (
    "optimizer_type",
    "scheduler_type",
    "optimizer_topology",
    "optimizer_factory",
    "optimizer_config",
    "scheduler_config",
    "optimizer_types",
    "scheduler_types",
    "optimizer_topologies",
    "optimizer_factories",
    "optimizer_configs",
    "scheduler_configs",
    "components",
    "precision",
    "compile",
    "distributed",
    "counters",
)


def republish(run_id: str, tag: Any, epoch: int = 5) -> None:
    """A concurrent writer (#537): a real ``NNCheckpoint.save`` publishes
    ``tag`` of ``run_id`` again — the next generation — now at ``epoch``,
    with the training state it holds (none for a weights-only checkpoint)."""
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    checkpoint, state = NNCheckpoint.load_with_training_state(run=run_id, type=tag)
    assert checkpoint is not None
    checkpoint = replace(checkpoint, idp=replace(checkpoint.idp, epoch_idx=epoch))
    if state is None:
        checkpoint.save(run=run_id, type=tag)
        return
    keywords = {_SAVE_KEYWORDS.get(key, key): state.get(key) for key in (*_SAVE_KEYWORDS, *_SAVED_AS_IS)}
    checkpoint.save(run=run_id, type=tag, completed_epoch=epoch, **keywords)


@contextlib.contextmanager
def republished_on_verify(run_id: str, tag: Any = "last", *, before: bool = False) -> Iterator[None]:
    """Wrap ``NNCheckpoint.verify`` so that :func:`republish` lands right
    after it verifies ``run_id``'s point — before the resume loads it — or,
    ``before=True``, just before it verifies (#537)."""
    from nnx.nn.params.nn_checkpoint import NNCheckpoint

    real_verify = NNCheckpoint.verify

    def verify(run: str, type: Any, root: Optional[str] = None) -> Any:
        if before and run == run_id:
            republish(run_id, tag)
        manifest = real_verify(run, type, root)
        if not before and run == run_id:
            republish(run_id, tag)
        return manifest

    NNCheckpoint.verify = staticmethod(verify)  # type: ignore[method-assign]
    try:
        yield
    finally:
        NNCheckpoint.verify = staticmethod(real_verify)  # type: ignore[method-assign]
