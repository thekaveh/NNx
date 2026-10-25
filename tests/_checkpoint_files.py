"""Helpers for tests that edit a checkpoint's files on disk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def restamp(sidecar: Path) -> None:
    """Record a deliberately edited sidecar in its resume-point manifest (#394),
    as if the run had written that content: the test then exercises the check
    it targets rather than the digest check."""
    manifest_path = sidecar.parent / (sidecar.name.split(".opt.")[0] + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
