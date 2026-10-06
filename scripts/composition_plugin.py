"""A pytest plugin recording every test phase's outcome (FEAT-039).

``scripts/check_feature_composition.py --run`` loads it with
``-p scripts.composition_plugin`` and reads the JSON it writes to
``$NNX_COMPOSITION_PHASES``: for each node id, the outcome of its setup,
call and teardown (``passed`` / ``failed`` / ``skipped``) and whether it
was an expected failure. Nothing else is recorded."""

from __future__ import annotations

import json
import os
from typing import Any

_PHASES: dict[str, dict[str, Any]] = {}


def pytest_runtest_logreport(report: Any) -> None:
    entry = _PHASES.setdefault(report.nodeid, {"setup": None, "call": None, "teardown": None, "xfail": False})
    entry[report.when] = report.outcome
    if getattr(report, "wasxfail", None) is not None:
        entry["xfail"] = True


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    path = os.environ.get("NNX_COMPOSITION_PHASES")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_PHASES, handle, sort_keys=True)
