"""Shared pytest configuration for the nnx test suite.

Today this file registers session-wide hygiene (suppressing tqdm output
so pytest's runs stay clean) and per-test env_snapshot cache resets so
tests that chdir don't pollute each other's metadata.yaml. When other
real shared fixtures emerge from repeated boilerplate across multiple
tests, add them here. Don't add fixtures pre-emptively — unused fixtures
are dead code that mislead future contributors about what's shared in
practice.
"""

from __future__ import annotations

import os

# macOS + torch + faiss-cpu both bundle libomp.dylib statically. When both
# end up in the same process, faiss aborts on its first kernel call with
# "OMP Error #15: Initializing libomp.dylib, but found libomp.dylib
# already initialized." or segfaults outright. Setting this env var
# BEFORE either library imports tells libomp to tolerate the collision.
# Set early — conftest is imported before any test module — so the FAISS
# export tests don't have to scramble. Harmless on Linux CI where the
# duplicate doesn't occur. ``setdefault`` preserves any caller override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Even with KMP_DUPLICATE_LIB_OK, faiss-cpu on macOS segfaults inside
# its parallel search kernel when torch's libomp got loaded first.
# Pinning OMP to a single thread sidesteps the buggy parallel path
# entirely. Tests don't need multi-threaded BLAS / FAISS to be fast;
# the suite already runs in ~15s single-threaded. Linux CI doesn't hit
# either problem but is also unaffected by the pin (still fast).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_tqdm_in_tests():
    """Set NNX_TQDM_DISABLE=1 for the entire test session so progress
    bars don't pollute pytest output. Autouse so individual tests don't
    have to remember."""
    os.environ["NNX_TQDM_DISABLE"] = "1"
    yield
    os.environ.pop("NNX_TQDM_DISABLE", None)


@pytest.fixture(autouse=True)
def _reset_env_snapshot_cache():
    """Clear the env_snapshot memo before/after every test.

    The cache is correct for production (env doesn't change during a
    single training run, and we want to avoid re-running `git rev-parse`
    once per epoch in NNRun.save). But tests that `monkeypatch.chdir` to
    a fresh tmp_path between runs would otherwise see the previous
    test's git_commit/git_dirty values cached.

    Autouse so individual tests don't have to remember. The
    test_v3_env_snapshot_is_cached_across_calls test still works because
    it exercises caching within a single test body (where this fixture
    is set up and torn down once).
    """
    from nnx import seeding

    seeding._ENV_SNAPSHOT_CACHE = None
    yield
    seeding._ENV_SNAPSHOT_CACHE = None


def _skip_if_dynamo_dispatch_error(exc: BaseException) -> None:
    """If ``exc`` is a torch dynamo-ONNX dispatch error (no ONNX function
    registered for one of the prims/aten ops the current torch emits),
    convert it into ``pytest.skip`` with an explanatory reason. Otherwise
    re-raise.

    The dynamo ONNX exporter (``torch.onnx.export(..., dynamo=True)``) is
    opt-in (NNx defaults to the legacy TorchScript path) and its op
    coverage depends on which ONNX functions the installed ``onnxscript``
    registers for the ATen / prims ops the current torch version's
    decomposition table emits. Newer torch releases occasionally emit
    prims/aten ops (e.g. ``prims.view_of``, ``aten.sym_size.int``) that
    the installed onnxscript hasn't dispatched yet — and vice versa. The
    failure surfaces as
    ``torch.onnx._internal.exporter._errors.ConversionError`` wrapping a
    ``DispatchError``.

    Tests that exercise the dynamo path use this helper so a torch /
    onnxscript drift mismatch in CI or a contributor's environment doesn't
    fail the suite over a known upstream incompatibility. The legacy path
    (the NNx default) is unaffected and has its own tests.
    """
    msg = str(exc)
    if "DispatchError" in msg or "No ONNX function found" in msg:
        pytest.skip(f"installed torch/onnxscript can't dispatch dynamo ONNX path (upstream skew): {msg[:200]}")
    raise exc


@pytest.fixture(scope="session")
def skip_on_dynamo_dispatch_error():
    """Yield ``_skip_if_dynamo_dispatch_error`` so tests can call it from
    an ``except`` block around their ``torch.onnx.export(..., dynamo=True)``
    invocation. See the helper's docstring for the upstream-skew rationale."""
    return _skip_if_dynamo_dispatch_error
