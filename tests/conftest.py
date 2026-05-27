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
