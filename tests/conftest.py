"""Shared pytest configuration for the nnx test suite.

Today this file only registers session-wide hygiene (suppressing tqdm
output so pytest's runs stay clean). When real shared fixtures emerge
from repeated boilerplate across multiple tests, add them here. Don't
add fixtures pre-emptively — unused fixtures are dead code that mislead
future contributors about what's shared in practice.
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
