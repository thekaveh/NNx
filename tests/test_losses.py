"""Tests for the Losses enum factory.

Each variant must return the correct torch.nn loss class. Catches the regression
where MEAN_SQUARED_ERROR and BINARY_CROSS_ENTROPY were swapped in the match
expression — a silent bug shipped between commits 3962995 and the next release.
"""
from __future__ import annotations

from torch import nn

from nnx.nn.enum.losses import Losses


def test_cross_entropy_returns_ce_loss():
    loss = Losses.CROSS_ENTROPY()
    assert isinstance(loss, nn.CrossEntropyLoss)


def test_mean_squared_error_returns_mse_loss():
    """Regression: was returning nn.BCELoss() in a swap with BCE."""
    loss = Losses.MEAN_SQUARED_ERROR()
    assert isinstance(loss, nn.MSELoss)


def test_binary_cross_entropy_returns_bce_loss():
    """Regression: was returning nn.MSELoss() in a swap with MSE."""
    loss = Losses.BINARY_CROSS_ENTROPY()
    assert isinstance(loss, nn.BCELoss)


def test_negative_log_likelihood_returns_nll_loss():
    loss = Losses.NEGATIVE_LOG_LIKELIHOOD()
    assert isinstance(loss, nn.NLLLoss)


def test_all_enum_variants_have_a_factory_branch():
    """If a new variant is added to Losses, this test fails until the factory
    is updated. Belt-and-braces against another silent miss in __call__."""
    for variant in Losses:
        loss = variant()
        assert loss is not None
        assert isinstance(loss, nn.Module)
