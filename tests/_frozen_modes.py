"""Helpers for tests that a frozen KD teacher or DPO reference runs every
forward in eval mode, whatever switched it back to train mode."""

from __future__ import annotations

import torch

from nnx import Callback


class TrainModeAtEpochStart(Callback):
    """Switches ``module`` back to train mode at every epoch start, as user code might."""

    def __init__(self, module: torch.nn.Module):
        self.module = module

    def on_epoch_begin(self, ctx) -> None:
        self.module.train()


def record_forward_modes(net: torch.nn.Module) -> list[tuple[bool, bool]]:
    """Record ``(training, output == eval-mode output)`` at every forward of ``net``."""
    seen: list[tuple[bool, bool]] = []
    replaying = [False]

    def hook(module, inputs, output):
        if replaying[0]:  # the eval-mode replay below
            return
        was, replaying[0] = module.training, True
        try:
            module.eval()
            with torch.no_grad():
                expected = module(*inputs)
            module.train(was)
        finally:
            replaying[0] = False
        seen.append((was, torch.equal(output, expected)))

    net.register_forward_hook(hook)
    return seen


def record_mode_switches(monkeypatch, net: torch.nn.Module) -> list[bool]:
    """Record the ``mode`` of every later ``net.train(mode)`` / ``net.eval()`` call."""
    switches: list[bool] = []
    original = net.train

    def counting_train(mode: bool = True):
        switches.append(mode)
        return original(mode)

    monkeypatch.setattr(net, "train", counting_train)
    return switches
