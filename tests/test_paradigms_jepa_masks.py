"""``random_block_mask`` validates its ranges and never returns an empty
context."""

from __future__ import annotations

import pytest
import torch

from nnx.paradigms.jepa import random_block_mask


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_scale": (0.0, 0.2)}, "block_scale"),
        ({"block_scale": (0.3, 0.2)}, "block_scale"),
        ({"block_scale": (0.2, 1.0)}, "block_scale"),
        ({"block_aspect": (0.0, 1.0)}, "block_aspect"),
        ({"block_aspect": (2.0, 1.0)}, "block_aspect"),
    ],
)
def test_out_of_range_block_shapes_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        random_block_mask(n_patches=16, grid_size=4, **kwargs)


def test_a_block_covering_the_whole_grid_is_refused():
    with pytest.raises(ValueError, match="no context"):
        random_block_mask(n_patches=1, grid_size=1)


def test_masks_are_complementary_and_the_context_is_never_empty():
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        context, target = random_block_mask(n_patches=16, grid_size=4, generator=generator)
        assert torch.equal(context, ~target) and context.any() and target.any()
