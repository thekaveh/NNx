"""Preference dataset for DPO.

Each example is a ``(prompt, chosen, rejected)`` triple where the
``chosen`` response is preferred over ``rejected`` for the given
``prompt``. The dataset encodes all three through an
:class:`NNTokenizerParams`, optionally truncates/pads to fixed
lengths, and yields ``(prompt_ids, chosen_ids, rejected_ids)`` as
``torch.LongTensor``\\ s — the shape expected by
:func:`nnx.paradigms.dpo.dpo_train_step_factory`.

For DPO experimentation, plug a published HF Hub preference dataset
(Anthropic HH, OpenAssistant, UltraFeedback, …) into the three lists
directly. The NNx scope is honestly "TinyStories-class preference
tuning, end-to-end on a laptop" — see ``docs/dpo.md`` for the
production-RLHF caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..._validation import require_batch_sizes
from .nn_dataset_base import NNDatasetBase


class _PreferenceCorpus(Dataset):
    """Internal ``torch.utils.data.Dataset`` over encoded preference triples.

    Holds pre-encoded long tensors so ``__getitem__`` is just a slice
    rather than per-call tokenization — DPO's training step already
    pays a 2-forward-passes-per-row cost (policy + reference), and we
    don't want to compound that with re-encoding the same prompt every
    epoch.
    """

    def __init__(
        self,
        prompt_ids: list[list[int]],
        chosen_ids: list[list[int]],
        rejected_ids: list[list[int]],
        max_prompt_len: int,
        max_response_len: int,
        pad_token_id: int,
    ):
        if not (len(prompt_ids) == len(chosen_ids) == len(rejected_ids)):
            raise ValueError(
                "NNPreferenceDataset: prompts, chosen, rejected must align — got "
                f"{len(prompt_ids)} / {len(chosen_ids)} / {len(rejected_ids)}"
            )
        if max_prompt_len <= 0:
            raise ValueError(f"max_prompt_len must be positive, got {max_prompt_len}")
        if max_response_len <= 0:
            raise ValueError(f"max_response_len must be positive, got {max_response_len}")

        # Pre-pad / truncate to fixed lengths so the DataLoader can
        # stack rows without a custom collate_fn. Truncation is right-
        # side (keep the prompt's start, keep the response's start) —
        # the LM cares more about what it conditioned on than the
        # tail of an oversize prompt.
        self._prompt = torch.stack([_pad_or_truncate(p, max_prompt_len, pad_token_id) for p in prompt_ids])
        self._chosen = torch.stack([_pad_or_truncate(c, max_response_len, pad_token_id) for c in chosen_ids])
        self._rejected = torch.stack([_pad_or_truncate(r, max_response_len, pad_token_id) for r in rejected_ids])

    def __len__(self) -> int:
        return self._prompt.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._prompt[idx], self._chosen[idx], self._rejected[idx]


def _pad_or_truncate(ids: list[int], length: int, pad_id: int) -> torch.Tensor:
    """Right-truncate to ``length`` and right-pad with ``pad_id``."""
    truncated = list(ids[:length])
    if len(truncated) < length:
        truncated = truncated + [pad_id] * (length - len(truncated))
    return torch.tensor(truncated, dtype=torch.long)


@dataclass(frozen=True, kw_only=True, slots=True)
class NNPreferenceDataset(NNDatasetBase):
    """Wrap parallel lists of (prompt, chosen, rejected) strings as DPO loaders.

    Tokenizes every triple through ``tokenizer.encode`` once at
    construction, pads/truncates to fixed lengths, then splits into
    train / val / test ``DataLoader``\\ s with the same shape as the
    rest of :class:`NNDatasetBase` (so callbacks and the standard
    training loop work unchanged).

    Each batch yielded is ``(prompt_ids, chosen_ids, rejected_ids)``
    where each entry is ``(B, T_*)`` ``torch.LongTensor``.

    ``batch_sizes`` is a ``(train, val, test)`` tuple. ``None`` (the
    default for every slot) means *one batch holding the complete split* —
    one DPO step per epoch for the default train loader; pass an explicit
    train size such as ``batch_sizes=(8, None, None)`` for mini-batches. A
    positive integer is kept verbatim (larger than the split → one smaller
    batch); NumPy integers are normalized to ``int``. Zero, ``False``,
    negatives, floats, strings and anything but a 3-tuple are rejected with
    the ``batch_sizes[i] (split)`` slot named before ``tokenizer.encode``
    is called. Zero never disables a split: `val_proportion=0.0` /
    `test_proportion=0.0` do (that loader is then ``None`` with a
    placeholder ``1`` in the resolved ``batch_sizes``).
    """

    prompts: list[str]
    chosen: list[str]
    rejected: list[str]
    tokenizer: object  # NNTokenizerParams — typed `object` to keep the lm extra optional

    max_prompt_len: int = 64
    max_response_len: int = 64
    pad_token_id: int = 0

    batch_sizes: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)
    val_proportion: float = 0.1
    test_proportion: float = 0.1
    name_override: Optional[str] = None
    seed: Optional[int] = field(default=None)

    def __post_init__(self):
        if not 0.0 <= self.val_proportion < 1.0:
            raise ValueError(f"val_proportion must be in [0, 1), got {self.val_proportion}")
        if not 0.0 <= self.test_proportion < 1.0:
            raise ValueError(f"test_proportion must be in [0, 1), got {self.test_proportion}")
        if self.val_proportion + self.test_proportion >= 1.0:
            raise ValueError(
                f"val_proportion + test_proportion must be < 1, got {self.val_proportion + self.test_proportion}"
            )
        if not (len(self.prompts) == len(self.chosen) == len(self.rejected)):
            raise ValueError(
                "NNPreferenceDataset: prompts, chosen, rejected must align — got "
                f"{len(self.prompts)} / {len(self.chosen)} / {len(self.rejected)}"
            )
        if len(self.prompts) == 0:
            raise ValueError("NNPreferenceDataset requires non-empty input lists")
        # Validate the batch-size request before a single tokenizer call
        # (FIX-022): `None` is the only full-split sentinel.
        requested = require_batch_sizes(self.batch_sizes, owner="NNPreferenceDataset")

        # Encode everything via the tokenizer up-front.
        encode = self.tokenizer.encode  # type: ignore[attr-defined]
        p_ids = [encode(p) for p in self.prompts]
        c_ids = [encode(c) for c in self.chosen]
        r_ids = [encode(r) for r in self.rejected]

        full = _PreferenceCorpus(
            prompt_ids=p_ids,
            chosen_ids=c_ids,
            rejected_ids=r_ids,
            max_prompt_len=self.max_prompt_len,
            max_response_len=self.max_response_len,
            pad_token_id=self.pad_token_id,
        )

        n_total = len(full)
        n_val = int(n_total * self.val_proportion)
        n_test = int(n_total * self.test_proportion)
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"NNPreferenceDataset: not enough rows ({n_total}) for the requested "
                f"val/test split ({self.val_proportion}, {self.test_proportion})"
            )

        # Deterministic split when seed is given — matches NNTabularDataset's
        # implicit contract that the same inputs round-trip to the same
        # train/val/test ids when seeded externally.
        # seed=None must genuinely fall back to the global torch RNG: a
        # fresh torch.Generator() always carries the same fixed default
        # seed, which would make unseeded splits bit-identical across
        # runs and deaf to torch.manual_seed.
        gen = torch.Generator().manual_seed(int(self.seed)) if self.seed is not None else torch.default_generator
        train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

        object.__setattr__(self, "name", self.name_override or "NNPreferenceDataset")

        # `None` → one batch holding the complete split (placeholder 1 for
        # an empty optional split); an explicit positive size is kept
        # verbatim — an `is None` test, not truthiness (FIX-022).
        train_batch_size = n_train if requested[0] is None else requested[0]
        val_batch_size = max(1, n_val) if requested[1] is None else requested[1]
        test_batch_size = max(1, n_test) if requested[2] is None else requested[2]
        object.__setattr__(self, "batch_sizes", (train_batch_size, val_batch_size, test_batch_size))

        object.__setattr__(
            self,
            "train_loader",
            DataLoader(train_ds, batch_size=train_batch_size, shuffle=True),
        )
        object.__setattr__(
            self,
            "val_loader",
            DataLoader(val_ds, batch_size=val_batch_size, shuffle=False) if n_val > 0 else None,
        )
        object.__setattr__(
            self,
            "test_loader",
            DataLoader(test_ds, batch_size=test_batch_size, shuffle=False) if n_test > 0 else None,
        )

        # input_dim / output_dim: there's no "feature dimension" for a
        # raw text-pair dataset — report the tokenizer's vocab size so
        # downstream sizing checks have a number to look at.
        vocab_size = int(self.tokenizer.vocab_size)  # type: ignore[attr-defined]
        object.__setattr__(self, "input_dim", vocab_size)
        object.__setattr__(self, "output_dim", vocab_size)

        object.__setattr__(
            self,
            "_state",
            dict(
                name=self.name,
                vocab_size=vocab_size,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                max_prompt_len=self.max_prompt_len,
                max_response_len=self.max_response_len,
                pad_token_id=self.pad_token_id,
            ),
        )
