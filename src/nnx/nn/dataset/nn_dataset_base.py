from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Protocol, TypeVar


class DatasetProtocol(Protocol):
    def __iter__(self) -> Iterable: ...

    def __len__(self) -> int: ...


DatasetType = TypeVar("DatasetType", bound=DatasetProtocol)


@dataclass(frozen=True, kw_only=True, slots=True)
class NNDatasetBase(ABC):
    name: str = field(init=False)

    input_dim: int = field(init=False)
    output_dim: int = field(init=False)

    train_loader: DatasetProtocol = field(init=False)
    # val/test may be None: the tabular and preference subclasses set
    # them to None for empty splits (their documented contract).
    val_loader: Optional[DatasetProtocol] = field(init=False)
    test_loader: Optional[DatasetProtocol] = field(init=False)

    _state: dict = field(repr=False, init=False)

    def state(self) -> dict:
        return self._state
