from __future__ import annotations

from enum import Enum


class Checkpoints(Enum):
    Q1      = "q1"
    Q2      = "q2"
    Q3      = "q3"
    BEST    = "best"
    LAST    = "last"
    FIRST   = "first"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)
