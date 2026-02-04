from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Field:
    name: str

@dataclass(frozen=True)
class Const:
    value: float | int | bool | str

@dataclass(frozen=True)
class Call:
    fn: str
    args: Tuple["Node", ...]

Node = Field | Const | Call