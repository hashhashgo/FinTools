from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, TypeAlias

@dataclass(frozen=True)
class Field:
    name: str

@dataclass(frozen=True)
class Const:
    value: float | int | bool | str

@dataclass(frozen=True)
class ParamAssign:
    name: str
    value: Node

@dataclass(frozen=True)
class Call:
    fn: str
    args: Tuple["Node", ...]
    kwargs: Dict[str, "Node"]

Node: TypeAlias = Field | Const | Call | ParamAssign