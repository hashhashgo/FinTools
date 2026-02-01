from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Window = Literal["second", "minute", "hour", "day"]

@dataclass(frozen=True)
class Policy:
    # 并发控制（进程内）
    max_concurrency: int = 1

    # 速率/配额（由 limits 负责；None 表示不限制）
    # 例如：max_calls=20, window="second" 表示 20 QPS
    max_calls: int | None = None
    window: Window = "second"

DEFAULT_POLICY = Policy(max_concurrency=1, max_calls=None, window="second")
SOURCE_ENDPOINT = "__source__"
