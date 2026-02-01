from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from limits.storage import MemoryStorage, Storage
from limits.strategies import MovingWindowRateLimiter
from limits import (
    RateLimitItemPerSecond,
    RateLimitItemPerMinute,
    RateLimitItemPerHour,
    RateLimitItemPerDay,
)

from .policy import Policy

def _make_item(policy: Policy):
    if policy.max_calls is None:
        return None
    if policy.window == "second":
        return RateLimitItemPerSecond(policy.max_calls)
    if policy.window == "minute":
        return RateLimitItemPerMinute(policy.max_calls)
    if policy.window == "hour":
        return RateLimitItemPerHour(policy.max_calls)
    if policy.window == "day":
        return RateLimitItemPerDay(policy.max_calls)
    raise ValueError(f"Unknown window: {policy.window}")

class LimitsBackend:
    """
    配额引擎：只负责 hit() 判断是否允许；不负责并发。
    Storage 可换成 RedisStorage 实现跨进程/跨机器共享。
    """
    def __init__(self, storage: Storage | None = None):
        self.storage = storage or MemoryStorage()
        self.limiter = MovingWindowRateLimiter(self.storage)

    def hit(self, key: str, policy: Policy) -> bool:
        item = _make_item(policy)
        if item is None:
            return True
        return self.limiter.hit(item, key)
