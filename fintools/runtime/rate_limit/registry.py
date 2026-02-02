from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, Tuple
import time

from .policy import Policy, DEFAULT_POLICY, SOURCE_ENDPOINT
from .concurrency import ConcurrencyLimiter
from .limits_backend import LimitsBackend

import random


class RateLimitExceeded(RuntimeError):
    pass

class RateLimitRegistry:
    def __init__(self, backend: LimitsBackend | None = None):
        self.backend = backend or LimitsBackend()

        self._source_policy: Dict[str, Policy] = {}
        self._endpoint_policy: Dict[Tuple[str, str], Policy] = {}

        self._source_conc: Dict[str, ConcurrencyLimiter] = {}
        self._endpoint_conc: Dict[Tuple[str, str], ConcurrencyLimiter] = {}

    # ---- policy set ----
    def set_source_policy(self, source: str, policy: Policy):
        self._source_policy[source] = policy
        self._source_conc[source] = ConcurrencyLimiter(policy.max_concurrency)

    def set_endpoint_policy(self, source: str, endpoint: str, policy: Policy):
        self._endpoint_policy[(source, endpoint)] = policy
        self._endpoint_conc[(source, endpoint)] = ConcurrencyLimiter(policy.max_concurrency)

    def _resolve_source_policy(self, source: str) -> Policy:
        return self._source_policy.get(source, DEFAULT_POLICY)

    def _resolve_endpoint_policy(self, source: str, endpoint: str) -> Policy:
        return self._endpoint_policy.get((source, endpoint), self._resolve_source_policy(source))

    def guard(self, source: str, endpoint: str, raise_on_exceed: bool = True, max_wait: float | None = None):
        """
        同时施加：
        1) 源级并发/配额（全方法共享）
        2) endpoint 级并发/配额（方法额外限制）
        """
        src_pol = self._resolve_source_policy(source)
        ep_pol = self._resolve_endpoint_policy(source, endpoint)

        # 并发：源级 -> endpoint 级
        src_conc = self._source_conc.get(source) or ConcurrencyLimiter(src_pol.max_concurrency)
        ep_conc = self._endpoint_conc.get((source, endpoint)) or ConcurrencyLimiter(ep_pol.max_concurrency)

        # 配额 key：建议源级和 endpoint 级分开计数
        src_key = f"{source}:{SOURCE_ENDPOINT}"
        ep_key = f"{source}:{endpoint}"

        @contextmanager
        def _ctx():
            with src_conc.acquire():
                if not self.backend.hit(src_key, src_pol):
                    policy = self._resolve_source_policy(source)
                    if raise_on_exceed: raise RateLimitExceeded(f"source rate limit exceeded: {source}")
                    else:
                        start_wait = time.time()
                        while not self.backend.hit(src_key, src_pol):
                            if max_wait is not None and (time.time() - start_wait) >= max_wait:
                                raise RateLimitExceeded(f"source rate limit exceeded (max_wait={max_wait}s): {source}")
                            if policy.window == 'second': w = 1
                            else: w = 60
                            time.sleep(random.random() * w)
                with ep_conc.acquire():
                    if not self.backend.hit(ep_key, ep_pol):
                        policy = self._resolve_endpoint_policy(source, endpoint)
                        if raise_on_exceed: raise RateLimitExceeded(f"endpoint rate limit exceeded: {source}.{endpoint}")
                        else:
                            start_wait = time.time()
                            while not self.backend.hit(ep_key, ep_pol):
                                if max_wait is not None and (time.time() - start_wait) >= max_wait:
                                    raise RateLimitExceeded(f"endpoint rate limit exceeded (max_wait={max_wait}s): {source}.{endpoint}")
                                if policy.window == 'second': w = 1
                                else: w = 60
                                time.sleep(random.random() * w)
                    yield

        return _ctx()

# 全局单例（全库共享）
GLOBAL_REGISTRY = RateLimitRegistry()
