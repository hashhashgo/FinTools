from __future__ import annotations
from typing import Dict, Tuple

from .policy import Policy, DEFAULT_POLICY, SOURCE_ENDPOINT
from .concurrency import ConcurrencyLimiter
from .limits_backend import LimitsBackend

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

    def guard(self, source: str, endpoint: str):
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

        # 这里用 contextmanager 风格返回一个可用的上下文
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            with src_conc.acquire():
                if not self.backend.hit(src_key, src_pol):
                    raise RateLimitExceeded(f"source rate limit exceeded: {source}")
                with ep_conc.acquire():
                    if not self.backend.hit(ep_key, ep_pol):
                        raise RateLimitExceeded(f"endpoint rate limit exceeded: {source}.{endpoint}")
                    yield

        return _ctx()

# 全局单例（全库共享）
GLOBAL_REGISTRY = RateLimitRegistry()
