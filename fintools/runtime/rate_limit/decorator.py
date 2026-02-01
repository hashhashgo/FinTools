from __future__ import annotations
import wrapt
from .registry import GLOBAL_REGISTRY

def rate_limited(source: str, endpoint: str | None = None):
    @wrapt.decorator
    def _wrapper(wrapped, instance, args, kwargs):
        ep = endpoint or wrapped.__name__
        with GLOBAL_REGISTRY.guard(source, ep):
            return wrapped(*args, **kwargs)
    return _wrapper
