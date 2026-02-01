from __future__ import annotations
import wrapt
from .registry import GLOBAL_REGISTRY

class ClientProxy:
    def __init__(self, client, source: str):
        self._client = client
        self._source = source

    def __getattr__(self, name: str):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        @wrapt.decorator
        def _wrapped(wrapped, instance, args, kwargs):
            with GLOBAL_REGISTRY.guard(self._source, name):
                return wrapped(*args, **kwargs)

        return _wrapped(attr)
