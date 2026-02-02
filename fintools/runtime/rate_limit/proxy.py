from __future__ import annotations

from typing import cast
import pandas as pd
import threading

from .registry import GLOBAL_REGISTRY

class ClientProxy:
    def __init__(self, client, source: str, raise_on_exceed: bool = True, max_wait: float | None = None):
        self._client = client
        self._source = source
        self._raise_on_exceed = raise_on_exceed
        self._max_wait = max_wait

    def __getattr__(self, name: str):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs) -> pd.DataFrame:
            with GLOBAL_REGISTRY.guard(self._source, name, raise_on_exceed=self._raise_on_exceed, max_wait=self._max_wait):
                return cast(pd.DataFrame,attr(*args, **kwargs))

        return _wrapped