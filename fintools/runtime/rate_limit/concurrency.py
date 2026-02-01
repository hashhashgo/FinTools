from __future__ import annotations
import threading
from contextlib import contextmanager

class ConcurrencyLimiter:
    def __init__(self, max_concurrency: int):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._sem = threading.Semaphore(max_concurrency)

    @contextmanager
    def acquire(self):
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()
