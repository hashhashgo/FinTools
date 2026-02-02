from ..rate_limit import (
    Policy,
    GLOBAL_REGISTRY,
    RateLimitExceeded, rate_limited,
    ClientProxy
)
from typing import cast
from tenacity import retry, stop_after_attempt, wait_exponential, stop_never
import wrapt
from wrapt.wrappers import ObjectProxy

class RetryProxy(wrapt.ObjectProxy):
    def __init__(self, wrapped, attempts=-1, wait=wait_exponential(multiplier=0.1, min=0.1, max=2)):
        super().__init__(wrapped)
        self.__stop__ = stop_after_attempt(attempts) if attempts > 0 else stop_never
        self.__wait__ = wait

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        attr = super().__getattr__(name)
        if callable(attr):
            return retry(stop=self.__stop__, wait=self.__wait__)(attr)
        else: return attr

import dotenv
dotenv.load_dotenv()
import os
import tushare

GLOBAL_REGISTRY.set_source_policy("tushare", Policy(
    max_concurrency = 5000,
    max_calls = None,
    window = "minute"
))

GLOBAL_REGISTRY.set_endpoint_policy("tushare", "daily", Policy(
    max_concurrency=5000,
    max_calls=500,
    window="minute"
))

# pro = RetryProxy(tushare.pro_api(os.getenv("TUSHARE_API_KEY", "")), attempts=5)
pro = ClientProxy(
    client = RetryProxy(tushare.pro_api(os.getenv("TUSHARE_API_KEY", "")), attempts=5),
    source = "tushare",
    raise_on_exceed = False,
    max_wait = 120
)

__all__ = ["pro"]


if __name__ == "__main__":
    df = pro.daily(ts_code="000001.SZ", start_date="20220101", end_date="20221231")
    print(df)