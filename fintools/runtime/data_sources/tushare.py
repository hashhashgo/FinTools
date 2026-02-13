from ..rate_limit import (
    Policy,
    GLOBAL_REGISTRY,
    ClientProxy
)

from ..retry import RetryProxy


import os
import tushare

GLOBAL_REGISTRY.set_source_policy("tushare", Policy(
    max_concurrency = 5000,
    max_calls = None,
    window = "minute"
))

GLOBAL_REGISTRY.set_default_endpoint_policy("tushare", Policy(
    max_concurrency=5000,
    max_calls=500,
    window="minute"
))

GLOBAL_REGISTRY.set_endpoint_policy("tushare", "daily_basic", Policy(
    max_concurrency=5000,
    max_calls=700,
    window="minute"
))

GLOBAL_REGISTRY.set_endpoint_policy("tushare", "adj_factor", Policy(
    max_concurrency=5000,
    max_calls=1500,
    window="minute"
))

GLOBAL_REGISTRY.set_endpoint_policy("tushare", "stk_limit", Policy(
    max_concurrency=5000,
    max_calls=400,
    window="minute"
))

pro = ClientProxy(
    client = RetryProxy(tushare.pro_api(os.getenv("TUSHARE_API_KEY", "")), attempts=5),
    source = "tushare",
    raise_on_exceed = False,
    max_wait = None
)

__all__ = ["pro"]


if __name__ == "__main__":
    df = pro.daily(ts_code="000001.SZ", start_date="20220101", end_date="20221231")
    print(df)