from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import tqdm

from fintools.api.F.fin_history import get_data, DataFrequency, UnderlyingType
from fintools.utils.underlying import stock_basic

from fintools.runtime.rate_limit import GLOBAL_REGISTRY, Policy
from fintools.runtime.rate_limit.limits_backend import _make_item


def fetch_all_stock(last_trade_date = "20260130"):
    df_stocks = []
    underlyings = stock_basic()['ts_code'].unique()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(lambda u=underlying: get_data(
            datasource = "tushare",
            symbol = u,
            type = UnderlyingType.STOCK,
            freq = DataFrequency.DAILY,
            end = last_trade_date,
            only_standard_columns=False
        )) : underlying for underlying in underlyings}

        t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
        policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", "daily")
        item = _make_item(policy=policy)
        assert item is not None
        for future in t:
            t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, 'tushare:daily').remaining} / {policy.max_calls}"
            df = future.result()
            df_stocks.append(df)
    
    return pd.concat(df_stocks, ignore_index=True)