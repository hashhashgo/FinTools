from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import tqdm
import os
from pathlib import Path

from fintools.api.F.fin_history import get_data, DataFrequency, UnderlyingType
from fintools.data_sources.fin_history import DATASOURCES
from fintools.databases import DB_CONNECTIONS
from fintools.utils.underlying import stock_basic

from fintools.runtime.rate_limit import GLOBAL_REGISTRY, Policy
from fintools.runtime.rate_limit.limits_backend import _make_item
from fintools.runtime.data_sources.tushare import pro


def _fetch_all_stock_deep(df: pd.DataFrame):
    df_stocks = []
    underlyings = stock_basic()['ts_code'].unique()
    last_trade_dates = df.groupby('ts_code')['trade_date'].max().to_dict()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(lambda u=underlying: get_data(
            datasource = "tushare",
            symbol = u,
            type = UnderlyingType.STOCK,
            freq = DataFrequency.DAILY,
            start = last_trade_dates.get(u, 0),
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

def _fetch_all_stock_shallow(last_trade_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")):
    df_stocks = []
    offset = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        is_end = False
        while not is_end:
            futures = [executor.submit(lambda off=o: pro.daily(offset=off)) for o in range(offset, offset + 60000, 6000)]
            t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
            policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", "daily")
            item = _make_item(policy=policy)
            assert item is not None
            for future in t:
                t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, 'tushare:daily').remaining} / {policy.max_calls}"
                df = future.result()
                df_stocks.append(df)
                if df['trade_date'].max() < last_trade_date:
                    is_end = True
            offset += 60000
    df_all = pd.concat(df_stocks, ignore_index=True)
    return df_all

def fetch_all_stock():
    db = DB_CONNECTIONS['fintools.data_sources.fin_history.tushare:TushareDataSource.history']
    common_fields = {"type": UnderlyingType.STOCK, "freq": DataFrequency.DAILY}
    table_name, cursor = db.get_table_name_and_cursor(common_fields=common_fields)
    
    parquet_path = Path(os.getenv("PARQUET_STORAGE_PATH", "./parquet_storage")) / f'{table_name}.parquet'
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        cursor.execute(f"SELECT count(*) as cnt from {table_name}")
        total_count = cursor.fetchone()['cnt']
        df_stocks = []
        for df in tqdm.tqdm(pd.read_sql(f"SELECT * from {table_name}", cursor.connection, chunksize=100000), total=(total_count // 100000) + 1, desc="Loading data from DB"):
            df = db.format_dataframe(df, common_fields=common_fields)
            df_stocks.append(df)
        df = pd.concat(df_stocks, ignore_index=True)
        del df_stocks
    last_trade_date = df['date'].max()
    assert not pd.isna(last_trade_date), "No data found in the database."
    if datetime.now(ZoneInfo("Asia/Shanghai")) - last_trade_date > timedelta(days=10):
        return _fetch_all_stock_deep(df)
    data = _fetch_all_stock_shallow(last_trade_date=last_trade_date.strftime("%Y%m%d"))
    data['trade_date'] = pd.to_datetime(data['trade_date']).dt.tz_localize('Asia/Shanghai')
    data = DATASOURCES['tushare']()._format_dataframe(df=data)
    df = pd.concat([df, data], ignore_index=True).drop_duplicates(subset=['ts_code', 'date'])
    
    if os.getenv("PARQUET_STORAGE_PATH") is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)

    return df