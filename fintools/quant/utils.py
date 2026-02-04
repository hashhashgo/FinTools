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

_stock_cache = None

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
    global _stock_cache
    db = DB_CONNECTIONS['fintools.data_sources.fin_history.tushare:TushareDataSource.history']
    common_fields = {"type": UnderlyingType.STOCK, "freq": DataFrequency.DAILY}
    table_name, cursor = db.get_table_name_and_cursor(common_fields=common_fields)
    
    parquet_path = Path(os.getenv("PARQUET_STORAGE_PATH", "./parquet_storage")) / f'{table_name}.parquet'
    if _stock_cache is not None: df = _stock_cache
    elif parquet_path.exists():
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
    
    _stock_cache = df
    if os.getenv("PARQUET_STORAGE_PATH") is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)
    
    df = df.merge(stock_basic()[['ts_code', 'industry']], on='ts_code', how='left')
    df = df.rename(columns={'pct_chg': 'returns'})

    return df

_daily_basic = None
def fetch_daily_basic():
    global _daily_basic
    parquet_path = Path(os.getenv("PARQUET_STORAGE_PATH", "./parquet_storage")) / f'daily_basic.parquet'
    if _daily_basic is not None: data = _daily_basic
    elif parquet_path.exists():
        data = pd.read_parquet(parquet_path)
    else: data = pd.DataFrame(columns=['ts_code', 'date'])
    last_trade_date = data['date'].max()
    if pd.isna(last_trade_date):
        last_trade_date = "20000101"

    policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", "daily_basic")
    item = _make_item(policy=policy)
    assert item is not None
    if datetime.now(ZoneInfo("Asia/Shanghai")) - pd.to_datetime(last_trade_date) < timedelta(days=10):
        df_stocks = []
        offset = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            is_end = False
            while not is_end:
                futures = [executor.submit(lambda off=o: pro.daily_basic(offset=off)) for o in range(offset, offset + 60000, 6000)]
                t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
                for future in t:
                    t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, 'tushare:daily_basic').remaining} / {policy.max_calls}"
                    df = future.result()
                    df.rename(columns={'trade_date': 'date'}, inplace=True)
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize('Asia/Shanghai')
                    if not df.empty: df_stocks.append(df)
                    if df['date'].max() < last_trade_date:
                        is_end = True
                offset += 60000
        data = pd.concat([data, *df_stocks], ignore_index=True)
    else:
        df_stocks = []
        underlyings = stock_basic()['ts_code'].unique()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(lambda u=underlying: pro.daily_basic(ts_code=u)) : underlying for underlying in underlyings}

            t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
            policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", "daily_basic")
            for future in t:
                df = future.result()
                df.rename(columns={'trade_date': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize('Asia/Shanghai')
                if not df.empty: df_stocks.append(df)
                t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, 'tushare:daily_basic').remaining} / {policy.max_calls}"
        data = pd.concat(df_stocks, ignore_index=True)

    data = data.drop_duplicates(subset=['ts_code', 'date'])

    _daily_basic = data
    if os.getenv("PARQUET_STORAGE_PATH") is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(parquet_path)
    
    return data

def fetch_everything():
    df_stock = fetch_all_stock()
    df_daily_basic = fetch_daily_basic()
    return df_stock.merge(df_daily_basic, on=['ts_code', 'date'], suffixes=('', '_daily_basic'), how='left')