from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import cast
from zoneinfo import ZoneInfo
import pandas as pd
import polars as pl
import tqdm
import os
from pathlib import Path

from fintools.api.F.fin_history import get_data, DataFrequency, UnderlyingType
from fintools.data_sources.fin_history import DATASOURCES, STANDARD_COLUMN_NAMES
from fintools.databases import DB_CONNECTIONS
from fintools.utils.underlying import stock_basic

from fintools.runtime.rate_limit import GLOBAL_REGISTRY, Policy
from fintools.runtime.rate_limit.limits_backend import _make_item
from fintools.runtime.data_sources.tushare import pro

# _stock_cache: pl.DataFrame | None = None

# def _fetch_all_stock_deep(df: pl.DataFrame, last_trade_date = datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(days=10)) -> pl.DataFrame:
#     df_stocks = []
#     underlyings = stock_basic()['ts_code'].unique()
#     last_trade_dates = dict(zip(*df.group_by('ts_code').agg(pl.col('date').max())))
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         futures = {executor.submit(lambda u=underlying: get_data(
#             datasource = "tushare",
#             symbol = u,
#             type = UnderlyingType.STOCK,
#             freq = DataFrequency.DAILY,
#             only_standard_columns=False
#         )) : underlying for underlying in underlyings if last_trade_dates.get(underlying, datetime(1970,1,1, tzinfo=ZoneInfo("Asia/Shanghai"))) < last_trade_date}

#         t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
#         policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", "daily")
#         item = _make_item(policy=policy)
#         assert item is not None
#         for future in t:
#             t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, 'tushare:daily').remaining} / {policy.max_calls}"
#             df = pl.from_pandas(future.result())
#             if df.is_empty(): continue
#             df = df.with_columns(pl.col('date').dt.cast_time_unit("ns").dt.replace_time_zone("Asia/Shanghai"))
#             df_stocks.append(df)
    
#     return pl.concat(df_stocks, how="vertical")

# def _fetch_all_stock_shallow(last_trade_date) -> pl.DataFrame:
#     df_stocks = []
#     offset = 0
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         is_end = False
#         while not is_end:
#             futures = [executor.submit(lambda off=o: pro.daily(offset=off)) for o in range(offset, offset + 60000, 6000)]
#             t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
#             policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", "daily")
#             item = _make_item(policy=policy)
#             assert item is not None
#             for future in t:
#                 t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, 'tushare:daily').remaining} / {policy.max_calls}"
#                 df = pl.from_pandas(future.result())
#                 df = df.rename({"trade_date": "date"})
#                 df = df.with_columns(pl.col('date').str.strptime(pl.Datetime, format="%Y%m%d").dt.cast_time_unit("ns").dt.replace_time_zone("Asia/Shanghai"))
#                 df_stocks.append(df)
#                 max_date = df['date'].max()
#                 assert isinstance(max_date, datetime)
#                 if max_date < last_trade_date:
#                     is_end = True
#             offset += 60000
#     return pl.concat(df_stocks, how="vertical")

# def fetch_all_stock(fetch_new: bool = True):
#     global _stock_cache
#     db = DB_CONNECTIONS['fintools.data_sources.fin_history.tushare:TushareDataSource.history']
#     common_fields = {"type": UnderlyingType.STOCK, "freq": DataFrequency.DAILY}
#     table_name, cursor = db.get_table_name_and_cursor(common_fields=common_fields)
    
#     parquet_path = Path(os.getenv("PARQUET_STORAGE_PATH", "./parquet_storage")) / f'{table_name}.parquet'
#     if _stock_cache is not None: df = _stock_cache
#     elif parquet_path.exists():
#         df = pl.read_parquet(parquet_path)
#     else:
#         cursor.execute(f"SELECT count(*) as cnt from {table_name}")
#         total_count = cursor.fetchone()['cnt']
#         df_stocks = []
#         for df in tqdm.tqdm(pd.read_sql(f"SELECT * from {table_name}", cursor.connection, chunksize=100000), total=(total_count // 100000) + 1, desc="Loading data from DB"):
#             df = db.format_dataframe(df, common_fields=common_fields)
#             df_stocks.append(pl.from_pandas(df))
#         df = pl.concat(df_stocks, how="vertical")
#         del df_stocks
#     if fetch_new:
#         df = pl.DataFrame(df)
#         last_trade_date = df['date'].max()
#         if not isinstance(last_trade_date, datetime) \
#             or datetime.now(ZoneInfo("Asia/Shanghai")) - last_trade_date > timedelta(days=10) \
#             or df['ts_code'].unique().len() < stock_basic()['ts_code'].nunique():
#             data = _fetch_all_stock_deep(df)
#             df = pl.concat([df[data.columns], data], how="vertical").unique(subset=['ts_code', 'date'])
#             last_trade_date = df['date'].max()
#         assert isinstance(last_trade_date, datetime)
#         data = _fetch_all_stock_shallow(last_trade_date=last_trade_date)
#         data = data.rename({'vol': 'volume'})
#         df = pl.concat([df[data.columns], data], how="vertical").unique(subset=['ts_code', 'date'])
    
#     _stock_cache = df
#     if fetch_new:
#         if os.getenv("PARQUET_STORAGE_PATH") is not None:
#             parquet_path.parent.mkdir(parents=True, exist_ok=True)
#             df.write_parquet(parquet_path)
    
#     df = df.join(
#         pl.from_pandas(stock_basic()[['ts_code', 'industry']]), 
#         on='ts_code', 
#         how='left'
#     )
#     df = df.rename({"pct_chg": "returns"})

#     return df

_cache = defaultdict(lambda: pl.DataFrame(schema={'ts_code': pl.Utf8, 'date': pl.Datetime(time_unit='ns', time_zone="Asia/Shanghai")}))
def fetch_all(api: str, fetch_new: bool = True) -> pl.DataFrame:
    global _cache
    parquet_path = Path(os.getenv("PARQUET_STORAGE_PATH", "./parquet_storage")) / f'{api}.parquet'
    if not _cache[api].is_empty(): data = _cache[api]
    elif parquet_path.exists():
        data = pl.read_parquet(parquet_path)
    else: data = _cache[api]

    if fetch_new:
        last_trade_date = data['date'].max()
        if not isinstance(last_trade_date, datetime):
            last_trade_date = datetime(1990, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))

        policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", api)
        item = _make_item(policy=policy)
        if datetime.now(ZoneInfo("Asia/Shanghai")) - last_trade_date < timedelta(days=10):
            df_stocks = []
            offset = 0
            with ThreadPoolExecutor(max_workers=10) as executor:
                is_end = False
                while not is_end:
                    futures = [executor.submit(lambda off=o: eval(f"pro.{api}")(offset=off)) for o in range(offset, offset + 60000, 6000)]
                    t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
                    for future in t:
                        if item: t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, f'tushare:{api}').remaining} / {policy.max_calls}"
                        df = pl.from_pandas(future.result())
                        df = df.rename({'trade_date': 'date'})
                        df = df.with_columns(pl.col('date').str.strptime(pl.Datetime, format="%Y%m%d").dt.cast_time_unit("ns").dt.replace_time_zone("Asia/Shanghai"))
                        if not df.is_empty():
                            df_stocks.append(df)
                            if cast(datetime, df['date'].max()) < last_trade_date:
                                is_end = True
                    offset += 60000
            data = pl.concat([data, *df_stocks], how="vertical")
        else:
            df_stocks = []
            underlyings = stock_basic()['ts_code'].unique()
            with ThreadPoolExecutor(max_workers=10) as executor:
                q = deque()
                for underlying in underlyings: q.append((underlying, datetime.now().strftime("%Y%m%d")))
                while len(q):
                    futures = {executor.submit(lambda u=u: eval(f"pro.{api}")(ts_code=u, end_date=t)): (u, t) for u, t in q}
                    q.clear()
                    t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
                    for future in t:
                        df = future.result()
                        df = df.rename(columns={'trade_date': 'date'})
                        df['date'] = pd.to_datetime(df['date'], format="%Y%m%d").dt.as_unit("ns").dt.tz_localize("Asia/Shanghai")
                        if not df.empty:
                            df_stocks.append(df)
                            if df.shape[0] >= 6000:
                                q.append((futures[future][0], df['date'].min().strftime("%Y%m%d")))
                        if item: t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, f'tushare:{api}').remaining} / {policy.max_calls}"
            data = pd.concat(df_stocks)
            data = pl.from_pandas(data)

    data = data.unique(subset=['ts_code', 'date'])

    _cache[api] = data
    if fetch_new:
        if os.getenv("PARQUET_STORAGE_PATH") is not None:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            data.write_parquet(parquet_path)
    
    return data

def fetch_everything(fetch_new: bool = True) -> pl.DataFrame:
    df_stock = fetch_all(api='daily', fetch_new=fetch_new).lazy()
    df_daily_basic = fetch_all(api="daily_basic", fetch_new=fetch_new).lazy()
    df_adj_factor = fetch_all(api="adj_factor", fetch_new=fetch_new).lazy()
    df_all = df_stock\
        .join(pl.from_pandas(stock_basic())[['ts_code', 'industry']].lazy(), on='ts_code', how='left')\
        .join(df_daily_basic, on=['ts_code', 'date'], how='left')\
        .join(df_adj_factor, on=['ts_code', 'date'], how='left')\
        .sort(['ts_code', 'date'], descending=[False, False])\
        .with_columns(
            pl.col('adj_factor').forward_fill().backward_fill().over('ts_code')
        )
    df_all = df_all.with_columns(
        (pl.col('open') * pl.col('adj_factor')).alias('open_adj'),
        (pl.col('high') * pl.col('adj_factor')).alias('high_adj'),
        (pl.col('low') * pl.col('adj_factor')).alias('low_adj'),
        (pl.col('close') * pl.col('adj_factor')).alias('close_adj'),
    )
    return df_all.collect()

def make_dataset(fetch_new: bool = True):
    from .registry import DATA_SCHEMA
    df = fetch_everything(fetch_new=fetch_new)
    df = df.rename({
        "circ_mv": "cap", "ts_code": "symbol",
        })
    df = df.with_columns(
        (pl.col("amount") / pl.col("volume")).alias("vwap"),
        pl.col('date').dt.date().alias('date')
    )
    df = df.sort(by=['symbol', 'date'], descending=[False, False])
    df = df.cast(DATA_SCHEMA)
    return df.select(DATA_SCHEMA.keys())