from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import cast, Set, List, Iterable
from zoneinfo import ZoneInfo
import pandas as pd
import polars as pl
import tqdm
import os
from pathlib import Path

from fintools.utils.underlying import stock_basic, index_basic

from fintools.runtime.rate_limit import GLOBAL_REGISTRY, Policy
from fintools.runtime.rate_limit.limits_backend import _make_item
from fintools.runtime.data_sources.tushare import pro


_cache = defaultdict(lambda: pl.DataFrame(schema={'ts_code': pl.Utf8, 'date': pl.Datetime(time_unit='ns', time_zone="Asia/Shanghai")}))

def fetch_all(api: str, underlyings: Set[str], *, allow_no_params: bool = True, fetch_new: bool = True) -> pl.DataFrame:
    global _cache
    parquet_path = Path(os.getenv("PARQUET_STORAGE_PATH", "./parquet_storage")) / f'{api}.parquet'
    if not _cache[api].is_empty(): data: pl.DataFrame = _cache[api]
    elif parquet_path.exists():
        data = pl.read_parquet(parquet_path)
    else: data = _cache[api]

    try:
        if fetch_new:
            last_trade_date = data['date'].max()
            if not isinstance(last_trade_date, datetime):
                last_trade_date = datetime(1970, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))

            policy = GLOBAL_REGISTRY._resolve_endpoint_policy("tushare", api)
            item = _make_item(policy=policy)
            
            if datetime.now(ZoneInfo("Asia/Shanghai")) - last_trade_date > timedelta(days=10)\
            or len(underlyings - set(data['ts_code'].unique().to_list())) > 0:
                df_stocks_pd = [data.to_pandas()]
                last_trade_dates = dict(zip(*data.group_by('ts_code').agg(pl.col('date').max())))
                try:
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        q = deque()
                        for underlying in underlyings:
                            if last_trade_dates.get(underlying, datetime(1970,1,1, tzinfo=ZoneInfo("Asia/Shanghai"))) < datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(days=10):
                                q.append((underlying, datetime.now().strftime("%Y%m%d")))
                        while len(q):
                            futures = {executor.submit(lambda u=u, t=t: eval(f"pro.{api}")(ts_code=u, end_date=t)): (u, t) for u, t in q}
                            q.clear()
                            t = tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fetching stock data")
                            for future in t:
                                df = future.result()
                                df = df.rename(columns={'trade_date': 'date'})
                                df['date'] = pd.to_datetime(df['date'], format="%Y%m%d").dt.as_unit("ns").dt.tz_localize("Asia/Shanghai")
                                if not df.empty:
                                    df_stocks_pd.append(df)
                                    if df.shape[0] >= 6000:
                                        q.append((futures[future][0], df['date'].min().strftime("%Y%m%d")))
                                if item: t.postfix = f"Remaining: {GLOBAL_REGISTRY.backend.limiter.get_window_stats(item, f'tushare:{api}').remaining} / {policy.max_calls}"
                except Exception as e:
                    raise e
                finally:
                    data = pl.from_pandas(pd.concat(df_stocks_pd, ignore_index=True))
            if allow_no_params:
                last_trade_date = data['date'].max()
                if not isinstance(last_trade_date, datetime):
                    last_trade_date = datetime(1990, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
                df_stocks: List[pl.DataFrame] = [data]
                offset = 0
                try:
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        is_end = False
                        while not is_end and offset < 100000:
                            futures = [executor.submit(lambda off=o: eval(f"pro.{api}")(offset=off)) for o in range(offset, min(offset + 60000, 100000), 6000)]
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
                except Exception as e:
                    raise e
                finally:
                    data = pl.concat(df_stocks, how="vertical")

    except Exception as e:
        raise e
    finally:
        data = data.unique(subset=['ts_code', 'date'])
        _cache[api] = data
        if fetch_new:
            if os.getenv("PARQUET_STORAGE_PATH") is not None:
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                data.write_parquet(parquet_path)
    
    return data

def fetch_everything(*, fetch_new: bool = True) -> pl.DataFrame:
    df_stock = fetch_all(api='daily', underlyings=set(stock_basic()['ts_code'].unique().tolist()), fetch_new=fetch_new).lazy()
    df_daily_basic = fetch_all(api="daily_basic", underlyings=set(stock_basic()['ts_code'].unique().tolist()), fetch_new=fetch_new).lazy()
    df_adj_factor = fetch_all(api="adj_factor", underlyings=set(stock_basic()['ts_code'].unique().tolist()), fetch_new=fetch_new).lazy()
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
    ).drop(['open', 'high', 'low', 'close']).rename({
        'open_adj': 'open', 'high_adj': 'high', 'low_adj': 'low', 'close_adj': 'close'
    })
    return df_all.collect()

def make_dataset(*, drop_days: int = 365, only_SHSZ: bool = True, fetch_new: bool = True) -> pl.DataFrame:
    from .registry import DATA_SCHEMA
    df = fetch_everything(fetch_new=fetch_new).lazy()
    df = df.rename({
        "circ_mv": "cap", "ts_code": "symbol",
        'vol': 'volume', 'amount': 'amount',
        'pct_chg': 'returns'
    }).with_columns(
        pl.col('returns').fill_null(0.0) / 100
    )
    df = df.with_columns(
        (pl.col("amount") / pl.col("volume")).alias("vwap"),
        pl.col('date').dt.date().alias('date')
    )
    df = df.filter(
        pl.col('date') > pl.col('date').min().over('symbol') + timedelta(days=drop_days)
    )
    if only_SHSZ:
        df = df.filter(
            pl.col('symbol').str.starts_with("6") | pl.col('symbol').str.starts_with("0") | pl.col('symbol').str.starts_with("3")
        )
    df = df.sort(by=['symbol', 'date'], descending=[False, False])
    df = df.cast(DATA_SCHEMA)
    return df.collect()