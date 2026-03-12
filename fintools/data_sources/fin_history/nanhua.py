from .base import OHLCDataSource, UnderlyingType, DataFrequency
from fintools.databases.history_db import history_cache
import pandas as pd
import sqlite3
import os
from typing import Optional, Callable, Union
from datetime import datetime, date, timedelta
from importlib.resources import files

import requests

class NanHuaDataSource(OHLCDataSource):

    name = "nanhua"

    freq_map = {
        DataFrequency.MINUTE1: 'MIN1',
        DataFrequency.MINUTE5: 'MIN5',
        DataFrequency.MINUTE15: 'MIN15',
        DataFrequency.MINUTE30: 'MIN30',
        DataFrequency.MINUTE60: 'MIN60',
        DataFrequency.MINUTE120: 'MIN120',
        DataFrequency.MINUTE240: 'MIN240',
        DataFrequency.DAILY: 'DAY1',
        DataFrequency.WEEKLY: 'WEEK1',
        DataFrequency.MONTHLY: 'MONTH1'
    }
    column_names = ["date", "open", "high", "low", "close", "volume"]

    extra_symbols = {
        'TL_NH': 'CBA21801.CS',
        'T_NH': 'CBA04501.CS',
    }

    def __init__(self, data_server_url: str = os.getenv("NANHUA_SERVER_URL", "http://127.0.0.1:13200/")):
        if not data_server_url.endswith('/'):
            data_server_url += '/'
        self.data_server_url = data_server_url


    @history_cache(
        table_basename=name,
        db_path=os.getenv("FINTOOLS_DB", ""),
        key_fields=("symbol",),
        common_fields= ("freq", ),
        except_fields=("type", ),
    )
    def history(self, symbol: str, type: UnderlyingType = UnderlyingType.INDEX, start: Union[str, datetime, date, int] = 0, end: Union[str, datetime, date, int] = datetime.now(), freq: DataFrequency = DataFrequency.DAILY) -> pd.DataFrame:
        assert type == UnderlyingType.INDEX, "NanHuaDataSource only supports UnderlyingType.INDEX"
        nh_freq = self._map_frequency(freq)
        data_raw = requests.get(f'{self.data_server_url}?ticker={symbol}&freq={nh_freq}').json()
        df = pd.DataFrame(data_raw)
        df["date"] = pd.to_datetime(df['quoteTime'], unit='ms', utc=True)
        df = df.rename(columns={
            "turnOver": "amount",
        })

        start_date = self._parse_datetime(start)
        end_date = self._parse_datetime(end)

        if symbol in self.extra_symbols:
            assert freq == DataFrequency.DAILY, "Only daily frequency is supported for extra symbols"
            earliest = df.loc[df['date'].idxmin()]
            assert isinstance(earliest, pd.Series)
            if start_date < earliest['date']:
                earliest_date = earliest['date'].date()
                earliest_close = earliest['close']
                df_pre = pd.read_csv(str(files("fintools").joinpath(f"data/daily_{self.extra_symbols[symbol]}.csv")), encoding="utf-8")
                df_pre['date'] = pd.to_datetime(df_pre['date'], utc=True).dt.date
                ratio = earliest_close / df_pre[df_pre["date"] == earliest_date]["close"].values[0]
                df_pre[['open', 'high', 'low', 'close']] = df_pre[['open', 'high', 'low', 'close']] * ratio
                df_pre = df_pre[df_pre['date'] < earliest_date]
                df_pre['date'] = pd.to_datetime(df_pre['date'], utc=True)
                df = pd.concat([df_pre, df], ignore_index=True)

        if start_date.time() == datetime.min.time() and end_date.time() == datetime.min.time():
            end_date = end_date + self._datetime_shift_base(freq)
        df = pd.DataFrame(df[(df["date"] >= start_date) & (df["date"] <= end_date)])
        return self._format_dataframe(df)


    def subscribe(self, symbol: str, interval: str, callback: Callable) -> None:
        raise NotImplementedError("NanHuaDataSource does not support real-time data subscription")

    def unsubscribe(self, symbol: str, interval: str) -> None:
        raise NotImplementedError("NanHuaDataSource does not support real-time data unsubscription")