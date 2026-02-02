from .base import OHLCDataSource, UnderlyingType, DataFrequency
from fintools.databases.history_db import history_cache
import pandas as pd
import sqlite3
import os
from typing import Optional, Callable, Union
from datetime import datetime, date

from ...runtime.data_sources.tushare import pro

class TushareDataSource(OHLCDataSource):

    name = "tushare"

    freq_map = {
        DataFrequency.MINUTE1: '1min',
        DataFrequency.MINUTE5: '5min',
        DataFrequency.MINUTE15: '15min',
        DataFrequency.MINUTE30: '30min',
        DataFrequency.MINUTE60: '60min',
        DataFrequency.DAILY: 'daily',
        DataFrequency.WEEKLY: 'weekly',
        DataFrequency.MONTHLY: 'monthly'
    }
    column_names = ["trade_date", "open", "high", "low", "close", "vol"]

    extra_symbols = {
        "NHCI": "南华综合指数",
        "NHII": "南华工业品指数",
        "NHECI": "南华能化指数",
        "NHMI": "南华金属指数",
        "NHAI": "南华农产品指数",
        "NHPMI": "南华贵金属指数",
        "NHEI": "南华能源指数",
        "NHPCI": "南华石油化工指数",
        "NHCCI": "南华煤制化工指数",
        "NHNFI": "南华有色金属指数",
        "NHFI": "南华黑色产业指数",
        "NHFMI": "南华黑色原材料指数",
        "NHBMI": "南华建材指数",
        "NHOOI": "南华油脂油料指数",
        "NHAECI": "南华经济作物指数",
        "NHNMI": "南华新材料指数",
        "NHCIMi": "南华迷你综合指数",
        "NHRECI": "南华风险均衡商品指数",
        "NHQFII": "南华QFII商品指数"
    }


    @history_cache(
        table_basename=name,
        db_path=os.getenv("FINTOOLS_DB", ""),
        key_fields=("symbol",),
        common_fields=("type", "freq"),
        except_fields=(),
        missing_threshold=0
    )
    def history(self, symbol: str, type: UnderlyingType, start: Union[str, datetime, date, int] = 0, end: Union[str, datetime, date, int] = datetime.now(), freq: DataFrequency = DataFrequency.DAILY) -> pd.DataFrame:
        if type == UnderlyingType.STOCK: return self._format_dataframe(self._history_stock(symbol, start, end, freq))
        elif type == UnderlyingType.INDEX: return self._format_dataframe(self._history_index(symbol, start, end, freq))
        elif type == UnderlyingType.FOREX: return self._format_dataframe(self._history_forex(symbol, start, end, freq))
        elif type == UnderlyingType.COMMODITY: return self._format_dataframe(self._history_commodity(symbol, start, end, freq))
        elif type == UnderlyingType.FUND: return self._format_dataframe(self._history_etf(symbol, start, end, freq))
        else: raise NotImplementedError(f"Data type {type} not supported in Tushare")

    def _history_stock(self, symbol: str, start: Union[str, datetime, date, int], end: Union[str, datetime, date, int], freq: DataFrequency) -> pd.DataFrame:
        ts_freq = self._map_frequency(freq)
        start_date = self._parse_datetime(start).strftime("%Y%m%d")
        end_date = self._parse_datetime(end).strftime("%Y%m%d")
        if ts_freq == "daily":
            df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        elif ts_freq == "weekly":
            df = pro.weekly(ts_code=symbol, start_date=start_date, end_date=end_date)
        elif ts_freq == "monthly":
            df = pro.monthly(ts_code=symbol, start_date=start_date, end_date=end_date)
        elif ts_freq in ['1min', '5min', '15min', '30min', '60min']:
            start_date = self._parse_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
            end_date = self._parse_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
            df = pro.stk_mins(ts_code=symbol, freq=self._map_frequency(freq), start_date=start_date, end_date=end_date)
        else:
            raise NotImplementedError(f"Frequency {freq} not supported for stock data in Tushare")
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.tz_localize('Asia/Shanghai')
        return df

    def _history_index(self, symbol: str, start: Union[str, datetime, date, int], end: Union[str, datetime, date, int], freq: DataFrequency) -> pd.DataFrame:
        ts_freq = self._map_frequency(freq)
        start_date = self._parse_datetime(start).strftime("%Y%m%d")
        end_date = self._parse_datetime(end).strftime("%Y%m%d")
        if symbol in ['XIN9', 'HSI', 'HKTECH', 'HKAH', 'DJI', 'SPX', 'IXIC', 'FTSE', 'FCHI', 'GDAXI', 'N225', 'KS11', 'AS51', 'SENSEX', 'IBOVESPA', 'RTS', 'TWII', 'CKLSE', 'SPTSX', 'CSX5P', 'RUT']:
            if ts_freq == "daily":
                df = pro.index_global(ts_code=symbol, start_date=start_date, end_date=end_date)
            else:
                raise NotImplementedError(f"Frequency {freq} not supported for global index data in Tushare")
        else:
            if ts_freq == "daily":
                df = pro.index_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            elif ts_freq == "weekly":
                df = pro.index_weekly(ts_code=symbol, start_date=start_date, end_date=end_date)
            elif ts_freq == "monthly":
                df = pro.index_monthly(ts_code=symbol, start_date=start_date, end_date=end_date)
            else:
                raise NotImplementedError(f"Frequency {freq} not supported for index data in Tushare")
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.tz_localize('Asia/Shanghai')
        return df
        
    def _history_forex(self, symbol: str, start: Union[str, datetime, date, int], end: Union[str, datetime, date, int], freq: DataFrequency) -> pd.DataFrame:
        ts_freq = self._map_frequency(freq)
        start_date = self._parse_datetime(start).strftime("%Y%m%d")
        end_date = self._parse_datetime(end).strftime("%Y%m%d")
        if ts_freq == "daily":
            df = pro.fx_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            df['open'] = (df['bid_open'] + df['ask_open']) / 2
            df['high'] = (df['bid_high'] + df['ask_high']) / 2
            df['low'] = (df['bid_low'] + df['ask_low']) / 2
            df['close'] = (df['bid_close'] + df['ask_close']) / 2
            df['vol'] = df['tick_qty']
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.tz_localize('Asia/Shanghai')
            return df
        else:
            raise NotImplementedError(f"Frequency {freq} not supported for forex data in Tushare")

    def _history_commodity(self, symbol: str, start: Union[str, datetime, date, int], end: Union[str, datetime, date, int], freq: DataFrequency) -> pd.DataFrame:
        ts_freq = self._map_frequency(freq)
        start_date = self._parse_datetime(start).strftime("%Y%m%d")
        end_date = self._parse_datetime(end).strftime("%Y%m%d")
        if ts_freq == "daily":
            df = pro.fut_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            return df
        elif ts_freq == "weekly":
            df = pro.fut_weekly_monthly(ts_code=symbol, start_date=start_date, end_date=end_date, freq="week")
            return df
        elif ts_freq == "monthly":
            df = pro.fut_weekly_monthly(ts_code=symbol, start_date=start_date, end_date=end_date, freq="month")
            return df
        elif ts_freq in ['1min', '5min', '15min', '30min', '60min']:
            start_date = self._parse_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
            end_date = self._parse_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
            df = pro.fut_mins(ts_code=symbol, freq=self._map_frequency(freq), start_date=start_date, end_date=end_date)
            return df
        else:
            raise NotImplementedError(f"Frequency {freq} not supported for commodity data in Tushare")
    
    def _history_etf(self, symbol: str, start: Union[str, datetime, date, int], end: Union[str, datetime, date, int], freq: DataFrequency) -> pd.DataFrame:
        ts_freq = self._map_frequency(freq)
        start_date = self._parse_datetime(start).strftime("%Y%m%d")
        end_date = self._parse_datetime(end).strftime("%Y%m%d")
        if ts_freq == "daily":
            df = pro.fund_daily(ts_code=symbol, start_date=start_date, end_date=end_date)
        elif ts_freq in ['1min', '5min', '15min', '30min', '60min']:
            start_date = self._parse_datetime(start).strftime("%Y-%m-%d %H:%M:%S")
            end_date = self._parse_datetime(end).strftime("%Y-%m-%d %H:%M:%S")
            df = pro.stk_mins(ts_code=symbol, freq=self._map_frequency(freq), start_date=start_date, end_date=end_date)
        else:
            raise NotImplementedError(f"Frequency {freq} not supported for ETF data in Tushare")
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.tz_localize('Asia/Shanghai')
        return df

    def subscribe(self, symbol: str, interval: str, callback: Callable) -> None:
        # Tushare does not support real-time data subscription
        raise NotImplementedError("Tushare does not support real-time data subscription")

    def unsubscribe(self, symbol: str, interval: str) -> None:
        # Tushare does not support real-time data unsubscription
        raise NotImplementedError("Tushare does not support real-time data unsubscription")