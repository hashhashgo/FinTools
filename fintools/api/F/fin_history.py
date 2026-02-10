from typing import Dict, Annotated

from importlib.resources import files
from datetime import datetime, date
import pandas as pd
import tzlocal

from fintools.data_sources.fin_history import (
    STANDARD_COLUMN_NAMES, OHLCDataSource,
    DATASOURCES
)
from .types import UnderlyingType, DataFrequency

import logging

logger = logging.getLogger(__name__)

datasources: Dict[str, OHLCDataSource] = {}

def get_data(
    datasource: Annotated[str, "data source, selected from: " + ", ".join(DATASOURCES.keys())],
    symbol: Annotated[str, "symbol code of the index in the data source"],
    type: Annotated[UnderlyingType, "type of the symbol, selected from: " + " / ".join([t.value for t in UnderlyingType])],
    freq: Annotated[DataFrequency, "data frequency, supports: " + " / ".join([f.value for f in DataFrequency])] = DataFrequency.DAILY,
    indicators: Annotated[list[str], "list of technical indicators to compute, supported by stockstats, e.g., macd, rsi, boll"] = [],
    start: Annotated[str | datetime | date | int, "start time, supports str: YYYY-MM-DD / YYYY-MM-DD HH:MM:SS | datetime | date | int (timestamp)"] = 0,
    end: Annotated[str | datetime | date | int, "end time, supports str: YYYY-MM-DD / YYYY-MM-DD HH:MM:SS | datetime | date | int (timestamp)"] = datetime.now(),
    only_standard_columns: Annotated[bool, "if True, only return standard columns"] = True
) -> pd.DataFrame:
    """
    Get historical financial data for a given symbol from specified data source.
    
    Except from the index symbols listed by list_indices, you can also refer to the data source documentation for valid symbols.
    For example, for yfinance, although symbols like AAPL are not listed in list_indices, they are valid symbols.
    For Chinese stocks, it's better to use tushare data source.
    For Global indexes, you can use symbols like ^GSPC for S&P 500 in yfinance and usd-cny for US Dollar to Chinese Yuan exchange rate in investing.com.

    If you want to compute technical indicators, please specify them in the 'indicators' parameter as a comma separated list.
    The indicators will be computed using stockstats library.

    Returns: record DataFrame, with standard columns and specific columns
    - date: pd.Timestamp
    - other standard columns: float
    - specific columns: refer to data source documentation.

    Parameters:
    - datasource: str, data source to fetch data from
    - symbol: str, symbol code of the index in the data source
    - type: UnderlyingType, type of the symbol
    - freq: DataFrequency, data frequency
    - indicators: list of str, technical indicators to compute, supported by stockstats
    - start: str | datetime | date | int, start time
    - end: str | datetime | date | int, end time
    - only_standard_columns: bool, if True, only return standard columns

    Time range is [start, end), i.e., start is inclusive, end is exclusive.
    So if you want data up to and including 2025-01-01, please set end to 2025-01-02.

    indicators supported by stockstats:
    - close_50_sma, close_200_sma, close_10_ema
    - macd, macds, macdh
    - rsi
    - boll, boll_ub, boll_lb, atr
    - vwma
    """
    
    if datasource not in DATASOURCES:
        raise ValueError(f"Unknown datasource: {datasource}\n\nSupported datasources: {' / '.join(DATASOURCES.keys())}")

    if datasource not in datasources:
        datasources[datasource] = DATASOURCES[datasource]()
    ds = datasources[datasource]

    logger.info(f"Fetching history: datasource={datasource}, symbol={symbol}, type={type}, start={start}, end={end}, freq={freq}, only_standard_columns={only_standard_columns}")

    df: pd.DataFrame = ds.history(
        symbol=symbol,
        type=type,
        start=start,
        end=end,
        freq=freq,
    )

    if df.empty:
        logger.warning("No data fetched, returning empty DataFrame")
        logger.warning(f"Parameters: datasource={datasource}, symbol={symbol}, type={type}, start={start}, end={end}, freq={freq}")
        return pd.DataFrame(columns=STANDARD_COLUMN_NAMES)

    logger.debug("Raw data fetched:")
    logger.debug(df)

    if only_standard_columns:
        df = pd.DataFrame(df[STANDARD_COLUMN_NAMES])
    
    from stockstats import wrap
    df = df.sort_values(by='date', ascending=True)
    df = wrap(df)
    for indicator in [ind.strip().lower() for ind in indicators if ind.strip()]:
        try:
            _ = df[indicator]
        except Exception as e:
            logger.warning(f"Warning: Failed to compute indicator '{indicator}': {e}")
    
    df = df.reset_index()
    df = pd.DataFrame(df)
    df = df.sort_values(by='date', ascending=True)
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert(tzlocal.get_localzone())

    return df
    
def list_indices() -> pd.DataFrame:
    """
    List known financial indices with their types and available data sources.
    Returns: index info DataFrame, containing:
    - name: str, Name of the index
    - type: str, Type of the index
    - datasource1: str, Symbol code in datasource1
    - datasource2: str, Symbol code in datasource2 (if available)
    - ... (other data sources)
    """
    logger.info("Listing known financial indices")
    df = pd.read_csv(files("fintools").joinpath("data/known_indices.csv").open(encoding="utf-8"))
    df = df[['name', 'type'] + list(DATASOURCES.keys())]
    ret = []
    for _, row in df.iterrows():
        entry = {col: row[col] for col in df.columns if pd.notna(row[col])}
        ret.append(entry)
    return pd.DataFrame(ret)


__all__ = [
    "get_data",
    "list_indices"
]
