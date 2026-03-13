from typing import Annotated, Dict, List, Literal, TypedDict, DefaultDict, Set, cast
from collections import defaultdict
from datetime import datetime, date
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import os
import efinance as ef

from concurrent.futures import ThreadPoolExecutor, as_completed

import importlib
import logging
logger = logging.getLogger(__name__)

from .types import parse_datetime

from ..runtime.data_sources.tushare import pro

_fetch_cache: Dict[str, pd.DataFrame] = {}
def fetch_with_offsets(func_name) -> pd.DataFrame:
    global _fetch_cache
    if func_name in _fetch_cache:
        return _fetch_cache[func_name]
    df_list = []
    offset = 0
    while True:
        df = eval(f"pro.{func_name}")(offset=offset)
        if df.empty:
            break
        df_list.append(df)
        offset += len(df)
    result = pd.concat(df_list, ignore_index=True)
    _fetch_cache[func_name] = result
    return result

index_basic = lambda: fetch_with_offsets("index_basic")
stock_basic = lambda: fetch_with_offsets("stock_basic")
us_basic = lambda: fetch_with_offsets("us_basic")
hk_basic = lambda: fetch_with_offsets("hk_basic")
fx_obasic = lambda: fetch_with_offsets("fx_obasic")


global_index_map = {
    'XIN9': '富时中国A50指数',
    'HSI': '恒生指数',
    'HKTECH': '恒生科技指数',
    'HKAH': '恒生AH股H指数',
    'DJI': '道琼斯工业指数',
    'SPX': '标普500指数',
    'IXIC': '纳斯达克指数',
    'FTSE': '富时100指数',
    'FCHI': '法国CAC40指数',
    'GDAXI': '德国DAX指数',
    'N225': '日经225指数',
    'KS11': '韩国综合指数',
    'AS51': '澳大利亚标普200指数',
    'SENSEX': '印度孟买SENSEX指数',
    'IBOVESPA': '巴西IBOVESPA指数',
    'RTS': '俄罗斯RTS指数',
    'TWII': '台湾加权指数',
    'CKLSE': '马来西亚指数',
    'SPTSX': '加拿大S&P/TSX指数',
    'CSX5P': 'STOXX欧洲50指数',
    'RUT': '罗素2000指数'
}

class SYMBOL_SEARCH_RESULT(TypedDict):
    type: Literal['stock', 'index', 'fund', 'forex', 'unknown']
    symbol: str
    name: str
    source: Literal['tushare', 'choice', 'nanhua', 'efinance']

_symbol_search_cache: DefaultDict[str, List[SYMBOL_SEARCH_RESULT]] = defaultdict(list)
_nanhua_codes: pd.DataFrame | None = None
_nanhua_category: Dict = {}
_ef_fund_cache: pd.DataFrame | None = None
def symbol_search_all(
    keyword: Annotated[str, "The symbol code or name to search for."] = "",
    strict: Annotated[bool, "Whether to perform a strict search. If True, only exact matches will be returned."] = True,
    timeout: Annotated[float, "Maximum time to wait for the search operation. -1 means wait indefinitely."] = -1
) -> List[SYMBOL_SEARCH_RESULT]:
    assert keyword, "Either symbol or name must be provided."
    global _symbol_search_cache
    if keyword in _symbol_search_cache:
        return _symbol_search_cache[keyword]
    
    ret: List[SYMBOL_SEARCH_RESULT] = []
    # Try to search in stocks
    if keyword in stock_basic()['symbol'].values:
        res = stock_basic()[stock_basic()['symbol'] == keyword].iloc[0]
        ret.append({'type': 'stock', 'symbol': res['ts_code'], 'name': res['name'], 'source': 'tushare'})
    elif keyword in stock_basic()['ts_code'].values:
        res = stock_basic()[stock_basic()['ts_code'] == keyword].iloc[0]
        ret.append({'type': 'stock', 'symbol': res['ts_code'], 'name': res['name'], 'source': 'tushare'})
    elif keyword in stock_basic()['name'].values:
        res = stock_basic()[stock_basic()['name'] == keyword].iloc[0]
        ret.append({'type': 'stock', 'symbol': res['ts_code'], 'name': keyword, 'source': 'tushare'})
    elif keyword in us_basic()['ts_code'].values:
        res = us_basic()[us_basic()['ts_code'] == keyword].iloc[0]
        ret.append({'type': 'stock', 'symbol': res['ts_code'], 'name': res['name'], 'source': 'tushare'})
    elif keyword in hk_basic()['ts_code'].values:
        res = hk_basic()[hk_basic()['ts_code'] == keyword].iloc[0]
        ret.append({'type': 'stock', 'symbol': res['ts_code'], 'name': res['name'], 'source': 'tushare'})
    else:
        res_df = pro.stock_basic(ts_code=keyword)
        assert isinstance(res_df, pd.DataFrame)
        if res_df.empty: res_df = pro.stock_basic(name=keyword)
        assert isinstance(res_df, pd.DataFrame)
        if not res_df.empty:
            res = res_df.iloc[0]
            ret.append({'type': 'stock', 'symbol': res['ts_code'], 'name': res['name'], 'source': 'tushare'})
            global _stock_basic
            _stock_basic = pd.concat([stock_basic(), res_df], ignore_index=True)
    
    # Try to search in indexes
    from fintools.data_sources.fin_history.tushare import TushareDataSource
    if keyword in global_index_map or keyword in global_index_map.values():
        if keyword in global_index_map.values():
            keyword = [k for k, v in global_index_map.items() if v == keyword][0]
        ret.append({'type': 'index', 'symbol': keyword, 'name': global_index_map[keyword], 'source': 'tushare'})
    elif keyword and keyword in TushareDataSource.extra_index:
        ret.append({'type': 'index', 'symbol': keyword, 'name': TushareDataSource.extra_index[keyword], 'source': 'tushare'})
    elif keyword and keyword in TushareDataSource.extra_index.values():
        symbol = [k for k, v in TushareDataSource.extra_index.items() if v == keyword][0]
        ret.append({'type': 'index', 'symbol': symbol, 'name': keyword, 'source': 'tushare'})
    elif keyword and keyword in index_basic()['ts_code'].values:
        res = index_basic()[index_basic()['ts_code'] == keyword].iloc[0]
        ret.append({'type': 'index', 'symbol': keyword, 'name': res['name'], 'source': 'tushare'})
    elif keyword and keyword in index_basic()['name'].values:
        res = index_basic()[index_basic()['name'] == keyword].iloc[0]
        ret.append({'type': 'index', 'symbol': res['ts_code'], 'name': keyword, 'source': 'tushare'})
    else:
        res_df = pro.index_basic(ts_code=keyword)
        assert isinstance(res_df, pd.DataFrame)
        if res_df.empty: res_df = pro.index_basic(name=keyword)
        assert isinstance(res_df, pd.DataFrame)
        if not res_df.empty:
            res = res_df.iloc[0]
            ret.append({'type': 'index', 'symbol': res['ts_code'], 'name': res['name'], 'source': 'tushare'})
            global _index_basic
            _index_basic = pd.concat([index_basic(), res_df], ignore_index=True)

    if keyword in fx_obasic()['ts_code'].values:
        res = fx_obasic()[fx_obasic()['ts_code'] == keyword].iloc[0]
        ret.append({'type': 'forex', 'symbol': res['ts_code'], 'name': res['name'], 'source': 'tushare'})
    elif keyword in fx_obasic()['name'].values:
        res = fx_obasic()[fx_obasic()['name'] == keyword].iloc[0]
        ret.append({'type': 'forex', 'symbol': res['ts_code'], 'name': keyword, 'source': 'tushare'})
    
    res = pro.fund_daily(ts_code=keyword, limit=1)
    assert isinstance(res, pd.DataFrame)
    if not res.empty:
        res_ts = res.iloc[0]
        global _ef_fund_cache
        if _ef_fund_cache is None:
            _ef_fund_cache = ef.fund.get_fund_codes()
        res_ef = _ef_fund_cache[_ef_fund_cache['基金代码'] == keyword.split('.')[0]]
        if res_ef.empty: name = "UNKNOWN"
        else: name = res_ef.iloc[0]['基金简称']
        ret.append({'type': 'fund', 'symbol': res_ts['ts_code'], 'name': str(name), 'source': 'tushare'})
    
    try:
        ChoiceDataSource = importlib.import_module("fintools.data_sources.fin_history.choice").ChoiceDataSource
        with ChoiceDataSource.borrow_choice(timeout=timeout) as choice:
            data = choice.css(keyword, "NAME").Data
            if keyword in data:
                ret.append({'type': 'unknown', 'symbol': keyword, 'name': data[keyword][0], 'source': 'choice'})
    except: pass
    
    global _nanhua_codes, _nanhua_category
    try:
        if _nanhua_codes is None or _nanhua_codes.empty:
            nanhua_server_url = os.getenv("NANHUA_SERVER_URL", "http://127.0.0.1:13200/")
            if not nanhua_server_url.endswith("/"):
                nanhua_server_url = nanhua_server_url + '/'
            all_info = requests.get(nanhua_server_url + "contracts", timeout=0.5).json()
            _nanhua_codes = pd.DataFrame(all_info['base_info']['codes'])
            _nanhua_category = all_info['category']
    except: pass
    if _nanhua_codes is not None and not _nanhua_codes.empty:
        if keyword in _nanhua_codes['code'].values:
            res = _nanhua_codes[_nanhua_codes['code'] == keyword].iloc[0]
            ret.append({'type': 'index', 'symbol': res['code'], 'name': res['name'], 'source': 'nanhua'})
        elif keyword in _nanhua_codes['name'].values:
            res = _nanhua_codes[_nanhua_codes['name'] == keyword].iloc[0]
            ret.append({'type': 'index', 'symbol': res['code'], 'name': res['name'], 'source': 'nanhua'})
    
    res = ef.utils.search_quote(keyword.split('.')[0], count=200)
    if res:
        if not isinstance(res, list):
            res = [res]
        for r in res:
            if strict and keyword not in (r.code, r.name):
                continue
            code = r.code
            name = r.name
            if r.classify == 'Fund':
                type = 'fund'
            elif r.classify == 'Index':
                type = 'index'
            elif 'Stock' in r.classify:
                type = 'stock'
            else:
                type = r.classify
                logger.warning(f"Unknown classify '{r.classify}' from efinance for code '{code}'")
            ret.append({'type': type, 'symbol': code, 'name': name, 'source': 'efinance'})
    
    _symbol_search_cache[keyword] = ret
    return _symbol_search_cache[keyword]

def symbol_search(
    keyword: Annotated[str, "The symbol code or name to search for."] = "",
    source: Annotated[Literal['tushare', 'choice', 'nanhua', 'efinance', 'all'], "The data source to search in."] = 'all',
    strict: Annotated[bool, "Whether to perform a strict search. If True, only exact matches will be returned."] = True,
    timeout: Annotated[float, "Maximum time to wait for the search operation. -1 means wait indefinitely."] = -1
) -> SYMBOL_SEARCH_RESULT | None:
    results = symbol_search_all(keyword, strict=strict, timeout=timeout)
    if results:
        if source == 'all': return results[0]
        else:
            for res in results:
                if res['source'] == source:
                    return res
    return None

_index_components_cache: Dict[str, pd.DataFrame] = {}
def index_components(
    index_symbol: Annotated[str, "The symbol code of the index, e.g., '000905.SH' for CSI500 Index."],
    date: Annotated[str | datetime | date | int, "The date for which to get the index components. Can be a string in 'YYYYMMDD' format, a datetime/date object, or a Unix timestamp. If not provided, uses the latest available date."] = datetime.now().astimezone()
) -> pd.DataFrame:
    """
    Get the list of components for a given index symbol.

    Parameters:
    - index_symbol: str, The symbol code of the index, e.g., "000001.SH" for SSE Composite Index.

    Returns:
    A DataFrame containing the infos of component stocks.
    """
    global _index_components_cache
    if index_symbol in _index_components_cache:
        return _index_components_cache[index_symbol]
    date = parse_datetime(date).astimezone(ZoneInfo("Asia/Shanghai")).strftime('%Y%m%d')
    df = pro.index_weight(index_code=index_symbol, end_date=date)
    assert isinstance(df, pd.DataFrame)
    trade_date = df['trade_date'].max()
    df = df[df['trade_date'] == trade_date]

    stock_df = stock_basic()
    result_df = pd.merge(df, stock_df, left_on='con_code', right_on='ts_code', how='left')

    _index_components_cache[index_symbol] = result_df
    return result_df


def unpack_components(
    index_symbols: Annotated[List[str], "The list of symbol codes of the indexes, e.g., ['000300.SH', '000905.SH']."],
) -> Set[str]:
    """
    Get the list of component stock names for given index symbols.

    Parameters:
    - index_symbol: str, The symbol code of the index, e.g., "000001.SH" for SSE Composite Index.

    Returns:
    A set of names of component stocks.
    """
    components = set()
    with ThreadPoolExecutor() as executor:
        futures = [ executor.submit(index_components, symbol) for symbol in index_symbols ]
        for future in as_completed(futures):
            res = future.result()
            components.update(res['con_code'].tolist())
    stock_basic_df = stock_basic()
    all_names = set(stock_basic_df[stock_basic_df['ts_code'].isin(components)]['name'].tolist())
    logger.debug(f"Total {len(all_names)} unique components found for indexes {index_symbols}.")
    return all_names

def unpack_everything(
    keywords: Annotated[List[str], "The list of symbol codes or names of stocks/indexes, e.g., ['000001.SH', '上证指数']." ] ,
) -> Set[str]:
    """
    Get the list of stock names for given symbol codes or names of stocks/indexes.

    Parameters:
    - keywords: List[str], The list of symbol codes or names of stocks/indexes.

    Returns:
    A set of names of stocks.
    """
    all_names = set()
    index_symbols = []
    for keyword in keywords:
        res = symbol_search(keyword)
        if not res:
            all_names.add(keyword) # Try to search by the keyword itself
        elif res['type'] == 'stock' or res['type'] == 'etf':
            all_names.add(res['name'])
        elif res['type'] == 'index':
            index_symbols.append(res['symbol'])
            all_names.add(res['name']) # Also add the index name
        else:
            raise ValueError(f"Unknown type '{res['type']}' for symbol '{keyword}'.")
    if index_symbols:
        all_names.update(unpack_components(index_symbols))
    logger.debug(f"Total {len(all_names)} unique names found for keywords {keywords}.")
    return all_names

__all__ = [
    "index_basic", "stock_basic",
    "symbol_search", "symbol_search_all", "SYMBOL_SEARCH_RESULT",
    "index_components", "unpack_components", "unpack_everything",
]
