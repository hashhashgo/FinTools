if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.as_posix())

from fintools.data_sources.fin_history import DATASOURCES, UnderlyingType, DataFrequency
from datetime import datetime, date, timedelta
import dotenv
dotenv.load_dotenv()

def test_choice_history():
    ch = DATASOURCES['choice']()
    df_ch = ch.history("USDCNH.FX", type=UnderlyingType.STOCK, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_ch)
    df_ch = ch.history("000300.SH", type=UnderlyingType.INDEX, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_ch)

def test_tushare_history():
    tu = DATASOURCES['tushare']()
    df_tu = tu.history("600519.SH", type=UnderlyingType.STOCK, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_tu)
    df_tu = tu.history("IXIC", type=UnderlyingType.INDEX, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_tu)
    df_tu = tu.history("USDCNH.FXCM", type=UnderlyingType.FOREX, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_tu)

def test_efinance_history():
    ef = DATASOURCES['efinance']()
    df_ef = ef.history("600519", type=UnderlyingType.STOCK, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_ef)
    df_ef = ef.history("000300", type=UnderlyingType.INDEX, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_ef)
    df_ef = ef.history("510300", type=UnderlyingType.FUND, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_ef)

def test_yahoo_finance_history():
    yf = DATASOURCES['yahoo_finance']()
    df_yf = yf.history("AAAA", type=UnderlyingType.STOCK, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_yf)
    df_yf = yf.history("AAAA", type=UnderlyingType.STOCK, start=0, end=datetime.now(), freq=DataFrequency.MINUTE60)
    assert len(df_yf)

# def test_investing_history():
#     ic = DATASOURCES['investing.com']()
#     df_ic = ic.history("usd-cny", type=UnderlyingType.FOREX, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
#     assert len(df_ic)
#     df_ic = ic.history("usd-cny", type=UnderlyingType.FOREX, start=0, end=datetime.now(), freq=DataFrequency.MINUTE60)
#     assert len(df_ic)

def test_nanhua_history():
    nh = DATASOURCES['nanhua']()
    df_nh = nh.history("PP_NH", type=UnderlyingType.COMMODITY, start=0, end=datetime.now(), freq=DataFrequency.DAILY)
    assert len(df_nh)
    df_nh = nh.history("PP_NH", type=UnderlyingType.COMMODITY, start=0, end=datetime.now(), freq=DataFrequency.MINUTE60)
    assert len(df_nh)

if __name__ == "__main__":
    # test_investing_history()
    # test_choice_history()
    test_tushare_history()
    test_yahoo_finance_history()
    test_nanhua_history()