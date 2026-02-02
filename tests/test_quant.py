if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.as_posix())

def test_get_all_stock():
    from fintools.quant.utils import fetch_all_stock
    import pandas as pd

    df_stocks = fetch_all_stock()

    # Basic checks
    assert isinstance(df_stocks, pd.DataFrame)
    assert not df_stocks.empty
    assert 'ts_code' in df_stocks.columns

    print(df_stocks['ts_code'].unique())


if __name__ == "__main__":
    test_get_all_stock()
    print("All tests passed.")