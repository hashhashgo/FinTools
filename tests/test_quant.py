if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.as_posix())

def test_fetch_everything():
    from fintools.quant.utils import fetch_everything
    import polars as pl

    df_stocks = fetch_everything()

    # Basic checks
    assert isinstance(df_stocks, pl.DataFrame)
    assert not df_stocks.is_empty()
    assert 'ts_code' in df_stocks.columns

def test_compile_expression(pre_hash: list[str] = [], raise_if_error=True):
    from fintools.quant.parser import ParserError, Parser
    from fintools.quant.validate import ValidationError, validate, normalize, ast_to_hash
    from fintools.quant.compiler import CompileError, compile_expr
    from fintools.quant.registry import DATA_SCHEMA
    import polars as pl
    import hashlib

    df = pl.DataFrame(schema=DATA_SCHEMA)

    with open("tests/alpha101.txt", "r") as f:
        lines = f.readlines()

        sha1 = hashlib.sha1()
        for l in lines:
            sha1.update(l.encode('utf-8'))
        if pre_hash and sha1.hexdigest() == pre_hash[-1]:
            return
        pre_hash.append(sha1.hexdigest())

        total = 0
        passed = 0
        memo = {}
        for i, each in enumerate(lines):
            each = each.strip()
            total += 1
            try:
                ast = Parser(expression=each).parse()
                ast = normalize(ast)
                validate(ast)
                hashed = ast_to_hash(ast)
                ldf = df.lazy()
                compile_expr(ldf, ast, extra_columns=memo)
                passed += 1
            
            except ParserError as pe:
                if raise_if_error:
                    raise pe
                else:
                    print(f"ParserError compiling expression #{i + 1}:")
                    print(pe.pretty())
                    print("\n")
                
            except ValidationError as ve:
                if raise_if_error:
                    raise ve
                else:
                    print(f"ValidationError compiling expression #{i + 1}: {ve}")

            except CompileError as ce:
                if raise_if_error:
                    raise ce
                else:
                    print(f"CompileError compiling expression #{i + 1}: {ce}")

            except Exception as e:
                raise e
        if raise_if_error is False:
            print(f"Compiled {passed}/{total} expressions successfully.")

if __name__ == "__main__":
    import time
    import os
    import dotenv; dotenv.load_dotenv()

    from fintools.quant.utils import fetch_all
    from fintools.utils.underlying import stock_basic, index_basic

    fetch_all(api='stock_st', underlyings=set(stock_basic()['ts_code'].unique().tolist()), fetch_new=True)
    
    from fintools.quant.engine import QuantEngine
    engine = QuantEngine(fetch_new_data=False)
    

    ########## auto test alpha101.txt ##########
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    observer = Observer()
    class ReloadHandler(FileSystemEventHandler):
        def __init__(self, pre_hash: list[str] = []):
            super().__init__()
            self.pre_hash = pre_hash

        def on_modified(self, event):
            event_path = os.path.normpath(event.src_path)
            assert isinstance(event_path, str)
            if event_path.endswith('alpha101.txt'):
                try:
                    test_compile_expression(pre_hash=self.pre_hash, raise_if_error=False)
                except Exception as e:
                    print(f"Tests failed: {e}")
    file_hash = []
    test_compile_expression(pre_hash=file_hash, raise_if_error=False)
    handler = ReloadHandler(pre_hash=file_hash)
    observer.schedule(handler, path="tests/", recursive=True)
    observer.start()
    print("Watching for changes in alpha101.txt...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()