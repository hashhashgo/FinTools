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
    from fintools.quant.engine import compile_expression
    from fintools.quant.parser import ParserError
    from fintools.quant.validate import ValidationError
    from fintools.quant.compiler import CompileError
    import hashlib

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
        for i, each in enumerate(lines):
            each = each.strip()
            total += 1
            try:
                result = compile_expression(each)
                passed += 1

                assert 'ast' in result
                assert 'aid' in result
                assert result['aid'] is not None
            
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
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    # test_fetch_everything()
    # test_compile_expression(raise_if_error=False)
    # print("All tests passed.")
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