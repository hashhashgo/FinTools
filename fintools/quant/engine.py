from dataclasses import dataclass
from typing import List
import polars as pl
import numpy as np
from .parser import Parser, Node
from .validate import normalize, validate, ast_to_hash
from .compiler import compile_expr
from .utils import make_dataset
from .registry import ScheduleColume, Schedule, GroupBy
from .config import REAL, INTEGER, STRING
from datetime import date

@dataclass
class FactorRecord:
    fid: str
    expr: str

    horizon: int

    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_pos_rate: float
    quality: float

    parent_fid: str | None

class QuantEngine:
    def __init__(
        self,
        test_start: date = date(2024, 1, 1),
        fetch_new_data: bool = False,
    ):
        dataset = make_dataset(fetch_new=fetch_new_data)
        self.dataset = dataset
        self._lazy_res_cols: List[pl.LazyFrame] = []
        self._result = self.dataset[['date', 'symbol']]
    
    def result(self) -> pl.DataFrame:
        if len(self._lazy_res_cols) == 0: return self._result
        self._result = pl.concat([self._result.lazy(), *self._lazy_res_cols], how='horizontal', parallel=True).collect()
        return self._result

    def reset(self):
        self._lazy_res_cols = []
        self._result = self.dataset[['date', 'symbol']]

    def add(self, expressions: List[str] | str):
        if isinstance(expressions, str): expressions = [expressions]
        for expr in expressions:
            lazy_df = self.dataset.lazy()
            ast = Parser(expression=expr).parse()
            ast = normalize(ast)
            validate(ast)
            aid = ast_to_hash(ast)
            col = self.ast_to_col(lazy_df, aid, ast)
            self._lazy_res_cols.append(col.cast(REAL))

    def ast_to_col(self, df: pl.LazyFrame, alias: str, ast: Node) -> pl.LazyFrame:
        extra_columns = {}
        df, compiled = compile_expr(df, ast, extra_columns=extra_columns)
        if isinstance(compiled, Schedule):
            compiled_expr = compiled.expr
            if compiled.over != GroupBy.NONE:
                compiled_expr = compiled.expr.over(compiled.over.value)
        else:
            compiled_expr = pl.lit(compiled)
        return df.select(compiled_expr.alias(alias))