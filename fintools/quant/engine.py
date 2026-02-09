from dataclasses import dataclass
from typing import List
import polars as pl
import numpy as np
from .parser import Parser, Node
from .validate import normalize, validate, ast_to_hash
from .compiler import compile_expr
from .utils import make_dataset
from .registry import ScheduleColume, Schedule, GroupBy
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
        test_start: date = date(2024, 1, 1)
    ):
        dataset = make_dataset()
        self.dataset = dataset
        self.result = dataset[['date', 'symbol', 'returns']]

    def add(self, expressions: List[str]):
        cols: List[pl.LazyFrame] = [self.result.lazy()]
        for expr in expressions:
            lazy_df = self.dataset.lazy()
            ast = Parser(expression=expr).parse()
            ast = normalize(ast)
            validate(ast)
            aid = ast_to_hash(ast)
            col = self.ast_to_col(lazy_df, aid, ast)
            cols.append(col)
        self.result = pl.concat(cols, how='horizontal', parallel=True)

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