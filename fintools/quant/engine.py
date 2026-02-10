from dataclasses import dataclass
from typing import List, Literal
import polars as pl
import numpy as np
from .parser import Parser, Node
from .validate import normalize, validate, ast_to_hash
from .compiler import compile_expr
from .utils import make_dataset
from .registry import ScheduleColume, Schedule, GroupBy
from .config import REAL, INTEGER, STRING
from datetime import date
from pathlib import Path
import tqdm
import json


class QuantEngine:
    def __init__(
        self,
        test_start: date = date(2024, 1, 1),
        alphas_cache: Path | None = Path("./results/alphas.json"),
        alpha_records_cache: Path | None = Path("./results/alpha_records.parquet"),
        norm_alpha_cache: Path | None = Path("./results/alpha_cache.parquet"),
        raw_values_cache: Path | None = Path("./results/raw_alpha_values.parquet"),
        fetch_new_data: bool = True,
    ):
        dataset = make_dataset(fetch_new=fetch_new_data)
        self.dataset = dataset
        self.test_start = test_start
        self._lazy_res_cols: List[pl.LazyFrame] = []
        self._alpha = self.dataset[['date', 'symbol']]
        self._raw_alpha = self.dataset[['date', 'symbol']]
        self.raw_values_cache = raw_values_cache
        if raw_values_cache is not None and raw_values_cache.exists():
            self._raw_alpha = pl.read_parquet(raw_values_cache)
        self.norm_alpha_cache = norm_alpha_cache
        if norm_alpha_cache is not None and norm_alpha_cache.exists():
            self._alpha = pl.read_parquet(norm_alpha_cache)
        self.alpha_records_cache = alpha_records_cache
        self._alpha_records = pl.DataFrame(schema=[
            ('fid', STRING),
            ('expr', STRING),
            ('horizon', INTEGER),
            ('on', STRING),
            ('ic_mean', REAL),
            ('ic_median', REAL),
            ('ic_std', REAL),
            ('ic_ir', REAL),
            ('ic_win_rate', REAL),
            ('ic_loss_rate', REAL),
        ])
        if alpha_records_cache is not None and alpha_records_cache.exists():
            self._alpha_records = pl.read_parquet(alpha_records_cache)
        self._alphas = {}
        if alphas_cache is not None and alphas_cache.exists():
            with open(alphas_cache, 'r') as f:
                self._alphas = json.load(f)
        self.alpha_pool = set(self._alphas.keys())
    
    def evaluate(self, horizon: int = 5, on: Literal['train', 'test'] = 'train') -> pl.DataFrame:
        normalized_alpha = self.alpha().lazy().join(
            self.dataset.lazy().with_columns(
                (pl.col('vwap') / (pl.col('vwap').shift(horizon) + 1e-9)).over('symbol').fill_null(0.0).alias(f'ret_h{horizon}')
            ).select(['date', 'symbol', f'ret_h{horizon}']),
            on=['date', 'symbol'],
            how='inner'
        )
        IC = normalized_alpha.group_by('date').agg([
            pl.corr(pl.col(c), pl.col(f'ret_h{horizon}')).alias(c) for c in self.alpha_pool
        ]).filter(pl.col('date') >= self.test_start if on == 'test' else pl.col('date') < self.test_start)
        IC_mean = IC.select(
            (pl.col(c).drop_nans().drop_nulls().mean().cast(REAL).alias(c) for c in self.alpha_pool),
        ).unpivot(value_name='ic_mean', variable_name='fid')
        IC_median = IC.select(
            (pl.col(c).drop_nans().drop_nulls().median().cast(REAL).alias(c) for c in self.alpha_pool),
        ).unpivot(value_name='ic_median', variable_name='fid')
        IC_std = IC.select(
            (pl.col(c).drop_nans().drop_nulls().std().cast(REAL).alias(c) for c in self.alpha_pool),
        ).unpivot(value_name='ic_std', variable_name='fid')
        IC_ir = IC_mean.join(IC_std, on='fid').with_columns(
            (pl.col('ic_mean') / (pl.col('ic_std') + 1e-9)).cast(REAL).alias('ic_ir')
        ).select(['fid', 'ic_ir'])
        IC_win_rate = IC.select(
            (pl.col(c).drop_nans().drop_nulls().filter(pl.col(c) > 0).count() / (pl.col(c).drop_nans().drop_nulls().count() + 1e-9)).cast(REAL).alias(c) for c in self.alpha_pool
        ).unpivot(value_name='ic_win_rate', variable_name='fid')
        IC_loss_rate = IC.select(
            (pl.col(c).drop_nans().drop_nulls().filter(pl.col(c) < 0).count() / (pl.col(c).drop_nans().drop_nulls().count() + 1e-9)).cast(REAL).alias(c) for c in self.alpha_pool
        ).unpivot(value_name='ic_loss_rate', variable_name='fid')
        records = IC_mean \
            .join(IC_median, on='fid')\
            .join(IC_std, on='fid')\
            .join(IC_ir, on='fid')\
            .join(IC_win_rate, on='fid')\
            .join(IC_loss_rate, on='fid')
        records = records.with_columns(
            pl.col('fid').map_elements(lambda x: self._alphas.get(x, 'unknown')).alias('expr'),
            pl.lit(horizon).alias('horizon'),
            pl.lit(on).alias('on'),
        ).select(self._alpha_records.columns)
        self._alpha_records = pl.concat([self._alpha_records.lazy(), records], how='vertical').collect()
        self._alpha_records = self._alpha_records.unique(subset=['fid', 'horizon', 'on'], keep='last')
        if self.alpha_records_cache is not None:
            self.alpha_records_cache.parent.mkdir(parents=True, exist_ok=True)
            self._alpha_records.write_parquet(self.alpha_records_cache)
        return self._alpha_records

    def alpha(self, max_parallel: int | None = None) -> pl.DataFrame:
        raw_alpha = self.raw_alpha(max_parallel=max_parallel)
        all_cols = set(raw_alpha.columns) - {'date', 'symbol'}
        calc_cols = all_cols - set(self._alpha.columns)
        if len(calc_cols) == 0: return self._alpha
        res = self.normalize_alpha(raw_alpha.lazy(), cols=list(calc_cols))
        self._alpha = pl.concat([self._alpha.lazy(), res.select(list(calc_cols))], how='horizontal', parallel=True).collect()
        if self.norm_alpha_cache is not None:
            self._alpha.write_parquet(self.norm_alpha_cache)
        return self._alpha
    
    def raw_alpha(self, max_parallel: int | None = None) -> pl.DataFrame:
        if max_parallel is None: max_parallel = len(self._lazy_res_cols)
        if len(self._lazy_res_cols) == 0: return self._raw_alpha
        for i in tqdm.trange(0, len(self._lazy_res_cols), max_parallel):
            self._raw_alpha = pl.concat([self._raw_alpha.lazy(), *self._lazy_res_cols[i:i+max_parallel]], how='horizontal', parallel=True).collect()
        self._lazy_res_cols = []
        if self.raw_values_cache is not None:
            self.raw_values_cache.parent.mkdir(parents=True, exist_ok=True)
            self._raw_alpha.write_parquet(self.raw_values_cache)
        return self._raw_alpha

    def add(self, expressions: List[str] | str):
        if isinstance(expressions, str): expressions = [expressions]
        for expr in expressions:
            lazy_df = self.dataset.lazy()
            ast = Parser(expression=expr).parse()
            ast = normalize(ast)
            validate(ast)
            aid = ast_to_hash(ast)
            self._alphas[aid] = expr
            if aid in self._raw_alpha.columns: continue
            col = self.ast_to_col(lazy_df, aid, ast)
            self._lazy_res_cols.append(col)
        with open("./results/alphas.json", 'w') as f:
            json.dump(self._alphas, f, indent=2)
    
    @classmethod
    def normalize_alpha(cls, df: pl.LazyFrame, cols: List[str]) -> pl.LazyFrame:
        if not isinstance(cols, list): cols = [cols]
        for c in cols:
            median = pl.col(c).median()
            mad = (pl.col(c) - median).abs().median()
            z = (pl.col(c) - median) / (mad * 1.4826 + 1e-9)
            z = z.clip(-5, 5).fill_null(0).fill_nan(0)
            df = df.with_columns(z.alias(c))
        return df

    @classmethod
    def ast_to_col(cls, df: pl.LazyFrame, alias: str, ast: Node) -> pl.LazyFrame:
        extra_columns = {}
        df, compiled = compile_expr(df, ast, extra_columns=extra_columns)
        if isinstance(compiled, Schedule):
            compiled_expr = compiled.expr
            if compiled.over != GroupBy.NONE:
                compiled_expr = compiled.expr.over(compiled.over.value)
        else:
            compiled_expr = pl.lit(compiled)
        return df.select(compiled_expr.cast(REAL).alias(alias))