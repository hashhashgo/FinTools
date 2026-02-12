from dataclasses import dataclass
from typing import List, Literal, Set
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
        start_date: date = date(2014, 1, 1),
        test_start: date = date(2024, 1, 1),
        alphas_cache: Path | None = Path("./results/alphas.json"),
        alpha_records_cache: Path | None = Path("./results/alpha_records.parquet"),
        norm_alpha_cache: Path | None = Path("./results/alpha_cache.parquet"),
        raw_values_cache: Path | None = Path("./results/raw_alpha_values.parquet"),
        fetch_new_data: bool = False,
    ):
        dataset = make_dataset(fetch_new=fetch_new_data).filter(pl.col('date') >= start_date)
        self.dataset = dataset
        self.test_start = test_start
        self._lazy_res_cols: List[pl.LazyFrame] = []
        self._norm_alpha = self.dataset[['date', 'symbol']]
        self._raw_alpha = self.dataset[['date', 'symbol']]
        self.raw_values_cache = raw_values_cache
        if not fetch_new_data and raw_values_cache is not None and raw_values_cache.exists():
            self._raw_alpha = pl.read_parquet(raw_values_cache)
        self.norm_alpha_cache = norm_alpha_cache
        if not fetch_new_data and norm_alpha_cache is not None and norm_alpha_cache.exists():
            self._norm_alpha = pl.read_parquet(norm_alpha_cache)
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
        if not fetch_new_data and alpha_records_cache is not None and alpha_records_cache.exists():
            self._alpha_records = pl.read_parquet(alpha_records_cache)
        self._all_alphas = {}
        if not fetch_new_data and alphas_cache is not None and alphas_cache.exists():
            with open(alphas_cache, 'r') as f:
                self._all_alphas = json.load(f)
        self.alpha_pool = set()

    def pool_add(self, fid: str, horizon: int = 5):
        if fid in self.alpha_pool: return
        if fid not in self._all_alphas: raise ValueError(f"Alpha with fid {fid} not found")
        if fid not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid}")
        evaluated = self.evaluate(alphas=self.alpha_pool | {fid}, horizon=horizon, on='train')

    def pool_relevance(self, fid: str, pool: Set[str] | None = None, on: Literal['train', 'test'] = 'train') -> pl.DataFrame:
        if pool is None:
            pool = self.alpha_pool
        if len(pool) == 0: return pl.DataFrame()
        if fid not in self._all_alphas: raise ValueError(f"Alpha with fid {fid} not found")
        if fid not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid}")
        for fid2 in pool:
            if fid2 not in self._all_alphas: raise ValueError(f"Alpha with fid {fid2} not found")
            if fid2 not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid2}")
        normalized_alpha = self.alpha().lazy().filter(pl.col('date') >= self.test_start if on == 'test' else pl.col('date') < self.test_start).select(['date', 'symbol', fid, *pool])
        relevance = normalized_alpha.select([
            pl.corr(pl.col(fid), pl.col(fid2), method='spearman').over('date')\
                .drop_nans().drop_nulls().mean().cast(REAL).alias(fid2) for fid2 in pool
        ])
        return relevance.collect()
    
    def evaluate(self, alphas: Set[str] | None = None, horizon: int = 5, on: Literal['train', 'test'] = 'train') -> pl.DataFrame:
        if alphas is None:
            alphas = set(self._all_alphas.keys())
        evaluated = set(self._alpha_records.filter((pl.col('horizon') == horizon) & (pl.col('on') == on))['fid'].to_list())
        pending = alphas - evaluated
        normalized_alpha = self.alpha().lazy().join(
            self.dataset.lazy().with_columns(
                (pl.col('vwap').shift(-horizon + 1) / (pl.col('vwap').shift(1) + 1e-9)).over('symbol').fill_null(0.0).alias(f'ret_h{horizon}')
            ).select(['date', 'symbol', f'ret_h{horizon}']),
            on=['date', 'symbol'],
            how='inner'
        )
        IC = normalized_alpha.group_by('date').agg([
            pl.corr(pl.col(c), pl.col(f'ret_h{horizon}')).alias(c) for c in pending
        ]).filter(pl.col('date') >= self.test_start if on == 'test' else pl.col('date') < self.test_start)
        IC_mean = IC.select(
            (pl.col(c).drop_nans().drop_nulls().mean().cast(REAL).alias(c) for c in pending),
        ).unpivot(value_name='ic_mean', variable_name='fid')
        IC_median = IC.select(
            (pl.col(c).drop_nans().drop_nulls().median().cast(REAL).alias(c) for c in pending),
        ).unpivot(value_name='ic_median', variable_name='fid')
        IC_std = IC.select(
            (pl.col(c).drop_nans().drop_nulls().std().cast(REAL).alias(c) for c in pending),
        ).unpivot(value_name='ic_std', variable_name='fid')
        IC_ir = IC_mean.join(IC_std, on='fid').with_columns(
            (pl.col('ic_mean') / (pl.col('ic_std') + 1e-9)).cast(REAL).alias('ic_ir')
        ).select(['fid', 'ic_ir'])
        IC_win_rate = IC.select(
            (pl.col(c).filter(pl.col(c) > 0).drop_nans().drop_nulls().count() / (pl.col(c).drop_nans().drop_nulls().count() + 1e-9)).cast(REAL).alias(c) for c in pending
        ).unpivot(value_name='ic_win_rate', variable_name='fid')
        IC_loss_rate = IC.select(
            (pl.col(c).filter(pl.col(c) < 0).drop_nans().drop_nulls().count() / (pl.col(c).drop_nans().drop_nulls().count() + 1e-9)).cast(REAL).alias(c) for c in pending
        ).unpivot(value_name='ic_loss_rate', variable_name='fid')
        records = IC_mean \
            .join(IC_median, on='fid')\
            .join(IC_std, on='fid')\
            .join(IC_ir, on='fid')\
            .join(IC_win_rate, on='fid')\
            .join(IC_loss_rate, on='fid')
        records = records.with_columns(
            pl.col('fid').map_elements(lambda x: self._all_alphas.get(x, 'unknown')).alias('expr'),
            pl.lit(horizon).alias('horizon'),
            pl.lit(on).alias('on'),
        ).select(self._alpha_records.columns)
        self._alpha_records = pl.concat([self._alpha_records.lazy(), records], how='vertical').collect()
        self._alpha_records = self._alpha_records.unique(subset=['fid', 'horizon', 'on'], keep='last')
        if self.alpha_records_cache is not None:
            self.alpha_records_cache.parent.mkdir(parents=True, exist_ok=True)
            self._alpha_records.write_parquet(self.alpha_records_cache)
        return self._alpha_records.filter(pl.col('fid').is_in(alphas))

    def alpha(self, max_parallel: int | None = None) -> pl.DataFrame:
        raw_alpha = self.raw_alpha(max_parallel=max_parallel)
        all_cols = set(raw_alpha.columns) - {'date', 'symbol'}
        calc_cols = all_cols - set(self._norm_alpha.columns)
        if len(calc_cols) == 0: return self._norm_alpha
        res = self.normalize_alpha(raw_alpha.lazy(), cols=list(calc_cols))
        self._norm_alpha = pl.concat([self._norm_alpha.lazy(), res.select(list(calc_cols))], how='horizontal', parallel=True).collect()
        if self.norm_alpha_cache is not None:
            self._norm_alpha.write_parquet(self.norm_alpha_cache)
        return self._norm_alpha
    
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
            self._all_alphas[aid] = expr
            if aid in self._raw_alpha.columns: continue
            col = self.ast_to_col(lazy_df, aid, ast)
            self._lazy_res_cols.append(col)
        with open("./results/alphas.json", 'w') as f:
            json.dump(self._all_alphas, f, indent=2)

    def backtest_alpha(self, fid: str, long_top_pct: float = 0.2, delay: int = 1):
        if fid not in self._all_alphas: raise ValueError(f"Alpha with fid {fid} not found")
        if fid not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid}")
        df_alpha = self.dataset.lazy().join(
            self.alpha().lazy().select(['date', 'symbol', fid]),
            on=['date', 'symbol'],
            how='inner'
        ).with_columns(
            ((pl.col(fid).rank(method = 'max') - 1) / (pl.len() - 1)).over('date').alias('rank'),
        ).filter(pl.col('rank').shift(delay) >= 1 - long_top_pct).group_by('date').agg(
            (pl.col('returns') / 100).mean().alias('alpha_daily_return')
        ).sort('date').with_columns(
            ((pl.col('alpha_daily_return') + 1).cum_prod() - 1).alias('alpha_cumulative_return')
        )
        df_excess = self.dataset.lazy().group_by('date').agg(
            (pl.col('returns') / 100).mean().alias('baseline_daily_return')
        ).sort('date').with_columns(
            ((pl.col('baseline_daily_return') + 1).cum_prod() - 1).alias('baseline_cumulative_return')
        )
        alpha_all = df_alpha.join(df_excess, on='date', how='inner').with_columns(
            (pl.col('alpha_daily_return') - pl.col('baseline_daily_return')).alias('excess_daily_return'),
            ((pl.col('alpha_cumulative_return') + 1) / (pl.col('baseline_cumulative_return') + 1) - 1).alias('excess_cumulative_return')
        )
        ann_ret = alpha_all.select(pl.col('alpha_daily_return').drop_nulls().mean() * 252)
        ann_vol = alpha_all.select(pl.col('alpha_daily_return').drop_nulls().std() * np.sqrt(252))
        max_drawdown = df_alpha.select(
            (pl.col('alpha_cumulative_return') - pl.col('alpha_cumulative_return').cum_max()).min().alias('drawdown')
        )
        final_return = df_alpha.select(pl.col('alpha_cumulative_return').last())
        excess_sharpe = alpha_all.select(
            (pl.col('excess_daily_return').drop_nulls().mean() / (pl.col('excess_daily_return').drop_nulls().std() + 1e-9) * np.sqrt(252)).alias('excess_sharpe')
        )
        alpha_all, ann_ret, ann_vol, max_drawdown, final_return, excess_sharpe = pl.collect_all([alpha_all, ann_ret, ann_vol, max_drawdown, final_return, excess_sharpe])
        ann_ret = ann_ret.item()
        ann_vol = ann_vol.item()
        max_drawdown = max_drawdown.item()
        final_return = final_return.item()
        calmar = -ann_ret / (max_drawdown + 1e-9)
        sharpe = ann_ret / (ann_vol + 1e-9)
        excess_sharpe = excess_sharpe.item()
        return {
            'alpha_all': alpha_all,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'max_drawdown': max_drawdown,
            'final_return': final_return,
            'calmar': calmar,
            'sharpe': sharpe,
            'excess_sharpe': excess_sharpe,
        }
    
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