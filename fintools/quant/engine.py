from dataclasses import dataclass
from gc import collect
from typing import List, Literal, Set, Iterable, Tuple, cast
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

import logging
logger = logging.getLogger(__name__)


class QuantEngine:
    def __init__(
        self,
        *,
        start_date: date = date(2014, 1, 1),
        val_start: date = date(2022, 1, 1),
        test_start: date = date(2024, 1, 1),
        init_alphas: Iterable[str] | Path | None = None,
        alphas_cache: Path | None = Path("./results/alphas.json"),
        alpha_records_cache: Path | None = Path("./results/alpha_records.parquet"),
        norm_alpha_cache: Path | None = Path("./results/alpha_cache.parquet"),
        raw_values_cache: Path | None = Path("./results/raw_alpha_values.parquet"),
        fetch_new_data: bool = False,
    ):
        self.dataset = make_dataset(fetch_new=fetch_new_data).filter(pl.col('date') >= start_date)
        self.train_start = start_date
        self.val_start = val_start
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
            ('ic_ir', REAL),
            ('ic_win_rate', REAL),
            ('excess_ann_ret', REAL),
            ('excess_sharpe', REAL),
            ('excess_calmar', REAL),
        ])
        if not fetch_new_data and alpha_records_cache is not None and alpha_records_cache.exists():
            self._alpha_records = pl.read_parquet(alpha_records_cache)
        self._all_alphas = {}
        if not fetch_new_data and alphas_cache is not None and alphas_cache.exists():
            with open(alphas_cache, 'r') as f:
                self._all_alphas = json.load(f)
        self.alpha_pool = set()
        if init_alphas is not None:
            alphas: Set[str] = set()
            try:
                if isinstance(init_alphas, Path):
                    with open(init_alphas, 'r') as f:
                        alphas |= set(self.add(f.readlines()))
                else:
                    alphas |= set(self.add(list(init_alphas)))
            except Exception as e:
                logger.error(f"Error adding initial alphas: {e}")
                raise e
            error_alphas = self.evaluate(alphas=alphas, horizon=5, on='train')\
                .filter(pl.col('ic_mean').is_nan() | pl.col('ic_mean').is_null() | (pl.col('ic_win_rate') < 1e-6))['fid'].to_list()
            for ea in error_alphas:
                logger.warning(f"Alpha with fid {ea} has invalid IC, and will not be added to the pool. Expr: {self._all_alphas.get(ea, 'unknown')}")
                alphas.discard(ea)
            self.alpha_pool |= alphas
    
    def __timerange__(self, on: Literal['train', 'val', 'test']) -> tuple[date, date]:
        if on == 'train':
            return (self.train_start, self.val_start)
        elif on == 'val':
            return (self.val_start, self.test_start)
        elif on == 'test':
            return (self.test_start, self.dataset.select(pl.col('date').max()).item())
        else:
            raise ValueError(f"Invalid on value: {on}")
    
    def __timerange_expr__(self, on: Literal['train', 'val', 'test']) -> pl.Expr:
        start, end = self.__timerange__(on)
        return (pl.col('date') >= start) & (pl.col('date') < end)

    def pool_add(self, fid: str, horizon: int = 5):
        if fid in self.alpha_pool: return
        if fid not in self._all_alphas: raise ValueError(f"Alpha with fid {fid} not found")
        if fid not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid}")
        evaluated = self.evaluate(alphas=self.alpha_pool | {fid}, horizon=horizon, on='train')

    def pool_relevance(self, fid: str, pool: Set[str] | None = None, on: Literal['train', 'val', 'test'] = 'train') -> pl.DataFrame:
        if pool is None:
            pool = self.alpha_pool
        if len(pool) == 0: return pl.DataFrame()
        if fid not in self._all_alphas: raise ValueError(f"Alpha with fid {fid} not found")
        if fid not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid}")
        for fid2 in pool:
            if fid2 not in self._all_alphas: raise ValueError(f"Alpha with fid {fid2} not found")
            if fid2 not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid2}")
        normalized_alpha = self.alpha().lazy().filter(self.__timerange_expr__(on)).select(['date', 'symbol', fid, *pool])
        relevance = normalized_alpha.select([
            pl.corr(pl.col(fid), pl.col(fid2), method='spearman').over('date')\
                .drop_nans().drop_nulls().mean().cast(REAL).alias(fid2) for fid2 in pool
        ])
        return relevance.collect()
    
    def evaluate(self, alphas: Set[str] | None = None, horizon: int = 5, rebalance_period: int = 7, rebalance_delay: int = 1, long_pct: float = 0.8, on: Literal['train', 'val', 'test'] = 'train') -> pl.DataFrame:
        if alphas is None:
            alphas = set(self._all_alphas.keys())
        evaluated = set(self._alpha_records.filter((pl.col('horizon') == horizon) & (pl.col('on') == on))['fid'].to_list())
        pending = alphas - evaluated
        if len(pending) == 0:
            return self._alpha_records.filter((pl.col('horizon') == horizon) & (pl.col('on') == on) & pl.col('fid').is_in(alphas))
        # normalized_alpha = self.alpha().lazy().join(
        normalized_alpha = self.alpha().select(['date', 'symbol', *pending]).join(
            # self.dataset.lazy().with_columns(
            self.dataset.with_columns(
                (pl.col('vwap').shift(-horizon) / (pl.col('vwap').shift(-1) + 1e-9) - 1).over('symbol').fill_null(0.0).alias(f'ret_h{horizon}')
            ).select(['date', 'symbol', f'ret_h{horizon}', 'buyable', 'sellable', 'returns']),
            on=['date', 'symbol'],
            how='inner'
        ).with_columns(
            (pl.col('date').cast(INTEGER) / rebalance_period).floor().cast(INTEGER).alias('period')
        ).with_columns(
            pl.col('buyable').first().over(['period', 'symbol']).alias('buyable'),
            # pl.col('sellable').first().over(['period', 'symbol']).alias('sellable'),
        )

        IC = normalized_alpha.group_by('date').agg([
            pl.corr(pl.col(c), pl.col(f'ret_h{horizon}'), method='spearman').alias(c) for c in pending
        ]).filter(self.__timerange_expr__(on))
        df_daily_ret = normalized_alpha.sort('date').with_columns(
            (
                (((pl.col(fid).rank(method = 'max') - 1) / (pl.len() - 1)).over('date') >= long_pct) &\
                (pl.col('buyable') == True)
            ).alias(fid) for fid in pending
        ).with_columns(
            pl.col(fid).first().over(['period', 'symbol']).alias(fid) for fid in pending
        ).with_columns(
            pl.when(
                pl.col(fid).shift(rebalance_delay).over('symbol')
            ).then(pl.col('returns')).otherwise(None).alias(fid) for fid in pending
        ).group_by('date').agg(
            [(pl.col(fid)).mean().alias(fid) for fid in pending] +
            [(pl.col('returns')).mean().alias('baseline_daily_return')]
        ).sort('date').filter(self.__timerange_expr__(on))
        df_NAV = df_daily_ret.select(
            (
                (pl.col(fid) + 1).cum_prod() /
                (pl.col('baseline_daily_return') + 1).cum_prod()
            ).alias(fid) for fid in pending # excess NAV
        )

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

        # excess_ann_ret = df_NAV.select(
        #     (pl.col(fid).last()  / pl.len() * 252).alias(fid) for fid in pending
        # ).unpivot(value_name='excess_ann_ret', variable_name='fid')
        excess_ann_ret = df_daily_ret.select(
            (((1 + pl.col(fid)) / (1 + pl.col('baseline_daily_return')) - 1).drop_nulls().mean() * (252)).alias(fid) for fid in pending
        ).unpivot(value_name='excess_ann_ret', variable_name='fid')
        excess_ann_vol = df_daily_ret.select(
            (((1 + pl.col(fid)) / (1 + pl.col('baseline_daily_return')) - 1).drop_nulls().std() * np.sqrt(252)).alias(fid) for fid in pending
        ).unpivot(value_name='excess_ann_vol', variable_name='fid')
        excess_max_drawdown = df_NAV.select(
            ((pl.col(fid) - pl.col(fid).cum_max()) / ((pl.col(fid)).cum_max() + 1e-9)).min().alias(fid) for fid in pending
        ).unpivot(value_name='excess_max_drawdown', variable_name='fid')

        records = IC_mean \
            .join(IC_median, on='fid')\
            .join(IC_ir, on='fid')\
            .join(IC_win_rate, on='fid')\
            .join(excess_ann_ret, on='fid')\
            .join(excess_ann_vol, on='fid')\
            .join(excess_max_drawdown, on='fid')\
            .with_columns(
                (pl.col('excess_ann_ret') / (pl.col('excess_ann_vol') + 1e-9)).alias('excess_sharpe'),
                (-pl.col('excess_ann_ret') / (pl.col('excess_max_drawdown') + 1e-9)).alias('excess_calmar'),
            )
        records = records.with_columns(
            pl.col('fid').map_elements(lambda x: self._all_alphas.get(x, 'unknown')).alias('expr'),
            pl.lit(horizon).alias('horizon'),
            pl.lit(on).alias('on'),
        ).select(self._alpha_records.columns).cast(self._alpha_records.schema)
        # self._alpha_records = pl.concat([self._alpha_records.lazy(), records], how='vertical')\
        self._alpha_records = pl.concat([self._alpha_records, records], how='vertical')\
            .unique(subset=['fid', 'horizon', 'on'], keep='last')
            # .unique(subset=['fid', 'horizon', 'on'], keep='last').collect()
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

    def add(self, expressions: List[str] | str) -> List[str]:
        fids = []
        if isinstance(expressions, str): expressions = [expressions]
        for expr in expressions:
            expr = expr.strip()
            if len(expr) == 0 or expr.startswith('#'): continue
            lazy_df = self.dataset.lazy()
            ast = Parser(expression=expr).parse()
            ast = normalize(ast)
            validate(ast)
            aid = ast_to_hash(ast)
            fids.append(aid)
            self._all_alphas[aid] = expr
            if aid in self._raw_alpha.columns: continue
            col = self.ast_to_col(lazy_df, aid, ast)
            self._lazy_res_cols.append(col)
        with open("./results/alphas.json", 'w') as f:
            json.dump(self._all_alphas, f, indent=2)
        return fids

    def plot_top_20_backtest(
            self,
            fid: str,
            *,
            rebalance_period: int = 7,
            delay: int = 1,
            figsettings: dict = {'figsize': (12,6), 'dpi':200},
        ):
        import matplotlib.pyplot as plt

        back_test_result = self.backtest_details(fid=fid, rebalance_period=rebalance_period, pct_ranges=[(0.8, 1.0)], rebalance_delay=delay)[0]
        fig = plt.figure(**figsettings)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(back_test_result['alpha_all']['date'], back_test_result['alpha_all']['alpha_cumulative_return'] + 1, label='NAV', color='blue')
        ax.plot(back_test_result['alpha_all']['date'], back_test_result['alpha_all']['baseline_cumulative_return'] + 1, label='Baseline NAV', color='black', linestyle='--')
        ax2 = ax.twinx()
        ax2.plot(back_test_result['alpha_all']['date'], back_test_result['alpha_all']['excess_cumulative_return'] + 1, label='Excess NAV', color='orange')
        ax2.legend(loc='upper right')
        ax.axvline(x=cast(float, self.val_start), color='gray', linestyle='--', label='Val Start')
        ax.axvline(x=cast(float, self.test_start), color='red', linestyle='--', label='Test Start')
        ax.set_title(fid)
        ax.legend(loc='upper left')
        fig.tight_layout()
        return fig

    def plot_quintile_backtest(
            self,
            fid: str,
            *,
            rebalance_period: int = 7,
            delay: int = 1,
            figsettings: dict = {'figsize': (12,6), 'dpi':200},
        ):
        import matplotlib.pyplot as plt
        
        pct_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        backtest_results = self.backtest_details(fid, rebalance_period=rebalance_period, pct_ranges=pct_ranges, rebalance_delay=delay)
        fig = plt.figure(**figsettings)
        ax = fig.add_subplot(1, 1, 1)
        if pct_ranges is None: pct_ranges = [(np.inf, np.inf)] * len(backtest_results)
        for i, (ran, res) in enumerate(zip(pct_ranges, backtest_results)):
            ax.plot(res['alpha_all']['date'], res['alpha_all']['alpha_cumulative_return'], label=f"{ran[0] * 100:.0f}%-{ran[1] * 100:.0f}%" if ran[0] is not np.inf and ran[1] is not np.inf else str(i))
        ax.plot(backtest_results[0]['alpha_all']['date'], backtest_results[0]['alpha_all']['baseline_cumulative_return'], label='Baseline', color='black', linestyle='--')
        ax.axvline(x=cast(float, self.val_start), color='gray', linestyle='--', label='Val Start')
        ax.axvline(x=cast(float, self.test_start), color='red', linestyle='--', label='Test Start')
        ax.set_title(fid)
        ax.legend()
        fig.tight_layout()
        return fig

    def backtest_details(
            self,
            fid: str,
            *,
            rebalance_period: int = 7,
            pct_ranges: List[Tuple[float, float]] = [(0.8, 1.0)],
            rebalance_delay: int = 1
        ):
        if fid not in self._all_alphas: raise ValueError(f"Alpha with fid {fid} not found")
        if fid not in self.alpha().columns: raise RuntimeError(f"Alpha shoud be computed, but not found in alpha dataframe. fid: {fid}")
        ret = []
        all_lazy = []
        for long_pct_bottom, long_pct_top in pct_ranges:
            df_alpha = self.dataset.lazy().join(
                self.alpha().lazy().select(['date', 'symbol', fid]),
                on=['date', 'symbol'],
                how='inner'
            ).with_columns(
                (pl.col('date').cast(INTEGER) / rebalance_period).floor().cast(INTEGER).alias('period')
            ).sort('date').with_columns(
                ((pl.col(fid).rank(method = 'max') - 1) / (pl.len() - 1)).over('date').alias('rank'),
            ).with_columns(
                pl.col('rank').first().over(['period', 'symbol']).alias('rank'),
                pl.col('buyable').first().over(['period', 'symbol']).alias('buyable'),
                pl.col('sellable').first().over(['period', 'symbol']).alias('sellable'),
            ).with_columns(
                pl.col('rank').shift(rebalance_delay).over('symbol').alias('rank'),
                pl.col('buyable').shift(rebalance_delay).over('symbol').alias('buyable'),
                pl.col('sellable').shift(rebalance_delay).over('symbol').alias('sellable'),
            ).filter(
                (long_pct_bottom <= pl.col('rank')) &
                ((pl.col('rank') < long_pct_top) if long_pct_top < 1.0 else pl.lit(True)) &
                (pl.col('buyable') == True)
            ).group_by('date').agg(
                (pl.col('returns')).mean().alias('alpha_daily_return')
            ).sort('date').with_columns(
                ((pl.col('alpha_daily_return') + 1).cum_prod() - 1).alias('alpha_cumulative_return')
            )
            df_baseline = self.dataset.lazy().group_by('date').agg(
                (pl.col('returns')).mean().alias('baseline_daily_return')
            ).sort('date').with_columns(
                ((pl.col('baseline_daily_return') + 1).cum_prod() - 1).alias('baseline_cumulative_return')
            )
            alpha_all = df_alpha.join(df_baseline, on='date', how='inner').with_columns(
                ((1 + pl.col('alpha_daily_return')) / (1 + pl.col('baseline_daily_return')) - 1).alias('excess_daily_return'),
                ((pl.col('alpha_cumulative_return') + 1) / (pl.col('baseline_cumulative_return') + 1) - 1).alias('excess_cumulative_return')
            )
            excess_max_drawdown = alpha_all.select(
                ((pl.col('excess_cumulative_return') - pl.col('excess_cumulative_return').cum_max()) / (1 + (pl.col('excess_cumulative_return')).cum_max() + 1e-9)).min()
            )
            excess_final_return = alpha_all.select(pl.col('excess_cumulative_return').last())
            excess_ann_ret = alpha_all.select(pl.col('excess_daily_return').drop_nulls().mean() * 252)
            excess_ann_vol = alpha_all.select(pl.col('excess_daily_return').drop_nulls().std() * np.sqrt(252))
            all_lazy.extend([alpha_all, excess_max_drawdown, excess_final_return, excess_ann_ret, excess_ann_vol])
        all_collected = pl.collect_all(all_lazy)
        for i in range(len(pct_ranges)):
            alpha_all, excess_max_drawdown, excess_final_return, excess_ann_ret, excess_ann_vol = all_collected[i*5:(i+1)*5]
            excess_max_drawdown = excess_max_drawdown.item()
            excess_final_return = excess_final_return.item()
            excess_ann_ret = excess_ann_ret.item()
            excess_ann_vol = excess_ann_vol.item()
            excess_sharpe = excess_ann_ret / (excess_ann_vol + 1e-9)
            excess_calmar = -excess_ann_ret / (excess_max_drawdown + 1e-9)
            ret.append({
                'expr': self._all_alphas.get(fid, 'unknown'),
                'alpha_all': alpha_all,
                'excess_max_drawdown': excess_max_drawdown,
                'excess_final_return': excess_final_return,
                'excess_calmar': excess_calmar,
                'excess_ann_ret': excess_ann_ret,
                'excess_ann_vol': excess_ann_vol,
                'excess_sharpe': excess_sharpe,
            })
        return ret

    def fid_to_expr(self, fid: str) -> str:
        return self._all_alphas.get(fid, 'unknown')
    
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