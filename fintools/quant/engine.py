from typing import Dict, List, Literal, Set, Iterable, Tuple, cast, IO, Callable
import polars as pl
import numpy as np
from .parser import Parser, Node
from .validate import normalize, validate, ast_to_hash, ast_to_expression
from .compiler import compile_expr
from .utils import make_dataset
from .registry import ScheduleColume, Schedule, GroupBy
from .config import REAL, INTEGER, STRING
from datetime import date
from pathlib import Path
from importlib.resources import files
import tqdm
import json
from os import PathLike

import logging
logger = logging.getLogger(__name__)


class QuantEngine:
    def __init__(
        self,
        *,
        dataset: pl.DataFrame | Callable[..., pl.DataFrame] = lambda fetch_new: make_dataset(fetch_new=fetch_new),
        start_date: date = date(2014, 1, 1),
        val_start: date = date(2022, 1, 1),
        test_start: date = date(2024, 1, 1),
        init_alphas: Iterable[str] | str | PathLike[str] | IO[str] | None = None,
        alphas_cache: Path | None = None, # Path("./results/alphas.json"),
        alpha_pool_cache: Path | None = None, # Path("./results/alpha_pool.txt"),
        alpha_records_cache: Path | None = None, # Path("./results/alpha_records.parquet"),
        norm_alpha_cache: Path | None = None, # Path("./results/alpha_cache.parquet"),
        raw_values_cache: Path | None = None, # Path("./results/raw_alpha_values.parquet"),
        fetch_new_data: bool = False,
    ):
        if isinstance(dataset, pl.DataFrame):
            self.dataset = dataset.filter(pl.col('date') >= start_date)
        elif callable(dataset):
            self.dataset = dataset(fetch_new=fetch_new_data).filter(pl.col('date') >= start_date)
        self.train_start = start_date
        self.val_start = val_start
        self.test_start = test_start
        self._lazy_res_cols: List[pl.LazyFrame] = []
        self._lazy_res_cols_fids: Set[str] = set()
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
        self._all_alphas: Dict[str, str] = {}
        self.alphas_cache = alphas_cache
        if not fetch_new_data and alphas_cache is not None and alphas_cache.exists():
            with open(alphas_cache, 'r') as f:
                self._all_alphas = json.load(f)
        self.alpha_pool: Set[str] = set()
        self.alpha_pool_cache = alpha_pool_cache
        if alpha_pool_cache is not None and alpha_pool_cache.exists():
            with open(alpha_pool_cache, 'r') as f:
                self.alpha_pool = set(self.add(f.readlines()))
        if init_alphas is not None and len(self.alpha_pool) == 0:
            alphas: Set[str] = set()
            try:
                if isinstance(init_alphas, (str, PathLike)):
                    if init_alphas == "alpha101":
                        init_alphas = cast(PathLike, files('fintools').joinpath('data/alpha101.txt'))
                    with open(init_alphas, 'r') as f:
                        alphas |= set(self.add(f.readlines()))
                elif hasattr(init_alphas, 'readlines'):
                    alphas |= set(self.add(cast(IO[str], init_alphas).readlines()))
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
            self.alpha_pool_cache = alpha_pool_cache
            if alpha_pool_cache is not None:
                with open(alpha_pool_cache, 'w') as f:
                    for fid in sorted(self.alpha_pool):
                        assert fid in self._all_alphas, f"fid {fid} not found in _all_alphas"
                        f.write(self._all_alphas.get(fid, fid) + '\n')

    def clear_cache(self):
        self.alpha_pool = set()
        if self.alpha_pool_cache is not None and self.alpha_pool_cache.exists():
            with open(self.alpha_pool_cache, 'r') as f:
                self.alpha_pool = set(self.add(f.readlines()))
        self.add([self._all_alphas.get(alpha, alpha) for alpha in self.alpha_pool])
        self.alpha()
        self._lazy_res_cols = []
        self._lazy_res_cols_fids = set()
        self._norm_alpha = self._norm_alpha.select(['date', 'symbol', *self.alpha_pool]).rechunk()
        self._raw_alpha = self._raw_alpha.select(['date', 'symbol', *self.alpha_pool]).rechunk()
        self._alpha_records = self._alpha_records.rechunk()
    
    def __timerange__(self, on: Literal['train', 'val', 'test', 'all']) -> tuple[date, date]:
        if on == 'train':
            return (self.train_start, self.val_start)
        elif on == 'val':
            return (self.val_start, self.test_start)
        elif on == 'test':
            return (self.test_start, self.dataset.select(pl.col('date').max()).item())
        elif on == 'all':
            return (self.train_start, self.dataset.select(pl.col('date').max()).item())
        else:
            raise ValueError(f"Invalid on value: {on}")
    
    def __timerange_expr__(self, on: Literal['train', 'val', 'test', 'all']) -> pl.Expr:
        start, end = self.__timerange__(on)
        return (pl.col('date') >= start) & (pl.col('date') < end)

    def pool_add_batch(
        self,
        new_fids: Iterable[str] | str,
        *,
        relevance_similar_threshold: float = 0.5,
        relevance_same_threshold: float = 0.8,
        top_k: int = 5,
        pool: Set[str] | None = None,
        on: Literal["train", "val", "test", "all"] = "train",
        date_equal_weight: bool = True,   # True: 每个date等权平均；False: 按有效样本数加权
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Add a batch of new alpha FIDs to the pool, evaluating their relevance and performance.

        Returns a tuple of three sets:
        - good_fids: FIDs that are added to the pool
        - pool: Updated pool after adding good_fids and removing drop_fids
        - drop_fids: FIDs that are removed from the pool due to low relevance or performance
        """
        apply_to_pool = False
        if pool is None:
            pool = self.alpha_pool
            apply_to_pool = True
        pool = set(pool)

        if isinstance(new_fids, str):
            new_fids = [new_fids]

        new_fids = set(new_fids) - pool
        if not new_fids:
            return set(), pool, set()

        relevance_df = self.pool_relevance_batch(
            new_fids=new_fids,
            pool=pool,
            on=on,
            date_equal_weight=date_equal_weight,
        ).unpivot(
            index="pool_fid",
            variable_name="new_fid",
            value_name="relevance"
        ).drop_nans("relevance")
        relevance_df = relevance_df.join(
            self.evaluate(pool),
            left_on='pool_fid',
            right_on='fid',
            how='left'
        ).join(
            self.evaluate(new_fids),
            left_on='new_fid',
            right_on='fid',
            how='left',
            suffix="_new"
        )

        rel_similar = relevance_df.filter(pl.col('relevance') >= relevance_similar_threshold)
        rel_same = relevance_df.filter(pl.col('relevance') >= relevance_same_threshold)

        bad_fids = set(
            rel_similar.with_columns([
                pl.col("ic_mean").abs().top_k(top_k).min().over('new_fid').alias("min_abs_ic"),
                pl.col("excess_sharpe").abs().top_k(top_k).min().over('new_fid').alias("min_abs_excess_sharpe"),
            ]).filter(
                (pl.col('ic_mean_new').abs() < pl.col('min_abs_ic')) &
                (pl.col('excess_sharpe_new').abs() < pl.col('min_abs_excess_sharpe'))
            ).select('new_fid').unique().to_series()
        )

        good_fids = set(new_fids) - bad_fids

        drop_fids = set(
            rel_same.with_columns([
                pl.col("ic_mean_new").abs().top_k(top_k).min().over('pool_fid').alias("min_abs_ic"),
                pl.col("excess_sharpe_new").abs().top_k(top_k).min().over('pool_fid').alias("min_abs_excess_sharpe"),
            ]).filter(
                (pl.col('ic_mean').abs() < pl.col('min_abs_ic')) &
                (pl.col('excess_sharpe').abs() < pl.col('min_abs_excess_sharpe'))
            ).select('pool_fid').unique().to_series()
        )

        pool = (pool - drop_fids) | good_fids

        if apply_to_pool:
            self.alpha_pool = pool
            if self.alpha_pool_cache is not None:
                with open(self.alpha_pool_cache, 'w') as f:
                    for fid in sorted(self.alpha_pool):
                        assert fid in self._all_alphas, f"fid {fid} not found in _all_alphas"
                        f.write(self._all_alphas.get(fid, fid) + '\n')
            self._write_alpha_cache()
        
        return good_fids, pool, drop_fids

    def pool_relevance_batch(
        self,
        new_fids: Iterable[str] | str,
        *,
        pool: Set[str] | None = None,
        on: Literal["train", "val", "test", "all"] = "train",
        date_equal_weight: bool = True,   # True: 每个date等权平均；False: 按有效样本数加权
    ) -> pl.DataFrame:
        if pool is None:
            pool = self.alpha_pool
        pool = set(pool)

        if isinstance(new_fids, str):
            new_fids = set([new_fids])
        new_fids = set(new_fids)

        if not new_fids or not pool:
            # 输出格式：pool做行
            return pl.DataFrame({"pool_fid": sorted(pool), **{nf: [] for nf in new_fids}})

        # 校验列存在
        cols = set(self.alpha().columns)
        for fid in list(pool) + list(new_fids):
            if fid not in self._all_alphas:
                raise ValueError(f"Alpha with fid {fid} not found")
            if fid not in cols:
                self.add([self._all_alphas.get(fid, fid)])

        # 防止新因子里有跟pool重复（会导致列重复）
        pool_only = [f for f in sorted(pool) if f not in set(new_fids)]
        if not pool_only:
            # pool 全被 new 覆盖了，那行集合为空就没意义；这里按你的需求也可以改成仍然输出
            pool_only = sorted(pool)

        new_fids = list(new_fids)
        all_cols = pool_only + new_fids

        lf = (
            self.alpha()
            .lazy()
            .filter(self.__timerange_expr__(on))
            .select(["date", "symbol", *all_cols])
        )

        # 1) Spearman：先 rank（按date横截面）
        lf = lf.with_columns(
            pl.col(all_cols)
            .rank(method="average")
            .over("date")
            .cast(REAL)
        )

        # 2) 再对rank做zscore（按date横截面）
        lf = lf.with_columns(
            (
                (pl.col(all_cols) - pl.col(all_cols).mean().over("date"))
                / pl.col(all_cols).std(ddof=1).over("date")
            ).cast(REAL)
        )

        df = lf.collect()

        # 3) 按 date 分块做矩阵相关
        # partition_by 在 Rust 侧分块，通常比 python groupby 快/省事
        parts = df.partition_by("date", as_dict=False)

        N = len(pool_only)
        M = len(new_fids)

        sum_corr = np.zeros((N, M), dtype=np.float64)
        cnt = np.zeros((N, M), dtype=np.int32)

        for dfi in tqdm.tqdm(parts, desc="Calculate Relevance..."):
            A = dfi.select(pool_only).to_numpy()  # (n_symbol, N)
            B = dfi.select(new_fids).to_numpy()   # (n_symbol, M)

            # mask：有效值（排除 null/NaN）
            Am = np.isfinite(A)
            Bm = np.isfinite(B)

            # NaN 置0，方便 matmul；有效样本数用 mask 来算
            A0 = np.where(Am, A, 0.0)
            B0 = np.where(Bm, B, 0.0)

            # numerator = sum(zA * zB) over symbols
            num = A0.T @ B0  # (N, M)

            # valid_n = count(valid) over symbols
            valid_n = (Am.astype(np.int32).T) @ (Bm.astype(np.int32))  # (N, M)

            # 相关：cov(zA,zB)= sum(zA*zB)/(n-1) （因为zscore用ddof=1）
            denom = valid_n - 1
            ok = denom > 0

            corr = np.full((N, M), np.nan, dtype=np.float64)
            corr[ok] = num[ok] / denom[ok]

            if date_equal_weight:
                # 每个date等权平均（与你原来的 over('date').mean() 更接近）
                sum_corr[ok] += corr[ok]
                cnt[ok] += 1
            else:
                # 按有效样本数加权：权重=denom（或valid_n都行）
                w = denom.astype(np.float64)
                sum_corr[ok] += corr[ok] * w[ok]
                cnt[ok] += w[ok].astype(np.int32)

        out = sum_corr / np.where(cnt == 0, np.nan, cnt)

        # 输出：行=pool因子，列=new因子
        out_df = pl.DataFrame(out, schema=new_fids).with_columns(
            pl.Series("pool_fid", pool_only)
        ).select(["pool_fid", *new_fids])

        return out_df
    
    def evaluate(self, alphas: Set[str] | None = None, horizon: int = 5, rebalance_period: int = 7, rebalance_delay: int = 1, long_pct: float = 0.8, on: Literal['train', 'val', 'test'] = 'train') -> pl.DataFrame:
        if alphas is None:
            alphas = self.alpha_pool
        evaluated = set(self._alpha_records.filter((pl.col('horizon') == horizon) & (pl.col('on') == on))['fid'].to_list())
        pending = alphas - evaluated
        if len(pending) == 0:
            return self._alpha_records.filter((pl.col('horizon') == horizon) & (pl.col('on') == on) & pl.col('fid').is_in(alphas))
        # normalized_alpha = self.alpha().select(['date', 'symbol', *pending]).join(
        normalized_alpha = self.alpha().lazy().select(['date', 'symbol', *pending]).join(
            # self.dataset.with_columns(
            self.dataset.lazy().with_columns(
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
        # self._alpha_records = pl.concat([self._alpha_records, records], how='vertical')\
        #     .unique(subset=['fid', 'horizon', 'on'], keep='last')
        self._alpha_records = pl.concat([self._alpha_records.lazy(), records], how='vertical')\
            .unique(subset=['fid', 'horizon', 'on'], keep='last').collect()
        if self.alpha_records_cache is not None:
            self.alpha_records_cache.parent.mkdir(parents=True, exist_ok=True)
            self._alpha_records.write_parquet(self.alpha_records_cache)
        return self._alpha_records.filter(pl.col('fid').is_in(alphas) & (pl.col('horizon') == horizon) & (pl.col('on') == on))

    def _write_alpha_cache(self):
        self.add([self._all_alphas.get(aid, aid) for aid in self.alpha_pool])
        self.alpha()
        if self.raw_values_cache is not None:
            self.raw_values_cache.parent.mkdir(parents=True, exist_ok=True)
            self._raw_alpha.select('date', 'symbol', *self.alpha_pool).write_parquet(self.raw_values_cache)
        if self.norm_alpha_cache is not None:
            self.norm_alpha_cache.parent.mkdir(parents=True, exist_ok=True)
            self._norm_alpha.select('date', 'symbol', *self.alpha_pool).write_parquet(self.norm_alpha_cache)

    def alpha(self, max_parallel: int | None = None) -> pl.DataFrame:
        raw_alpha = self.raw_alpha(max_parallel=max_parallel)
        all_cols = set(raw_alpha.columns) - {'date', 'symbol'}
        calc_cols = all_cols - set(self._norm_alpha.columns)
        if len(calc_cols) == 0: return self._norm_alpha
        res = self.normalize_alpha(raw_alpha.lazy(), cols=list(calc_cols))
        self._norm_alpha = pl.concat([self._norm_alpha.lazy(), res.select(list(calc_cols))], how='horizontal', parallel=True).collect()
        return self._norm_alpha
    
    def raw_alpha(self, max_parallel: int | None = None) -> pl.DataFrame:
        if max_parallel is None: max_parallel = len(self._lazy_res_cols)
        if len(self._lazy_res_cols) == 0: return self._raw_alpha
        for i in tqdm.trange(0, len(self._lazy_res_cols), max_parallel):
            self._raw_alpha = pl.concat([self._raw_alpha.lazy(), *self._lazy_res_cols[i:i+max_parallel]], how='horizontal', parallel=True).collect()
        self._lazy_res_cols = []
        self._lazy_res_cols_fids = set()
        return self._raw_alpha

    def add(self, expressions: Iterable[str] | str) -> List[str]:
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
            self._all_alphas[aid] = ast_to_expression(ast)
            if aid in self._raw_alpha.columns or aid in self._lazy_res_cols_fids: continue
            col = self.ast_to_col(lazy_df, aid, ast)
            self._lazy_res_cols.append(col)
            self._lazy_res_cols_fids.add(aid)
        if self.alphas_cache is not None:
            with open(self.alphas_cache, 'w') as f:
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
            rebalance_delay: int = 1,
            on: Literal['train', 'val', 'test', 'all'] = 'all',
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
            ).filter(self.__timerange_expr__(on=on)).with_columns(
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
            df_baseline = self.dataset.lazy().filter(self.__timerange_expr__(on=on)).group_by('date').agg(
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
        return df.with_columns(compiled_expr.cast(REAL).alias(alias)).select(alias)
