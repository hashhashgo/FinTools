from __future__ import annotations

from ast import expr
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, cast, get_type_hints, get_args, get_origin, Annotated, TypeAlias
from enum import Enum
import inspect
from functools import wraps
import logging
logger = logging.getLogger(__name__)

import polars as pl
from scipy import stats
from .AST import Node, Const, Call
from .config import STRING, INTEGER, REAL

class ValidationError(Exception): pass

DATE_COL = "date"
SYMBOL_COL = "symbol"

DATA_SCHEMA: pl.Schema = pl.Schema({
    'symbol': STRING,
    'date': pl.Date,
    'open': REAL,
    'high': REAL,
    'low': REAL,
    'close': REAL,
    'volume': REAL,
    'amount': REAL,
    'returns': REAL,
    'vwap': REAL,
    'cap': REAL,
    'industry': STRING
})
GROUP_FIELDS = {'industry'}

class LLMHidden: pass

@dataclass
class OpSpec:
    name: str
    min_arity: int
    max_arity: int
    field_type: Tuple[type, ...]
    normalize: Callable[[Tuple[Node, ...]], Tuple[Node, ...]]
    func: ScheduleFunc | None
    hint_func_doc: str
    hint_func_sig: str
    hint_params: List[str]

OPS: Dict[str, OpSpec] = {}

AnyNumerical: TypeAlias = pl.Expr | float | int | bool
AnyConstant: TypeAlias = float | int | bool | str
AnyInput: TypeAlias = pl.Expr | AnyConstant
VALIDATABLE_TYPES = {int, float, str, bool}
def register_func(func: Callable[..., pl.Expr]) -> Callable[..., pl.Expr]:
    sig = inspect.signature(func)
    name = func.__name__.strip('_')
    params = sig.parameters
    min_arity = 0
    max_arity = 0
    type_hints = {k: v if v in VALIDATABLE_TYPES else object for k, v in get_type_hints(func).items()}
    fileds = []
    for p in params.values():
        max_arity += 1
        if p.default is p.empty:
            min_arity += 1
        dtype = type_hints.get(p.name, object)
        if p.kind == p.VAR_POSITIONAL:
            max_arity = 100000  # Arbitrary large number for *args
            for _ in range(min_arity, max_arity):
                fileds.append(object)
        elif p.kind == p.VAR_KEYWORD:
            raise ValidationError(f"Function '{name}' cannot have **kwargs parameter")
        else:
            fileds.append(dtype)
    field_type = tuple(fileds)
    
    def norm_args(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
        args_list = list(args)
        for i, p in enumerate(params.values()):
            if i >= len(args_list):
                if p.default is not p.empty:
                    default_value = p.default
                    if isinstance(default_value, (int, float, bool, str)):
                        args_list.append(Const(default_value))
                    else:
                        raise ValidationError(f"Unsupported default value type for parameter '{p.name}' in function '{name}'")
                else:
                    raise ValidationError(f"Missing required argument '{p.name}' for function '{name}'")
            else:
                arg = args_list[i]
                if field_type[i] != object and (not isinstance(arg, Const) or type(arg.value) != field_type[i]):
                    if not isinstance(arg, Const):
                        raise ValidationError(f"Argument {i + 1} of function '{name}' expected a constant value")
                    logger.warning(f"Argument {i + 1} of function '{name}' expected type {field_type[i].__name__}, got {type(arg.value).__name__}")
        return tuple(args_list)

    # for LLM
    type_hints_extra = get_type_hints(func, include_extras=True)
    param_list = []
    hint_params = []
    for p in params.values():
        if p.name in type_hints_extra and get_origin(type_hints_extra[p.name]) is Annotated:
            annotated_args = get_args(type_hints_extra[p.name])
            dtype = annotated_args[0] if annotated_args[0] in VALIDATABLE_TYPES else object
            if LLMHidden in annotated_args[1:]: continue
            param_list.append((p.name, dtype, p.default if p.default is not p.empty else None))
            if len(annotated_args) >= 2:
                hint_params.append(f"{p.name}: " + ', '.join(str(arg) for arg in annotated_args[1:]))
                continue
        if p.name in type_hints and type_hints[p.name] is not object:
            param_list.append((p.name, type_hints[p.name], p.default if p.default is not p.empty else None))
            hint_params.append(f"{p.name}: {type_hints[p.name].__name__}")
            continue
        hint_params.append(f"{p.name}: expression or constant")
        param_list.append((p.name, object, p.default if p.default is not p.empty else None))
    func_sig = name + "(" + ", ".join([name + (f": {dtype.__name__}" if dtype is not object else "") + (f" = {default}" if default is not None else "") for name, dtype, default in param_list]) + ")"

    OPS[name] = OpSpec(
        name=name,
        min_arity=min_arity,
        max_arity=max_arity,
        field_type=field_type,
        normalize=norm_args,
        func=None,
        hint_func_doc='\n'.join([l.strip() for l in (func.__doc__ or "").splitlines() if l.strip()]),
        hint_func_sig=func_sig,
        hint_params=hint_params
    )

    return func

def func_doc(func_name: str) -> str:
    if func_name not in OPS:
        raise ValidationError(f"Unknown function '{func_name}'")
    ops = OPS[func_name]
    return f"""
{ops.hint_func_sig}
{ops.hint_func_doc}
Parameters:
""" + "\n".join([f"- {p}" for p in ops.hint_params])


NORMAL_FUNC = []
TS_FUNC = []
CS_FUNC = []
GROUP_FUNC = []

class GroupBy(Enum):
    OVERSYMBOL = SYMBOL_COL
    OVERDATE = DATE_COL
    NONE = 'none'

@dataclass(frozen = True)
class Schedule:
    expr: pl.Expr
    aid: str
    over: GroupBy

@dataclass(frozen=True)
class ScheduleColume:
    col: pl.Expr
    aid: str

ScheduleFunc: TypeAlias = Callable[[str, pl.LazyFrame, Dict[str, ScheduleColume], AnyConstant | Schedule], Tuple[pl.LazyFrame, Schedule]]

def make_schedule_func(func: Callable[..., pl.Expr], over: GroupBy, eager: bool) -> ScheduleFunc:
    @wraps(func)
    def wrapper(aid: str, df: pl.LazyFrame, memory: Dict[str, ScheduleColume], *args: Schedule | AnyConstant) -> Tuple[pl.LazyFrame, Schedule]:
        nonlocal over
        arg_expr: List[AnyInput] = []
        for arg in args:
            if isinstance(arg, Schedule):
                if arg.aid in memory:
                    expr = memory[arg.aid].col
                elif arg.over == GroupBy.NONE:
                    expr = arg.expr
                else:
                    if over == GroupBy.NONE or (over == arg.over and not eager): 
                        over = arg.over
                        expr = arg.expr
                    else:
                        df = df.with_columns(
                            arg.expr.over(arg.over.value).alias(arg.aid)
                        )
                        expr = pl.col(arg.aid)
                        memory[arg.aid] = ScheduleColume(
                            col = expr,
                            aid = arg.aid
                        )
            else:
                expr = arg
            arg_expr.append(expr)
        output = func(*arg_expr)
        if eager:
            if over != GroupBy.NONE:
                output = output.over(over.value)
                over = GroupBy.NONE
            df = df.with_columns(
                output.alias(aid)
            )
            output = pl.col(aid)
            memory[aid] = ScheduleColume(
                col = output,
                aid = aid
            )
        return df, Schedule(
            expr = output,
            aid = aid,
            over = over
        )
    
    return wrapper

def quant_func(func: Callable[..., pl.Expr]) -> ScheduleFunc:
    name = func.__name__.strip('_')
    NORMAL_FUNC.append(name)
    func = register_func(func)

    wrapper = make_schedule_func(func=func, over=GroupBy.NONE, eager=False)

    OPS[name].func = wrapper
    return wrapper

def quant_ts_func(func: Callable[..., pl.Expr]) -> ScheduleFunc:
    name = func.__name__.strip('_')
    TS_FUNC.append(name)
    func = register_func(func)

    wrapper = make_schedule_func(func=func, over=GroupBy.OVERSYMBOL, eager=False)

    OPS[name].func = wrapper
    return wrapper

def quant_cs_func(func: Callable[..., pl.Expr]) -> ScheduleFunc:
    name = func.__name__.strip('_')
    CS_FUNC.append(name)
    func = register_func(func)

    wrapper = make_schedule_func(func=func, over=GroupBy.OVERDATE, eager=False)

    OPS[name].func = wrapper
    return wrapper

def quant_eager_func(func: Callable[..., pl.Expr]) -> ScheduleFunc:
    name = func.__name__.strip('_')
    GROUP_FUNC.append(name)
    func = register_func(func)

    wrapper = make_schedule_func(func=func, over=GroupBy.NONE, eager=True)

    OPS[name].func = wrapper
    return wrapper

################ Arithmetic Operators ################
@quant_func
def _densify(x: pl.Expr) -> pl.Expr:
    """
    Converts a grouping field of many buckets into lesser number of only available buckets.
    e.g. [1, 2, 3, 99, 2, 1] -> [1, 2, 3, 4, 2, 1]
    """
    return x.rank(method="dense")

@quant_func
def _abs(x: AnyNumerical) -> pl.Expr:
    """
    Absolute value of x
    """
    if not isinstance(x, pl.Expr):
        return pl.lit(abs(x))
    else:
        return x.abs()

@quant_func
def _neg(x: AnyNumerical) -> pl.Expr:
    """
    Equivalent of unary minus operator -x
    """
    if not isinstance(x, pl.Expr):
        return pl.lit(-x)
    else:
        return x.cast(REAL).neg()

@quant_func
def _add(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of binary addition operator x + y
    """
    if not isinstance(x, pl.Expr) and not isinstance(y, pl.Expr):
        return pl.lit(x + y)
    elif not isinstance(x, pl.Expr):
        return pl.lit(x) + y
    elif not isinstance(y, pl.Expr):
        return x + pl.lit(y)
    else:
        return x + y

@quant_func
def _sub(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of binary subtraction operator x - y
    """
    if not isinstance(x, pl.Expr) and not isinstance(y, pl.Expr):
        return pl.lit(x - y)
    elif not isinstance(x, pl.Expr):
        return pl.lit(x) - y
    elif not isinstance(y, pl.Expr):
        return x - pl.lit(y)
    else:
        return x - y

@quant_func
def _mul(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of binary multiplication operator x * y
    """
    if not isinstance(x, pl.Expr) and not isinstance(y, pl.Expr):
        return pl.lit(x * y)
    elif not isinstance(x, pl.Expr):
        return pl.lit(x) * y
    elif not isinstance(y, pl.Expr):
        return x * pl.lit(y)
    else:
        return x * y

@quant_func
def _div(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of binary division operator x / y
    """
    if not isinstance(x, pl.Expr) and not isinstance(y, pl.Expr):
        return pl.lit(x / y)
    elif not isinstance(x, pl.Expr):
        return pl.lit(x) / y
    elif not isinstance(y, pl.Expr):
        return x / pl.lit(y)
    else:
        return x / y

@quant_func
def _pow(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of binary exponentiation operator x ** y
    """
    if not isinstance(x, pl.Expr) and not isinstance(y, pl.Expr):
        return pl.lit(x ** y)
    elif not isinstance(x, pl.Expr):
        return pl.lit(x) ** y
    elif not isinstance(y, pl.Expr):
        return x ** pl.lit(y)
    else:
        return x ** y

@quant_func
def _log(x: AnyNumerical) -> pl.Expr:
    """
    natural logarithm
    """
    if not isinstance(x, pl.Expr):
        return pl.lit(x).log()
    else:
        return x.log()

@quant_func
def _max(*args: AnyNumerical) -> pl.Expr:
    """
    Maximum value among the arguments
    """
    return pl.max_horizontal(*args)

@quant_func
def _min(*args: AnyNumerical) -> pl.Expr:
    """
    Minimum value among the arguments
    """
    return pl.min_horizontal(*args)

@quant_func
def _sign(x: AnyNumerical) -> pl.Expr:
    """
    Sign of x
    * -1 if x < 0.
    *  1 if x > 0.
    *  x otherwise (typically 0, but could be NaN if the input is).
    """
    if not isinstance(x, pl.Expr):
        return pl.lit(x).sign()
    else:
        return x.sign()

@quant_func
def _signed_pow(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Signed power function
    Computes x raised to the power of y, preserving the sign of x.
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return pl.when(x >= 0).then(x ** y).otherwise(-( (-x) ** y))

@quant_func
def _sqrt(x: AnyNumerical) -> pl.Expr:
    """
    Square root of x
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    return x.sqrt()

# ################ Logical Operators ################
@quant_func
def _and(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Logical AND operation between x and y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(bool(x))
    if not isinstance(y, pl.Expr):
        y = pl.lit(bool(y))
    return x.cast(pl.Boolean) & y.cast(pl.Boolean)

@quant_func
def _or(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Logical OR operation between x and y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(bool(x))
    if not isinstance(y, pl.Expr):
        y = pl.lit(bool(y))
    return x.cast(pl.Boolean) | y.cast(pl.Boolean)

@quant_func
def _not(x: AnyNumerical) -> pl.Expr:
    """
    Logical NOT operation on x
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(bool(x))
    return ~x.cast(pl.Boolean)

@quant_func
def _where(cond: AnyNumerical, x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Conditional selection
    Returns x where cond is true, and y otherwise.
    """
    if not isinstance(cond, pl.Expr):
        cond = pl.lit(bool(cond))
    return pl.when(cond.cast(pl.Boolean)).then(x).otherwise(y)

@quant_func
def _gt(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of greater-than operator x > y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return x > y

@quant_func
def _lt(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of less-than operator x < y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return x < y

@quant_func
def _ge(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of greater-than-or-equal-to operator x >= y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return x >= y

@quant_func
def _le(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of less-than-or-equal-to operator x <= y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return x <= y

@quant_func
def _eq(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of equality operator x == y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return x == y

@quant_func
def _ne(x: AnyNumerical, y: AnyNumerical) -> pl.Expr:
    """
    Equivalent of not-equal-to operator x != y
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return x != y

@quant_func
def _is_nan(x: AnyNumerical) -> pl.Expr:
    """
    Check if x is NaN
    """
    if not isinstance(x, pl.Expr):
        x = pl.lit(x)
    return x.is_nan()

# ################ Time Series Operators ################
@quant_eager_func
def _days_from_last_change(x: pl.Expr) -> pl.Expr:
    """
    Number of days since the last change in the value of x
    """
    seg = (x != x.shift(-1)).fill_null(True).cast(INTEGER).cum_sum()
    pos = x.cum_count().over([seg, pl.col(SYMBOL_COL)])
    return pos.shift(1)

@quant_ts_func
def _kth_element(x: pl.Expr, d: Annotated[int, "lookback"], k: Annotated[int, "k"] = 1) -> pl.Expr:
    """
    Returns K-th valid value of input by looking through lookback days.
    This operator can be used to backfill missing data if k=1.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    if k > 1:
        return x.reverse().drop_nulls().get(k - 1).rolling(index_column="date", period=f"{d}d")
    else:
        return x.fill_null(strategy="forward", limit=d - 1)

@quant_ts_func
def _last_diff_value(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Returns last x value not equal to current x value from last d days
    **This operator is slow**
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    idx = pl.col("date").cum_count().reverse()
    idx_diff = pl.min_horizontal(pl.when(x.shift(i) != x).then(idx + i).otherwise(None) for i in range(1, d))
    v_diff = pl.min_horizontal(pl.when(idx.shift(i) == idx_diff).then(x.shift(i)).otherwise(None) for i in range(1, d))
    return v_diff

@quant_ts_func
def _ts_min(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Minimum value of x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.rolling_min(window_size=d, min_samples=1)

@quant_ts_func
def _ts_max(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Maximum value of x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.rolling_max(window_size=d, min_samples=1)

@quant_ts_func
def _ts_argmax(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Returns the relative index of the max value in the time series for the past d days.
    If the current day has the max value for the past d days, it returns 0.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    idx = pl.col("date").cum_count().reverse()
    maxv = x.rolling_max(window_size=d, min_samples=1)
    last_max_idx = pl.min_horizontal(pl.when(x.shift(i) == maxv).then(idx + i).otherwise(None) for i in range(d))
    return (last_max_idx - idx)

@quant_ts_func
def _ts_argmin(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Returns the relative index of the min value in the time series for the past d days.
    If the current day has the min value for the past d days, it returns 0.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    idx = pl.col("date").cum_count().reverse()
    minv = x.rolling_min(window_size=d, min_samples=1)
    last_min_idx = pl.min_horizontal(pl.when(x.shift(i) == minv).then(idx + i).otherwise(None) for i in range(d))
    return (last_min_idx - idx)

@quant_ts_func
def _ts_av_diff(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Returns x - tsmean(x, d), but deals with NaNs carefully.
    That is NaNs are ignored during mean computation.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x - x.rolling_mean(window_size=d, min_samples=1)

@quant_ts_func
def _ts_backfill(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Backfill missing values in x by looking back through the past d days.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.fill_null(strategy="forward", limit=d - 1)

@quant_ts_func
def _ts_corr(x: pl.Expr, y: pl.Expr, d: Annotated[int, "lookback"], ddof: Annotated[int, "delta degrees of freedom", LLMHidden] = 1) -> pl.Expr:
    """
    Pearson correlation coefficient between x and y over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return pl.rolling_corr(x, y, window_size=d, min_samples=2, ddof=ddof)

@quant_ts_func
def _ts_count_nans(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Count of NaN values in x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.is_nan().cast(INTEGER).rolling_sum(window_size=d, min_samples=1)

@quant_ts_func
def _ts_cov(x: pl.Expr, y: pl.Expr, d: Annotated[int, "lookback"], ddof: Annotated[int, "delta degrees of freedom", LLMHidden] = 1) -> pl.Expr:
    """
    Covariance between x and y over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return pl.rolling_cov(x, y, window_size=d, min_samples=2, ddof=ddof)

@quant_ts_func
def _ts_decay_linear(x: pl.Expr, d: Annotated[int, "lookback"], dense: Annotated[bool, "dense=false means operator works in sparse mode and we treat NaN as 0. In dense mode we ignore NaN."] = False) -> pl.Expr:
    """
    Returns the linear decay on x for the past d days. 
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    x = x.cast(REAL)
    if dense:
        num = sum(
            pl.when(x.shift(i).is_not_null())
            .then(x.shift(i) * (d - i))
            .otherwise(0)
            for i in range(d)
        )
        
        den = sum(
            pl.when(x.shift(i).is_not_null())
            .then(d - i)
            .otherwise(0)
            for i in range(d)
        )
        
        return cast(pl.Expr, num / den)
    else:
        return cast(pl.Expr, sum(x.shift(i).cast(REAL).fill_null(0.) * (d - i) for i in range(d)) / (d * (d + 1) / 2))

@quant_ts_func
def _ts_delay(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Delay the time series x by d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.shift(d)

@quant_ts_func
def _ts_delta(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Difference between current x and x d days ago
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x - x.shift(d)

@quant_ts_func
def _ts_mean(x: pl.Expr, d: Annotated[int, 'lookback']) -> pl.Expr:
    """
    Mean of x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.rolling_mean(window_size=d, min_samples=1)

@quant_ts_func
def _ts_product(x: pl.Expr, d: Annotated[int, 'lookback']) -> pl.Expr:
    """
    Product of x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.log().rolling_sum(window_size=d, min_samples=1).exp()

@quant_ts_func
def _ts_quantile(x: pl.Expr, d: Annotated[int, 'lookback'], driver: Annotated[str, 'distribution: "gaussian" / "uniform" / "cauchy"'] = "gaussian") -> pl.Expr:
    """
    It calculates ts_rank and apply to its value an inverse cumulative density function from driver distribution.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    rr = (x.rolling_rank(window_size=d, method="min", min_samples=1) - 1) / (d - 1)
    rr = 1 / d + rr * (1 - 2 / d)
    if driver == 'gaussian':
        return x.map_batches(lambda s: pl.Series(stats.norm.ppf(s))).cast(REAL)
    elif driver == 'uniform':
        return rr - rr.rolling_mean(window_size=d, min_samples=1)
    elif driver == 'cauchy':
        return x.map_batches(lambda s: pl.Series(stats.cauchy.ppf(s))).cast(REAL)
    else:
        raise ValidationError(f"Unknown driver {driver} for ts_quantile")

@quant_ts_func
def _ts_rank(x: pl.Expr, d: Annotated[int, 'lookback'], constant: Annotated[float, 'constant'] = 0.0) -> pl.Expr:
    """
    Rank of x over the past d days, normalized to [0, 1], then return the rank of the current value + constant.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return (x.rolling_rank(window_size=d, method="min", min_samples=1) - 1) / (d - 1) + pl.lit(constant)

@quant_ts_func
def _ts_regression(y: pl.Expr, x: pl.Expr, d: Annotated[int, 'lookback'], lag: Annotated[int, 'y_i=\\beta x_{i-lag}+\\alpha'] = 0, rettype: Annotated[int, 'what to return'] = 0) -> pl.Expr:
    """
    Perform linear regression of y on x over the past d days with optional lag and return specified regression component.
    y_i = β * x_{i-lag} + α
    rettype options:
    - 0: residuals (y - y_)
    - 1: alpha (intercept)
    - 2: beta (slope)
    - 3: y-estimate (y_)
    - 4: SSE (Sum of Squared Errors)
    - 5: SST (Total Sum of Squares)
    - 6: R^2 (Coefficient of Determination)
    - 7: MSE (Mean Squared Error)
    - 8: Standard Error of β
    - 9: Standard Error of α
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    if lag < 0: raise ValidationError("lag must >= 0")
    lag = int(lag)
    rettype = int(rettype)

    x = x.shift(lag)
    mask = x.is_not_null() & y.is_not_null() & x.is_finite() & y.is_finite()
    x0  = pl.when(mask).then(x).otherwise(None)
    y0  = pl.when(mask).then(y).otherwise(None)
    xx0 = pl.when(mask).then(x * x).otherwise(None)
    yy0 = pl.when(mask).then(y * y).otherwise(None)
    xy0 = pl.when(mask).then(x * y).otherwise(None)
    n0  = pl.when(mask).then(pl.lit(1.0)).otherwise(None)

    Sx  = x0.rolling_sum(window_size=d, min_samples=2)
    Sy  = y0.rolling_sum(window_size=d, min_samples=2)
    Sxx = xx0.rolling_sum(window_size=d, min_samples=2)
    Syy = yy0.rolling_sum(window_size=d, min_samples=2)
    Sxy = xy0.rolling_sum(window_size=d, min_samples=2)
    n   = n0.rolling_sum(window_size=d, min_samples=2)

    num   = Sxy - (Sx * Sy) / n
    denom = Sxx - (Sx * Sx) / n
    sst   = Syy - (Sy * Sy) / n
    ssr   = (num * num) / denom
    sse = sst - ssr

    eps = 1e-12
    beta = num / denom

    mx = x0.rolling_mean(window_size=d, min_samples=2)
    my = y0.rolling_mean(window_size=d, min_samples=2)
    alpha = my - beta * mx

    y_ = x * beta + alpha

    mse = sse / (n - 2)

    if rettype == 0: # y - y_
        ret = y - y_
    elif rettype == 1: # alpha
        ret = alpha
    elif rettype == 2: # beta
        ret = beta
    elif rettype == 3: # y_
        ret = y_
    elif rettype == 4: # SSE
        ret = sst - ssr
    elif rettype == 5: # SST
        ret = sst
    elif rettype == 6: # R^2
        ret = 1 - sse/sst
    elif rettype == 7: # MSE
        ret = mse
    elif rettype == 8: # Standard Error of β
        ret = mse / denom
    elif rettype == 9: # Standard Error of α
        ret = (mse * (1/n + (Sx/n)**2 / denom)).sqrt()
    else:
        raise ValidationError(f"Error rettype: {rettype}")

    return pl.when((n >= 2) & denom.abs().gt(eps)).then(ret).otherwise(None)

@quant_ts_func
def _ts_scale(x: pl.Expr, d: Annotated[int, 'lookback'], constant: Annotated[float, 'constant'] = 0) -> pl.Expr:
    """
    Scale x over the past d days to the range [0, 1] + constant.
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    minx = x.rolling_max(window_size=d, min_samples=1)
    maxx = x.rolling_min(window_size=d, min_samples=1)
    return (x - minx) / (maxx - minx) + constant

@quant_ts_func
def _ts_stddev(x: pl.Expr, d: Annotated[int, 'lookback']) -> pl.Expr:
    """
    Standard deviation of x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.rolling_std(window_size=d, min_samples=1)

@quant_ts_func
def _ts_step() -> pl.Expr:
    """
    Returns days' counter. 0 for the most recent day, 1 for the previous day, and so on.
    """
    return pl.col('date').cum_count().reverse()

@quant_ts_func
def _ts_sum(x: pl.Expr, d: Annotated[int, 'lookback']) -> pl.Expr:
    """
    Sum of x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return x.rolling_sum(window_size=d, min_samples=1)

@quant_ts_func
def _ts_zscore(x: pl.Expr, d: Annotated[int, 'lookback']) -> pl.Expr:
    """
    Z-score normalization of x over the past d days
    """
    if d <= 0: raise ValidationError("lookback days must > 0")
    return (x - x.rolling_mean(window_size=d, min_samples=1)) / x.rolling_std(window_size=d, min_samples=1)

# ################ Cross-sectional Operators ################
@quant_cs_func
def _normalize(x: pl.Expr, useStd: Annotated[bool, 'divide standard deviation or not'] = False, limit: Annotated[float, 'result clip to [-limit, limit]'] = 0.0) -> pl.Expr:
    """
    Normalize x for each date by subtracting the mean and optionally dividing by the standard deviation.
    Optionally clip the result to the range [-limit, limit] if limit is non-zero.
    """
    ret = x - x.mean()
    if useStd:
        ret = ret / x.std(ddof=0)
    if abs(limit) > 1e-6:
        ret = ret.clip(-abs(limit), abs(limit))
    return ret

@quant_cs_func
def _quantile(x: pl.Expr, driver: Annotated[str, 'distribution: "gaussian" / "uniform" / "cauchy"'] = "gaussian", sigma: Annotated[float, 'scale on final value'] = 1.0) -> pl.Expr:
    """
    It calculates rank and apply to its value an inverse cumulative density function from driver distribution for each date.
    Finally scale the result by sigma.
    """
    rr = (x.rank(method="min") - 1) / (x.len() - 1)
    rr = 1 / x.len() + rr * (1 - 2 / x.len())
    if driver == 'gaussian':
        ret = x.map_batches(lambda s: pl.Series(stats.norm.ppf(s))).cast(REAL)
    elif driver == 'uniform':
        ret = rr - rr.mean()
    elif driver == 'cauchy':
        ret = x.map_batches(lambda s: pl.Series(stats.cauchy.ppf(s))).cast(REAL)
    else:
        raise ValidationError(f"Unknown driver {driver} for quantile")
    return ret * sigma

@quant_cs_func
def _rank(x: pl.Expr, rate: Annotated[int, '10**rate buckets'] = 2) -> pl.Expr:
    """
    Rank x for each date, normalized to [0, 1], then re-rank into 10**rate buckets and normalize again.
    For precise sort, use the rate as 0.
    """
    n = x.len()
    r1 = (x.rank(method='min') - 1) / (n - 1)
    rate = int(rate)
    if rate < 0: raise ValidationError("rate must >= 0 for rank")
    if rate == 0:
        return r1.cast(REAL)
    bucket_cnt = pl.lit(10 ** rate)
    bucket = (r1 * bucket_cnt).floor()
    r2 = (bucket.rank(method='min') - 1) / (n - 1)
    return r2.cast(REAL)

@quant_cs_func
def _scale(x: pl.Expr, scale: Annotated[float, 'scale for all'] = 1.0, longscale: Annotated[float, 'scale for long position'] = 1.0, shortscale: Annotated[float, 'scale for short position'] = 1.0) -> pl.Expr:
    """
    Scale x for each date such that the sum of absolute values equals scale.
    If scale is not 1.0, both longscale and shortscale are ignored.
    If scale is 1.0, longscale and shortscale are used to scale positive and negative values separately.
    """
    pos = pl.when(x > 0).then(x).otherwise(0.0)
    neg = pl.when(x < 0).then(x).otherwise(0.0)
    long_sum = pos.sum()
    short_sum = neg.abs().sum()
    if abs(scale - 1.0) > 1e-6:
        denom = x.abs().sum()
        return scale * x / denom
    
    return pl.when(x > 0).then(longscale * x / long_sum) \
             .when(x < 0).then(shortscale * x / short_sum) \
             .otherwise(0.0)

@quant_cs_func
def _winsorize(x: pl.Expr, std: Annotated[float, 'multiple of std'] = 4.0) -> pl.Expr:
    """
    Winsorize x for each date by clipping values to the range [mean - std * stddev, mean + std * stddev].
    """
    mu = x.mean()
    sigma = x.std(ddof=0)
    return x.clip(mu - std * sigma, mu + std * sigma)

@quant_cs_func
def _zscore(x: pl.Expr) -> pl.Expr:
    """
    Z-score normalization of x for each date
    """
    return (x - x.mean()) / x.std(ddof=0)

# ################ Group Operators ################
@quant_eager_func
def _group_mean(x: pl.Expr, group: Annotated[str, 'group by']) -> pl.Expr:
    """
    Mean of x within each group for each date
    """
    if group not in DATA_SCHEMA: raise ValidationError(f"Cannot group by {group}")
    return x.mean().over([DATE_COL, group])

@quant_eager_func
def _group_neutralize(x: pl.Expr, group: Annotated[str, 'group by']) -> pl.Expr:
    """
    Neutralize x within each group for each date by subtracting the group mean and dividing by the group standard deviation.
    """
    if group not in DATA_SCHEMA: raise ValidationError(f"Cannot group by {group}")
    ret = (x - x.mean()) / x.std(ddof=0)
    return ret.over([DATE_COL, group])

@quant_eager_func
def _group_rank(x: pl.Expr, group: Annotated[str, 'group by']) -> pl.Expr:
    """
    Rank of x within each group for each date, normalized to [0, 1]
    """
    if group not in DATA_SCHEMA: raise ValidationError(f"Cannot group by {group}")
    return ((x.rank(method='min') - 1) / (x.len() - 1)).over([DATE_COL, group])

@quant_eager_func
def _group_scale(x: pl.Expr, group: Annotated[str, 'group by']) -> pl.Expr:
    """
    Scale x within each group for each date to the range [0, 1]
    """
    if group not in DATA_SCHEMA: raise ValidationError(f"Cannot group by {group}")
    minx = x.min()
    maxx = x.max()
    return ((x - minx) / (maxx - minx)).over([DATE_COL, group])

@quant_eager_func
def _group_zscore(x: pl.Expr, group: Annotated[str, 'group by']) -> pl.Expr:
    """
    Z-score normalization of x within each group for each date
    """
    if group not in DATA_SCHEMA: raise ValidationError(f"Cannot group by {group}")
    return ((x - x.mean()) / x.std(ddof=0)).over([DATE_COL, group])
