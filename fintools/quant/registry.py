from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, cast, get_type_hints, get_args, get_origin, Annotated
import inspect
from functools import wraps
import logging
logger = logging.getLogger(__name__)

import polars as pl
from .AST import Node, Const, Call

class ValidationError(Exception): pass

class LLMHidden: pass

@dataclass
class OpSpec:
    name: str
    min_arity: int
    max_arity: int
    field_type: Tuple[type, ...]
    normalize: Callable[[Tuple[Node, ...]], Tuple[Node, ...]]
    func: Callable[..., pl.Expr]
    hint_func_doc: str
    hint_func_sig: str
    hint_params: List[str]

OPS: Dict[str, OpSpec] = {}

def quant_func(func: Callable[..., pl.Expr]) -> Callable[..., pl.Expr]:
    sig = inspect.signature(func)
    name = func.__name__.strip('_')
    params = sig.parameters
    min_arity = 0
    max_arity = 0
    type_hints = {k: v if not v is pl.Expr else object for k, v in get_type_hints(func).items()}
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
                if field_type[i] != object and type(args_list[i]) != field_type[i]:
                    logger.warning(f"Argument {i + 1} of function '{name}' expected type {field_type[i].__name__}, got {type(args_list[i]).__name__}")
        return tuple(args_list)

    # for LLM
    type_hints_extra = get_type_hints(func, include_extras=True)
    param_list = []
    hint_params = []
    for p in params.values():
        if p.name in type_hints_extra and get_origin(type_hints_extra[p.name]) is Annotated:
            annotated_args = get_args(type_hints_extra[p.name])
            dtype = annotated_args[0] if get_origin(annotated_args[0]) is not pl.Expr else object
            if LLMHidden in annotated_args[1:]: continue
            param_list.append((p.name, dtype, p.default if p.default is not p.empty else None))
            if len(annotated_args) >= 2:
                hint_params.append(f"{p.name}: {dtype} ({annotated_args[1:]})")
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
        func=func,
        hint_func_doc='\n'.join([l.strip() for l in (func.__doc__ or "").splitlines() if l.strip()]),
        hint_func_sig=func_sig,
        hint_params=hint_params
    )

    return func

def quant_ts_func(func: Callable[..., pl.Expr]) -> Callable[..., pl.Expr]:
    @wraps(quant_func(func))
    def wrapper(*args) -> pl.Expr:
        result = func(*args)
        return result.over(pl.col("symbol"))
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
def _abs(x: pl.Expr) -> pl.Expr:
    """
    Absolute value of x
    """
    return x.abs()

@quant_func
def _neg(x: pl.Expr) -> pl.Expr:
    """
    Equivalent of unary minus operator -x
    """
    return x.neg()

@quant_func
def _add(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of binary addition operator x + y
    """
    return x + y

@quant_func
def _sub(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of binary subtraction operator x - y
    """
    return x - y

@quant_func
def _mul(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of binary multiplication operator x * y
    """
    return x * y

@quant_func
def _div(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of binary division operator x / y
    """
    return x / y

@quant_func
def _pow(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of binary exponentiation operator x ** y
    """
    return x ** y

@quant_func
def _log(x: pl.Expr) -> pl.Expr:
    """
    natural logarithm
    """
    return x.log()

@quant_func
def _max(*args: pl.Expr) -> pl.Expr:
    """
    Maximum value among the arguments
    """
    return pl.max_horizontal(*args)

@quant_func
def _min(*args: pl.Expr) -> pl.Expr:
    """
    Minimum value among the arguments
    """
    return pl.min_horizontal(*args)

@quant_func
def _sign(x: pl.Expr) -> pl.Expr:
    """
    Sign of x
    * -1 if x < 0.
    *  1 if x > 0.
    *  x otherwise (typically 0, but could be NaN if the input is).
    """
    return x.sign()

@quant_func
def _signed_pow(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Signed power function
    Computes x raised to the power of y, preserving the sign of x.
    """
    return pl.when(x >= 0).then(x ** y).otherwise(-( (-x) ** y))

@quant_func
def _sqrt(x: pl.Expr) -> pl.Expr:
    """
    Square root of x
    """
    return x.sqrt()

# ################ Logical Operators ################
@quant_func
def _and(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Logical AND operation between x and y
    """
    return x.cast(pl.Boolean) & y.cast(pl.Boolean)

@quant_func
def _or(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Logical OR operation between x and y
    """
    return x.cast(pl.Boolean) | y.cast(pl.Boolean)

@quant_func
def _not(x: pl.Expr) -> pl.Expr:
    """
    Logical NOT operation on x
    """
    return ~x.cast(pl.Boolean)

@quant_func
def _where(cond: pl.Expr, x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Conditional selection
    Returns x where cond is true, and y otherwise.
    """
    return pl.when(cond.cast(pl.Boolean)).then(x).otherwise(y)

@quant_func
def _gt(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of greater-than operator x > y
    """
    return x > y

@quant_func
def _lt(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of less-than operator x < y
    """
    return x < y

@quant_func
def _ge(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of greater-than-or-equal-to operator x >= y
    """
    return x >= y

@quant_func
def _le(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of less-than-or-equal-to operator x <= y
    """
    return x <= y

@quant_func
def _eq(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of equality operator x == y
    """
    return x == y

@quant_func
def _ne(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """
    Equivalent of not-equal-to operator x != y
    """
    return x != y

@quant_func
def _is_nan(x: pl.Expr) -> pl.Expr:
    """
    Check if x is NaN
    """
    return x.is_nan()

# ################ Time Series Operators ################
# @quant_ts_func
# def _days_from_last_change(x: pl.Expr) -> pl.Expr:
#     """
#     Number of days since the last change in the value of x
#     """
#     seg = (x != x.shift(-1)).fill_null(True).cum_sum()
#     pos = x.cum_count().over(seg)
#     return pos.shift(1)

@quant_ts_func
def _kth_element(x: pl.Expr, d: Annotated[int, "lookback"], k: Annotated[int, "k"] = 1) -> pl.Expr:
    """
    Returns K-th valid value of input by looking through lookback days.
    This operator can be used to backfill missing data if k=1.
    """
    if k > 1:
        return x.reverse().drop_nulls().get(k - 1).rolling(index_column="date", period=f"{d}d")
    else:
        return x.fill_null(strategy="forward", limit=d - 1)

# @quant_ts_func
# def _last_diff_value(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
#     """
#     Returns last x value not equal to current x value from last d days
#     **This operator is slow**
#     """
#     idx = pl.col("date").cum_count().reverse()
#     idx_diff = pl.min_horizontal(pl.when(x.shift(i) != x).then(idx + i).otherwise(None) for i in range(1, d))
#     v_diff = pl.min_horizontal(pl.when(idx.shift(i) == idx_diff).then(x.shift(i)).otherwise(None) for i in range(1, d))
#     return v_diff

@quant_ts_func
def _ts_min(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Minimum value of x over the past d days
    """
    return x.rolling_min(window_size=d, min_samples=1)

@quant_ts_func
def _ts_max(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Maximum value of x over the past d days
    """
    return x.rolling_max(window_size=d, min_samples=1)

@quant_ts_func
def _ts_argmax(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Returns the relative index of the max value in the time series for the past d days.
    If the current day has the max value for the past d days, it returns 0.
    """
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
    return x - x.rolling_mean(window_size=d, min_samples=1)

@quant_ts_func
def _ts_backfill(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Backfill missing values in x by looking back through the past d days.
    """
    return x.fill_null(strategy="forward", limit=d - 1)

@quant_ts_func
def _ts_corr(x: pl.Expr, y: pl.Expr, d: Annotated[int, "lookback"], ddof: Annotated[int, "delta degrees of freedom", LLMHidden] = 1) -> pl.Expr:
    """
    Pearson correlation coefficient between x and y over the past d days
    """
    x = pl.when(x.is_nan()).then(None).otherwise(x)
    y = pl.when(y.is_nan()).then(None).otherwise(y)

    ex  = x.rolling_mean(window_size=d, min_samples=1)
    ey  = y.rolling_mean(window_size=d, min_samples=1)
    exy = (x * y).rolling_mean(window_size=d, min_samples=1)
    n = (x.is_not_null() & y.is_not_null()).cast(pl.Int32).rolling_sum(window_size=d, min_samples=1)

    cov = (exy - ex * ey) * (n / (n - pl.lit(ddof)))
    
    ex2 = (x * x).rolling_mean(window_size=d, min_samples=1)
    ey2 = (y * y).rolling_mean(window_size=d, min_samples=1)

    varx = (ex2 - ex * ex) * (n / (n - pl.lit(ddof)))
    vary = (ey2 - ey * ey) * (n / (n - pl.lit(ddof)))

    return cov / (varx.sqrt() * vary.sqrt())

@quant_ts_func
def _ts_count_nans(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Count of NaN values in x over the past d days
    """
    return x.is_nan().cast(pl.Int64).rolling_sum(window_size=d, min_samples=1)

@quant_ts_func
def _ts_cov(x: pl.Expr, y: pl.Expr, d: Annotated[int, "lookback"], ddof: Annotated[int, "delta degrees of freedom", LLMHidden] = 1) -> pl.Expr:
    """
    Covariance between x and y over the past d days
    """
    ex = x.rolling_mean(window_size=d, min_samples=1)
    ey = y.rolling_mean(window_size=d, min_samples=1)
    exy = (x * y).rolling_mean(window_size=d, min_samples=1)
    n = (x.is_not_null() & y.is_not_null()).cast(pl.Int32).rolling_sum(window_size=d, min_samples=1)
    return (exy - ex * ey) * n / (n - ddof)

@quant_ts_func
def _ts_decay_linear(x: pl.Expr, d: Annotated[int, "lookback"], dense: Annotated[bool, "dense=false means operator works in sparse mode and we treat NaN as 0. In dense mode we ignore NaN."] = False) -> pl.Expr:
    """
    Returns the linear decay on x for the past d days. 
    """
    assert d > 0, "Lookback period d must be positive."
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
        return cast(pl.Expr, sum(x.shift(i).fill_null(0) * (d - i) for i in range(d)) / (d * (d + 1) / 2))

@quant_ts_func
def _ts_delay(x: pl.Expr, d: Annotated[int, "lookback"]) -> pl.Expr:
    """
    Delay the time series x by d days
    """
    return x.shift(d)

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_delta(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args


# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_mean(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_product(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 3, field_type=(object, int, str))
# def _norm_ts_quantile(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args if len(args) == 3 else args + (Const("gaussian"),)

# @wrap_op(2, 3, field_type=(object, int, float))
# def _norm_ts_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args if len(args) == 3 else args + (Const(0.),)

# @wrap_op(3, 5, field_type=(object, object, int, int, int))
# def _norm_ts_regression(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 4:
#         args += (Const(0),)
#     if len(args) < 5:
#         args += (Const(0),)
#     return args

# @wrap_op(2, 3, field_type=(object, int, float))
# def _norm_ts_scale(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args if len(args) == 3 else args + (Const(0.),)

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_stddev(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(1, 1)
# def _norm_ts_step_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_sum(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_zscore(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# ################ Cross-sectional Operators ################
# @wrap_op(1, 3, field_type=(object, bool, float))
# def _norm_normalize(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 2:
#         args += (Const(False),)
#     if len(args) < 3:
#         args += (Const(0.),)
#     return args

# @wrap_op(1, 3, field_type=(object, str, float))
# def _norm_quantile(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 2:
#         args += (Const("gaussian"),)
#     if len(args) < 3:
#         args += (Const(1.0),)
#     return args

# @wrap_op(1, 2, field_type=(object, int))
# def _norm_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 2:
#         args += (Const(2),)
#     return args

# @wrap_op(1, 4, field_type=(object, int, int, int))
# def _norm_scale(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 2:
#         args += (Const(1),)
#     if len(args) < 3:
#         args += (Const(1),)
#     if len(args) < 4:
#         args += (Const(1),)
#     return args

# @wrap_op(1, 2, field_type=(object, int))
# def _norm_winsorize(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 2:
#         args += (Const(4),)
#     return args

# @wrap_op(1, 1)
# def _norm_zscore(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(1, 2, field_type=(object, float))
# def _norm_clip_pct(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 2:
#         args += (Const(0.01),)
#     return args

# ################ Group Operators ################
# @wrap_op(3, 4, field_type=(object, str, int, float))
# def _norm_group_backfill(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     if len(args) < 4:
#         args += (Const(4.0),)
#     return args

# @wrap_op(3, 3, field_type=(object, float, str))
# def _norm_group_mean(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, str))
# def _norm_group_neutralize(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, str))
# def _norm_group_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, str))
# def _norm_group_scale(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, str))
# def _norm_group_zscore(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args


FIELDS = {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'returns', 'vwap', 'cap', 'industry'}