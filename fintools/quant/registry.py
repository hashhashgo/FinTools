from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, get_type_hints
import inspect
from functools import wraps
import logging
logger = logging.getLogger(__name__)

import polars as pl
from .AST import Node, Const, Call

class ValidationError(Exception): pass

@dataclass
class OpSpec:
    name: str
    min_arity: int
    max_arity: int
    field_type: Tuple[type, ...]
    normalize: Callable[[Tuple[Node, ...]], Tuple[Node, ...]]
    func: Callable[..., pl.Expr]

OPS: Dict[str, OpSpec] = {}

def quant_func(func: Callable[..., pl.Expr]) -> Callable[..., pl.Expr]:
    sig = inspect.signature(func)
    name = func.__name__.strip('_')
    params = sig.parameters
    min_arity = 0
    max_arity = 0
    type_hints = get_type_hints(func)
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

    OPS[name] = OpSpec(
        name=name,
        min_arity=min_arity,
        max_arity=max_arity,
        field_type=field_type,
        normalize=norm_args,
        func=func
    )

    return func

################ Arithmetic Operators ################
@quant_func
def _abs(x: pl.Expr) -> pl.Expr:
    return x.abs()

@quant_func
def _neg(x: pl.Expr) -> pl.Expr:
    return x.neg()

@quant_func
def _add(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x + y

@quant_func
def _sub(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x - y

@quant_func
def _mul(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x * y

@quant_func
def _div(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x / y

@quant_func
def _pow(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x ** y

@quant_func
def _inverse(x: pl.Expr) -> pl.Expr:
    return 1 / x

@quant_func
def _log(x: pl.Expr) -> pl.Expr:
    return x.log()

@quant_func
def _max(*args: pl.Expr) -> pl.Expr:
    return pl.max_horizontal(*args)

@quant_func
def _min(*args: pl.Expr) -> pl.Expr:
    return pl.min_horizontal(*args)

@quant_func
def _sign(x: pl.Expr) -> pl.Expr:
    return x.sign()

@quant_func
def _signed_pow(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return pl.when(x >= 0).then(x ** y).otherwise(-( (-x) ** y))

@quant_func
def _sqrt(x: pl.Expr) -> pl.Expr:
    return x.sqrt()

# ################ Logical Operators ################
@quant_func
def _and(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x.cast(pl.Boolean) & y.cast(pl.Boolean)

@quant_func
def _or(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x.cast(pl.Boolean) | y.cast(pl.Boolean)

@quant_func
def _not(x: pl.Expr) -> pl.Expr:
    return ~x.cast(pl.Boolean)

@quant_func
def _where(cond: pl.Expr, x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return pl.when(cond.cast(pl.Boolean)).then(x).otherwise(y)

@quant_func
def _gt(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x > y

@quant_func
def _lt(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x < y

@quant_func
def _ge(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x >= y

@quant_func
def _le(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return x <= y

# @wrap_op(2, 2)
# def _norm_eq(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2)
# def _norm_ne(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(1, 1)
# def _norm_is_nan(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# ################ Time Series Operators ################
# @wrap_op(1, 1)
# def _norm_days_from_last_change(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(3, 3, field_type=(object, int, int))
# def _norm_kth_element(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_last_diff_value(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_min(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_max(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_argmax(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_argmin(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_av_diff(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_backfill(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(3, 3, field_type=(object, object, int))
# def _norm_ts_corr(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_count_nans(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(3, 3, field_type=(object, object, int))
# def _norm_ts_cov(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

# @wrap_op(2, 3, field_type=(object, int, bool))
# def _norm_ts_decay_linear(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args if len(args) == 3 else args + (Const(False),)

# @wrap_op(2, 2, field_type=(object, int))
# def _norm_ts_delay(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
#     return args

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


FIELDS = {'open', 'high', 'low', 'close', 'volume', 'amount', 'returns', 'vwap', 'cap', 'industry'}