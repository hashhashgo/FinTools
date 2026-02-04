from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
from .AST import Node, Const, Call

class ValidationError(Exception): pass

@dataclass
class OpSpec:
    name: str
    min_arity: int
    max_arity: int
    field_type: Tuple[type, ...]
    normalize: Callable[[Tuple[Node, ...]], Tuple[Node, ...]]

OPS: Dict[str, OpSpec] = {}

def wrap_op(min_arity: int, max_arity: int, field_type: Tuple[type, ...] = ()):
    if len(field_type) < max_arity:
        field_type = field_type + (object,) * (max_arity - len(field_type))
    def decorator(norm_func: Callable[[Tuple[Node, ...]], Tuple[Node, ...]]):
        assert norm_func.__name__.startswith("_norm_"), "Normalization function name must start with '_norm_'"
        name = norm_func.__name__[6:]  # Remove '_norm_' prefix
        OPS[name] = OpSpec(name=name, min_arity=min_arity, max_arity=max_arity, field_type=field_type, normalize=norm_func)
        return norm_func
    return decorator

################ Arithmetic Operators ################
@wrap_op(1, 1)
def _norm_abs(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_neg(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_add(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_sub(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_mul(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_div(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_pow(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_inverse(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_log(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 100000)
def _norm_max(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 100000)
def _norm_min(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_sign(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_signed_pow(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_sqrt(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

################ Logical Operators ################
@wrap_op(2, 2)
def _norm_and(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_or(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_not(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(3, 3)
def _norm_where(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_gt(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_lt(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_ge(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_le(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_eq(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2)
def _norm_ne(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_is_nan(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

################ Time Series Operators ################
@wrap_op(1, 1)
def _norm_days_from_last_change(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(3, 3, field_type=(object, int, int))
def _norm_kth_element(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_last_diff_value(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_min(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_max(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_argmax(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_argmin(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_av_diff(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_backfill(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(3, 3, field_type=(object, object, int))
def _norm_ts_corr(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_count_nans(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(3, 3, field_type=(object, object, int))
def _norm_ts_cov(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 3, field_type=(object, int, bool))
def _norm_ts_decay_linear(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args if len(args) == 3 else args + (Const(False),)

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_delay(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_delta(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_mean(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_product(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 3, field_type=(object, int, str))
def _norm_ts_quantile(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args if len(args) == 3 else args + (Const("gaussian"),)

@wrap_op(2, 3, field_type=(object, int, float))
def _norm_ts_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args if len(args) == 3 else args + (Const(0.),)

@wrap_op(3, 5, field_type=(object, object, int, int, int))
def _norm_ts_regression(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 4:
        args += (Const(0),)
    if len(args) < 5:
        args += (Const(0),)
    return args

@wrap_op(2, 3, field_type=(object, int, float))
def _norm_ts_scale(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args if len(args) == 3 else args + (Const(0.),)

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_stddev(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 1)
def _norm_ts_step_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_sum(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, int))
def _norm_ts_zscore(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

################ Cross-sectional Operators ################
@wrap_op(1, 3, field_type=(object, bool, float))
def _norm_normalize(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 2:
        args += (Const(False),)
    if len(args) < 3:
        args += (Const(0.),)
    return args

@wrap_op(1, 3, field_type=(object, str, float))
def _norm_quantile(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 2:
        args += (Const("gaussian"),)
    if len(args) < 3:
        args += (Const(1.0),)
    return args

@wrap_op(1, 2, field_type=(object, int))
def _norm_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 2:
        args += (Const(2),)
    return args

@wrap_op(1, 4, field_type=(object, int, int, int))
def _norm_scale(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 2:
        args += (Const(1),)
    if len(args) < 3:
        args += (Const(1),)
    if len(args) < 4:
        args += (Const(1),)
    return args

@wrap_op(1, 2, field_type=(object, int))
def _norm_winsorize(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 2:
        args += (Const(4),)
    return args

@wrap_op(1, 1)
def _norm_zscore(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(1, 2, field_type=(object, float))
def _norm_clip_pct(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 2:
        args += (Const(0.01),)
    return args

################ Group Operators ################
@wrap_op(3, 4, field_type=(object, str, int, float))
def _norm_group_backfill(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    if len(args) < 4:
        args += (Const(4.0),)
    return args

@wrap_op(3, 3, field_type=(object, float, str))
def _norm_group_mean(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, str))
def _norm_group_neutralize(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, str))
def _norm_group_rank(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, str))
def _norm_group_scale(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args

@wrap_op(2, 2, field_type=(object, str))
def _norm_group_zscore(args: Tuple[Node, ...]) -> Tuple[Node, ...]:
    return args


FIELDS = {'open', 'high', 'low', 'close', 'volume', 'amount', 'returns', 'vwap', 'cap', 'industry'}