from __future__ import annotations
import polars as pl
from typing import Dict, Optional, Tuple, List

from .AST import Node, Field, Const, Call
from .validate import ast_to_hash
from .registry import OPS, Schedule, AnyConstant, GroupBy, ScheduleColume, DATA_SCHEMA

class CompileError(Exception): pass


def compile_expr(df: pl.LazyFrame, node: Node, *, extra_columns: Optional[Dict[str, ScheduleColume]] = None) -> Tuple[pl.LazyFrame, Schedule | AnyConstant]:
    if extra_columns is None:
        extra_columns = {}
    
    key = ast_to_hash(node=node)
    if key in extra_columns:
        return df, Schedule(
            expr = extra_columns[key].col,
            aid = extra_columns[key].aid,
            over = GroupBy.NONE
        )
    
    if isinstance(node, Const):
        return df, node.value
    elif isinstance(node, Field):
        if node.name not in DATA_SCHEMA:
            raise CompileError(f"Unknown field: {node.name}")
        result = Schedule(
            expr = pl.col(node.name),
            aid = key,
            over = GroupBy.NONE
        )
    elif isinstance(node, Call):
        fn = node.fn
        args = node.args
        compiled_args: List[Schedule | AnyConstant] = []
        for arg in args:
            df, res = compile_expr(df, arg, extra_columns=extra_columns)
            compiled_args.append(res)

        if fn not in OPS:
            raise CompileError(f"Unknown function: {fn}")
        
        try:
            processor = OPS[fn].func
            assert processor is not None, f"OPS[{fn}].func must not be None. Please check registry.py."
            df, result = processor(key, df, extra_columns, *compiled_args)
        except Exception as e:
            raise CompileError(f"Error compiling function '{fn}': {e}")
    else:
        raise CompileError(f"Unknown node type: {type(node)}")
    
    return df, result