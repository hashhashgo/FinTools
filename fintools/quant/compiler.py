from __future__ import annotations
import polars as pl
from typing import Dict, Any, Optional

from .AST import Node, Field, Const, Call
from .validate import ast_to_hash
from .registry import OPS, FIELDS

class CompileError(Exception): pass


def compile_expr(node: Node, *, memo: Optional[Dict[str, Any]] = None) -> Any:
    if memo is None:
        memo = {}
    
    key = ast_to_hash(node=node)
    if key in memo:
        return memo[key]
    
    if isinstance(node, Const):
        result = node.value
    elif isinstance(node, Field):
        result = pl.col(node.name)
    elif isinstance(node, Call):
        fn = node.fn
        args = node.args
        compiled_args = [compile_expr(arg, memo=memo) for arg in args]

        if fn not in OPS:
            raise CompileError(f"Unknown function: {fn}")
        
        try:
            result = OPS[fn].func(*compiled_args)
        except Exception as e:
            raise CompileError(f"Error compiling function '{fn}': {e}")
    else:
        raise CompileError(f"Unknown node type: {type(node)}")
    
    memo[key] = result
    return result