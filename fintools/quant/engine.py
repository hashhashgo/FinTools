import polars as pl
from .parser import Parser
from .validate import normalize, validate, ast_to_hash
from .compiler import compile_expr

def compile_expression(expr: str) -> dict:
    ast = Parser(expression=expr).parse()
    ast = normalize(ast)
    validate(ast)
    hashed = ast_to_hash(ast)
    compiled = compile_expr(ast)
    return {
        "ast": ast,
        "aid": hashed,
        "compiled": compiled
    }