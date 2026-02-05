import polars as pl
from .parser import Parser
from .validate import normalize, validate, ast_to_hash

def compile_expression(expr: str) -> dict:
    ast = Parser(expression=expr).parse()
    ast = normalize(ast)
    validate(ast)
    aid = ast_to_hash(ast)
    return {"ast": ast, "aid": aid}