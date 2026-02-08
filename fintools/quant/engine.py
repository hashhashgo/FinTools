import polars as pl
from .parser import Parser
from .validate import normalize, validate, ast_to_hash
from .compiler import compile_expr
from .utils import make_dataset

class QuantEngine:
    def __init__(self):
        self.expr_cache = {}

    def init(self):
        self.dataset = make_dataset()

    def compile(self, expr: str) -> dict:
        ast = Parser(expression=expr).parse()
        ast = normalize(ast)
        validate(ast)
        hashed = ast_to_hash(ast)
        compiled = compile_expr(ast, memo=self.expr_cache)
        return {
            "ast": ast,
            "aid": hashed,
            "compiled": compiled,
            "expr": expr
        }