from __future__ import annotations

import re
from typing import Iterable, Tuple
from .AST import Field, Const, Call, Node


class ParserError(Exception): 
    def __init__(self, message: str, source: str | None = None, pos: int | None = None):
        super().__init__(message)
        self.message = message
        self.source = source
        self.pos = pos
    
    def pretty(self, context_lines: int = 0) -> str:
        """
        输出类似编译器的报错：
        <line>| <code>
              ^--- message
        """

        if self.source is None or self.pos is None:
            return self.message

        s = self.source
        pos = max(0, min(self.pos, len(s)))

        # 找到 pos 所在行
        line_start = s.rfind("\n", 0, pos) + 1
        line_end = s.find("\n", pos)
        if line_end == -1:
            line_end = len(s)

        line = s[line_start:line_end]
        col = pos - line_start  # 0-based column

        # 行号（1-based）
        line_no = s.count("\n", 0, line_start) + 1

        prefix = f"{line_no} | "
        caret_line = " " * len(prefix) + " " * col + "^"

        return f"{prefix}{line}\n{caret_line}  {self.message}"

_token_re = re.compile(
    r"""\s*(
        \d+(?:\.\d*)? |            # Number
        [A-Za-z_]\w*  |            # Identifier
        \"[^\"]*\"    |            # String literal
        true | false  |            # Boolean literals
        >= | <= | == | != | > | < |# Comparison operators
        \*\*          |            # power
        [+\-*/(),]                 # Operators and parentheses
    )""",
    re.VERBOSE
)

_unexpected_char_re = re.compile(r"\s*.")

def tokenize(expression: str) -> Iterable[Tuple[str, str]]:
    pos = 0
    while pos < len(expression):
        match = _token_re.match(expression, pos)
        if not match:
            match = _unexpected_char_re.match(expression, pos)
            pos = match.end() - 1
            raise ParserError(f"Unexpected character at position {pos}: '{expression[pos]}'", expression, pos)
        tok = match.group(1)
        pos = match.end()
        if re.fullmatch(r"\d+(?:\.\d*)?", tok):
            yield ("NUMBER", tok)
        elif re.fullmatch(r'"[^"]*"', tok):
            yield ("STRING", tok[1:-1])
        elif tok in {"true", "false"}:
            yield ("BOOLEAN", tok)
        elif re.fullmatch(r"[A-Za-z_]\w*", tok):
            yield ("IDENTIFIER", tok)
        elif tok == "(":
            yield ("LPAREN", tok)
        elif tok == ")":
            yield ("RPAREN", tok)
        elif tok == ",":
            yield ("COMMA", tok)
        elif tok in {"+", "-", "*", "/", "**", ">", "<", ">=", "<=", "==", "!="}:
            yield ("OPERATOR", tok)
        else:
            raise ParserError(f"Unknown token: {tok}", expression, pos)
    yield ("EOF", "")

BP = {
    "+": (10, 11),
    "-": (10, 11),
    "*": (20, 21),
    "/": (20, 21),
    "**": (30, 29),
    ">":  (5, 6),
    "<":  (5, 6),
    ">=": (5, 6),
    "<=": (5, 6),
    "==": (5, 6),
    "!=": (5, 6),
}

OP_TO_FUNC = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "**": "pow",
    ">":  "gt",
    "<":  "lt",
    ">=": "ge",
    "<=": "le",
    "==": "eq",
    "!=": "ne",
}

class Parser:
    def __init__(self, expression: str) -> None:
        self.tokens = list(tokenize(expression))
        self.pos = 0
    
    def peek(self) -> Tuple[str, str]:
        return self.tokens[self.pos]
    
    def kind(self) -> str:
        return self.tokens[self.pos][0]

    def val(self) -> str:
        return self.tokens[self.pos][1]

    def eat(self, kind: str, val: str | None = None) -> str:
        k, v = self.peek()
        if k != kind:
            raise ParserError(f"Expected token {kind}, got {(k, v)}")
        if val is not None and v != val:
            raise ParserError(f"Expected token value {val}, got {v}")
        self.pos += 1
        return v
    
    def parse(self) -> Node:
        node = self.expr(0)
        self.eat("EOF")
        return node
    
    def expr(self, min_bp: int) -> Node:
        lhs = self.nud()

        while True:
            if self.kind() != "OPERATOR":
                break

            op = self.val()
            if op not in BP:
                break

            lbp, rbp = BP[op]
            if lbp < min_bp:
                break

            self.eat("OPERATOR", op)
            rhs = self.expr(rbp)
            lhs = Call(OP_TO_FUNC[op], (lhs, rhs))
        
        return lhs
        
    def nud(self) -> Node:
        k, v = self.peek()
        if k == "NUMBER":
            self.eat("NUMBER")
            return Const(float(v))
        
        elif k == "STRING":
            self.eat("STRING")
            return Const(v)
        
        elif k == "BOOLEAN":
            self.eat("BOOLEAN")
            return Const(v == "true")
        
        elif k == "OPERATOR" and v == "-":
            self.eat("OPERATOR", "-")
            right = self.expr(25)
            return Call("neg", (right,))
        
        elif k == "LPAREN":
            self.eat("LPAREN")
            node = self.expr(0)
            self.eat("RPAREN")
            return node
        
        elif k == "IDENTIFIER":
            self.eat("IDENTIFIER")
            if self.kind() == "LPAREN":
                self.eat("LPAREN")
                args = []
                if self.kind() != "RPAREN":
                    while True:
                        args.append(self.expr(0))
                        if self.kind() == "COMMA":
                            self.eat("COMMA")
                        else:
                            break
                self.eat("RPAREN")
                return Call(v, tuple(args))
            else:
                return Field(v)
        else:
            raise ParserError(f"Unexpected token in nud: {(k, v)}")

if __name__ == "__main__":
    expr = "(rank(Ts_ArgMax(SignedPower(where(returns < 0, stddev(returns, 20), close), 2.), 5)) - 0.5)"
    # expr = "-5 ** 2"
    parser = Parser(expr)
    ast = parser.parse()
    print(ast)