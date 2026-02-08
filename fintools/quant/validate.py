from __future__ import annotations

import json, hashlib
from .AST import Node, Field, Const, Call
from .registry import OPS, FIELDS, ValidationError

MAX_NODES = 200
MAX_DEPTH = 30
MAX_WINDOW = 2520

def normalize(node: Node) -> Node:
    if isinstance(node, (Field, Const)):
        return node
    assert isinstance(node, Call)

    if node.fn not in OPS:
        raise ValidationError(f"Unknown operator: {node.fn}")
    
    spec = OPS[node.fn]
    norm_args = tuple(normalize(arg) for arg in node.args)
    if not (spec.min_arity <= len(norm_args) <= spec.max_arity):
        raise ValidationError(f"Operator '{node.fn}' expects between {spec.min_arity} and {spec.max_arity} arguments, got {len(norm_args)}")
    
    canon_args = spec.normalize(norm_args)
    return Call(fn=node.fn, args=canon_args)

def _require_int_const(node: Node, what: str) -> int:
    if not isinstance(node, Const) or not isinstance(node.value, (float, int)):
        raise ValidationError(f"{what} must be a numeric constant")
    v = node.value
    if abs(v - round(v)) > 1e-9:
        raise ValidationError(f"{what} must be an integer constant")
    return int(v)

def validate(node: Node, depth = 0, allow_float_for_int=False) -> int:
    total_nodes = 1
    if depth > MAX_DEPTH:
        raise ValidationError(f"Expression is too deep: {depth} (max {MAX_DEPTH})")
    
    if isinstance(node, Call):
        for i, dtype in enumerate(OPS[node.fn].field_type):
            if i >= len(node.args):
                break
            if dtype == object:
                t = validate(node.args[i], depth + 1)
                total_nodes += t
                if total_nodes > MAX_NODES:
                    raise ValidationError(f"Expression has too many nodes: {total_nodes} (max {MAX_NODES})")
            elif dtype == int:
                if not allow_float_for_int: _require_int_const(node.args[i], f"{i}-th argument of '{node.fn}'")
                arg = node.args[i]
                if not isinstance(arg, Const) or not isinstance(arg.value, (float, int)):
                    raise ValidationError(f"{i}-th argument of '{node.fn}' must be a numeric constant")
            elif dtype == float:
                arg = node.args[i]
                if not isinstance(arg, Const) or not isinstance(arg.value, (float)):
                    raise ValidationError(f"{i}-th argument of '{node.fn}' must be a numeric constant")
            elif dtype == str:
                arg = node.args[i]
                if not isinstance(arg, Const) or not isinstance(arg.value, str):
                    raise ValidationError(f"{i}-th argument of '{node.fn}' must be a string constant")
            elif dtype == bool:
                arg = node.args[i]
                if not isinstance(arg, Const) or not isinstance(arg.value, bool):
                    raise ValidationError(f"{i}-th argument of '{node.fn}' must be a boolean constant")
            else:
                raise ValidationError(f"Unsupported field type in operator spec: {dtype}")
    elif isinstance(node, Field):
        if node.name not in FIELDS:
            raise ValidationError(f"Unknown field: {node.name}")
    elif isinstance(node, Const):
        pass
    else:
        raise ValidationError(f"Unknown node type: {type(node)}")
    
    return total_nodes
            
def ast_to_canonical_json(node: Node) -> str:
    def node_to_dict(n: Node) -> dict:
        if isinstance(n, Field):
            return {"type": "Field", "name": n.name}
        elif isinstance(n, Const):
            if isinstance(n.value, float) and int(n.value) == n.value:
                return {"type": "Const", "value": int(n.value)}
            return {"type": "Const", "value": n.value}
        elif isinstance(n, Call):
            return {"type": "Call", "fn": n.fn, "args": [node_to_dict(arg) for arg in n.args]}
        else:
            raise ValueError(f"Unknown node type: {type(n)}")
    
    node_dict = node_to_dict(node)
    return json.dumps(node_dict, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def ast_to_hash(node: Node) -> str:
    canonical_json = ast_to_canonical_json(node)
    return hashlib.sha1(canonical_json.encode('utf-8')).hexdigest()