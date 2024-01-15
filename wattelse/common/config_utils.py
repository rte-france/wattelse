import ast
from typing import Any


def parse_literal(expr: Any) -> Any:
    """Allows to convert easily something like {'a': "2", 'b': "3", 3:'xyz', "c":"0.5", "z": "(1,1)"} into {'a': 2, 'b': 3, 3: 'xyz', 'c': 0.5, 'z': (1, 1)}"""
    if type(expr) is dict:
        return {k: parse_literal(v) for k, v in expr.items()}
    else:
        try:
            return ast.literal_eval(expr)
        except:
            return expr
