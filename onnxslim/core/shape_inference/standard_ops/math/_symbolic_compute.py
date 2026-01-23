# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shared symbolic computation helper for math operators."""

import sympy

from ...utils import is_literal


def infer_symbolic_compute_ops(node, ctx):
    """Handles symbolic computation operations for given node based on predefined functions."""
    funcs = {
        "Add": lambda l: l[0] + l[1],
        "Div": lambda l: (int(l[0] // l[1]) if isinstance(l[0] // l[1], float) else l[0] // l[1]),
        "Equal": lambda l: l[0] == l[1],
        "Floor": lambda l: sympy.floor(l[0]),
        "Max": lambda l: (
            l[1]
            if is_literal(l[0]) and int(l[0]) < -ctx.int_max_
            else (l[0] if is_literal(l[1]) and int(l[1]) < -ctx.int_max_ else sympy.Max(l[0], l[1]))
        ),
        "Min": lambda l: (
            l[1]
            if is_literal(l[0]) and int(l[0]) > ctx.int_max_
            else (l[0] if is_literal(l[1]) and int(l[1]) > ctx.int_max_ else sympy.Min(l[0], l[1]))
        ),
        "Mul": lambda l: (int(l[0] * l[1]) if isinstance(l[0] * l[1], float) else l[0] * l[1]),
        "Sub": lambda l: l[0] - l[1],
        "Where": lambda l: l[1] if l[0] else l[2],
        "Neg": lambda l: -l[0],
    }
    assert node.op_type in funcs
    ctx.compute_on_sympy_data(node, funcs[node.op_type])
