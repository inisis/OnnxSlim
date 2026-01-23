# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Slice operator."""

import logging

import numpy as np
import sympy
from onnx import helper

from onnxslim.third_party._sympy.solve import try_solve

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import (
    as_list,
    get_attribute,
    get_opset,
    get_shape_from_sympy_shape,
    is_literal,
)

logger = logging.getLogger(__name__)


class SliceHandler(ShapeHandler):
    """Handler for Slice operator."""

    @property
    def op_type(self) -> str:
        return "Slice"

    def infer_shape(self, node, ctx) -> None:
        def flatten_min(expr):
            """Returns a list with expressions split by min() for inequality proof."""
            assert isinstance(expr, sympy.Add), f"Expected a sum of two arguments, got {expr}"
            min_positions = [idx for idx in range(len(expr.args)) if isinstance(expr.args[idx], sympy.Min)]
            if len(min_positions) == 1:
                min_pos = min_positions[0]

                def replace_min_with_arg(arg_idx):
                    replaced = list(expr.args)
                    assert isinstance(replaced[min_pos], sympy.Min)
                    assert len(replaced[min_pos].args) == 2
                    replaced[min_pos] = replaced[min_pos].args[arg_idx]
                    return sympy.Add(*replaced)

                return [replace_min_with_arg(0), replace_min_with_arg(1)]
            return [expr]

        def less_equal(x, y):
            """Returns True if x is less than or equal to y."""
            try:
                return x <= y
            except TypeError:
                pass
            try:
                return y >= x
            except TypeError:
                pass
            try:
                return -x >= -y
            except TypeError:
                pass
            try:
                return -y <= -x
            except TypeError:
                pass
            try:
                return y - x >= 0
            except TypeError:
                return all(d >= 0 for d in flatten_min(y - x))

        def handle_negative_index(index, bound):
            """Normalizes a negative index to be in [0, bound)."""
            try:
                if not less_equal(0, index):
                    if is_literal(index) and index <= -ctx.int_max_:
                        return index
                    return bound + index
            except TypeError:
                logger.warning(f"Cannot determine if {index} < 0")
            return index

        if get_opset(ctx.out_mp_) <= 9:
            axes = get_attribute(node, "axes")
            starts = get_attribute(node, "starts")
            ends = get_attribute(node, "ends")
            if not axes:
                axes = list(range(len(starts)))
            steps = [1] * len(axes)
        else:
            starts = as_list(ctx.try_get_value(node, 1), keep_none=True)
            ends = as_list(ctx.try_get_value(node, 2), keep_none=True)
            axes = ctx.try_get_value(node, 3)
            steps = ctx.try_get_value(node, 4)
            if axes is None and (starts is not None or ends is not None):
                axes = list(range(len(starts if starts is not None else ends)))
            if steps is None and (starts is not None or ends is not None):
                steps = [1] * len(starts if starts is not None else ends)
            axes = as_list(axes, keep_none=True)
            steps = as_list(steps, keep_none=True)

        new_sympy_shape = ctx.get_sympy_shape(node, 0)
        if starts is None or ends is None:
            if axes is None:
                for i in range(len(new_sympy_shape)):
                    new_sympy_shape[i] = ctx.new_symbolic_dim_from_output(node, 0, i)
            else:
                new_sympy_shape = get_shape_from_sympy_shape(new_sympy_shape)
                for i in axes:
                    new_sympy_shape[i] = ctx.new_symbolic_dim_from_output(node, 0, i)
        else:
            for i, s, e, t in zip(axes, starts, ends, steps):
                if is_literal(e):
                    e = handle_negative_index(e, new_sympy_shape[i])
                if is_literal(e):
                    if e >= ctx.int_max_:
                        e = new_sympy_shape[i]
                    elif e <= -ctx.int_max_:
                        e = 0 if s > 0 else -1
                    elif is_literal(new_sympy_shape[i]):
                        if e < 0:
                            e = max(0, e + new_sympy_shape[i])
                        e = min(e, new_sympy_shape[i])
                    else:
                        if e > 0:
                            e = sympy.Min(e, new_sympy_shape[i]) if e > 1 else e
                else:
                    if is_literal(new_sympy_shape[i]):
                        if new_sympy_shape[i] < 0:
                            e = sympy.Min(e, new_sympy_shape[i])
                    else:
                        try:
                            if not less_equal(e, new_sympy_shape[i]):
                                e = new_sympy_shape[i]
                        except Exception:
                            if len(e.free_symbols) == 1:
                                if try_solve((e - new_sympy_shape[i]) >= 0, next(iter(e.free_symbols))) is None:
                                    logger.warning(
                                        f"Unable to solve if {e} <= {new_sympy_shape[i]}, treat as not equal"
                                    )
                            else:
                                logger.warning(f"Unable to determine if {e} <= {new_sympy_shape[i]}, treat as equal")
                                e = new_sympy_shape[i]

                s = handle_negative_index(s, new_sympy_shape[i])
                if is_literal(new_sympy_shape[i]) and is_literal(s):
                    s = max(0, min(s, new_sympy_shape[i]))

                new_sympy_shape[i] = sympy.simplify((e - s + t + (-1 if t > 0 else 1)) // t)

            ctx.update_computed_dims(new_sympy_shape)

        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )

        # handle sympy_data if needed, for slice in shape computation
        if (
            node.input[0] in ctx.sympy_data_
            and [0] == axes
            and starts is not None
            and len(starts) == 1
            and ends is not None
            and len(ends) == 1
            and steps is not None
            and len(steps) == 1
        ):
            input_sympy_data = ctx.sympy_data_[node.input[0]]
            if type(input_sympy_data) == list or (
                type(input_sympy_data) == np.array and len(input_sympy_data.shape) == 1
            ):
                ctx.sympy_data_[node.output[0]] = input_sympy_data[starts[0] : ends[0] : steps[0]]


register_shape_handler(SliceHandler())
