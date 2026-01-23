# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Range operator."""

import sympy
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import as_scalar, get_shape_from_sympy_shape


class RangeHandler(ShapeHandler):
    """Handler for Range operator."""

    @property
    def op_type(self) -> str:
        return "Range"

    def infer_shape(self, node, ctx) -> None:
        vi = ctx.known_vi_[node.output[0]]
        input_data = ctx.get_int_or_float_values(node)
        if all(i is not None for i in input_data):
            start = as_scalar(input_data[0])
            limit = as_scalar(input_data[1])
            delta = as_scalar(input_data[2])
            new_sympy_shape = [sympy.Max(sympy.ceiling((limit - start) / delta), 0)]
        else:
            new_sympy_shape = [ctx.new_symbolic_dim_from_output(node)]
        ctx.update_computed_dims(new_sympy_shape)
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )


register_shape_handler(RangeHandler())
