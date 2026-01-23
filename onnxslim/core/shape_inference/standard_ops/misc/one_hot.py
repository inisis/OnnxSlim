# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for OneHot operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_sympy_shape, handle_negative_axis, is_literal


class OneHotHandler(ShapeHandler):
    """Handler for OneHot operator."""

    @property
    def op_type(self) -> str:
        return "OneHot"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.get_sympy_shape(node, 0)
        depth = ctx.try_get_value(node, 1)
        axis = get_attribute(node, "axis", -1)
        axis = handle_negative_axis(axis, len(sympy_shape) + 1)
        new_shape = get_shape_from_sympy_shape(
            [
                *sympy_shape[:axis],
                depth if is_literal(depth) else ctx.new_symbolic_dim_from_output(node),
                *sympy_shape[axis:],
            ]
        )
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[2]].type.tensor_type.elem_type,
                new_shape,
            )
        )


register_shape_handler(OneHotHandler())
