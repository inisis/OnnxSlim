# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen unfold operator."""

from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..utils import get_shape_from_sympy_shape


class AtenUnfoldHandler(ShapeHandler):
    """Handler for ATen unfold operator."""

    @property
    def op_type(self) -> str:
        return "unfold"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.get_sympy_shape(node, 0)
        dimension = ctx.try_get_value(node, 1)
        size = ctx.try_get_value(node, 2)
        step = ctx.try_get_value(node, 3)
        if dimension is not None and size is not None and step is not None:
            assert dimension < len(sympy_shape)
            sympy_shape[dimension] = (sympy_shape[dimension] - size) // step + 1
            sympy_shape.append(size)
        else:
            rank = len(sympy_shape)
            sympy_shape = ctx.new_symbolic_shape(rank + 1, node)
        ctx.update_computed_dims(sympy_shape)
        if node.output[0]:
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(sympy_shape),
                )
            )


register_aten_handler(AtenUnfoldHandler())
