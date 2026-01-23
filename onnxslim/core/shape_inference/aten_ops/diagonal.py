# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen diagonal operator."""

import sympy
from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..utils import get_shape_from_sympy_shape, handle_negative_axis


class AtenDiagonalHandler(ShapeHandler):
    """Handler for ATen diagonal operator."""

    @property
    def op_type(self) -> str:
        return "diagonal"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.get_sympy_shape(node, 0)
        rank = len(sympy_shape)
        offset = ctx.try_get_value(node, 1)
        dim1 = ctx.try_get_value(node, 2)
        dim2 = ctx.try_get_value(node, 3)

        assert offset is not None and dim1 is not None and dim2 is not None
        dim1 = handle_negative_axis(dim1, rank)
        dim2 = handle_negative_axis(dim2, rank)

        new_shape = [val for dim, val in enumerate(sympy_shape) if dim not in {dim1, dim2}]
        shape1 = sympy_shape[dim1]
        shape2 = sympy_shape[dim2]
        if offset >= 0:
            diag_shape = sympy.Max(0, sympy.Min(shape1, shape2 - offset))
        else:
            diag_shape = sympy.Max(0, sympy.Min(shape1 + offset, shape2))
        new_shape.append(diag_shape)

        if node.output[0]:
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_shape),
                )
            )


register_aten_handler(AtenDiagonalHandler())
