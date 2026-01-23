# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen min/max operators."""

import onnx
from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..utils import get_shape_from_sympy_shape, handle_negative_axis


class AtenMinMaxHandler(ShapeHandler):
    """Handler for ATen min/max operators."""

    def __init__(self, op_name):
        super().__init__()
        self._op_type = op_name

    @property
    def op_type(self) -> str:
        return self._op_type

    def infer_shape(self, node, ctx) -> None:
        vi = ctx.known_vi_[node.output[0]]
        if len(node.input) == 1:
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    [],
                )
            )
        else:
            assert len(node.input) == 3
            keepdim = ctx.try_get_value(node, 2)
            assert keepdim is not None
            dim = ctx.try_get_value(node, 1)
            if dim is None:
                rank = ctx.get_shape_rank(node, 0)
                output_shape = ctx.new_symbolic_shape(rank if keepdim else rank - 1, node)
            else:
                shape = ctx.get_sympy_shape(node, 0)
                dim = handle_negative_axis(dim, len(shape))
                output_shape = shape[:dim]
                if keepdim:
                    output_shape += [1]
                output_shape += shape[dim + 1 :]

            output_shape = get_shape_from_sympy_shape(output_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    output_shape,
                )
            )
            vi1 = ctx.known_vi_[node.output[1]]
            vi1.CopyFrom(helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT64, output_shape))


register_aten_handler(AtenMinMaxHandler("max"))
register_aten_handler(AtenMinMaxHandler("min"))
