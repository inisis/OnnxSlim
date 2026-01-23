# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for DequantizeLinear operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class DequantizeLinearHandler(ShapeHandler):
    """Handler for DequantizeLinear operator."""

    @property
    def op_type(self) -> str:
        return "DequantizeLinear"

    def infer_shape(self, node, ctx) -> None:
        output_dtype = ctx.known_vi_[node.input[1]].type.tensor_type.elem_type
        output_shape = ctx.get_shape(node, 0)
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))


register_shape_handler(DequantizeLinearHandler())
