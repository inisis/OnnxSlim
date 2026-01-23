# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for QuantizeLinear operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class QuantizeLinearHandler(ShapeHandler):
    """Handler for QuantizeLinear operator."""

    @property
    def op_type(self) -> str:
        return "QuantizeLinear"

    def infer_shape(self, node, ctx) -> None:
        output_dtype = onnx.TensorProto.UINT8
        if len(node.input) > 2 and node.input[2]:
            output_dtype = ctx.known_vi_[node.input[2]].type.tensor_type.elem_type
        output_shape = ctx.get_shape(node, 0)
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))


register_shape_handler(QuantizeLinearHandler())
