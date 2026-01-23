# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for BiasSplitGelu operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class BiasSplitGeluHandler(ShapeHandler):
    """Handler for BiasSplitGelu operator."""

    @property
    def op_type(self) -> str:
        return "BiasSplitGelu"

    def infer_shape(self, node, ctx) -> None:
        input_shape = ctx.get_shape(node, 0)
        bias_shape = ctx.get_shape(node, 1)
        if input_shape and bias_shape and isinstance(bias_shape[0], int):
            output_shape = input_shape
            output_shape[2] = int(bias_shape[0] / 2)
            vi = ctx.known_vi_[node.output[0]]
            output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, output_shape))


register_shape_handler(BiasSplitGeluHandler())
