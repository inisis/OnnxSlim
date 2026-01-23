# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for RestorePadding operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class RestorePaddingHandler(ShapeHandler):
    """Handler for RestorePadding operator."""

    @property
    def op_type(self) -> str:
        return "RestorePadding"

    def infer_shape(self, node, ctx) -> None:
        shape_input = ctx.get_shape(node, 0)
        shape_token_offset = ctx.get_shape(node, 1)
        if shape_input and len(shape_input) == 2 and shape_token_offset and len(shape_token_offset) == 2:
            output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = ctx.known_vi_[node.output[0]]
            output_shape = [shape_token_offset[0], shape_token_offset[1], shape_input[1]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))


register_shape_handler(RestorePaddingHandler())
