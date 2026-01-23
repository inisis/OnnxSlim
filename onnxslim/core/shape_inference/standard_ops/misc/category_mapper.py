# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for CategoryMapper operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class CategoryMapperHandler(ShapeHandler):
    """Handler for CategoryMapper operator."""

    @property
    def op_type(self) -> str:
        return "CategoryMapper"

    def infer_shape(self, node, ctx) -> None:
        input_type = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        if input_type == onnx.TensorProto.STRING:
            output_type = onnx.TensorProto.INT64
        else:
            output_type = onnx.TensorProto.STRING
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_type, ctx.get_shape(node, 0)))


register_shape_handler(CategoryMapperHandler())
