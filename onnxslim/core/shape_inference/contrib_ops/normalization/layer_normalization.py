# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for LayerNormalization operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, handle_negative_axis


class LayerNormalizationHandler(ShapeHandler):
    """Handler for LayerNormalization operator."""

    @property
    def op_type(self) -> str:
        return "LayerNormalization"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node)
        if len(node.output) > 1:
            axis = get_attribute(node, "axis")
            if axis is None:
                axis = -1
            x_shape = ctx.get_shape(node, 0)
            if x_shape is not None:
                rank = len(x_shape)
                axis = handle_negative_axis(axis, rank)
                mean_shape = x_shape[:axis] + [1 for _ in range(rank - axis)]
                mean_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
                if mean_dtype in {onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}:
                    mean_dtype = onnx.TensorProto.FLOAT
                vi = ctx.known_vi_[node.output[1]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[1], mean_dtype, mean_shape))
                if len(node.output) > 2:
                    vi = ctx.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(node.output[2], mean_dtype, mean_shape))


register_shape_handler(LayerNormalizationHandler())
