# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for RemovePadding operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class RemovePaddingHandler(ShapeHandler):
    """Handler for RemovePadding operator."""

    @property
    def op_type(self) -> str:
        return "RemovePadding"

    def infer_shape(self, node, ctx) -> None:
        shape = ctx.get_shape(node, 0)
        if shape and len(shape) == 3:
            output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, ["token_count", shape[2]]))

            vi_token_offset = ctx.known_vi_[node.output[1]]
            vi_token_offset.CopyFrom(
                helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT32, [shape[0], shape[1]])
            )

            vi_cumulated_seq_len = ctx.known_vi_[node.output[2]]
            vi_cumulated_seq_len.CopyFrom(
                helper.make_tensor_value_info(node.output[2], onnx.TensorProto.INT32, ["batch_size + 1"])
            )

            vi_max_seq_len = ctx.known_vi_[node.output[3]]
            vi_max_seq_len.CopyFrom(helper.make_tensor_value_info(node.output[3], onnx.TensorProto.INT32, [1]))


register_shape_handler(RemovePaddingHandler())
