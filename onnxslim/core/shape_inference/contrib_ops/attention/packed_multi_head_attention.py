# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for PackedMultiHeadAttention operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class PackedMultiHeadAttentionHandler(ShapeHandler):
    """Handler for PackedMultiHeadAttention operator."""

    @property
    def op_type(self) -> str:
        return "PackedMultiHeadAttention"

    def infer_shape(self, node, ctx) -> None:
        shape_value = ctx.try_get_shape(node, 2)
        if shape_value is not None and len(shape_value) == 2:
            output_shape = shape_value
        else:
            shape_query = ctx.get_shape(node, 0)
            assert shape_query is not None and len(shape_query) == 4
            output_shape = [shape_query[0], shape_query[1] * shape_query[3]]

        output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))


register_shape_handler(PackedMultiHeadAttentionHandler())
