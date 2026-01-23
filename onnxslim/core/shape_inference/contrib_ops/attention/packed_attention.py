# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for PackedAttention operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class PackedAttentionHandler(ShapeHandler):
    """Handler for PackedAttention operator."""

    @property
    def op_type(self) -> str:
        return "PackedAttention"

    def infer_shape(self, node, ctx) -> None:
        shape = ctx.get_shape(node, 0)
        shape_weights = ctx.get_shape(node, 1)
        shape_bias = ctx.try_get_shape(node, 2)
        if shape_bias is not None:
            assert len(shape_bias) == 1
        tripled_hidden_size = shape_bias[0] if shape_bias is not None else shape_weights[1]
        if shape and len(shape) == 2:
            qkv_hidden_sizes_attr = get_attribute(node, "qkv_hidden_sizes")
            if qkv_hidden_sizes_attr is not None:
                assert len(qkv_hidden_sizes_attr) == 3
                shape[1] = int(qkv_hidden_sizes_attr[2])
            elif isinstance(tripled_hidden_size, int):
                shape[1] = int(tripled_hidden_size / 3)
            output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, shape))


register_shape_handler(PackedAttentionHandler())
