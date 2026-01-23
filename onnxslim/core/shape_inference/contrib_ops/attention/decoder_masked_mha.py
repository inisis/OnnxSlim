# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for DecoderMaskedMultiHeadAttention operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class DecoderMaskedMultiHeadAttentionHandler(ShapeHandler):
    """Handler for DecoderMaskedMultiHeadAttention operator."""

    @property
    def op_type(self) -> str:
        return "DecoderMaskedMultiHeadAttention"

    def infer_shape(self, node, ctx) -> None:
        query_shape = ctx.get_shape(node, 0)
        if query_shape is not None:
            output_shape = query_shape
            output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            assert output_dtype is not None
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            if len(node.output) > 2 and node.output[1] and node.output[2]:
                past_shape = ctx.try_get_shape(node, 5)
                if past_shape is not None:
                    vi = ctx.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))
                    vi = ctx.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))


register_shape_handler(DecoderMaskedMultiHeadAttentionHandler())
