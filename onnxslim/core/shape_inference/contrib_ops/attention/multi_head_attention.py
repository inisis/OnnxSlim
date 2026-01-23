# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for MultiHeadAttention operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class MultiHeadAttentionHandler(ShapeHandler):
    """Handler for MultiHeadAttention operator."""

    @property
    def op_type(self) -> str:
        return "MultiHeadAttention"

    def infer_shape(self, node, ctx) -> None:
        query_shape = ctx.get_shape(node, 0)
        total_sequence_length = None
        output_dtype = None
        if query_shape is not None:
            if len(query_shape) == 3:
                key_shape = ctx.try_get_shape(node, 1)
                output_shape = query_shape
                if key_shape is not None and len(key_shape) == 3:
                    value_shape = ctx.try_get_shape(node, 2)
                    if value_shape is not None and len(value_shape) == 3:
                        output_shape[2] = value_shape[2]
                    total_sequence_length = key_shape[1]

                output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
                vi = ctx.known_vi_[node.output[0]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            elif len(query_shape) == 5:
                if isinstance(query_shape[2], int) and isinstance(query_shape[4], int):
                    output_shape = [query_shape[0], query_shape[1], query_shape[2] * query_shape[4]]
                else:
                    output_shape = [query_shape[0], query_shape[1], f"{query_shape[2]}*{query_shape[4]}"]

                total_sequence_length = query_shape[1]

                output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
                vi = ctx.known_vi_[node.output[0]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))

            if len(node.output) > 1:
                batch_size = query_shape[0]
                num_heads = get_attribute(node, "num_heads")

                head_size = None
                if len(query_shape) == 3:
                    head_size = (
                        int(query_shape[2] / num_heads)
                        if isinstance(query_shape[2], int)
                        else f"{query_shape[2]}/{num_heads}"
                    )
                else:
                    head_size = query_shape[4]

                past_shape = ctx.try_get_shape(node, 6)

                if past_shape is not None:
                    if isinstance(past_shape[2], int) and isinstance(total_sequence_length, int):
                        total_sequence_length = past_shape[2] + total_sequence_length
                    else:
                        total_sequence_length = f"{past_shape[2]}+{total_sequence_length}"

                present_shape = [batch_size, num_heads, total_sequence_length, head_size]

                assert output_dtype is not None
                if len(node.output) > 2 and node.output[1] and node.output[2]:
                    vi = ctx.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))
                    vi = ctx.known_vi_[node.output[2]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))


register_shape_handler(MultiHeadAttentionHandler())
