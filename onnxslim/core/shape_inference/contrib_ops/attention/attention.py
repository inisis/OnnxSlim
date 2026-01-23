# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Attention operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class AttentionHandler(ShapeHandler):
    """Handler for Attention operator."""

    @property
    def op_type(self) -> str:
        return "Attention"

    def infer_shape(self, node, ctx) -> None:
        shape = ctx.get_shape(node, 0)
        shape_weights = ctx.get_shape(node, 1)
        shape_bias = ctx.try_get_shape(node, 2)
        if shape_bias is not None:
            assert len(shape_bias) == 1
        tripled_hidden_size = shape_bias[0] if shape_bias is not None else shape_weights[1]
        if shape and len(shape) == 3:
            qkv_hidden_sizes_attr = get_attribute(node, "qkv_hidden_sizes")
            if qkv_hidden_sizes_attr is not None:
                assert len(qkv_hidden_sizes_attr) == 3
                shape[2] = int(qkv_hidden_sizes_attr[2])
            elif isinstance(tripled_hidden_size, int):
                shape[2] = int(tripled_hidden_size / 3)
            output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, shape))

            if len(node.output) > 1:
                input_shape = ctx.get_shape(node, 0)
                past_shape = ctx.get_shape(node, 4) if len(node.input) > 4 and node.input[4] else []
                mask_shape = ctx.get_shape(node, 3) if len(node.input) > 3 and node.input[3] else []

                if past_shape and len(past_shape) == 5:
                    if mask_shape and len(mask_shape) in {2, 3}:
                        past_shape[3] = mask_shape[-1]
                    elif input_shape and len(input_shape) == 3:
                        if isinstance(input_shape[1], int) and isinstance(past_shape[3], int):
                            past_shape[3] = input_shape[1] + past_shape[3]
                        else:
                            past_shape[3] = f"{past_shape[3]}+{input_shape[1]}"
                    vi = ctx.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, past_shape))
                else:
                    num_heads = get_attribute(node, "num_heads")
                    head_size = input_shape[2] // num_heads
                    present_shape = [2, input_shape[0], num_heads, input_shape[1], head_size]
                    vi = ctx.known_vi_[node.output[1]]
                    vi.CopyFrom(helper.make_tensor_value_info(vi.name, output_dtype, present_shape))


register_shape_handler(AttentionHandler())
