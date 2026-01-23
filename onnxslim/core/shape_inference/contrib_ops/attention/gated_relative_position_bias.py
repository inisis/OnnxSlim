# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for GatedRelativePositionBias operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class GatedRelativePositionBiasHandler(ShapeHandler):
    """Handler for GatedRelativePositionBias operator."""

    @property
    def op_type(self) -> str:
        return "GatedRelativePositionBias"

    def infer_shape(self, node, ctx) -> None:
        num_heads = get_attribute(node, "num_heads")
        token_offset_shape = ctx.try_get_shape(node, 6)
        if token_offset_shape is not None:
            output_shape = [token_offset_shape[0], num_heads, token_offset_shape[1], token_offset_shape[1]]
        else:
            query_layer_shape = ctx.get_shape(node, 0)
            assert query_layer_shape is not None and len(query_layer_shape) == 3
            output_shape = [query_layer_shape[0], num_heads, query_layer_shape[1], query_layer_shape[1]]

        output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))


register_shape_handler(GatedRelativePositionBiasHandler())
