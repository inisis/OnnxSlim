# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for RelativePositionBias operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class RelativePositionBiasHandler(ShapeHandler):
    """Handler for RelativePositionBias operator."""

    @property
    def op_type(self) -> str:
        return "RelativePositionBias"

    def infer_shape(self, node, ctx) -> None:
        seq_len = ctx.try_get_value(node, 1)
        real_seq_len = ctx.try_get_value(node, 2)
        if seq_len is None or real_seq_len is None:
            return
        num_heads = ctx.get_sympy_shape(node, 0)[1]
        new_shape = [1, num_heads, str(seq_len), str(real_seq_len)]
        output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))


register_shape_handler(RelativePositionBiasHandler())
