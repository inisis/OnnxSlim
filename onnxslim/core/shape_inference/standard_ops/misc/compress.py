# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Compress operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, handle_negative_axis


class CompressHandler(ShapeHandler):
    """Handler for Compress operator."""

    @property
    def op_type(self) -> str:
        return "Compress"

    def infer_shape(self, node, ctx) -> None:
        input_shape = ctx.get_shape(node, 0)
        compress_len = str(ctx.new_symbolic_dim_from_output(node))
        axis = get_attribute(node, "axis")
        if axis is None:
            output_shape = [compress_len]
        else:
            output_shape = input_shape
            output_shape[handle_negative_axis(axis, len(input_shape))] = compress_len
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )


register_shape_handler(CompressHandler())
