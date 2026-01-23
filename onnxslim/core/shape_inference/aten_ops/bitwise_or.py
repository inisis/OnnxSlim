# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen bitwise_or operator."""

from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler


class AtenBitwiseOrHandler(ShapeHandler):
    """Handler for ATen bitwise_or operator."""

    @property
    def op_type(self) -> str:
        return "bitwise_or"

    def infer_shape(self, node, ctx) -> None:
        shape0 = ctx.get_shape(node, 0)
        shape1 = ctx.get_shape(node, 1)
        new_shape = ctx.broadcast_shapes(shape0, shape1)
        t0 = ctx.known_vi_[node.input[0]]
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], t0.type.tensor_type.elem_type, new_shape))


register_aten_handler(AtenBitwiseOrHandler())
