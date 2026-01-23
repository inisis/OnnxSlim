# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for NhwcConv operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_shape_from_sympy_shape


class NhwcConvHandler(ShapeHandler):
    """Handler for NhwcConv operator (channels last format)."""

    @property
    def op_type(self) -> str:
        return "NhwcConv"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.compute_conv_pool_shape(node, channels_last=True)
        ctx.update_computed_dims(sympy_shape)
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )


register_shape_handler(NhwcConvHandler())
