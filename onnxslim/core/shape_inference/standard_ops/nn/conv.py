# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Conv operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_shape_from_sympy_shape


class ConvHandler(ShapeHandler):
    """Handler for Conv operator."""

    @property
    def op_type(self) -> str:
        return "Conv"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.compute_conv_pool_shape(node)
        ctx.update_computed_dims(sympy_shape)
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )


register_shape_handler(ConvHandler())
