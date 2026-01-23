# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Size operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import sympy_reduce_product


class SizeHandler(ShapeHandler):
    """Handler for Size operator."""

    @property
    def op_type(self) -> str:
        return "Size"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.get_sympy_shape(node, 0)
        ctx.sympy_data_[node.output[0]] = sympy_reduce_product(sympy_shape)
        ctx.known_vi_[node.output[0]].CopyFrom(
            helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [])
        )


register_shape_handler(SizeHandler())
