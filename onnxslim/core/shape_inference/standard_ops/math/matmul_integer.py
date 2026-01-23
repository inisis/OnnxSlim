# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for MatMulInteger16 operator."""

import onnx

from ...base import ShapeHandler
from ...registry import register_shape_handler


class MatMulIntegerHandler(ShapeHandler):
    """Handler for MatMulInteger16 operator."""

    @property
    def op_type(self) -> str:
        return "MatMulInteger16"

    def infer_shape(self, node, ctx) -> None:
        ctx.compute_matmul_shape(node, onnx.TensorProto.INT32)


register_shape_handler(MatMulIntegerHandler())
