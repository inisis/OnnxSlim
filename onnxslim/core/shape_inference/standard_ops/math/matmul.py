# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for MatMul operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class MatMulHandler(ShapeHandler):
    """Handler for MatMul operator."""

    @property
    def op_type(self) -> str:
        return "MatMul"

    def infer_shape(self, node, ctx) -> None:
        ctx.compute_matmul_shape(node)


register_shape_handler(MatMulHandler())
