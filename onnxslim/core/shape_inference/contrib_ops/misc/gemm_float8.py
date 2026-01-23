# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for GemmFloat8 operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class GemmFloat8Handler(ShapeHandler):
    """Handler for GemmFloat8 operator."""

    @property
    def op_type(self) -> str:
        return "GemmFloat8"

    def infer_shape(self, node, ctx) -> None:
        ctx.compute_matmul_shape(node)


register_shape_handler(GemmFloat8Handler())
