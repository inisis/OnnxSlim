# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Cast operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class CastHandler(ShapeHandler):
    """Handler for Cast operator."""

    @property
    def op_type(self) -> str:
        return "Cast"

    def infer_shape(self, node, ctx) -> None:
        ctx.pass_on_sympy_data(node)


register_shape_handler(CastHandler())
