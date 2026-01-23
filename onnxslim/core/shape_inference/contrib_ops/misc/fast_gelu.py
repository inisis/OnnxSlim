# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for FastGelu operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class FastGeluHandler(ShapeHandler):
    """Handler for FastGelu operator."""

    @property
    def op_type(self) -> str:
        return "FastGelu"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node)


register_shape_handler(FastGeluHandler())
