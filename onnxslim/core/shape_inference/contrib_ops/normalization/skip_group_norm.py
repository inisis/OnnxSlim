# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SkipGroupNorm operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class SkipGroupNormHandler(ShapeHandler):
    """Handler for SkipGroupNorm operator."""

    @property
    def op_type(self) -> str:
        return "SkipGroupNorm"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node, 0, 0)
        if len(node.output) > 1:
            ctx.propagate_shape_and_type(node, 0, 1)


register_shape_handler(SkipGroupNormHandler())
