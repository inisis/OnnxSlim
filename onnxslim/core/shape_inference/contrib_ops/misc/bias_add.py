# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for BiasAdd operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class BiasAddHandler(ShapeHandler):
    """Handler for BiasAdd operator."""

    @property
    def op_type(self) -> str:
        return "BiasAdd"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node)


register_shape_handler(BiasAddHandler())
