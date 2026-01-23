# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SkipLayerNormalization operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class SkipLayerNormalizationHandler(ShapeHandler):
    """Handler for SkipLayerNormalization operator."""

    @property
    def op_type(self) -> str:
        return "SkipLayerNormalization"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node)

        # If the SkipLayerNormalization node contains the optional
        # output for inference, infer the shape and type for it too
        if len(node.output) > 3:
            ctx.propagate_shape_and_type(node, 0, 3)


register_shape_handler(SkipLayerNormalizationHandler())
