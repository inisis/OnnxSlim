# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SkipSimplifiedLayerNormalization operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler
from .skip_layer_normalization import SkipLayerNormalizationHandler


class SkipSimplifiedLayerNormalizationHandler(ShapeHandler):
    """Handler for SkipSimplifiedLayerNormalization operator."""

    @property
    def op_type(self) -> str:
        return "SkipSimplifiedLayerNormalization"

    def infer_shape(self, node, ctx) -> None:
        # Reuse SkipLayerNormalization handler
        SkipLayerNormalizationHandler().infer_shape(node, ctx)


register_shape_handler(SkipSimplifiedLayerNormalizationHandler())
