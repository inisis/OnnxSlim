# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SimplifiedLayerNormalization operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler
from .layer_normalization import LayerNormalizationHandler


class SimplifiedLayerNormalizationHandler(ShapeHandler):
    """Handler for SimplifiedLayerNormalization operator."""

    @property
    def op_type(self) -> str:
        return "SimplifiedLayerNormalization"

    def infer_shape(self, node, ctx) -> None:
        # Reuse LayerNormalization handler
        LayerNormalizationHandler().infer_shape(node, ctx)


register_shape_handler(SimplifiedLayerNormalizationHandler())
