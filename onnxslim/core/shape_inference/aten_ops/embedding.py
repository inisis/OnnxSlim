# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen embedding operator."""

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..standard_ops.tensor.gather import GatherHandler


class AtenEmbeddingHandler(ShapeHandler):
    """Handler for ATen embedding operator (reuses Gather logic)."""

    @property
    def op_type(self) -> str:
        return "embedding"

    def infer_shape(self, node, ctx) -> None:
        # Embedding uses the same logic as Gather
        GatherHandler().infer_shape(node, ctx)


register_aten_handler(AtenEmbeddingHandler())
