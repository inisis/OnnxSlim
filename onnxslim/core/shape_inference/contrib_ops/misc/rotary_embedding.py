# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for RotaryEmbedding operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class RotaryEmbeddingHandler(ShapeHandler):
    """Handler for RotaryEmbedding operator."""

    @property
    def op_type(self) -> str:
        return "RotaryEmbedding"

    def infer_shape(self, node, ctx) -> None:
        if len(node.output) == 1:
            ctx.propagate_shape_and_type(node)
        elif len(node.output) == 2:
            # Extraneous constant nodes outputted by RotaryEmbedding function made with `export_modules_as_functions`
            ctx.propagate_shape_and_type(node, input_index=1, output_index=0)
            ctx.propagate_shape_and_type(node, input_index=0, output_index=1)
        elif len(node.output) == 3:
            # Extraneous constant nodes outputted by RotaryEmbedding function made with `export_modules_as_functions`
            ctx.propagate_shape_and_type(node, input_index=1, output_index=0)
            ctx.propagate_shape_and_type(node, input_index=1, output_index=1)
            ctx.propagate_shape_and_type(node, input_index=0, output_index=2)


register_shape_handler(RotaryEmbeddingHandler())
