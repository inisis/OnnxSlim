# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for LongformerAttention operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class LongformerAttentionHandler(ShapeHandler):
    """Handler for LongformerAttention operator."""

    @property
    def op_type(self) -> str:
        return "LongformerAttention"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node)


register_shape_handler(LongformerAttentionHandler())
