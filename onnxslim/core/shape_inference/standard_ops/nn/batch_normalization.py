# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for BatchNormalization operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class BatchNormalizationHandler(ShapeHandler):
    """Handler for BatchNormalization operator."""

    @property
    def op_type(self) -> str:
        return "BatchNormalization"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node)

        # this works for opsets < 14 and 14 since we check i < len(node.output) in the loop
        for i in {1, 2, 3, 4}:
            if i < len(node.output) and node.output[i]:
                ctx.propagate_shape_and_type(node, input_index=1, output_index=i)


register_shape_handler(BatchNormalizationHandler())
