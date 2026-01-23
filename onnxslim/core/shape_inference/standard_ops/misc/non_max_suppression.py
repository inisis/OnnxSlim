# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for NonMaxSuppression operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class NonMaxSuppressionHandler(ShapeHandler):
    """Handler for NonMaxSuppression operator."""

    @property
    def op_type(self) -> str:
        return "NonMaxSuppression"

    def infer_shape(self, node, ctx) -> None:
        selected = str(ctx.new_symbolic_dim_from_output(node))
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, [selected, 3]))


register_shape_handler(NonMaxSuppressionHandler())
