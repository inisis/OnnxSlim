# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Constant operator."""

from onnx import numpy_helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class ConstantHandler(ShapeHandler):
    """Handler for Constant operator."""

    @property
    def op_type(self) -> str:
        return "Constant"

    def infer_shape(self, node, ctx) -> None:
        t = get_attribute(node, "value")
        t.name = node.output[0]
        ctx.initializers_[node.output[0]] = t
        ctx.sympy_data_[node.output[0]] = numpy_helper.to_array(t)


register_shape_handler(ConstantHandler())
