# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ReduceProd operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, sympy_reduce_product


class ReduceProdHandler(ShapeHandler):
    """Handler for ReduceProd operator."""

    @property
    def op_type(self) -> str:
        return "ReduceProd"

    def infer_shape(self, node, ctx) -> None:
        axes = get_attribute(node, "axes")
        keep_dims = get_attribute(node, "keepdims", 1)
        if keep_dims == 0 and axes == [0]:
            data = ctx.get_int_or_float_values(node)[0]
            if data is not None:
                ctx.sympy_data_[node.output[0]] = sympy_reduce_product(data)


register_shape_handler(ReduceProdHandler())
