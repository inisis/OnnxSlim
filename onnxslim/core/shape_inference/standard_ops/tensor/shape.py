# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Shape operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class ShapeOpHandler(ShapeHandler):
    """Handler for Shape operator."""

    @property
    def op_type(self) -> str:
        return "Shape"

    def infer_shape(self, node, ctx) -> None:
        start = get_attribute(node, "start", 0)
        end = get_attribute(node, "end", None)

        full_sympy_shape = ctx.get_sympy_shape(node, 0)
        num_dims = len(full_sympy_shape)

        if start < 0:
            start = num_dims + start
        if end is None:
            end = num_dims
        elif end < 0:
            end = num_dims + end

        assert 0 <= start <= end <= num_dims, f"reshape start/end invalid: start={start}, end={end}, total_dims={num_dims}"

        target_sympy_shape = full_sympy_shape[start:end]
        ctx.sympy_data_[node.output[0]] = target_sympy_shape


register_shape_handler(ShapeOpHandler())
