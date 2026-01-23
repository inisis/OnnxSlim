# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Transpose operator."""

import numpy as np

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class TransposeHandler(ShapeHandler):
    """Handler for Transpose operator."""

    @property
    def op_type(self) -> str:
        return "Transpose"

    def infer_shape(self, node, ctx) -> None:
        if node.input[0] in ctx.sympy_data_:
            data_shape = ctx.get_shape(node, 0)
            perm = get_attribute(node, "perm", reversed(list(range(len(data_shape)))))
            input_data = ctx.sympy_data_[node.input[0]]
            ctx.sympy_data_[node.output[0]] = (
                np.transpose(np.array(input_data).reshape(*data_shape), axes=tuple(perm)).flatten().tolist()
            )


register_shape_handler(TransposeHandler())
