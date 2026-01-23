# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Gather operator."""

import numpy as np
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, handle_negative_axis


class GatherHandler(ShapeHandler):
    """Handler for Gather operator."""

    @property
    def op_type(self) -> str:
        return "Gather"

    def infer_shape(self, node, ctx) -> None:
        data_shape = ctx.get_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", 0), len(data_shape))
        indices_shape = ctx.get_shape(node, 1)
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                data_shape[:axis] + indices_shape + data_shape[axis + 1 :],
            )
        )
        # for 1D input, do some sympy compute
        if node.input[0] in ctx.sympy_data_ and len(data_shape) == 1 and get_attribute(node, "axis", 0) == 0:
            idx = ctx.try_get_value(node, 1)
            if idx is not None:
                data = ctx.sympy_data_[node.input[0]]
                if type(data) == list:
                    if type(idx) == np.ndarray and len(idx.shape) == 1:
                        ctx.sympy_data_[node.output[0]] = [data[int(i)] for i in idx]
                    else:
                        ctx.sympy_data_[node.output[0]] = data[int(idx)]
                else:
                    assert idx in {0, -1}
                    ctx.sympy_data_[node.output[0]] = data


register_shape_handler(GatherHandler())
