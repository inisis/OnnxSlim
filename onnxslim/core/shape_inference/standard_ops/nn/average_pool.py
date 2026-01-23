# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for AveragePool operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_shape_from_sympy_shape


class PoolHandler(ShapeHandler):
    """Handler for pooling operators (AveragePool, MaxPool)."""

    def __init__(self, op_type_name):
        super().__init__()
        self._op_type = op_type_name

    @property
    def op_type(self) -> str:
        return self._op_type

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.compute_conv_pool_shape(node)
        ctx.update_computed_dims(sympy_shape)
        for o in node.output:
            if not o:
                continue
            vi = ctx.known_vi_[o]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    o,
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(sympy_shape),
                )
            )


register_shape_handler(PoolHandler("AveragePool"))
