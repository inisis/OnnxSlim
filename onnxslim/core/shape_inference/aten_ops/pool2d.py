# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen pooling operators."""

import onnx
from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..utils import get_shape_from_sympy_shape


class AtenPool2dHandler(ShapeHandler):
    """Handler for ATen pooling operators (max_pool2d_with_indices, avg_pool2d, _adaptive_avg_pool2d)."""

    def __init__(self, op_name):
        super().__init__()
        self._op_type = op_name

    @property
    def op_type(self) -> str:
        return self._op_type

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.get_sympy_shape(node, 0)
        assert len(sympy_shape) == 4
        sympy_shape[-2:] = [ctx.new_symbolic_dim_from_output(node, 0, i) for i in {2, 3}]
        ctx.update_computed_dims(sympy_shape)
        for i, o in enumerate(node.output):
            if not o:
                continue
            vi = ctx.known_vi_[o]
            elem_type = onnx.TensorProto.INT64 if i == 1 else ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi.CopyFrom(helper.make_tensor_value_info(o, elem_type, get_shape_from_sympy_shape(sympy_shape)))


register_aten_handler(AtenPool2dHandler("max_pool2d_with_indices"))
register_aten_handler(AtenPool2dHandler("avg_pool2d"))
register_aten_handler(AtenPool2dHandler("_adaptive_avg_pool2d"))
