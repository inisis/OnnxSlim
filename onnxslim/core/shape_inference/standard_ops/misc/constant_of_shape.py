# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ConstantOfShape operator."""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_sympy_shape, is_literal


class ConstantOfShapeHandler(ShapeHandler):
    """Handler for ConstantOfShape operator."""

    @property
    def op_type(self) -> str:
        return "ConstantOfShape"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.get_int_or_float_values(node)[0]
        vi = ctx.known_vi_[node.output[0]]
        if sympy_shape is not None:
            if type(sympy_shape) != list:
                sympy_shape = [sympy_shape]
            ctx.update_computed_dims(sympy_shape)
            if vi.type.tensor_type.elem_type == onnx.TensorProto.INT64 and all(is_literal(x) for x in sympy_shape):
                ctx.sympy_data_[node.output[0]] = np.ones(
                    [int(x) for x in sympy_shape], dtype=np.int64
                ) * numpy_helper.to_array(get_attribute(node, "value", 0))
        else:
            sympy_shape = ctx.new_symbolic_shape(ctx.get_shape(node, 0)[0], node)

        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )


register_shape_handler(ConstantOfShapeHandler())
