# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Flatten operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_sympy_shape, sympy_reduce_product


class FlattenHandler(ShapeHandler):
    """Handler for Flatten operator."""

    @property
    def op_type(self) -> str:
        return "Flatten"

    def infer_shape(self, node, ctx) -> None:
        shape = ctx.get_sympy_shape(node, 0)
        axis = get_attribute(node, "axis", 1)
        if axis < 0:
            axis += len(shape)
        assert 0 <= axis <= len(shape)

        output_shape = [
            sympy_reduce_product(shape[:axis]),
            sympy_reduce_product(shape[axis:]),
        ]
        ctx.update_computed_dims(output_shape)
        output_type = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        ctx.known_vi_[node.output[0]].CopyFrom(
            helper.make_tensor_value_info(
                node.output[0], output_type, get_shape_from_sympy_shape(output_shape)
            )
        )


register_shape_handler(FlattenHandler())
