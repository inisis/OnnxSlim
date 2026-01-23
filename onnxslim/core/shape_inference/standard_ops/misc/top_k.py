# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for TopK operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import as_scalar, get_attribute, get_opset, get_shape_from_sympy_shape, handle_negative_axis


class TopKHandler(ShapeHandler):
    """Handler for TopK operator."""

    @property
    def op_type(self) -> str:
        return "TopK"

    def infer_shape(self, node, ctx) -> None:
        rank = ctx.get_shape_rank(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis", -1), rank)
        new_shape = ctx.get_shape(node, 0)

        if get_opset(ctx.out_mp_) <= 9:
            k = get_attribute(node, "k")
        else:
            k = ctx.get_int_or_float_values(node)[1]

        k = ctx.new_symbolic_dim_from_output(node) if k is None else as_scalar(k)
        if type(k) in {int, str}:
            new_shape[axis] = k
        else:
            new_sympy_shape = ctx.get_sympy_shape(node, 0)
            new_sympy_shape[axis] = k
            ctx.update_computed_dims(new_sympy_shape)
            new_shape = get_shape_from_sympy_shape(new_sympy_shape)

        for i_o in range(len(node.output)):
            vi = ctx.known_vi_[node.output[i_o]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[i_o], vi.type.tensor_type.elem_type, new_shape))


register_shape_handler(TopKHandler())
