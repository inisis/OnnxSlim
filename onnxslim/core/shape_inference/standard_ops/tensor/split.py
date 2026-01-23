# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Split operator."""

import sympy
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_opset, get_shape_from_sympy_shape, handle_negative_axis


class SplitHandler(ShapeHandler):
    """Handler for Split operator."""

    @property
    def op_type(self) -> str:
        return "Split"

    def infer_shape(self, node, ctx) -> None:
        infer_split_common(node, ctx, helper.make_tensor_value_info)


def infer_split_common(node, ctx, make_value_info_func):
    """Infers the output shape for the Split operator."""
    input_sympy_shape = ctx.get_sympy_shape(node, 0)
    axis = handle_negative_axis(get_attribute(node, "axis", 0), len(input_sympy_shape))
    op_set = get_opset(ctx.out_mp_)

    if op_set < 13:
        split = get_attribute(node, "split")
        assert ctx.try_get_value(node, 1) is None
    else:
        split = ctx.try_get_value(node, 1)
        assert get_attribute(node, "split") is None

    if split is None:
        num_outputs = len(node.output)
        split = [input_sympy_shape[axis] / sympy.Integer(num_outputs)] * num_outputs
        ctx.update_computed_dims(split)
    else:
        split = [sympy.Integer(s) for s in split]

    for i_o in range(len(split)):
        vi = ctx.known_vi_[node.output[i_o]]
        vi.CopyFrom(
            make_value_info_func(
                node.output[i_o],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape([*input_sympy_shape[:axis], split[i_o], *input_sympy_shape[axis + 1 :]]),
            )
        )
        ctx.known_vi_[vi.name] = vi


register_shape_handler(SplitHandler())
