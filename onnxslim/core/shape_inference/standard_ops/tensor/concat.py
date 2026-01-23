# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Concat operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_sympy_shape, handle_negative_axis


class ConcatHandler(ShapeHandler):
    """Handler for Concat operator."""

    @property
    def op_type(self) -> str:
        return "Concat"

    def infer_shape(self, node, ctx) -> None:
        if any(i in ctx.sympy_data_ or i in ctx.initializers_ for i in node.input):
            values = ctx.get_int_or_float_values(node)
            if all(v is not None for v in values):
                assert get_attribute(node, "axis") == 0
                ctx.sympy_data_[node.output[0]] = []
                for i in range(len(node.input)):
                    value = values[i]
                    if isinstance(value, list):
                        ctx.sympy_data_[node.output[0]].extend(value)
                    else:
                        ctx.sympy_data_[node.output[0]].append(value)

        sympy_shape = ctx.get_sympy_shape(node, 0)
        axis = handle_negative_axis(get_attribute(node, "axis"), len(sympy_shape))
        for i_idx in range(1, len(node.input)):
            input_shape = ctx.get_sympy_shape(node, i_idx)
            if input_shape:
                sympy_shape[axis] = sympy_shape[axis] + input_shape[axis]
        ctx.update_computed_dims(sympy_shape)
        # merge symbolic dims for non-concat axes
        for d in range(len(sympy_shape)):
            if d == axis:
                continue
            dims = [ctx.get_shape(node, i_idx)[d] for i_idx in range(len(node.input)) if ctx.get_shape(node, i_idx)]
            if all(d == dims[0] for d in dims):
                continue
            merged = ctx.merge_symbols(dims)
            if type(merged) == str:
                sympy_shape[d] = ctx.symbolic_dims_[merged] if merged else None
            else:
                sympy_shape[d] = merged
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                get_shape_from_sympy_shape(sympy_shape),
            )
        )


register_shape_handler(ConcatHandler())
