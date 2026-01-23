# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ReduceSum operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_opset, get_shape_from_sympy_shape, handle_negative_axis


class ReduceSumHandler(ShapeHandler):
    """Handler for ReduceSum operator."""

    @property
    def op_type(self) -> str:
        return "ReduceSum"

    def infer_shape(self, node, ctx) -> None:
        keep_dims = get_attribute(node, "keepdims", 1)
        if get_opset(ctx.out_mp_) >= 13 and len(node.input) > 1:
            axes = ctx.try_get_value(node, 1)
            vi = ctx.known_vi_[node.output[0]]
            if axes is None:
                assert keep_dims
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        get_shape_from_sympy_shape(ctx.new_symbolic_shape(ctx.get_shape_rank(node, 0), node)),
                    )
                )
            else:
                shape = ctx.get_shape(node, 0)
                output_shape = []
                axes = [handle_negative_axis(a, len(shape)) for a in axes]
                for i, d in enumerate(shape):
                    if i in axes:
                        if keep_dims:
                            output_shape.append(1)
                    else:
                        output_shape.append(d)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        output_shape,
                    )
                )


register_shape_handler(ReduceSumHandler())
