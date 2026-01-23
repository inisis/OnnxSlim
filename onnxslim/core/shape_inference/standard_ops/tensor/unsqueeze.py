# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Unsqueeze operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_opset, handle_negative_axis


class UnsqueezeHandler(ShapeHandler):
    """Handler for Unsqueeze operator."""

    @property
    def op_type(self) -> str:
        return "Unsqueeze"

    def infer_shape(self, node, ctx) -> None:
        input_shape = ctx.get_shape(node, 0)
        op_set = get_opset(ctx.out_mp_)

        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert ctx.try_get_value(node, 1) is None
        else:
            axes = ctx.try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        output_rank = len(input_shape) + len(axes)
        axes = [handle_negative_axis(a, output_rank) for a in axes]

        input_axis = 0
        output_shape = []
        for i in range(output_rank):
            if i in axes:
                output_shape.append(1)
            else:
                output_shape.append(input_shape[input_axis])
                input_axis += 1

        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )
        ctx.pass_on_sympy_data(node)


register_shape_handler(UnsqueezeHandler())
