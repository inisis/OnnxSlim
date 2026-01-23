# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Squeeze operator."""

import logging

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_opset, handle_negative_axis

logger = logging.getLogger(__name__)


class SqueezeHandler(ShapeHandler):
    """Handler for Squeeze operator."""

    @property
    def op_type(self) -> str:
        return "Squeeze"

    def infer_shape(self, node, ctx) -> None:
        input_shape = ctx.get_shape(node, 0)
        op_set = get_opset(ctx.out_mp_)

        if op_set < 13:
            axes = get_attribute(node, "axes")
            assert ctx.try_get_value(node, 1) is None
        else:
            axes = ctx.try_get_value(node, 1)
            assert get_attribute(node, "axes") is None

        if axes is None:
            output_shape = [s for s in input_shape if s != 1]
            if ctx.verbose_ > 0:
                symbolic_dimensions = [s for s in input_shape if type(s) != int]
                if symbolic_dimensions:
                    logger.debug(
                        f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                        f"Assuming the following dimensions are never equal to 1: {symbolic_dimensions}"
                    )
        else:
            axes = [handle_negative_axis(a, len(input_shape)) for a in axes]
            output_shape = []
            for i in range(len(input_shape)):
                if i not in axes:
                    output_shape.append(input_shape[i])
                else:
                    assert input_shape[i] == 1 or type(input_shape[i]) != int
                    if ctx.verbose_ > 0 and type(input_shape[i]) != int:
                        logger.debug(
                            f"Symbolic dimensions in input shape of op: '{node.op_type}' node: '{node.name}'. "
                            f"Assuming the dimension '{input_shape[i]}' at index {i} of the input to be equal to 1."
                        )

        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                output_shape,
            )
        )
        ctx.pass_on_sympy_data(node)


register_shape_handler(SqueezeHandler())
