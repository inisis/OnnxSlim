# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen native_group_norm operator."""

from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..utils import as_scalar


class AtenGroupNormHandler(ShapeHandler):
    """Handler for ATen native_group_norm operator."""

    @property
    def op_type(self) -> str:
        return "native_group_norm"

    def infer_shape(self, node, ctx) -> None:
        ctx.propagate_shape_and_type(node)
        input_shape = ctx.get_shape(node, 0)
        N = input_shape[0] if input_shape is not None and len(input_shape) != 0 else None
        group = ctx.try_get_value(node, 6)
        output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        for i in {1, 2}:
            if node.output[i]:
                vi = ctx.known_vi_[node.output[i]]
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[i],
                        output_dtype,
                        [
                            (N if N is not None else str(ctx.new_symbolic_dim_from_output(node, i, 0))),
                            (as_scalar(group) if group is not None else str(ctx.new_symbolic_dim_from_output(node, i, 1))),
                        ],
                    )
                )


register_aten_handler(AtenGroupNormHandler())
