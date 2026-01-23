# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for NonZero operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class NonZeroHandler(ShapeHandler):
    """Handler for NonZero operator."""

    @property
    def op_type(self) -> str:
        return "NonZero"

    def infer_shape(self, node, ctx) -> None:
        input_rank = ctx.get_shape_rank(node, 0)
        nz_len = str(ctx.new_symbolic_dim_from_output(node, 0, 1))
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], vi.type.tensor_type.elem_type, [input_rank, nz_len]))


register_shape_handler(NonZeroHandler())
