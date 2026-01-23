# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen multinomial operator."""

import onnx
from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..utils import get_shape_from_sympy_shape


class AtenMultinomialHandler(ShapeHandler):
    """Handler for ATen multinomial operator."""

    @property
    def op_type(self) -> str:
        return "multinomial"

    def infer_shape(self, node, ctx) -> None:
        sympy_shape = ctx.get_sympy_shape(node, 0)
        rank = len(sympy_shape)
        assert rank in {1, 2}
        num_samples = ctx.try_get_value(node, 1)
        di = rank - 1
        last_dim = num_samples or str(ctx.new_symbolic_dim_from_output(node, 0, di))
        output_shape = [*sympy_shape[:-1], last_dim]
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                onnx.TensorProto.INT64,
                get_shape_from_sympy_shape(output_shape),
            )
        )


register_aten_handler(AtenMultinomialHandler())
