# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen argmax operator."""

import onnx
from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..utils import get_shape_from_sympy_shape, handle_negative_axis


class AtenArgmaxHandler(ShapeHandler):
    """Handler for ATen argmax operator."""

    @property
    def op_type(self) -> str:
        return "argmax"

    def infer_shape(self, node, ctx) -> None:
        new_shape = None
        if not node.input[1]:
            # The argmax of the flattened input is returned.
            new_shape = []
        else:
            dim = ctx.try_get_value(node, 1)
            keepdim = ctx.try_get_value(node, 2)
            if keepdim is not None:
                sympy_shape = ctx.get_sympy_shape(node, 0)
                if dim is not None:
                    dim = handle_negative_axis(dim, len(sympy_shape))
                    if keepdim:
                        sympy_shape[dim] = 1
                    else:
                        del sympy_shape[dim]
                else:
                    rank = len(sympy_shape)
                    sympy_shape = ctx.new_symbolic_shape(rank if keepdim else rank - 1, node)
                ctx.update_computed_dims(sympy_shape)
                new_shape = get_shape_from_sympy_shape(sympy_shape)
        if node.output[0] and new_shape is not None:
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, new_shape))


register_aten_handler(AtenArgmaxHandler())
