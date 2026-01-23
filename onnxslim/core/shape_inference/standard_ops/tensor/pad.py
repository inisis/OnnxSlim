# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Pad operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_opset, get_shape_from_sympy_shape


class PadHandler(ShapeHandler):
    """Handler for Pad operator."""

    @property
    def op_type(self) -> str:
        return "Pad"

    def infer_shape(self, node, ctx) -> None:
        if get_opset(ctx.out_mp_) <= 10:
            pads = get_attribute(node, "pads")
        else:
            pads = ctx.try_get_value(node, 1)

        sympy_shape = ctx.get_sympy_shape(node, 0)
        rank = len(sympy_shape)

        if pads is not None:
            assert len(pads) == 2 * rank
            new_sympy_shape = [d + pad_up + pad_down for d, pad_up, pad_down in zip(sympy_shape, pads[:rank], pads[rank:])]
            ctx.update_computed_dims(new_sympy_shape)
        else:
            new_sympy_shape = ctx.new_symbolic_shape(rank, node)
        output_tp = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type

        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_tp, get_shape_from_sympy_shape(new_sympy_shape)))


register_shape_handler(PadHandler())
