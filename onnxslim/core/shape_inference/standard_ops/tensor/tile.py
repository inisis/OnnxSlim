# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Tile operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_shape_from_sympy_shape


class TileHandler(ShapeHandler):
    """Handler for Tile operator."""

    @property
    def op_type(self) -> str:
        return "Tile"

    def infer_shape(self, node, ctx) -> None:
        repeats_value = ctx.try_get_value(node, 1)
        new_sympy_shape = []
        if repeats_value is not None:
            input_sympy_shape = ctx.get_sympy_shape(node, 0)
            for i, d in enumerate(input_sympy_shape):
                new_dim = d * repeats_value[i]
                new_sympy_shape.append(new_dim)
            ctx.update_computed_dims(new_sympy_shape)
        else:
            new_sympy_shape = ctx.new_symbolic_shape(ctx.get_shape_rank(node, 0), node)
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                vi.type.tensor_type.elem_type,
                get_shape_from_sympy_shape(new_sympy_shape),
            )
        )


register_shape_handler(TileHandler())
