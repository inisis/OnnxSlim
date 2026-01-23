# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Expand operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import as_list, get_shape_from_sympy_shape


class ExpandHandler(ShapeHandler):
    """Handler for Expand operator."""

    @property
    def op_type(self) -> str:
        return "Expand"

    def infer_shape(self, node, ctx) -> None:
        expand_to_shape = as_list(ctx.try_get_value(node, 1), keep_none=True)
        if expand_to_shape is not None:
            ctx.update_computed_dims(expand_to_shape)
            shape = ctx.get_shape(node, 0)
            new_shape = ctx.broadcast_shapes(shape, get_shape_from_sympy_shape(expand_to_shape))
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    new_shape,
                )
            )


register_shape_handler(ExpandHandler())
