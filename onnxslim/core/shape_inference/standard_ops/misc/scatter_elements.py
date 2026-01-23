# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ScatterElements operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class ScatterElementsHandler(ShapeHandler):
    """Handler for ScatterElements operator."""

    @property
    def op_type(self) -> str:
        return "ScatterElements"

    def infer_shape(self, node, ctx) -> None:
        data_shape = ctx.get_shape(node, 0)
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                data_shape,
            )
        )


register_shape_handler(ScatterElementsHandler())
