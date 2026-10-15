# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handlers for SpaceToDepth and DepthToSpace operators."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_sympy_shape


class SpaceToDepthHandler(ShapeHandler):
    """Handler for SpaceToDepth operator."""

    @property
    def op_type(self) -> str:
        return "SpaceToDepth"

    def infer_shape(self, node, ctx) -> None:
        shape = ctx.get_sympy_shape(node, 0)
        assert len(shape) == 4
        blocksize = get_attribute(node, "blocksize")
        output_shape = [
            shape[0],
            shape[1] * blocksize**2,
            shape[2] // blocksize,
            shape[3] // blocksize,
        ]
        ctx.update_computed_dims(output_shape)
        output_type = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        ctx.known_vi_[node.output[0]].CopyFrom(
            helper.make_tensor_value_info(
                node.output[0], output_type, get_shape_from_sympy_shape(output_shape)
            )
        )


class DepthToSpaceHandler(ShapeHandler):
    """Handler for DepthToSpace operator."""

    @property
    def op_type(self) -> str:
        return "DepthToSpace"

    def infer_shape(self, node, ctx) -> None:
        shape = ctx.get_sympy_shape(node, 0)
        assert len(shape) == 4
        blocksize = get_attribute(node, "blocksize")
        output_shape = [
            shape[0],
            shape[1] // blocksize**2,
            shape[2] * blocksize,
            shape[3] * blocksize,
        ]
        ctx.update_computed_dims(output_shape)
        output_type = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        ctx.known_vi_[node.output[0]].CopyFrom(
            helper.make_tensor_value_info(
                node.output[0], output_type, get_shape_from_sympy_shape(output_shape)
            )
        )


register_shape_handler(SpaceToDepthHandler())
register_shape_handler(DepthToSpaceHandler())
