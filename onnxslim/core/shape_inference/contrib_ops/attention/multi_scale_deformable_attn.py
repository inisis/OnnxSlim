# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for MultiScaleDeformableAttnTRT operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class MultiScaleDeformableAttnTRTHandler(ShapeHandler):
    """Handler for MultiScaleDeformableAttnTRT operator."""

    @property
    def op_type(self) -> str:
        return "MultiScaleDeformableAttnTRT"

    def infer_shape(self, node, ctx) -> None:
        shape_value = ctx.try_get_shape(node, 0)
        sampling_locations = ctx.try_get_shape(node, 3)
        output_shape = shape_value
        output_shape[1] = sampling_locations[1]
        output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, output_shape))


register_shape_handler(MultiScaleDeformableAttnTRTHandler())
