# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SoftmaxCrossEntropyLoss operator."""

import onnx
from onnx import helper

from ...base import MultiOpHandler, ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class SoftmaxCrossEntropyLossHandler(ShapeHandler):
    """Handler for SoftmaxCrossEntropyLoss operator."""

    @property
    def op_type(self) -> str:
        return "SoftmaxCrossEntropyLoss"

    def infer_shape(self, node, ctx) -> None:
        vi = ctx.known_vi_[node.output[0]]
        elem_type = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type

        specified_output_type = get_attribute(node, "output_type", None)
        if specified_output_type is not None:
            elem_type = specified_output_type

        vi.type.tensor_type.elem_type = elem_type
        vi.type.tensor_type.shape.CopyFrom(onnx.TensorShapeProto())

        if len(node.output) > 1:
            data_shape = ctx.get_shape(node, 0)
            vi = ctx.known_vi_[node.output[1]]
            vi.CopyFrom(helper.make_tensor_value_info(vi.name, elem_type, data_shape))


def _infer_softmax_cross_entropy(node, ctx):
    SoftmaxCrossEntropyLossHandler().infer_shape(node, ctx)


register_shape_handler(SoftmaxCrossEntropyLossHandler())
register_shape_handler(MultiOpHandler("SoftmaxCrossEntropyLossInternal", _infer_softmax_cross_entropy))
register_shape_handler(MultiOpHandler("NegativeLogLikelihoodLossInternal", _infer_softmax_cross_entropy))
