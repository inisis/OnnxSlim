# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SplitToSequence operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ..tensor.split import infer_split_common


class SplitToSequenceHandler(ShapeHandler):
    """Handler for SplitToSequence operator."""

    @property
    def op_type(self) -> str:
        return "SplitToSequence"

    def infer_shape(self, node, ctx) -> None:
        infer_split_common(node, ctx, helper.make_sequence_value_info)


register_shape_handler(SplitToSequenceHandler())
