# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SequenceInsert operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler


class SequenceInsertHandler(ShapeHandler):
    """Handler for SequenceInsert operator."""

    @property
    def op_type(self) -> str:
        return "SequenceInsert"

    def infer_shape(self, node, ctx) -> None:
        vi_seq = ctx.known_vi_[node.input[0]]
        vi_tensor = ctx.known_vi_[node.input[1]]
        vi_out_seq = ctx.known_vi_[node.output[0]]
        vi_out_seq.CopyFrom(vi_seq)
        vi_out_seq.name = node.output[0]
        ctx.fuse_tensor_type(node, 0, vi_out_seq.type, vi_tensor.type)


register_shape_handler(SequenceInsertHandler())
