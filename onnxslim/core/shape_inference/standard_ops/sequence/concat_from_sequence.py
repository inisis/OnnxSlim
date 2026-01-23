# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ConcatFromSequence operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, handle_negative_axis


class ConcatFromSequenceHandler(ShapeHandler):
    """Handler for ConcatFromSequence operator."""

    @property
    def op_type(self) -> str:
        return "ConcatFromSequence"

    def infer_shape(self, node, ctx) -> None:
        seq_shape = ctx.get_shape(node, 0)
        new_axis = 1 if get_attribute(node, "new_axis") else 0
        axis = handle_negative_axis(get_attribute(node, "axis"), len(seq_shape) + new_axis)
        concat_dim = str(ctx.new_symbolic_dim_from_output(node, 0, axis))
        new_shape = seq_shape
        if new_axis:
            new_shape = [*seq_shape[:axis], concat_dim, *seq_shape[axis:]]
        else:
            new_shape[axis] = concat_dim
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.sequence_type.elem_type.tensor_type.elem_type,
                new_shape,
            )
        )


register_shape_handler(ConcatFromSequenceHandler())
