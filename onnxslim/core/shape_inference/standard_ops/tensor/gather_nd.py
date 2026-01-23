# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for GatherND operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, is_literal


class GatherNDHandler(ShapeHandler):
    """Handler for GatherND operator."""

    @property
    def op_type(self) -> str:
        return "GatherND"

    def infer_shape(self, node, ctx) -> None:
        data_shape = ctx.get_shape(node, 0)
        data_rank = len(data_shape)
        indices_shape = ctx.get_shape(node, 1)
        last_index_dimension = indices_shape[-1]
        batch_dims = get_attribute(node, "batch_dims", 0)
        assert (
            is_literal(last_index_dimension)
            and is_literal(batch_dims)
            and (batch_dims + last_index_dimension) <= data_rank
        )
        new_shape = indices_shape[:-1] + data_shape[batch_dims + last_index_dimension :]
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                new_shape,
            )
        )


register_shape_handler(GatherNDHandler())
