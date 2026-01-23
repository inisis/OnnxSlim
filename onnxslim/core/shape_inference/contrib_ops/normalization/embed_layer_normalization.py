# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for EmbedLayerNormalization operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler


class EmbedLayerNormalizationHandler(ShapeHandler):
    """Handler for EmbedLayerNormalization operator."""

    @property
    def op_type(self) -> str:
        return "EmbedLayerNormalization"

    def infer_shape(self, node, ctx) -> None:
        input_ids_shape = ctx.get_shape(node, 0)
        word_embedding_shape = ctx.get_shape(node, 2)
        assert len(input_ids_shape) == 2 and len(word_embedding_shape) == 2
        output_shape = [*input_ids_shape, word_embedding_shape[1]]

        word_embedding_dtype = ctx.known_vi_[node.input[2]].type.tensor_type.elem_type
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], word_embedding_dtype, output_shape))

        if len(node.output) > 1 and node.output[1]:
            mask_index_shape = [input_ids_shape[0]]
            vi = ctx.known_vi_[node.output[1]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[1], onnx.TensorProto.INT32, mask_index_shape))

        if len(node.output) > 2:
            # Optional output of add before layer normalization is done
            vi = ctx.known_vi_[node.output[2]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[2], word_embedding_dtype, output_shape))


register_shape_handler(EmbedLayerNormalizationHandler())
