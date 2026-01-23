# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ZipMap operator."""

import onnx

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class ZipMapHandler(ShapeHandler):
    """Handler for ZipMap operator."""

    @property
    def op_type(self) -> str:
        return "ZipMap"

    def infer_shape(self, node, ctx) -> None:
        map_key_type = None
        if get_attribute(node, "classlabels_int64s") is not None:
            map_key_type = onnx.TensorProto.INT64
        elif get_attribute(node, "classlabels_strings") is not None:
            map_key_type = onnx.TensorProto.STRING

        assert map_key_type is not None
        new_vi = onnx.ValueInfoProto()
        new_vi.name = node.output[0]
        new_vi.type.sequence_type.elem_type.map_type.value_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        new_vi.type.sequence_type.elem_type.map_type.key_type = map_key_type
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(new_vi)


register_shape_handler(ZipMapHandler())
