# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for SequenceAt operator."""

import onnx

from ...base import ShapeHandler
from ...registry import register_shape_handler


class SequenceAtHandler(ShapeHandler):
    """Handler for SequenceAt operator."""

    @property
    def op_type(self) -> str:
        return "SequenceAt"

    def infer_shape(self, node, ctx) -> None:
        seq_shape = ctx.get_shape(node, 0)
        if seq_shape is not None:
            vi = ctx.known_vi_[node.output[0]]
            for di, d in enumerate(seq_shape):
                if d is not None:
                    continue
                new_dim = onnx.TensorShapeProto.Dimension()
                new_dim.dim_param = str(ctx.new_symbolic_dim_from_output(node, 0, di))
                vi.type.tensor_type.shape.dim[di].CopyFrom(new_dim)


register_shape_handler(SequenceAtHandler())
