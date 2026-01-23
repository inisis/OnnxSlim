# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen upsample operators."""

import numpy as np
from onnx import helper

from ..base import ShapeHandler
from ..registry import register_aten_handler


class AtenUpsampleHandler(ShapeHandler):
    """Handler for ATen upsample operators."""

    def __init__(self, op_name):
        super().__init__()
        self._op_type = op_name

    @property
    def op_type(self) -> str:
        return self._op_type

    def infer_shape(self, node, ctx) -> None:
        new_shape = None
        input_shape = ctx.get_shape(node, 0)
        if input_shape is not None:
            new_shape = input_shape[:2]
            output_size = ctx.try_get_value(node, 1)
            if output_size is not None:
                new_shape += [dim_size.item() if type(dim_size) == np.int64 else dim_size for dim_size in output_size]
            else:
                rank = len(input_shape)
                new_shape += [str(ctx.new_symbolic_dim_from_output(node, 0, i)) for i in range(2, rank)]
        if node.output[0] and new_shape is not None:
            output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
            vi = ctx.known_vi_[node.output[0]]
            vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))


register_aten_handler(AtenUpsampleHandler("upsample_nearest1d"))
register_aten_handler(AtenUpsampleHandler("upsample_nearest2d"))
register_aten_handler(AtenUpsampleHandler("upsample_nearest3d"))
register_aten_handler(AtenUpsampleHandler("upsample_bicubic2d"))
