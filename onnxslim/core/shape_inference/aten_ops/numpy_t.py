# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ATen numpy_T operator."""

from ..base import ShapeHandler
from ..registry import register_aten_handler
from ..standard_ops.tensor.transpose import TransposeHandler


class AtenNumpyTHandler(ShapeHandler):
    """Handler for ATen numpy_T operator (reuses Transpose logic)."""

    @property
    def op_type(self) -> str:
        return "numpy_T"

    def infer_shape(self, node, ctx) -> None:
        TransposeHandler().infer_shape(node, ctx)


register_aten_handler(AtenNumpyTHandler())
