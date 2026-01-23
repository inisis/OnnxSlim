# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Resize operator."""

import numpy as np
import sympy
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_opset, get_shape_from_sympy_shape


class ResizeHandler(ShapeHandler):
    """Handler for Resize operator."""

    @property
    def op_type(self) -> str:
        return "Resize"

    def infer_shape(self, node, ctx) -> None:
        vi = ctx.known_vi_[node.output[0]]
        input_sympy_shape = ctx.get_sympy_shape(node, 0)
        if get_opset(ctx.out_mp_) <= 10:
            scales = ctx.try_get_value(node, 1)
            if scales is not None:
                new_sympy_shape = [sympy.simplify(sympy.floor(d * s)) for d, s in zip(input_sympy_shape, scales)]
                ctx.update_computed_dims(new_sympy_shape)
                vi.CopyFrom(
                    helper.make_tensor_value_info(
                        node.output[0],
                        ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                        get_shape_from_sympy_shape(new_sympy_shape),
                    )
                )
        else:
            roi = ctx.try_get_value(node, 1)
            scales = ctx.try_get_value(node, 2)
            sizes = ctx.try_get_value(node, 3)
            if sizes is not None:
                new_sympy_shape = [sympy.simplify(round(s)) for s in sizes]
                ctx.update_computed_dims(new_sympy_shape)
            elif scales is not None:
                rank = len(scales)
                if get_attribute(node, "coordinate_transformation_mode") == "tf_crop_and_resize":
                    assert len(roi) == 2 * rank
                    roi_start = list(roi)[:rank]
                    roi_end = list(roi)[rank:]
                else:
                    roi_start = [0] * rank
                    roi_end = [1] * rank
                if isinstance(scales, np.ndarray):
                    scales = scales.tolist()
                else:
                    scales = list(scales)
                new_sympy_shape = [
                    sympy.floor(d * (end - start) * scale + sympy.Rational(1, 2))
                    for d, start, end, scale in zip(input_sympy_shape, roi_start, roi_end, scales)
                ]
                ctx.update_computed_dims(new_sympy_shape)
            else:
                new_sympy_shape = ctx.new_symbolic_shape(ctx.get_shape_rank(node, 0), node)

            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    ctx.known_vi_[node.input[0]].type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )


register_shape_handler(ResizeHandler())
