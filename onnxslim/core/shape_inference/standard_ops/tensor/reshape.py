# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Reshape operator."""

import sympy
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_shape_from_sympy_shape, is_literal


class ReshapeHandler(ShapeHandler):
    """Handler for Reshape operator."""

    @property
    def op_type(self) -> str:
        return "Reshape"

    def infer_shape(self, node, ctx) -> None:
        shape_value = ctx.try_get_value(node, 1)
        vi = ctx.known_vi_[node.output[0]]
        if shape_value is None:
            shape_shape = ctx.get_shape(node, 1)
            assert len(shape_shape) == 1
            shape_rank = shape_shape[0]
            assert is_literal(shape_rank)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(ctx.new_symbolic_shape(shape_rank, node)),
                )
            )
        else:
            input_sympy_shape = ctx.get_sympy_shape(node, 0)
            total = 1
            for d in input_sympy_shape:
                total = total * d
            new_sympy_shape = []
            deferred_dim_idx = -1
            non_deferred_size = 1
            for i, d in enumerate(shape_value):
                if type(d) == sympy.Symbol or d != 0:
                    new_sympy_shape.append(d)
                else:
                    new_sympy_shape.append(input_sympy_shape[i])
                    non_deferred_size = non_deferred_size * input_sympy_shape[i]
                if d == -1:
                    deferred_dim_idx = i
                elif d != 0:
                    non_deferred_size = non_deferred_size * d

            assert new_sympy_shape.count(-1) < 2
            if -1 in new_sympy_shape:
                new_dim = total // non_deferred_size
                new_sympy_shape[deferred_dim_idx] = new_dim

            ctx.update_computed_dims(new_sympy_shape)
            vi.CopyFrom(
                helper.make_tensor_value_info(
                    node.output[0],
                    vi.type.tensor_type.elem_type,
                    get_shape_from_sympy_shape(new_sympy_shape),
                )
            )

        ctx.pass_on_sympy_data(node)


register_shape_handler(ReshapeHandler())
