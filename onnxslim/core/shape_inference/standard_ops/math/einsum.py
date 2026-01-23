# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Einsum operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute


class EinsumHandler(ShapeHandler):
    """Handler for Einsum operator."""

    @property
    def op_type(self) -> str:
        return "Einsum"

    def infer_shape(self, node, ctx) -> None:
        equation = get_attribute(node, "equation")
        equation = equation.replace(b" ", b"")
        mid_index = equation.find(b"->")
        left_equation = equation[:mid_index] if mid_index != -1 else equation

        num_operands = 0
        num_ellipsis = 0
        num_ellipsis_indices = 0
        num_labels = 0
        ellipsis_flag = True
        dims_value = []
        ellipsis_dims_value = []

        label_maps = {}
        repeated_labels = set()

        terms = left_equation.split(b",")
        for term in terms:
            ellipsis_index = term.find(b"...")
            shape = ctx.get_shape(node, num_operands)
            rank = len(shape)
            ellipsis_dims = 0
            term_size = 0
            num_illegal_char = 0

            for i in range(len(term)):
                if term[i] != 46:
                    term_size = term_size + 1

            index = 0
            while index < len(term):
                if index == ellipsis_index:
                    ellipsis_dims = rank - term_size
                    if ellipsis_flag:
                        ellipsis_flag = False
                        for i in range(ellipsis_dims):
                            ellipsis_dims_value.append(shape[index + i - num_illegal_char])
                    else:
                        for i in range(ellipsis_dims):
                            shape_dim = shape[index + i - num_illegal_char]
                            current_dim = ellipsis_dims_value[i]
                            ellipsis_dims_value[i] = max(current_dim, shape_dim)

                    num_illegal_char += 3
                    index += 3
                    continue

                elif term[index] == 46:
                    num_illegal_char += 1
                    index += 1
                    continue

                char = term[index]
                if char not in label_maps:
                    label_maps[char] = num_labels
                    dims_value.append(shape[index + ellipsis_dims - num_illegal_char])
                    num_labels += 1
                else:
                    repeated_labels.add(char)

                index += 1

            if ellipsis_index != -1:
                if num_ellipsis == 0:
                    if rank < term_size:
                        raise ValueError("Ellipsis represents incompatible dimensions.")
                    num_ellipsis_indices = rank - term_size
                else:
                    if num_ellipsis_indices != rank - term_size:
                        raise ValueError("Ellipsis represents incompatible dimensions.")
                num_ellipsis += 1
            else:
                if rank != term_size:
                    raise ValueError("Rank of input ", num_operands, " does not match the equation indices.")
            num_operands += 1

        new_sympy_shape = []
        if mid_index != -1:
            right_equation = equation[mid_index + 2 :]
            right_ellipsis_index = right_equation.find(b"...")
            if right_ellipsis_index != -1:
                for i in range(num_ellipsis_indices):
                    new_sympy_shape.append(ellipsis_dims_value[i])
            for c in right_equation:
                if c != 46:
                    new_sympy_shape.append(dims_value[label_maps[c]])
        else:
            for i in range(num_ellipsis_indices):
                new_sympy_shape.append(ellipsis_dims_value[i])
            for label, idx in label_maps.items():
                if label not in repeated_labels:
                    new_sympy_shape.append(dims_value[idx])

        output_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_sympy_shape))


register_shape_handler(EinsumHandler())
