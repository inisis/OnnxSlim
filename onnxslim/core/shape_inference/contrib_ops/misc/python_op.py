# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for PythonOp operator."""

import onnx
from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_sympy_shape


class PythonOpHandler(ShapeHandler):
    """Handler for PythonOp operator."""

    @property
    def op_type(self) -> str:
        return "PythonOp"

    def infer_shape(self, node, ctx) -> None:
        output_tensor_types = get_attribute(node, "output_tensor_types")
        assert output_tensor_types, f"PythonOp '{node.name}' has no output_tensor_types attribute."
        output_tensor_ranks = get_attribute(node, "output_tensor_ranks")
        assert output_tensor_ranks, f"PythonOp '{node.name}' has no output_tensor_ranks attribute."

        try:
            from onnxruntime.capi._pybind_state import get_shape_inference_function

            func_name = get_attribute(node, "func_name").decode()
            shape_inferer = get_shape_inference_function(func_name)
        except ImportError:
            shape_inferer = None

        # Set the context output separately.
        # The first output is torch.autograd.Function's context.
        vi = ctx.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], onnx.TensorProto.INT64, []))

        if shape_inferer is not None:
            input_shapes = []
            input_dtypes = []
            for input_index in range(len(node.input)):
                shape = ctx.get_shape(node, input_index)
                input_shapes.append(shape)
                input_dtype = ctx.known_vi_[node.input[input_index]].type.tensor_type.elem_type
                input_dtypes.append(input_dtype)
            output_shapes, output_dtypes = shape_inferer(node, input_shapes, input_dtypes)
            assert len(output_shapes) == len(output_dtypes) == (len(node.output) - 1), (
                f"PythonOp '{func_name}' returned {len(output_shapes)} shapes and {len(output_dtypes)} dtypes, "
                f"but expected {len(node.output) - 1} outputs."
            )
            for i in range(len(node.output) - 1):
                output_index = i + 1
                vi = ctx.known_vi_[node.output[output_index]]
                vi.CopyFrom(helper.make_tensor_value_info(node.output[output_index], output_dtypes[i], output_shapes[i]))
        else:
            # General shape inference for PythonOp.
            for i in range(len(node.output) - 1):
                vi = ctx.known_vi_[node.output[i + 1]]
                sympy_shape = ctx.new_symbolic_shape(output_tensor_ranks[i], node)
                shape = get_shape_from_sympy_shape(sympy_shape)
                value_info = helper.make_tensor_value_info(node.output[i + 1], output_tensor_types[i], shape)
                vi.CopyFrom(value_info)


register_shape_handler(PythonOpHandler())
