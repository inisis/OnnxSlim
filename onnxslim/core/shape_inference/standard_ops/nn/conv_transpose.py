# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for ConvTranspose operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_sympy_shape


class ConvTransposeHandler(ShapeHandler):
    """Handler for ConvTranspose operator."""

    @property
    def op_type(self) -> str:
        return "ConvTranspose"

    def infer_shape(self, node, ctx) -> None:
        input_shape = ctx.get_sympy_shape(node, 0)
        weight_shape = ctx.get_sympy_shape(node, 1)
        rank = len(input_shape) - 2
        assert rank > 0 and len(weight_shape) == len(input_shape)

        group = get_attribute(node, "group", 1)
        output_shape = [input_shape[0], weight_shape[1] * group]
        spatial_shape = get_attribute(node, "output_shape")
        if spatial_shape is None:
            strides = get_attribute(node, "strides", [1] * rank)
            assert len(strides) == rank
            auto_pad = get_attribute(node, "auto_pad", b"NOTSET")
            auto_pad = (
                auto_pad.decode("utf-8") if isinstance(auto_pad, bytes) else auto_pad
            )
            if auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
                spatial_shape = [
                    dim * stride for dim, stride in zip(input_shape[2:], strides)
                ]
            else:
                kernel_shape = get_attribute(node, "kernel_shape", weight_shape[2:])
                dilations = get_attribute(node, "dilations", [1] * rank)
                output_padding = get_attribute(node, "output_padding", [0] * rank)
                pads = get_attribute(node, "pads", [0] * (2 * rank))
                assert (
                    len(kernel_shape) == len(dilations) == len(output_padding) == rank
                )
                assert len(pads) == 2 * rank
                spatial_shape = [
                    stride * (dim - 1)
                    + output_pad
                    + (kernel - 1) * dilation
                    + 1
                    - pad_start
                    - pad_end
                    for dim, stride, output_pad, kernel, dilation, pad_start, pad_end in zip(
                        input_shape[2:],
                        strides,
                        output_padding,
                        kernel_shape,
                        dilations,
                        pads[:rank],
                        pads[rank:],
                    )
                ]

        assert len(spatial_shape) == rank
        output_shape.extend(spatial_shape)
        ctx.update_computed_dims(output_shape)
        output_type = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
        ctx.known_vi_[node.output[0]].CopyFrom(
            helper.make_tensor_value_info(
                node.output[0], output_type, get_shape_from_sympy_shape(output_shape)
            )
        )


register_shape_handler(ConvTransposeHandler())
