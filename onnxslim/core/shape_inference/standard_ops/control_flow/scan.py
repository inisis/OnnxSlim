# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Scan operator."""

from onnx import helper

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_type_proto, handle_negative_axis


class ScanHandler(ShapeHandler):
    """Handler for Scan operator."""

    @property
    def op_type(self) -> str:
        return "Scan"

    def infer_shape(self, node, ctx) -> None:
        subgraph = get_attribute(node, "body")
        num_scan_inputs = get_attribute(node, "num_scan_inputs")
        scan_input_axes = get_attribute(node, "scan_input_axes", [0] * num_scan_inputs)
        num_scan_states = len(node.input) - num_scan_inputs
        scan_input_axes = [
            handle_negative_axis(ax, ctx.get_shape_rank(node, i + num_scan_states)) for i, ax in enumerate(scan_input_axes)
        ]

        assert len(subgraph.input) >= len(node.input)
        subgraph_inputs = subgraph.input[: len(node.input)]
        for i, si in enumerate(subgraph_inputs):
            subgraph_name = si.name
            si.CopyFrom(ctx.known_vi_[node.input[i]])
            if i >= num_scan_states:
                scan_input_dim = si.type.tensor_type.shape.dim
                scan_input_dim.remove(scan_input_dim[scan_input_axes[i - num_scan_states]])
            si.name = subgraph_name
        ctx.onnx_infer_subgraph(node, subgraph)
        num_scan_outputs = len(node.output) - num_scan_states
        scan_output_axes = get_attribute(node, "scan_output_axes", [0] * num_scan_outputs)
        scan_input_dim = get_shape_from_type_proto(ctx.known_vi_[node.input[-1]].type)[scan_input_axes[-1]]
        for i, o in enumerate(node.output):
            vi = ctx.known_vi_[o]
            if i >= num_scan_states:
                shape = get_shape_from_type_proto(subgraph.output[i].type)
                new_dim = handle_negative_axis(scan_output_axes[i - num_scan_states], len(shape) + 1)
                shape = [*shape[:new_dim], scan_input_dim, *shape[new_dim:]]
                vi.CopyFrom(helper.make_tensor_value_info(o, subgraph.output[i].type.tensor_type.elem_type, shape))
            else:
                vi.CopyFrom(subgraph.output[i])
            vi.name = o


register_shape_handler(ScanHandler())
