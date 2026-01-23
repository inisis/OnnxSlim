# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for If operator."""

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import as_scalar, get_attribute


class IfHandler(ShapeHandler):
    """Handler for If operator."""

    @property
    def op_type(self) -> str:
        return "If"

    def infer_shape(self, node, ctx) -> None:
        subgraphs = [
            get_attribute(node, "then_branch"),
            get_attribute(node, "else_branch"),
        ]

        cond = ctx.try_get_value(node, 0)

        for i_sub, subgraph in enumerate(subgraphs):
            subgraph_infer = ctx.onnx_infer_subgraph(node, subgraph, use_node_input=False)
            for i_out in range(len(node.output)):
                vi = ctx.known_vi_[node.output[i_out]]
                if i_sub == 0:
                    vi.CopyFrom(subgraph.output[i_out])
                    vi.name = node.output[i_out]
                else:
                    ctx.fuse_tensor_type(node, i_out, vi.type, subgraph.output[i_out].type)
                if (
                    cond is not None
                    and i_sub == (0 if as_scalar(cond) > 0 else 1)
                    and subgraph.output[i_out].name in subgraph_infer.sympy_data_
                ):
                    ctx.sympy_data_[vi.name] = subgraph_infer.sympy_data_[subgraph.output[i_out].name]


register_shape_handler(IfHandler())
