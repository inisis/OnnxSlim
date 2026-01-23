# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Loop operator."""

import logging

import onnx

from ...base import ShapeHandler
from ...registry import register_shape_handler
from ...utils import get_attribute, get_shape_from_value_info, is_sequence

logger = logging.getLogger(__name__)


class LoopHandler(ShapeHandler):
    """Handler for Loop operator."""

    @property
    def op_type(self) -> str:
        return "Loop"

    def infer_shape(self, node, ctx) -> None:
        subgraph = get_attribute(node, "body")
        assert len(subgraph.input) == len(node.input)
        num_loop_carried = len(node.input) - 2

        for i, si in enumerate(subgraph.input):
            si_name = si.name
            si.CopyFrom(ctx.known_vi_[node.input[i]])
            si.name = si_name

        ctx.onnx_infer_subgraph(node, subgraph)

        need_second_infer = False
        for i_out in range(1, num_loop_carried + 1):
            so = subgraph.output[i_out]
            so_shape = get_shape_from_value_info(so)
            if is_sequence(so.type):
                if so_shape and None in so_shape:
                    subgraph.input[i_out + 1].type.sequence_type.elem_type.CopyFrom(so.type.sequence_type.elem_type)
                    need_second_infer = True
            else:
                si = subgraph.input[i_out + 1]
                si_shape = get_shape_from_value_info(si)
                for di, dims in enumerate(zip(si_shape, so_shape)):
                    if dims[0] != dims[1]:
                        new_dim = onnx.TensorShapeProto.Dimension()
                        new_dim.dim_param = str(ctx.new_symbolic_dim_from_output(node, i_out, di))
                        si.type.tensor_type.shape.dim[di].CopyFrom(new_dim)
                        so.type.tensor_type.shape.dim[di].CopyFrom(new_dim)
                        need_second_infer = True

        if need_second_infer:
            if ctx.verbose_ > 2:
                logger.debug(f"Rerun Loop: {node.name}({node.output[0]}...), because of sequence in loop carried variables")
            ctx.onnx_infer_subgraph(node, subgraph, inc_subgraph_id=False)

        loop_iter_dim = str(ctx.new_symbolic_dim_from_output(node))
        for i in range(len(node.output)):
            vi = ctx.known_vi_[node.output[i]]
            vi.CopyFrom(subgraph.output[i + 1])
            if i >= num_loop_carried:
                assert not is_sequence(vi.type)
                subgraph_vi_dim = subgraph.output[i + 1].type.tensor_type.shape.dim
                vi.type.tensor_type.shape.ClearField("dim")
                vi_dim = vi.type.tensor_type.shape.dim
                vi_dim.add().dim_param = loop_iter_dim
                vi_dim.extend(list(subgraph_vi_dim))
            vi.name = node.output[i]


register_shape_handler(LoopHandler())
