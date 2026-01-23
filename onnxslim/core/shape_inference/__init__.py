# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Symbolic Shape Inference Module

This module provides symbolic shape inference for ONNX models. It replaces the
monolithic SymbolicShapeInference class with a modular, handler-based architecture.

Usage:
    from onnxslim.core.shape_inference import ShapeInferencer

    model = onnx.load("model.onnx")
    model_with_shapes = ShapeInferencer.infer_shapes(model)
"""

import logging

import onnx
import sympy
from onnx import helper

from .context import InferenceContext
from .registry import get_all_aten_handlers, get_all_shape_handlers, get_aten_handler, get_shape_handler
from .utils import (
    get_attribute,
    get_opset,
    get_shape_from_type_proto,
    get_shape_from_value_info,
    is_literal,
    is_sequence,
)

# Import all handlers to trigger registration
from . import aten_ops  # noqa: F401
from . import contrib_ops  # noqa: F401
from . import standard_ops  # noqa: F401

logger = logging.getLogger(__name__)


class ShapeInferencer:
    """Main class for performing symbolic shape inference on ONNX models."""

    def __init__(self, int_max=2**31 - 1, auto_merge=False, guess_output_rank=False, verbose=0, prefix=""):
        """Initialize the ShapeInferencer.

        Args:
            int_max: Maximum value for unbounded integers.
            auto_merge: Whether to automatically merge conflicting dimensions.
            guess_output_rank: Whether to guess output rank from input.
            verbose: Logging verbosity level.
            prefix: Prefix for generated symbolic dimension names.
        """
        self.int_max_ = int_max
        self.auto_merge_ = auto_merge
        self.guess_output_rank_ = guess_output_rank
        self.verbose_ = verbose
        self.prefix_ = prefix

    def _infer_impl(self, ctx, start_sympy_data=None):
        """Main inference implementation loop."""
        ctx.sympy_data_ = start_sympy_data or {}
        ctx.apply_suggested_merge(graph_input_only=True)
        ctx.input_symbols_ = set()

        # Process graph inputs
        for i in ctx.out_mp_.graph.input:
            input_shape = get_shape_from_value_info(i)
            if input_shape is None:
                continue

            if is_sequence(i.type):
                input_dims = i.type.sequence_type.elem_type.tensor_type.shape.dim
            else:
                input_dims = i.type.tensor_type.shape.dim

            for i_dim, dim in enumerate(input_shape):
                if dim is None:
                    input_dims[i_dim].dim_param = str(ctx.new_symbolic_dim(i.name, i_dim))

            ctx.input_symbols_.update([d for d in input_shape if type(d) == str])

        for s in ctx.input_symbols_:
            if s in ctx.suggested_merge_:
                s_merge = ctx.suggested_merge_[s]
                assert s_merge in ctx.symbolic_dims_
                ctx.symbolic_dims_[s] = ctx.symbolic_dims_[s_merge]
            else:
                ctx.symbolic_dims_[s] = sympy.Symbol(s, integer=True, positive=True)

        # Compute prerequisite for node for topological sort
        prereq_for_node = {}

        def get_prereq(node):
            names = {i for i in node.input if i}
            subgraphs = []
            if node.op_type == "If":
                subgraphs = [get_attribute(node, "then_branch"), get_attribute(node, "else_branch")]
            elif node.op_type in {"Loop", "Scan"}:
                subgraphs = [get_attribute(node, "body")]
            for g in subgraphs:
                g_outputs_and_initializers = {i.name for i in g.initializer}
                g_prereq = set()
                for n in g.node:
                    g_outputs_and_initializers.update(n.output)
                for n in g.node:
                    g_prereq.update([i for i in get_prereq(n) if i not in g_outputs_and_initializers])
                names.update(g_prereq)
                for i in g.input:
                    if i.name in names:
                        names.remove(i.name)
            return names

        for n in ctx.out_mp_.graph.node:
            prereq_for_node[n.output[0]] = get_prereq(n)

        # Topological sort nodes
        sorted_nodes = []
        sorted_known_vi = {i.name for i in list(ctx.out_mp_.graph.input) + list(ctx.out_mp_.graph.initializer)}
        if any(o.name in sorted_known_vi for o in ctx.out_mp_.graph.output):
            sorted_nodes = ctx.out_mp_.graph.node
        else:
            while any(o.name not in sorted_known_vi for o in ctx.out_mp_.graph.output):
                old_sorted_nodes_len = len(sorted_nodes)
                for node in ctx.out_mp_.graph.node:
                    if node.output[0] not in sorted_known_vi and all(
                        i in sorted_known_vi for i in prereq_for_node[node.output[0]] if i
                    ):
                        sorted_known_vi.update(node.output)
                        sorted_nodes.append(node)
                if old_sorted_nodes_len == len(sorted_nodes) and not all(
                    o.name in sorted_known_vi for o in ctx.out_mp_.graph.output
                ):
                    raise Exception("Invalid model with cyclic graph")

        # Get handlers
        shape_handlers = get_all_shape_handlers()
        aten_handlers = get_all_aten_handlers()

        # Process each node
        for node in sorted_nodes:
            assert all([i in ctx.known_vi_ for i in node.input if i])
            ctx.onnx_infer_single_node(node)
            known_aten_op = False

            # Try standard handlers first
            handler = get_shape_handler(node.op_type)
            if handler is not None:
                handler.infer_shape(node, ctx)
            elif node.op_type == "ConvTranspose":
                vi = ctx.known_vi_[node.output[0]]
                if len(vi.type.tensor_type.shape.dim) == 0:
                    vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
            elif node.op_type == "ATen" and node.domain == "org.pytorch.aten":
                for attr in node.attribute:
                    if attr.name == "operator":
                        aten_op_name = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                        aten_handler = get_aten_handler(aten_op_name)
                        if aten_handler is not None:
                            known_aten_op = True
                            aten_handler.infer_shape(node, ctx)
                        break

            if ctx.verbose_ > 2:
                logger.debug(node.op_type + ": " + node.name)
                for i, name in enumerate(node.input):
                    logger.debug(f"  Input {i}: {name} {'initializer' if name in ctx.initializers_ else ''}")

            # Handle dimension merging for broadcast ops
            if node.op_type in {
                "Add",
                "Sub",
                "Mul",
                "Div",
                "MatMul",
                "MatMulInteger",
                "MatMulInteger16",
                "Where",
                "Sum",
            }:
                vi = ctx.known_vi_[node.output[0]]
                out_rank = len(get_shape_from_type_proto(vi.type))
                in_shapes = [ctx.get_shape(node, i) for i in range(len(node.input))]
                for d in range(out_rank - (2 if node.op_type in {"MatMul", "MatMulInteger", "MatMulInteger16"} else 0)):
                    in_dims = [s[len(s) - out_rank + d] for s in in_shapes if len(s) + d >= out_rank]
                    if len(in_dims) > 1:
                        ctx.check_merged_dims(in_dims, allow_broadcast=True)

            # Process outputs
            for i_o in range(len(node.output)):
                if node.op_type in {"SkipLayerNormalization", "SkipSimplifiedLayerNormalization"} and i_o in {1, 2}:
                    continue
                if node.op_type == "RotaryEmbedding" and len(node.output) > 1:
                    continue

                vi = ctx.known_vi_[node.output[i_o]]
                out_type = vi.type
                out_type_kind = out_type.WhichOneof("value")

                if out_type_kind not in {"tensor_type", "sparse_tensor_type", None}:
                    if ctx.verbose_ > 2:
                        if out_type_kind == "sequence_type":
                            seq_cls_type = out_type.sequence_type.elem_type.WhichOneof("value")
                            if seq_cls_type == "tensor_type":
                                logger.debug(
                                    f"  {node.output[i_o]}: sequence of {str(get_shape_from_value_info(vi))} "
                                    f"{onnx.TensorProto.DataType.Name(vi.type.sequence_type.elem_type.tensor_type.elem_type)}"
                                )
                            else:
                                logger.debug(f"  {node.output[i_o]}: sequence of {seq_cls_type}")
                        else:
                            logger.debug(f"  {node.output[i_o]}: {out_type_kind}")
                    continue

                out_shape = get_shape_from_value_info(vi)
                out_type_undefined = out_type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED
                if ctx.verbose_ > 2:
                    logger.debug(
                        f"  {node.output[i_o]}: {out_shape!s} {onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)}"
                    )
                    if node.output[i_o] in ctx.sympy_data_:
                        logger.debug("  Sympy Data: " + str(ctx.sympy_data_[node.output[i_o]]))

                if (out_shape is not None and (None in out_shape or ctx.is_shape_contains_none_dim(out_shape))) or out_type_undefined:
                    if ctx.auto_merge_:
                        if node.op_type in {
                            "Add",
                            "Sub",
                            "Mul",
                            "Div",
                            "MatMul",
                            "MatMulInteger",
                            "MatMulInteger16",
                            "Concat",
                            "Where",
                            "Sum",
                            "Equal",
                            "Less",
                            "Greater",
                            "LessOrEqual",
                            "GreaterOrEqual",
                            "Min",
                            "Max",
                        }:
                            shapes = [ctx.get_shape(node, i) for i in range(len(node.input))]
                            if node.op_type in {"MatMul", "MatMulInteger", "MatMulInteger16"} and (
                                None in out_shape or ctx.is_shape_contains_none_dim(out_shape)
                            ):
                                if None in out_shape:
                                    idx = out_shape.index(None)
                                else:
                                    idx = out_shape.index(ctx.is_shape_contains_none_dim(out_shape))
                                dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                                assert len(shapes[0]) > 2 and dim_idx[0] < len(shapes[0]) - 2
                                assert len(shapes[1]) > 2 and dim_idx[1] < len(shapes[1]) - 2
                        elif node.op_type == "Expand":
                            shapes = [ctx.get_shape(node, 0), ctx.get_value(node, 1)]
                        else:
                            shapes = []

                        if shapes:
                            for idx in range(len(out_shape)):
                                if out_shape[idx] is not None and not ctx.is_none_dim(out_shape[idx]):
                                    continue
                                dim_idx = [len(s) - len(out_shape) + idx for s in shapes]
                                if dim_idx:
                                    ctx.add_suggested_merge(
                                        [s[i] if is_literal(s[i]) else str(s[i]) for s, i in zip(shapes, dim_idx) if i >= 0]
                                    )
                            ctx.run_ = True
                        else:
                            ctx.run_ = False
                    else:
                        ctx.run_ = False

                    if not ctx.run_ and handler is None and not known_aten_op:
                        is_unknown_op = out_type_undefined and (out_shape is None or len(out_shape) == 0)
                        if is_unknown_op:
                            out_rank = ctx.get_shape_rank(node, 0) if ctx.guess_output_rank_ else -1
                        else:
                            out_rank = len(out_shape)

                        if out_rank >= 0:
                            new_shape = ctx.new_symbolic_shape(out_rank, node, i_o)
                            if out_type_undefined:
                                out_dtype = ctx.known_vi_[node.input[0]].type.tensor_type.elem_type
                            else:
                                out_dtype = vi.type.tensor_type.elem_type
                            from .utils import get_shape_from_sympy_shape

                            vi.CopyFrom(
                                helper.make_tensor_value_info(vi.name, out_dtype, get_shape_from_sympy_shape(new_shape))
                            )

                            if ctx.verbose_ > 0:
                                if is_unknown_op:
                                    logger.debug(f"Possible unknown op: {node.op_type} node: {node.name}, guessing {vi.name} shape")
                                if ctx.verbose_ > 2:
                                    logger.debug(f"  {node.output[i_o]}: {new_shape!s} {vi.type.tensor_type.elem_type}")
                            ctx.run_ = True
                            continue

                    if ctx.verbose_ > 0 or not ctx.auto_merge_ or out_type_undefined:
                        logger.debug("Stopping at incomplete shape inference at " + node.op_type + ": " + node.name)
                        logger.debug("node inputs:")
                        for i in node.input:
                            if i in ctx.known_vi_:
                                logger.debug(ctx.known_vi_[i])
                            else:
                                logger.debug(f"not in known_vi_ for {i}")
                        logger.debug("node outputs:")
                        for o in node.output:
                            if o in ctx.known_vi_:
                                logger.debug(ctx.known_vi_[o])
                            else:
                                logger.debug(f"not in known_vi_ for {o}")
                        if ctx.auto_merge_ and not out_type_undefined:
                            logger.debug("Merging: " + str(ctx.suggested_merge_))
                    return False

        ctx.run_ = False
        return True

    def _update_output_from_vi(self, ctx):
        """Update output attributes using known value information dictionary."""
        for output in ctx.out_mp_.graph.output:
            if output.name in ctx.known_vi_:
                output.CopyFrom(ctx.known_vi_[output.name])

    @staticmethod
    def infer_shapes(in_mp, int_max=2**31 - 1, auto_merge=False, guess_output_rank=False, verbose=0):
        """Perform symbolic shape inference on an ONNX model.

        Args:
            in_mp: The input ONNX ModelProto.
            int_max: Maximum value for unbounded integers.
            auto_merge: Whether to automatically merge conflicting dimensions.
            guess_output_rank: Whether to guess output rank from input.
            verbose: Logging verbosity level.

        Returns:
            The model with inferred shapes.

        Raises:
            Exception: If shape inference is incomplete.
        """
        onnx_opset = get_opset(in_mp)
        if (not onnx_opset) or onnx_opset < 7:
            logger.warning("Only support models of onnx opset 7 and above.")
            return None

        inferencer = ShapeInferencer(int_max, auto_merge, guess_output_rank, verbose)

        # Create inference context
        ctx = InferenceContext(
            in_mp,
            int_max=int_max,
            auto_merge=auto_merge,
            guess_output_rank=guess_output_rank,
            verbose=verbose,
        )
        ctx.preprocess()

        all_shapes_inferred = False
        while ctx.run_:
            all_shapes_inferred = inferencer._infer_impl(ctx)

        inferencer._update_output_from_vi(ctx)

        if not all_shapes_inferred:
            raise Exception("Incomplete symbolic shape inference")

        return ctx.out_mp_


# For backward compatibility
SymbolicShapeInference = ShapeInferencer
