# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InferenceContext class for managing shape inference state."""

import logging

import numpy as np
import onnx
import sympy
from onnx import helper, numpy_helper, shape_inference

from onnxslim.third_party._sympy.functions import FloorDiv
from onnxslim.third_party._sympy.printers import PythonPrinter as _PythonPrinter

from .utils import (
    as_list,
    as_scalar,
    get_attribute,
    get_elem_type_from_type_proto,
    get_opset,
    get_shape_from_sympy_shape,
    get_shape_from_type_proto,
    get_shape_from_value_info,
    handle_negative_axis,
    is_literal,
    is_sequence,
    make_named_value_info,
)

logger = logging.getLogger(__name__)


class PythonPrinter(_PythonPrinter):
    """Custom Python printer for sympy expressions."""

    def doprint(self, expr: sympy.Expr, *, simplify: bool = True, p: bool = True) -> str:
        return super().doprint(expr)


pexpr = PythonPrinter().doprint


class InferenceContext:
    """Context object that encapsulates all state for shape inference.

    This class provides access to:
    - Known value info (known_vi_)
    - Symbolic dimensions (symbolic_dims_)
    - Sympy computed data (sympy_data_)
    - Initializers (initializers_)
    - Graph inputs (graph_inputs_)
    - Model opset and other configuration
    """

    def __init__(
        self,
        out_mp,
        int_max=2**31 - 1,
        auto_merge=False,
        guess_output_rank=False,
        verbose=0,
        prefix="",
    ):
        """Initialize the inference context.

        Args:
            out_mp: The ONNX ModelProto being processed.
            int_max: Maximum value for unbounded integers.
            auto_merge: Whether to automatically merge conflicting dimensions.
            guess_output_rank: Whether to guess output rank from input.
            verbose: Logging verbosity level.
            prefix: Prefix for generated symbolic dimension names.
        """
        self.out_mp_ = out_mp
        self.int_max_ = int_max
        self.auto_merge_ = auto_merge
        self.guess_output_rank_ = guess_output_rank
        self.verbose_ = verbose
        self.prefix_ = prefix
        self.subgraph_id_ = 0

        # State that needs to be initialized
        self.known_vi_ = {}
        self.symbolic_dims_ = {}
        self.sympy_data_ = {}
        self.initializers_ = {}
        self.graph_inputs_ = {}
        self.input_symbols_ = set()
        self.suggested_merge_ = {}
        self.run_ = True

    @property
    def opset(self):
        """Get the ONNX opset version of the model."""
        return get_opset(self.out_mp_)

    def preprocess(self):
        """Initialize data structures from the model."""
        self.graph_inputs_ = {i.name: i for i in list(self.out_mp_.graph.input)}
        self.initializers_ = {i.name: i for i in self.out_mp_.graph.initializer}
        self.known_vi_ = {i.name: i for i in list(self.out_mp_.graph.input)}
        self.known_vi_.update(
            {
                i.name: helper.make_tensor_value_info(i.name, i.data_type, list(i.dims))
                for i in self.out_mp_.graph.initializer
            }
        )
        self.known_vi_.update({i.name: i for i in list(self.out_mp_.graph.output)})

    # Shape retrieval methods
    def get_shape(self, node, idx):
        """Retrieve the shape of a tensor from a node's inputs."""
        name = node.input[idx]
        if name in self.known_vi_:
            vi = self.known_vi_[name]
            return get_shape_from_value_info(vi)
        else:
            assert name in self.initializers_
            return list(self.initializers_[name].dims)

    def try_get_shape(self, node, idx):
        """Attempts to retrieve the shape of the input node at the specified index."""
        if idx > len(node.input) - 1:
            return None
        name = node.input[idx]
        if name in self.known_vi_:
            vi = self.known_vi_[name]
            return get_shape_from_value_info(vi)
        if name in self.initializers_:
            return list(self.initializers_[name].dims)
        return None

    def get_shape_rank(self, node, idx):
        """Return the rank (number of dimensions) of the input tensor."""
        return len(self.get_shape(node, idx))

    def get_sympy_shape(self, node, idx):
        """Return the symbolic shape dimensions using SymPy."""
        sympy_shape = []
        for d in self.get_shape(node, idx):
            if type(d) == str:
                sympy_shape.append(
                    self.symbolic_dims_[d]
                    if d in self.symbolic_dims_
                    else sympy.Symbol(d, integer=True, nonnegative=True)
                )
            else:
                assert None is not d
                sympy_shape.append(d)
        return sympy_shape

    # Value retrieval methods
    def get_value(self, node, idx):
        """Retrieve the value associated with a node's input index."""
        name = node.input[idx]
        assert name in self.sympy_data_ or name in self.initializers_
        return self.sympy_data_[name] if name in self.sympy_data_ else numpy_helper.to_array(self.initializers_[name])

    def try_get_value(self, node, idx):
        """Try to retrieve the value associated with a node's input index."""
        if idx >= len(node.input):
            return None
        name = node.input[idx]
        if name in self.sympy_data_ or name in self.initializers_:
            return self.get_value(node, idx)
        return None

    # Symbolic dimension management
    def new_symbolic_dim(self, prefix, dim):
        """Create and return a new symbolic dimension."""
        new_dim = f"{prefix}_d{dim}"
        if new_dim in self.suggested_merge_:
            v = self.suggested_merge_[new_dim]
            new_symbolic_dim = sympy.Integer(int(v)) if is_literal(v) else v
        else:
            new_symbolic_dim = sympy.Symbol(new_dim, integer=True, nonnegative=True)
            self.symbolic_dims_[new_dim] = new_symbolic_dim
        return new_symbolic_dim

    def new_symbolic_dim_from_output(self, node, out_idx=0, dim=0):
        """Generates a new symbolic dimension for a given node's output."""
        return self.new_symbolic_dim(
            f"{node.op_type}{self.prefix_}_{list(self.out_mp_.graph.node).index(node)}_o{out_idx}_",
            dim,
        )

    def new_symbolic_shape(self, rank, node, out_idx=0):
        """Generate a new symbolic shape for a node output based on its rank."""
        return [self.new_symbolic_dim_from_output(node, out_idx, i) for i in range(rank)]

    def update_computed_dims(self, new_sympy_shape):
        """Update dimensions in new_sympy_shape based on suggested merges."""
        for i, new_dim in enumerate(new_sympy_shape):
            if not is_literal(new_dim) and type(new_dim) != str:
                str_dim = pexpr(new_dim)
                if str_dim in self.suggested_merge_:
                    if not is_literal(self.suggested_merge_[str_dim]):
                        new_sympy_shape[i] = self.symbolic_dims_[self.suggested_merge_[str_dim]]
                elif str_dim not in self.symbolic_dims_:
                    self.symbolic_dims_[str_dim] = new_dim

    # Dimension merging
    def add_suggested_merge(self, symbols, apply=False):
        """Add suggested merges for input symbols."""
        assert all((type(s) == str and s in self.symbolic_dims_) or is_literal(s) for s in symbols)
        symbols = set(symbols)
        for k, v in self.suggested_merge_.items():
            if k in symbols:
                symbols.remove(k)
                symbols.add(v)
        map_to = None
        # if there is literal, map to it first
        for s in symbols:
            if is_literal(s):
                map_to = s
                break
        # when no literals, map to input symbolic dims, then existing symbolic dims
        if map_to is None:
            for s in symbols:
                if s in self.input_symbols_:
                    map_to = s
                    break
        if map_to is None:
            for s in symbols:
                if type(self.symbolic_dims_[s]) == sympy.Symbol:
                    map_to = s
                    break
        # when nothing to map to, use the shorter one
        if map_to is None:
            if self.verbose_ > 0:
                logger.warning(f"Potential unsafe merge between symbolic expressions: ({','.join(symbols)})")
            symbols_list = list(symbols)
            lens = [len(s) for s in symbols_list]
            map_to = symbols_list[lens.index(min(lens))]
            symbols.remove(map_to)

        for s in symbols:
            if s == map_to:
                continue
            if is_literal(map_to) and is_literal(s):
                assert int(map_to) == int(s)
            self.suggested_merge_[s] = int(map_to) if is_literal(map_to) else map_to
            for k, v in self.suggested_merge_.items():
                if v == s:
                    self.suggested_merge_[k] = map_to
        if apply and self.auto_merge_:
            self.apply_suggested_merge()

    def apply_suggested_merge(self, graph_input_only=False):
        """Applies suggested merges to graph dimensions."""
        if not self.suggested_merge_:
            return
        for i in list(self.out_mp_.graph.input) + ([] if graph_input_only else list(self.out_mp_.graph.value_info)):
            for d in i.type.tensor_type.shape.dim:
                if d.dim_param in self.suggested_merge_:
                    v = self.suggested_merge_[d.dim_param]
                    if is_literal(v):
                        d.dim_value = int(v)
                    else:
                        d.dim_param = v

    def merge_symbols(self, dims):
        """Merge dimension symbols, handling automatic merging and validation."""
        if any(type(d) != str for d in dims):
            if not self.auto_merge_:
                return None
            unique_dims = list(set(dims))
            is_int = [is_literal(d) for d in unique_dims]
            assert sum(is_int) <= 1
            if sum(is_int) == 1:
                int_dim = is_int.index(1)
                if self.verbose_ > 0:
                    logger.debug(
                        f"dim {unique_dims[:int_dim] + unique_dims[int_dim + 1 :]} has been merged with value {unique_dims[int_dim]}"
                    )
                self.check_merged_dims(unique_dims, allow_broadcast=False)
                return unique_dims[int_dim]
            else:
                if self.verbose_ > 0:
                    logger.debug(f"dim {unique_dims[1:]} has been merged with dim {unique_dims[0]}")
                return dims[0]
        if all(d == dims[0] for d in dims):
            return dims[0]
        merged = [self.suggested_merge_[d] if d in self.suggested_merge_ else d for d in dims]
        if all(d == merged[0] for d in merged):
            assert merged[0] in self.symbolic_dims_
            return merged[0]
        else:
            return None

    def check_merged_dims(self, dims, allow_broadcast=True):
        """Checks merged dimensions for consistency."""
        if allow_broadcast:
            dims = [d for d in dims if not (is_literal(d) and int(d) <= 1)]
        if any(d != dims[0] for d in dims):
            self.add_suggested_merge(dims, apply=True)

    # Broadcasting
    def broadcast_shapes(self, shape1, shape2):
        """Broadcast two shapes from right to left."""
        new_shape = []
        rank1 = len(shape1)
        rank2 = len(shape2)
        new_rank = max(rank1, rank2)
        for i in range(new_rank):
            dim1 = shape1[rank1 - 1 - i] if i < rank1 else 1
            dim2 = shape2[rank2 - 1 - i] if i < rank2 else 1
            if dim1 in [1, dim2]:
                new_dim = dim2
            elif dim2 == 1:
                new_dim = dim1
            else:
                new_dim = self.merge_symbols([dim1, dim2])
                if not new_dim:
                    if self.auto_merge_:
                        self.add_suggested_merge([dim1, dim2], apply=True)
                    else:
                        logger.warning(f"unsupported broadcast between {dim1!s} {dim2!s}")
            new_shape = [new_dim, *new_shape]
        return new_shape

    # Shape computations
    def compute_conv_pool_shape(self, node, channels_last=False):
        """Calculate the output shape of a convolutional or pooling layer."""
        sympy_shape = self.get_sympy_shape(node, 0)
        if len(node.input) > 1:
            W_shape = self.get_sympy_shape(node, 1)
            rank = len(W_shape) - 2
            kernel_shape = W_shape[-rank - 1 : -1] if channels_last else W_shape[-rank:]
            sympy_shape[3 if channels_last else 1] = W_shape[0]
        else:
            W_shape = None
            kernel_shape = get_attribute(node, "kernel_shape")
            rank = len(kernel_shape)

        assert len(sympy_shape) == rank + 2

        spatial_shape = sympy_shape[-rank - 1 : -1] if channels_last else sympy_shape[-rank:]
        is_symbolic_dims = [not is_literal(i) for i in spatial_shape]

        if not any(is_symbolic_dims):
            shape = get_shape_from_value_info(self.known_vi_[node.output[0]])
            if len(shape) > 0:
                assert len(sympy_shape) == len(shape)
                if channels_last:
                    sympy_shape[-rank - 1 : -1] = [sympy.Integer(d) for d in shape[-rank - 1 : -1]]
                else:
                    sympy_shape[-rank:] = [sympy.Integer(d) for d in shape[-rank:]]
                return sympy_shape

        dilations = get_attribute(node, "dilations", [1] * rank)
        strides = get_attribute(node, "strides", [1] * rank)
        effective_kernel_shape = [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]
        pads = get_attribute(node, "pads")
        if pads is None:
            pads = [0] * (2 * rank)
            auto_pad = get_attribute(node, "auto_pad", b"NOTSET").decode("utf-8")
            if auto_pad not in {"VALID", "NOTSET"}:
                try:
                    residual = [sympy.Mod(d, s) for d, s in zip(sympy_shape[-rank:], strides)]
                    total_pads = [
                        max(0, (k - s) if r == 0 else (k - r))
                        for k, s, r in zip(effective_kernel_shape, strides, residual)
                    ]
                except TypeError:
                    total_pads = [max(0, (k - s)) for k, s in zip(effective_kernel_shape, strides)]
            elif auto_pad == "VALID":
                total_pads = []
            else:
                total_pads = [0] * rank
        else:
            assert len(pads) == 2 * rank
            total_pads = [p1 + p2 for p1, p2 in zip(pads[:rank], pads[rank:])]

        ceil_mode = get_attribute(node, "ceil_mode", 0)
        for i in range(rank):
            effective_input_size = sympy_shape[-rank + i + (-1 if channels_last else 0)]
            if len(total_pads) > 0:
                effective_input_size = effective_input_size + total_pads[i]
            if ceil_mode:
                strided_kernel_positions = sympy.ceiling(
                    (effective_input_size - effective_kernel_shape[i]) / strides[i]
                )
            else:
                strided_kernel_positions = FloorDiv((effective_input_size - effective_kernel_shape[i]), strides[i])
            sympy_shape[-rank + i + (-1 if channels_last else 0)] = strided_kernel_positions + 1
        return sympy_shape

    def compute_matmul_shape(self, node, output_dtype=None):
        """Compute the output shape for a matrix multiplication operation."""
        lhs_shape = self.get_shape(node, 0)
        rhs_shape = self.get_shape(node, 1)
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        lhs_reduce_dim = 0
        rhs_reduce_dim = 0
        assert lhs_rank > 0 and rhs_rank > 0
        if lhs_rank == 1 and rhs_rank == 1:
            new_shape = []
        elif lhs_rank == 1:
            rhs_reduce_dim = -2
            new_shape = [*rhs_shape[:rhs_reduce_dim], rhs_shape[-1]]
        elif rhs_rank == 1:
            lhs_reduce_dim = -1
            new_shape = lhs_shape[:lhs_reduce_dim]
        else:
            lhs_reduce_dim = -1
            rhs_reduce_dim = -2
            new_shape = [
                *self.broadcast_shapes(lhs_shape[:-2], rhs_shape[:-2]),
                lhs_shape[-2],
                rhs_shape[-1],
            ]
        self.check_merged_dims(
            [lhs_shape[lhs_reduce_dim], rhs_shape[rhs_reduce_dim]],
            allow_broadcast=False,
        )
        if output_dtype is None:
            output_dtype = self.known_vi_[node.input[0]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[0], output_dtype, new_shape))

    # Value operations
    def get_int_or_float_values(self, node, broadcast=False, allow_float_values=False):
        """Extracts integer or float values from a node."""

        def int_or_float(value, allow_float_values):
            return value if allow_float_values and value % 1 != 0 else int(value)

        values = [self.try_get_value(node, i) for i in range(len(node.input))]
        if all(v is not None for v in values):
            for i, v in enumerate(values):
                if type(v) != np.ndarray:
                    continue
                if len(v.shape) > 1:
                    new_v = None
                elif len(v.shape) == 0:
                    new_v = int_or_float(v.item(), allow_float_values)
                else:
                    assert len(v.shape) == 1
                    new_v = [int_or_float(vv, allow_float_values) for vv in v]
                values[i] = new_v
        values_len = [len(v) if isinstance(v, list) else 0 for v in values]
        max_len = max(values_len)
        if max_len >= 1 and broadcast:
            for i, v in enumerate(values):
                if v is None:
                    continue
                if isinstance(v, list):
                    if len(v) < max_len:
                        values[i] = v * max_len
                    else:
                        assert len(v) == max_len
                else:
                    values[i] = [v] * max_len
        return values

    def compute_on_sympy_data(self, node, op_func):
        """Calculate the result using Sympy data and a specified operation function."""
        assert len(node.output) == 1

        if node.op_type in {"Mul", "Div"}:
            values = self.get_int_or_float_values(node, broadcast=True, allow_float_values=True)
        else:
            values = self.get_int_or_float_values(node, broadcast=True)
        if all(v is not None for v in values):
            is_list = [isinstance(v, list) for v in values]
            as_list = any(is_list)
            if as_list:
                self.sympy_data_[node.output[0]] = [op_func(vs) for vs in zip(*values)]
            else:
                self.sympy_data_[node.output[0]] = op_func(values)

    def pass_on_sympy_data(self, node):
        """Pass Sympy data through a node."""
        assert len(node.input) == 1 or node.op_type in {
            "Reshape",
            "Unsqueeze",
            "Squeeze",
        }
        self.compute_on_sympy_data(node, lambda x: x[0])

    # Shape propagation
    def pass_on_shape_and_type(self, node):
        """Propagates the shape and type information from input to output."""
        vi = self.known_vi_[node.output[0]]
        vi.CopyFrom(
            helper.make_tensor_value_info(
                node.output[0],
                get_elem_type_from_type_proto(self.known_vi_[node.input[0]].type),
                self.get_shape(node, 0),
            )
        )

    def propagate_shape_and_type(self, node, input_index=0, output_index=0):
        """Propagates the shape and type information from input to output tensors."""
        shape = self.get_shape(node, input_index)
        output_dtype = self.known_vi_[node.input[input_index]].type.tensor_type.elem_type
        vi = self.known_vi_[node.output[output_index]]
        vi.CopyFrom(helper.make_tensor_value_info(node.output[output_index], output_dtype, shape))

    def fuse_tensor_type(self, node, out_idx, dst_type, src_type):
        """Update dst_tensor_type to be compatible with src_tensor_type."""
        dst_tensor_type = (
            dst_type.sequence_type.elem_type.tensor_type if is_sequence(dst_type) else dst_type.tensor_type
        )
        src_tensor_type = (
            src_type.sequence_type.elem_type.tensor_type if is_sequence(src_type) else src_type.tensor_type
        )
        if dst_tensor_type.elem_type != src_tensor_type.elem_type:
            node_id = node.name or node.op_type
            raise ValueError(
                f"For node {node_id}, dst_tensor_type.elem_type != src_tensor_type.elem_type: "
                f"{onnx.onnx_pb.TensorProto.DataType.Name(dst_tensor_type.elem_type)} vs "
                f"{onnx.onnx_pb.TensorProto.DataType.Name(src_tensor_type.elem_type)}"
            )
        if dst_tensor_type.HasField("shape"):
            for di, ds in enumerate(zip(dst_tensor_type.shape.dim, src_tensor_type.shape.dim)):
                if ds[0] != ds[1]:
                    new_dim = onnx.TensorShapeProto.Dimension()
                    if not is_sequence(dst_type):
                        new_dim.dim_param = str(self.new_symbolic_dim_from_output(node, out_idx, di))
                    dst_tensor_type.shape.dim[di].CopyFrom(new_dim)
        else:
            dst_tensor_type.CopyFrom(src_tensor_type)

    # ONNX inference helpers
    def onnx_infer_single_node(self, node):
        """Performs ONNX shape inference for a single node."""
        skip_infer = node.op_type in {
            "If",
            "Loop",
            "Scan",
            "SplitToSequence",
            "ZipMap",
            "Attention",
            "BiasGelu",
            "EmbedLayerNormalization",
            "FastGelu",
            "Gelu",
            "GemmFastGelu",
            "LayerNormalization",
            "LongformerAttention",
            "DequantizeLinear",
            "QuantizeLinear",
            "RelativePositionBias",
            "RemovePadding",
            "RestorePadding",
            "SimplifiedLayerNormalization",
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "PackedAttention",
            "PythonOp",
            "MultiHeadAttention",
            "GroupNorm",
            "SkipGroupNorm",
            "BiasSplitGelu",
            "BiasAdd",
            "NhwcConv",
            "QuickGelu",
            "RotaryEmbedding",
        }

        if not skip_infer:
            initializers = []
            if (get_opset(self.out_mp_) >= 9) and (
                node.op_type == "Unsqueeze"
                or node.op_type == "ReduceMax"
                or node.op_type == "ReduceMean"
                or node.op_type == "DFT"
                or node.op_type == "ReduceL2"
                or node.op_type == "ReduceMin"
            ):
                initializers = [
                    self.initializers_[name]
                    for name in node.input
                    if (name in self.initializers_ and name not in self.graph_inputs_)
                ]

            if (
                node.op_type
                in {
                    "Add",
                    "Sub",
                    "Mul",
                    "Div",
                    "MatMul",
                    "MatMulInteger",
                    "MatMulInteger16",
                    "Where",
                    "Sum",
                }
                and node.output[0] in self.known_vi_
            ):
                vi = self.known_vi_[node.output[0]]
                out_rank = len(get_shape_from_type_proto(vi.type))
                in_shapes = [self.get_shape(node, i) for i in range(len(node.input))]
                for d in range(out_rank - (2 if node.op_type in {"MatMul", "MatMulInteger", "MatMulInteger16"} else 0)):
                    in_dims = [s[len(s) - out_rank + d] for s in in_shapes if len(s) + d >= out_rank]
                    if len(in_dims) > 1:
                        self.check_merged_dims(in_dims, allow_broadcast=True)

            tmp_graph = helper.make_graph(
                [node],
                "tmp",
                [self.known_vi_[i] for i in node.input if i],
                [make_named_value_info(i) for i in node.output],
                initializers,
            )

            kwargs = {}
            kwargs["opset_imports"] = self.out_mp_.opset_import
            kwargs["ir_version"] = self.out_mp_.ir_version

            model = helper.make_model(tmp_graph, **kwargs)
            model = shape_inference.infer_shapes(model)

        for i_o in range(len(node.output)):
            o = node.output[i_o]
            if o:
                out = model.graph.output[i_o]
                if not out.type.WhichOneof("value") and o in self.known_vi_:
                    continue

                vi = self.out_mp_.graph.value_info.add()
                if not skip_infer:
                    vi.CopyFrom(out)
                else:
                    vi.name = o
                self.known_vi_[o] = vi

    # Helper methods for checking none dims
    def is_none_dim(self, dim_value):
        """Check if dimension value is unknown."""
        if type(dim_value) != str:
            return False
        return dim_value not in self.symbolic_dims_ if "unk__" in dim_value else False

    def is_shape_contains_none_dim(self, out_shape):
        """Check if any dimension in the given shape is unknown."""
        for out in out_shape:
            if self.is_none_dim(out):
                return out
        return None
