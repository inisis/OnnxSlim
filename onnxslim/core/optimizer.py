import contextlib
from collections import Counter, OrderedDict
from typing import List, Union

import numpy as np
import onnx

import onnxslim.onnx_graphsurgeon as gs
from onnxslim.onnx_graphsurgeon.exporters.onnx_exporter import dtype_to_onnx
from onnxslim.onnx_graphsurgeon.ir.graph import Graph
from onnxslim.onnx_graphsurgeon.ir.tensor import Constant, Variable
from onnxslim.utils import logger

DEFAULT_FUSION_PATTERNS = OrderedDict()


def register_fusion_pattern(layer_type):
    """Registers a fusion pattern function for a specified layer type in the DEFAULT_FUSION_PATTERNS dictionary."""

    def insert(fn):
        if layer_type in DEFAULT_FUSION_PATTERNS.keys():
            raise
        DEFAULT_FUSION_PATTERNS[layer_type] = fn
        return fn

    return insert


def get_fusion_patterns(skip_fusion_patterns: str = None):
    """Returns a copy of the default fusion patterns, optionally excluding specific patterns."""
    default_fusion_patterns = DEFAULT_FUSION_PATTERNS.copy()
    if skip_fusion_patterns:
        for pattern in skip_fusion_patterns:
            default_fusion_patterns.pop(pattern)

    return default_fusion_patterns


def get_node_users(node):
    """Retrieve the list of nodes that use the outputs of the given node."""
    users = []
    for output in node.outputs:  # output is a Variable
        users.extend(iter(output.outputs))
    return users


def get_node_feeds(node):
    """Retrieve the list of nodes that provide inputs to the given node."""
    feeds = []
    for input in node.inputs:  # input is a Variable
        feeds.extend(iter(input.inputs))
    return feeds


def get_previous_node_by_type(node, op_type, trajectory=None):
    """Recursively find and return the first preceding node of a specified type in the computation graph."""
    if trajectory is None:
        trajectory = []
    node_feeds = get_node_feeds(node)
    for node_feed in node_feeds:
        if node_feed.op == op_type:
            trajectory.append(node_feed)
            return trajectory
        else:
            trajectory.append(node_feed)
            return get_previous_node_by_type(node_feed, op_type, trajectory)


def get_constant_variable(node, return_idx=False):
    """Return the first constant variable found in a node's inputs, optionally including the index."""
    for idx, input in enumerate(list(node.inputs)):
        if isinstance(input, Constant):
            return (idx, input) if return_idx else input


def delete_node(node, input_var_idx=0, output_var_idx=0):
    """Delete a node from the computation graph while re-linking its input and output to maintain graph integrity."""
    input_variable = node.inputs[input_var_idx]
    node_variable = node.outputs[output_var_idx]
    next_nodes = get_node_users(node)
    if next_nodes:
        for next_node in next_nodes:
            index = next_node.inputs.index(node_variable)
            next_node.inputs.pop(index)
            next_node.inputs.insert(index, input_variable)
    else:
        input_node = node.i()
        input_node.outputs.remove(node.inputs[input_var_idx])
        input_node.outputs.append(node.outputs[output_var_idx])
        node.outputs.clear()


def check_shape(shapes):
    """Verify that 'shapes' contains exactly one string and all other elements are positive integers."""
    string_count = 0
    non_negative_int_count = 0

    for item in shapes:
        if isinstance(item, str):
            string_count += 1
        elif isinstance(item, int) and item > 0:
            non_negative_int_count += 1

    return string_count == 1 and non_negative_int_count == len(shapes) - 1


def graph_constant_fold_inplace(graph):
    """Perform in-place constant folding optimizations on the given computational graph by eliminating redundant
    nodes.
    """
    for subgraph in graph.subgraphs():
        graph_constant_fold_inplace(subgraph)

    for node in graph.nodes:
        if node.op in ["Identity", "Dropout"]:
            delete_node(node)

        elif node.op == "Pad":
            if len(node.inputs) > 1 and isinstance(node.inputs[1], Constant):
                pad_value = node.inputs[1].values.tolist()
                pad_value = pad_value if isinstance(pad_value, list) else [pad_value]
                if all(value == 0 for value in pad_value):
                    delete_node(node)
        elif node.op == "Cast":
            inp_dtype = [dtype_to_onnx(input.dtype) for input in node.inputs][0]
            if inp_dtype == node.attrs["to"]:
                delete_node(node)
        elif node.op == "Reshape":
            if (node.inputs[0].shape and len(node.inputs[0].shape) == 1) and (
                node.outputs[0].shape and len(node.outputs[0].shape) == 1
            ):
                delete_node(node)
            else:
                node_output_shape = node.outputs[0].shape
                if node_output_shape and check_shape(node_output_shape):
                    shapes = [shape if isinstance(shape, int) else -1 for shape in node_output_shape]
                    reshape_const = gs.Constant(
                        f"{node.inputs[1].name}_",
                        values=np.array(shapes, dtype=np.int64),
                    )
                    node.inputs.pop(1)
                    node.inputs.insert(1, reshape_const)
        elif node.op == "Mul":
            if (isinstance(node.inputs[1], Constant) and isinstance(node.inputs[0], Variable)) or (
                isinstance(node.inputs[0], Constant) and isinstance(node.inputs[1], Variable)
            ):
                idx, constant_variable = get_constant_variable(node, return_idx=True)
                if np.all(constant_variable.values == 1):
                    var_idx = 0 if idx == 1 else 1
                    delete_node(node, var_idx)
        elif node.op == "Add":
            if (isinstance(node.inputs[1], Constant) and isinstance(node.inputs[0], Variable)) or (
                isinstance(node.inputs[0], Constant) and isinstance(node.inputs[1], Variable)
            ):
                idx, constant_variable = get_constant_variable(node, return_idx=True)
                if np.all(constant_variable.values == 0):
                    idx = 0 if idx == 1 else 1
                    delete_node(node, idx)
        elif node.op == "Expand":
            if len(node.inputs) > 1 and isinstance(node.inputs[1], Constant) and np.all(node.inputs[1].values == 1):
                idx = 0 if idx == 1 else 1
                delete_node(node, idx)


@register_fusion_pattern("FusionPadConv")
def find_conv_nodes(node, opset):
    """Identify and match convolution nodes following a padding operation to update padding attributes for fusion
    purposes.
    """
    """
             x
             |
            Pad
             |
            Conv
    """
    # fmt: on
    match = {}
    if node.op == "Conv" and node.i(0).op == "Pad":
        pad_node = node.i(0)
        if isinstance(pad_node.inputs[1], Constant):
            pad_value = pad_node.inputs[1].values.tolist()
            input_variable = node.i(0).inputs[0]
            input_variable.outputs.remove(pad_node)

            pad_variable = node.i(0).outputs[0]  # pad output variable
            index = node.inputs.index(pad_variable)
            node.inputs.pop(index)
            node.inputs.insert(index, input_variable)

            inputs = list(node.inputs)
            outputs = list(node.outputs)
            attrs = node.attrs

            node.inputs.clear()
            node.outputs.clear()
            pad_node.inputs.clear()
            pad_node.outputs.clear()
            conv_pads = attrs["pads"]
            len_conv_pads = len(conv_pads) // 2

            len_pads = len(pad_value) // 2
            pads = pad_value[len_pads - len_conv_pads : len_pads] + pad_value[len_pads + len_conv_pads :]

            pads = [pad + conv_pad for pad, conv_pad in zip(pads, conv_pads)]
            attrs["pads"] = pads
            match[node.name] = {
                "op": "Conv",
                "inputs": inputs,
                "outputs": outputs,
                "name": node.name,
                "attrs": node.attrs,
                "domain": None,
            }

    return match


@register_fusion_pattern("FusionConvBN")
def find_conv_transpose_nodes(node, opset):
    # fmt: off
    """X | Conv/ConvTranspose | BatchNormalization."""
    # fmt: on
    match = {}
    if node.op == "BatchNormalization" and node.i(0).op in [
        "ConvTranspose",
        "Conv",
    ]:
        conv_transpose_node = node.i(0)
        conv_transpose_node_users = get_node_users(conv_transpose_node)
        if len(conv_transpose_node_users) == 1:
            conv_transpose_weight = conv_transpose_node.inputs[1].values
            bn_node = node
            bn_scale = bn_node.inputs[1].values
            bn_bias = bn_node.inputs[2].values
            bn_running_mean = bn_node.inputs[3].values
            bn_running_var = bn_node.inputs[4].values
            bn_eps = bn_node.attrs["epsilon"]

            if len(conv_transpose_node.inputs) == 2:
                conv_transpose_bias = np.zeros_like(bn_running_mean)
            else:
                conv_transpose_bias = conv_transpose_node.inputs[2].values

            bn_var_rsqrt = 1.0 / np.sqrt(bn_running_var + bn_eps)
            shape = [1] * len(conv_transpose_weight.shape)
            if node.i(0).op == "Conv":
                shape[0] = -1
            else:
                shape[1] = -1
            conv_w = conv_transpose_weight * (bn_scale * bn_var_rsqrt).reshape(shape)
            conv_b = (conv_transpose_bias - bn_running_mean) * bn_var_rsqrt * bn_scale + bn_bias

            inputs = []
            inputs.append(list(conv_transpose_node.inputs)[0])
            weight_name = list(conv_transpose_node.inputs)[1].name
            if weight_name.endswith("weight"):
                bias_name = f"{weight_name[:-6]}bias"
            else:
                bias_name = weight_name + "_bias"
            inputs.extend(
                (
                    gs.Constant(weight_name, values=conv_w),
                    gs.Constant(bias_name, values=conv_b),
                )
            )
            outputs = list(bn_node.outputs)

            conv_transpose_node.outputs.clear()
            bn_node.inputs.clear()
            bn_node.outputs.clear()

            match[conv_transpose_node.name] = {
                "op": conv_transpose_node.op,
                "inputs": inputs,
                "outputs": outputs,
                "name": conv_transpose_node.name,
                "attrs": conv_transpose_node.attrs,
                "domain": None,
            }

    return match


@register_fusion_pattern("EliminationSlice")
def find_slice_nodes(node, opset):
    """Identify and combine consecutive 'Slice' nodes in a computational graph for optimization purposes."""
    """
             x
             |
           Slice
             |
           Slice
    """
    # fmt: on
    match = {}
    if node.op == "Slice" and node.i(0).op == "Slice":
        first_slice_node = node.i(0)
        first_slice_node_inputs = list(first_slice_node.inputs)
        if all(isinstance(input, Constant) for input in first_slice_node_inputs[1:]):
            first_slice_node_users = get_node_users(first_slice_node)
            if all(
                user.op == "Slice" and all(isinstance(input, Constant) for input in list(user.inputs)[1:])
                for user in first_slice_node_users
            ):
                first_slice_node_starts = first_slice_node_inputs[1].values.tolist()
                first_slice_node_ends = first_slice_node_inputs[2].values.tolist()
                first_slice_node_axes = first_slice_node_inputs[3].values.tolist()
                first_slice_node_steps = first_slice_node_inputs[4].values.tolist()

                for user_node in first_slice_node_users:
                    second_slice_node = user_node
                    second_slice_node_inputs = list(second_slice_node.inputs)
                    second_slice_node_starts = second_slice_node_inputs[1].values.tolist()
                    second_slice_node_ends = second_slice_node_inputs[2].values.tolist()
                    second_slice_node_axes = second_slice_node_inputs[3].values.tolist()
                    second_slice_node_steps = second_slice_node_inputs[4].values.tolist()

                    new_starts = first_slice_node_starts + second_slice_node_starts
                    new_ends = first_slice_node_ends + second_slice_node_ends
                    new_axes = first_slice_node_axes + second_slice_node_axes
                    new_steps = first_slice_node_steps + second_slice_node_steps

                    if len(new_axes) != len(set(new_axes)):
                        continue

                    inputs = []
                    inputs.extend(
                        (
                            list(first_slice_node.inputs)[0],
                            gs.Constant(
                                second_slice_node_inputs[1].name,
                                values=np.array(new_starts, dtype=np.int64),
                            ),
                            gs.Constant(
                                second_slice_node_inputs[2].name,
                                values=np.array(new_ends, dtype=np.int64),
                            ),
                            gs.Constant(
                                second_slice_node_inputs[3].name,
                                values=np.array(new_axes, dtype=np.int64),
                            ),
                            gs.Constant(
                                second_slice_node_inputs[4].name,
                                values=np.array(new_steps, dtype=np.int64),
                            ),
                        )
                    )
                    outputs = list(second_slice_node.outputs)

                    first_slice_node.outputs.clear()
                    second_slice_node.inputs.clear()
                    second_slice_node.outputs.clear()

                    if len(first_slice_node_users) == 1:
                        match[first_slice_node.name] = {
                            "op": "Slice",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": first_slice_node.name,
                            "attrs": first_slice_node.attrs,
                            "domain": None,
                        }
                    else:
                        match[second_slice_node.name] = {
                            "op": "Slice",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": second_slice_node.name,
                            "attrs": second_slice_node.attrs,
                            "domain": None,
                        }

    return match


@register_fusion_pattern("EliminationReshape")
def find_reshape_nodes(node, opset):
    """Identify consecutive 'Reshape' nodes in the computational graph for potential fusion, returning a matching
    dictionary when criteria are met.
    """
    """
             x
             |
           Reshape
             |
           Reshape
    """
    # fmt: on
    match = {}
    if node.op == "Reshape" and node.i(0).op == "Reshape":
        first_reshape_node = node.i(0)
        first_reshape_node_inputs = list(first_reshape_node.inputs)
        first_reshape_node_users = get_node_users(first_reshape_node)
        if len(first_reshape_node_users) == 1:
            second_reshape_node = node

            def check_constant_mergeable(reshape_node):
                if isinstance(reshape_node.inputs[1], Constant):
                    input_shape = reshape_node.inputs[0].shape
                    reshape_shape = reshape_node.inputs[1].values
                    if input_shape != None and np.any(reshape_shape == 0):
                        shape = [
                            input_shape[i] if dim_size == 0 else dim_size for i, dim_size in enumerate(reshape_shape)
                        ]
                        if not all(isinstance(item, int) for item in shape):
                            return False
                return True

            if check_constant_mergeable(first_reshape_node) and check_constant_mergeable(second_reshape_node):
                inputs = []
                inputs.append(first_reshape_node_inputs[0])
                inputs.append(second_reshape_node.inputs[1])
                outputs = list(second_reshape_node.outputs)
                first_reshape_node.outputs.clear()
                second_reshape_node.inputs.clear()
                second_reshape_node.outputs.clear()

                match[first_reshape_node.name] = {
                    "op": "Reshape",
                    "inputs": inputs,
                    "outputs": outputs,
                    "name": first_reshape_node.name,
                    "attrs": first_reshape_node.attrs,
                    "domain": None,
                }

    return match


# @register_fusion_pattern("EliminationTranspose")
def find_slice_nodes(node, opset):
    """Identifies and processes patterns of consecutive Transpose nodes in a computational graph."""
    """
             x
             |
          Transpose
             |
          Transpose
    """
    if node.op == "Transpose":
        previous_nodes = get_previous_node_by_type(node, "Transpose")
        if previous_nodes:
            if len(previous_nodes) == 1:
                delete_node(node)
                delete_node(previous_nodes[-1])
            else:
                delete_node(node)
                previous_transpose_node = previous_nodes[-1]
                last_node = previous_nodes[-2]
                slice_axis = gs.Constant(
                    f"{last_node.name}_slice_axis",
                    values=np.array([2]).astype(np.int64),
                )
                last_node.inputs.pop(3)
                last_node.inputs.insert(3, slice_axis)
                previous_transpose_node_variable = previous_transpose_node.outputs[0]  # pad output variable
                previous_transpose_node_variable.outputs.remove(last_node)
                last_node.inputs.insert(0, previous_transpose_node.inputs[0])
                for node in previous_nodes:
                    for output in node.outputs:
                        if isinstance(output, Constant):
                            continue
                        output.shape = None

    return {}


@register_fusion_pattern("FusionGemm")
def find_matmul_add_nodes(node, opset):
    """Identifies and returns a pattern match for MatMul followed by Add operations for optimization in a computational
    graph.
    """
    """
             x
             |
           MatMul
             |
            Add
    """
    # fmt: on
    match = {}
    if node.op == "Add" and (
        (isinstance(node.inputs[1], Constant) and node.i(0).op == "MatMul")
        or (isinstance(node.inputs[0], Constant) and node.i(1).op == "MatMul")
    ):
        matmul_node = node.i(0) if isinstance(node.inputs[1], Constant) else node.i(1)
        matmul_bias_variable = get_constant_variable(matmul_node)
        input_variable = matmul_node.inputs[0] if isinstance(matmul_node.inputs[1], Constant) else matmul_node.inputs[1]
        users = get_node_users(matmul_node)
        if len(users) == 1 and matmul_bias_variable:
            if (
                input_variable.shape
                and len(input_variable.shape) > 2
                and all([isinstance(value, int) for value in input_variable.shape])
            ):
                pre_reshape_const = gs.Constant(
                    matmul_node.name + "_pre_reshape_in",
                    values=np.array([-1, matmul_bias_variable.values.shape[0]], dtype=np.int64),
                )
                inputs = []
                inputs.append(input_variable)
                inputs.append(pre_reshape_const)

                reshape_out_variable = gs.Variable(
                    matmul_node.name + "_pre_reshape_out",
                    dtype=input_variable.dtype,
                )
                outputs = [reshape_out_variable]

                match.update(
                    {
                        matmul_node.name + "_pre_reshape": {
                            "op": "Reshape",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": matmul_node.name + "_pre_reshape",
                            "domain": None,
                        }
                    }
                )

                add_node = node
                add_bias_variable = get_constant_variable(add_node)

                output_variable = add_node.inputs[0]
                output_variable.outputs.remove(add_node)

                matmul_bias_transpose_constant = gs.Constant(
                    matmul_bias_variable.name, values=matmul_bias_variable.values.T
                )

                inputs = []
                inputs.append(reshape_out_variable)
                inputs.append(matmul_bias_transpose_constant)
                inputs.append(add_bias_variable)

                gemm_out_variable = gs.Variable(matmul_node.name + "_gemm_out", dtype=output_variable.dtype)
                outputs = [gemm_out_variable]

                match.update(
                    {
                        matmul_node.name: {
                            "op": "Gemm",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": matmul_node.name,
                            "attrs": {
                                "alpha": 1.0,
                                "beta": 1.0,
                                "transA": 0,
                                "transB": 1,
                            },
                            "domain": None,
                        }
                    }
                )

                values = input_variable.shape[:-1] + [matmul_bias_variable.values.shape[-1]]
                post_reshape_const = gs.Constant(
                    matmul_node.name + "_post_reshape_in",
                    values=np.array(values, dtype=np.int64),
                )

                inputs = []
                inputs.append(gemm_out_variable)
                inputs.append(post_reshape_const)
                outputs = list(add_node.outputs)

                matmul_node.outputs.clear()
                add_node.inputs.clear()
                add_node.outputs.clear()

                match.update(
                    {
                        matmul_node.name + "_post_reshape": {
                            "op": "Reshape",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": matmul_node.name + "_post_reshape",
                            "domain": None,
                        }
                    }
                )
            elif (
                input_variable.shape
                and len(input_variable.shape) == 2
                and all([isinstance(value, int) for value in input_variable.shape])
            ):
                add_node = node
                add_bias_variable = get_constant_variable(add_node)

                output_variable = add_node.inputs[0]
                output_variable.outputs.remove(add_node)

                matmul_bias_transpose_constant = gs.Constant(
                    matmul_bias_variable.name, values=matmul_bias_variable.values.T
                )

                inputs = []
                inputs.append(input_variable)
                inputs.append(matmul_bias_transpose_constant)
                inputs.append(add_bias_variable)

                outputs = list(add_node.outputs)
                add_node.inputs.clear()
                add_node.outputs.clear()
                match.update(
                    {
                        matmul_node.name: {
                            "op": "Gemm",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": matmul_node.name,
                            "attrs": {
                                "alpha": 1.0,
                                "beta": 1.0,
                                "transA": 0,
                                "transB": 1,
                            },
                            "domain": None,
                        }
                    }
                )
    return match


# @register_fusion_pattern("FusionGelu")
def find_gelu_nodes(node, opset):
    """Identifies GELU (Gaussian Error Linear Unit) activation pattern nodes in a computational graph based on given
    conditions.
    """
    """
             x
         /      \
         |     Div
         |      |
         |     Erf
         |      |
         |     Add
         \      /
            Mul
             |
            Mul
    """
    # fmt: on
    match = {}
    if node.op == "Mul" and (
        node.i(0).op == "Mul"
        and node.i(0).i(1).op == "Add"
        and node.i(0).i(1).i(0).op == "Erf"
        and node.i(0).i(1).i(0).i(0).op == "Div"
    ):
        input_variable = node.i(0).i(1).i(0).i(0).inputs[0]
        mul_node = node.i(0)
        div_node = node.i(0).i(1).i(0).i(0)

        input_variable.outputs.remove(mul_node)
        input_variable.outputs.remove(div_node)

        output_variable = node.outputs[0]
        output_variable.inputs.clear()
        match[node.name] = {
            "op": "Gelu",
            "inputs": [input_variable],
            "outputs": [output_variable],
            "domain": None,
        }

    return match


@register_fusion_pattern("FusionReduce")
def find_slice_nodes(node, opset):
    """Find and return a dictionary of matching 'ReduceSum' followed by 'Unsqueeze' nodes that match specific conditions
    in the graph.
    """
    """
             x
             |
         ReduceSum
             |
         Unsqueeze
    """
    # fmt: on
    match = {}
    if node.op == "Unsqueeze" and node.i(0).op == "ReduceSum":
        reduce_node = node.i(0)
        reduce_node_node_users = get_node_users(reduce_node)
        if len(reduce_node_node_users) == 1:
            unsqueeze_node = node

            reduce_node_axes = reduce_node.attrs.get("axes", None)
            reduce_node_keepdims = reduce_node.attrs.get("keepdims", 1)
            unsqueeze_node_axes = unsqueeze_node.attrs.get("axes", None)

            if opset < 13 and reduce_node_axes == [-1] and unsqueeze_node_axes == [-1] and reduce_node_keepdims == 0:
                inputs = list(reduce_node.inputs)
                outputs = list(unsqueeze_node.outputs)
                attrs = reduce_node.attrs
                reduce_node.outputs.clear()
                unsqueeze_node.inputs.clear()
                unsqueeze_node.outputs.clear()
                attrs["keepdims"] = 1
                match[reduce_node.name] = {
                    "op": reduce_node.op,
                    "inputs": inputs,
                    "outputs": outputs,
                    "name": reduce_node.name,
                    "attrs": attrs,
                    "domain": None,
                }

    return match


@gs.Graph.register()
def replace_custom_layer(
    self,
    op: str,
    inputs,
    outputs: List[str],
    name: str,
    attrs: dict = None,
    domain: str = "ai.onnx.contrib",
):
    return self.layer(
        op=op,
        inputs=inputs,
        outputs=outputs,
        name=name,
        attrs=attrs,
        domain=domain,
    )


def find_matches(graph: Graph, fusion_patterns: dict):
    """Find matching patterns in the graph based on provided fusion patterns."""
    opset = graph.opset
    match_map = {}
    counter = Counter()
    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for layer_type, func in fusion_patterns.items():
                with contextlib.suppress(IndexError):
                    matches = func(node, opset)
                    if matches:
                        logger.debug(f"matched pattern {layer_type}")
                        for _, match in matches.items():
                            if "op" not in match:
                                match.update({"op": layer_type})
                            if "name" not in match:
                                match.update({"name": f"{layer_type.lower()}_{counter[layer_type]}"})
                            counter.update([layer_type])
                        match_map.update(matches)

    return match_map


def find_and_remove_replaceable_nodes(nodes):
    """Find and remove duplicate or replaceable nodes in a given list of computational graph nodes."""

    def get_node_key(node):
        input_names = []
        for input_node in node.inputs:
            if isinstance(input_node, Variable):
                input_names.append(input_node.name)
        return "_".join(input_names) if input_names else None

    def replace_node_references(existing_node, to_be_removed_node):
        users = get_node_users(to_be_removed_node)
        for user in users:
            for idx, inp in enumerate(user.inputs):
                if inp in to_be_removed_node.outputs:
                    index = user.inputs.index(inp)
                    user.inputs.pop(index)
                    user.inputs.insert(index, existing_node.outputs[0])

        to_be_removed_node.inputs.clear()
        to_be_removed_node.outputs.clear()

    node_dict = {}
    for node in nodes:
        key = get_node_key(node)
        if key:
            if key in node_dict:
                node_dict[key].append(node)
            else:
                node_dict[key] = [node]

    for key, bucketed_nodes in node_dict.items():
        if len(bucketed_nodes) > 1:
            keep_nodes = [True] * len(bucketed_nodes)
            for i, node in enumerate(bucketed_nodes):
                if keep_nodes[i]:
                    for j in range(i + 1, len(bucketed_nodes)):
                        if keep_nodes[j]:
                            logger.debug(f"node.op {bucketed_nodes[0].op} idx i: {i}, idx j: {j}")
                            if can_be_replaced(node, bucketed_nodes[j]):
                                keep_nodes[j] = False
                                existing_node = node
                                to_be_removed_node = bucketed_nodes[j]
                                replace_node_references(existing_node, to_be_removed_node)
                                logger.debug(f"Node {to_be_removed_node.name} can be replaced by {existing_node.name}")


def sequences_equal(seq1, seq2):
    """Check if two sequences are equal by comparing their lengths and elements."""
    length_match = len(seq1) == len(seq2)
    if not length_match:
        return False

    return all(elem1 == elem2 for elem1, elem2 in zip(seq1, seq2))


def can_be_replaced(node, other_node):
    """Check if two nodes can be replaced based on their operations, attributes, and inputs."""
    attrs_match = node.op == other_node.op and node.attrs == other_node.attrs
    inputs_match = sequences_equal(node.inputs, other_node.inputs)

    return attrs_match and inputs_match


def subexpression_elimination(graph):
    """Perform subexpression elimination on a computational graph to optimize node operations."""
    nodes_by_op = {}

    for node in graph.nodes:
        op = node.op
        if op not in nodes_by_op:
            nodes_by_op[op] = []
        nodes_by_op[op].append(node)

    for nodes in nodes_by_op.values():
        find_and_remove_replaceable_nodes(nodes)


def optimize_model(model: Union[onnx.ModelProto, gs.Graph], skip_fusion_patterns: str = None) -> onnx.ModelProto:
    graph = model if isinstance(model, gs.Graph) else gs.import_onnx(model)
    fusion_patterns = get_fusion_patterns(skip_fusion_patterns)
    fusion_pairs = find_matches(graph, fusion_patterns)
    for _, match in fusion_pairs.items():
        graph.replace_custom_layer(**match)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    graph_constant_fold_inplace(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    subexpression_elimination(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    model = gs.export_onnx(graph)

    return model
