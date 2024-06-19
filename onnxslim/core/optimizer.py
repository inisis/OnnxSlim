from collections import Counter, OrderedDict
from typing import List, Union

import numpy as np
import onnx

import onnxslim.onnx_graphsurgeon as gs
from onnxslim.onnx_graphsurgeon.exporters.onnx_exporter import dtype_to_onnx
from onnxslim.onnx_graphsurgeon.ir.graph import Graph
from onnxslim.onnx_graphsurgeon.ir.tensor import Constant, Variable
from onnxslim.utils import logger
from onnxslim.core.graph_rewriter import PatternMatcher, Pattern, get_node_feeds, get_node_users

DEFAULT_FUSION_PATTERNS = OrderedDict()


def register_fusion_pattern(fusion_pattern):
    """Registers a fusion pattern function for a specified layer type in the DEFAULT_FUSION_PATTERNS dictionary."""
    layer_type = fusion_pattern.name

    if layer_type in DEFAULT_FUSION_PATTERNS.keys():
        raise
    DEFAULT_FUSION_PATTERNS[layer_type] = fusion_pattern


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
        trajectory.append(node_feed)
        if node_feed.op == op_type:
            return trajectory
        else:
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
        if node.op in {"Identity", "Dropout"}:
            delete_node(node)
        elif node.op == "Pad":
            if len(node.inputs) > 1 and isinstance(node.inputs[1], Constant):
                pad_value = node.inputs[1].values.tolist()
                pad_value = pad_value if isinstance(pad_value, list) else [pad_value]
                if all(value == 0 for value in pad_value):
                    delete_node(node)
                    logger.debug(f"removing Pad op: {node.name}")
        elif node.op == "Cast":
            inp_dtype = [dtype_to_onnx(input.dtype) for input in node.inputs][0]
            if inp_dtype == node.attrs["to"]:
                delete_node(node)
                logger.debug(f"removing Cast op: {node.name}")
        elif node.op == "Reshape":
            if (node.inputs[0].shape and len(node.inputs[0].shape) == 1) and (
                node.outputs[0].shape and len(node.outputs[0].shape) == 1
            ):
                delete_node(node)
                logger.debug(f"removing Reshape op: {node.name}")
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
                    logger.debug(f"removing Mul op: {node.name}")
        elif node.op == "Add":
            if (isinstance(node.inputs[1], Constant) and isinstance(node.inputs[0], Variable)) or (
                isinstance(node.inputs[0], Constant) and isinstance(node.inputs[1], Variable)
            ):
                idx, constant_variable = get_constant_variable(node, return_idx=True)
                if np.all(constant_variable.values == 0) and (node.inputs[0].shape == node.inputs[1].shape):
                    idx = 0 if idx == 1 else 1
                    delete_node(node, idx)
                    logger.debug(f"removing Add op: {node.name}")
        elif node.op == "Expand":
            idx, constant_variable = get_constant_variable(node, return_idx=True)
            if len(node.inputs) > 1 and isinstance(node.inputs[1], Constant) and np.all(node.inputs[1].values == 1):
                idx = 0 if idx == 1 else 1
                delete_node(node, idx)
                logger.debug(f"removing Expand op: {node.name}")
        elif node.op == "Concat":
            if len(node.inputs) == 1:
                delete_node(node)
                logger.debug(f"removing Concat op: {node.name}")
            else:
                for input in node.inputs:
                    if isinstance(input, Constant) and input.values.size == 0:
                        node.inputs.remove(input)


class PadConvMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            '''
            input  input  0 1 pad_0
            Pad    pad_0  1+ 1 input conv_0
            Conv   conv_0 1+ 1 pad_0 output
            output output 1 0 conv_0
            ''')
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionPadConv"

    def parameter_check(self):
        pad_node = self.pad_0
        if not isinstance(pad_node.inputs[1], Constant):
            return False

        return True

    def rewrite(self):
        match_case = {}
        node = self.conv_0
        pad_node = self.pad_0
        input_variable = self.pad_0.inputs[0]

        pad_value = pad_node.inputs[1].values.tolist()
        input_variable.outputs.remove(pad_node)

        pad_variable = pad_node.outputs[0]  # pad output variable
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

        match_case[node.name] = {
            "op": "Conv",
            "inputs": inputs,
            "outputs": outputs,
            "name": node.name,
            "attrs": node.attrs,
            "domain": None,
        }

        return match_case

register_fusion_pattern(PadConvMatcher(1))

class ConvBatchNormMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            '''
            input              input  0 1 conv_0
            Conv               conv_0 3 1 input ? ? bn_0
            BatchNormalization bn_0   5 1 conv_0 ? ? ? ? output
            output             output 1 0 bn_0
            ''')
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionConvBN"

    def rewrite(self):
        match_case = {}
        conv_transpose_node = self.conv_0
        conv_transpose_node_users = get_node_users(conv_transpose_node)
        node = self.bn_0
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

            match_case[conv_transpose_node.name] = {
                "op": conv_transpose_node.op,
                "inputs": inputs,
                "outputs": outputs,
                "name": conv_transpose_node.name,
                "attrs": conv_transpose_node.attrs,
                "domain": None,
            }

        return match_case

register_fusion_pattern(ConvBatchNormMatcher(1))

class SlicePatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            '''
            input  input   0 1 slice_0
            Slice  slice_0 5 1 input   ? ? ? ? slice_1
            Slice  slice_1 5 1 slice_0 ? ? ? ? output
            output output 1 0 slice_1
            ''')  # to check here slice_0
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "EliminationSlice"

    def rewrite(self):
        match_case = {}
        first_slice_node = self.slice_0
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
                        match_case[first_slice_node.name] = {
                            "op": "Slice",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": first_slice_node.name,
                            "attrs": first_slice_node.attrs,
                            "domain": None,
                        }
                    else:
                        match_case[second_slice_node.name] = {
                            "op": "Slice",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": second_slice_node.name,
                            "attrs": second_slice_node.attrs,
                            "domain": None,
                        }

        return match_case

register_fusion_pattern(SlicePatternMatcher(1))

class ReshapePatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            '''
            input    input   0 1 reshape_0
            Reshape  reshape_0 2 1 input   ? reshape_1
            Reshape  reshape_1 2 1 reshape_0 ? output
            output   output 1 0 reshape_1
            ''')
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "EliminationReshape"

    def rewrite(self):
        match_case = {}
        node = self.reshape_1
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

                match_case[first_reshape_node.name] = {
                    "op": "Reshape",
                    "inputs": inputs,
                    "outputs": outputs,
                    "name": first_reshape_node.name,
                    "attrs": first_reshape_node.attrs,
                    "domain": None,
                }

        return match_case

register_fusion_pattern(ReshapePatternMatcher(1))

class MatMulAddPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            '''
            input    input    0 1 matmul_0
            MatMul   matmul_0 2 1 input ? add_0
            Add      add_0    2 1 matmul_0 ? output
            output   output   1 0 add_0
            ''')
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionGemm"

    def rewrite(self):
        match_case = {}
        node = self.add_0
        matmul_node = self.matmul_0
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

                match_case.update(
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

                match_case.update(
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

                match_case.update(
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
                match_case.update(
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
        return match_case

register_fusion_pattern(MatMulAddPatternMatcher(1))

class GeluPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            '''
            input  input  0 2 mul_0 div_0
            Div    div_0  2 1 input ? erf_0
            Erf    erf_0  1 1 div_0 add_0
            Add    add_0  2 1 erf_0 ? mul_0
            Mul    mul_0  2 1 input add_0 mul_1
            Mul    mul_1  2 1 mul_0 ? output
            output output 1 0 mul_1
            ''')
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionGelu"

    def rewrite(self):
        match_case = {}
        input_variable = self.div_0.inputs[0]
        mul_node = self.mul_0
        div_node = self.div_0

        input_variable.outputs.remove(mul_node)
        input_variable.outputs.remove(div_node)

        output_variable = self.mul_1.outputs[0]
        output_variable.inputs.clear()

        match_case[self.mul_1.name] = {
            "op": "Gelu",
            "inputs": [input_variable],
            "outputs": [output_variable],
            "domain": None,
        }

        return match_case

# register_fusion_pattern(GeluPatternMatcher(1))

class ReducePatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            '''
            input     input       0 1 reduce_0
            ReduceSum reduce_0    1 1 input unsqueeze_0
            Unsqueeze unsqueeze_0 1 1 reduce_0 output
            output    output      1 0 unsqueeze_0
            ''')
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionReduce"

    def rewrite(self, opset=11):
        match_case = {}
        node = self.unsqueeze_0
        reduce_node = self.reduce_0
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
                match_case[reduce_node.name] = {
                    "op": reduce_node.op,
                    "inputs": inputs,
                    "outputs": outputs,
                    "name": reduce_node.name,
                    "attrs": attrs,
                    "domain": None,
                }

        return match_case

register_fusion_pattern(ReducePatternMatcher(1))

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
            for layer_type, pattern_matcher in fusion_patterns.items():
                match = pattern_matcher.match(node)
                if match:
                    match_case = pattern_matcher.rewrite()
                    logger.debug(f"matched pattern {layer_type}")
                    for _, match in match_case.items():
                        if "op" not in match:
                            match.update({"op": layer_type})
                        if "name" not in match:
                            match.update({"name": f"{layer_type.lower()}_{counter[layer_type]}"})
                        counter.update([layer_type])
                    match_map.update(match_case)

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
                            logger.debug(f"node.op {bucketed_nodes[i].op} idx i: {i}, idx j: {j}")
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
    for match in fusion_pairs.values():
        graph.replace_custom_layer(**match)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    graph_constant_fold_inplace(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    subexpression_elimination(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    model = gs.export_onnx(graph)

    return model
