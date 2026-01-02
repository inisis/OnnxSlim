import logging

import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.third_party.onnx_graphsurgeon.exporters.onnx_exporter import dtype_to_onnx
from onnxslim.third_party.onnx_graphsurgeon.ir.tensor import Constant, Variable

logger = logging.getLogger("onnxslim")


def dead_node_elimination(graph, is_subgraph=False):
    """Perform in-place constant folding optimizations on the given computational graph by eliminating redundant
    nodes.
    """
    for subgraph in graph.subgraphs():
        dead_node_elimination(subgraph, is_subgraph=True)

    for node in graph.nodes:
        if node.op in {"Identity", "Dropout"}:
            if not is_subgraph:
                node.erase()
                logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Pad":
            if len(node.inputs) > 1 and isinstance(node.inputs[1], Constant):
                pad_value = node.inputs[1].values.tolist()
                pad_value = pad_value if isinstance(pad_value, list) else [pad_value]
                if all(value == 0 for value in pad_value):
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Cast":
            inp_dtype = next(dtype_to_onnx(input.dtype) for input in node.inputs)
            if inp_dtype == node.attrs["to"]:
                node.erase()
                logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Reshape":
            if (node.inputs[0].shape and len(node.inputs[0].shape) == 1) and (
                node.outputs[0].shape and len(node.outputs[0].shape) == 1
            ):
                node.erase()
                logger.debug(f"removing {node.op} op: {node.name}")
            elif node.inputs[0].shape and node.outputs[0].shape and node.inputs[0].shape == node.outputs[0].shape:
                node.erase()
                logger.debug(f"removing {node.op} op: {node.name}")
            else:
                node_output_shape = node.outputs[0].shape
                if node_output_shape and check_shape(node_output_shape) and not isinstance(node.inputs[1], gs.Constant):
                    shapes = [shape if isinstance(shape, int) else -1 for shape in node_output_shape]
                    reshape_const = gs.Constant(
                        f"{node.inputs[1].name}_",
                        values=np.array(shapes, dtype=np.int64),
                    )
                    node.inputs.pop(1)
                    node.inputs.insert(1, reshape_const)
                    logger.debug(f"replacing {node.op} op: {node.name}")
        elif node.op == "Slice":
            if (node.inputs[0].shape and node.outputs[0].shape
                and node.inputs[0].shape == node.outputs[0].shape
                and all(isinstance(item, int) for item in node.inputs[0].shape)):

                # Check if slice is a no-op by analyzing parameters directly
                # Slice inputs: data, starts, ends, [axes], [steps]
                if is_noop_slice(node):
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")

        elif node.op == "Mul":
            if (isinstance(node.inputs[1], Constant) and isinstance(node.inputs[0], Variable)) or (
                isinstance(node.inputs[0], Constant) and isinstance(node.inputs[1], Variable)
            ):
                idx, constant_variable = get_constant_variable(node, return_idx=True)
                if np.all(constant_variable.values == 1):
                    var_idx = 0 if idx == 1 else 1
                    node.erase(var_idx, 0)
                    logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Add":
            if (isinstance(node.inputs[1], Constant) and isinstance(node.inputs[0], Variable)) or (
                isinstance(node.inputs[0], Constant) and isinstance(node.inputs[1], Variable)
            ):
                idx, constant_variable = get_constant_variable(node, return_idx=True)
                value = constant_variable.values
                var_idx = 0 if idx == 1 else 1
                if value.ndim == 0 and value == 0:
                    node.erase(var_idx, 0)
                    logger.debug(f"removing {node.op} op: {node.name}")
                elif np.all(value == 0) and (node.inputs[var_idx].shape == node.outputs[0].shape):
                    node.erase(var_idx, 0)
                    logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Expand":
            # tests/test_onnx_nets.py::TestTimmClass::test_timm[lambda_resnet26rpt_256]
            if len(node.inputs) > 1 and isinstance(node.inputs[1], Constant):
                constant_variable = node.inputs[1]
                value = constant_variable.values
                if node.inputs[0].shape is not None and node.inputs[0].shape == node.outputs[0].shape:
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")
                elif value.ndim == 0 and value == 1:
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Concat":
            if len(node.inputs) == 1:
                node.erase()
                logger.debug(f"removing {node.op} op: {node.name}")
            else:
                for input in node.inputs:
                    if isinstance(input, Constant) and input.values.size == 0:
                        node.inputs.remove(input)
        elif node.op == "Sub":
            if isinstance(node.inputs[1], Constant) and isinstance(node.inputs[0], Variable):
                constant_variable = node.inputs[1]
                value = constant_variable.values
                if value.ndim == 0 and value == 0:
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")
                elif np.all(value == 0) and (node.inputs[0].shape == node.outputs[0].shape):
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Div":
            if isinstance(node.inputs[1], Constant) and isinstance(node.inputs[0], Variable):
                constant_variable = node.inputs[1]
                value = constant_variable.values
                if value.ndim == 0 and value == 1:
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")
                elif np.all(value == 1) and (node.inputs[0].shape == node.outputs[0].shape):
                    node.erase()
                    logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Split":
            if (
                len(node.outputs) == 1
                and node.outputs[0].shape
                and node.inputs[0].shape
                and node.outputs[0].shape == node.inputs[0].shape
            ):
                node.erase()
                logger.debug(f"removing {node.op} op: {node.name}")
        elif node.op == "Resize":
            mode = node.attrs.get("mode")
            if mode is None:
                node.attrs["mode"] = "nearest"
                logger.debug(f"setting mode to nearest for {node.op} op: {node.name} since it is not set")


def check_shape(shapes):
    """Verify that 'shapes' contains exactly one string and all other elements are positive integers."""
    string_count = 0
    non_negative_int_count = 0

    for item in shapes:
        if isinstance(item, str):
            string_count += 1
        elif isinstance(item, int) and item > 0:
            non_negative_int_count += 1

    return (string_count == 1 and non_negative_int_count == len(shapes) - 1) or non_negative_int_count == len(shapes)


def get_constant_variable(node, return_idx=False):
    """Return the first constant variable found in a node's inputs, optionally including the index."""
    for idx, input in enumerate(list(node.inputs)):
        if isinstance(input, Constant):
            return (idx, input) if return_idx else input


def is_noop_slice(node):
    """Check if a Slice node is a no-op by analyzing its parameters directly.

    A Slice is a no-op when it extracts the entire tensor, i.e., for each sliced axis:
    - start == 0 (or equivalent negative index)
    - end >= dim_size (or is INT_MAX-like value)
    - step == 1
    """
    # Slice inputs: data, starts, ends, [axes], [steps]
    if len(node.inputs) < 3:
        return False

    data_shape = node.inputs[0].shape
    if not data_shape or not all(isinstance(d, int) for d in data_shape):
        return False

    # Get starts and ends (required)
    starts_input = node.inputs[1]
    ends_input = node.inputs[2]

    if not isinstance(starts_input, Constant) or not isinstance(ends_input, Constant):
        return False

    starts = starts_input.values.flatten().tolist()
    ends = ends_input.values.flatten().tolist()

    # Get axes (optional, defaults to [0, 1, 2, ...])
    if len(node.inputs) > 3 and isinstance(node.inputs[3], Constant):
        axes = node.inputs[3].values.flatten().tolist()
    else:
        axes = list(range(len(starts)))

    # Get steps (optional, defaults to [1, 1, 1, ...])
    if len(node.inputs) > 4 and isinstance(node.inputs[4], Constant):
        steps = node.inputs[4].values.flatten().tolist()
    else:
        steps = [1] * len(starts)

    # Check each axis
    ndim = len(data_shape)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        # Normalize negative axis
        if axis < 0:
            axis = ndim + axis

        if axis < 0 or axis >= ndim:
            return False

        dim_size = data_shape[axis]

        # Step must be 1 for no-op
        if step != 1:
            return False

        # Normalize negative start index
        if start < 0:
            start = max(0, dim_size + start)

        # Start must be 0
        if start != 0:
            return False

        # Normalize negative end index
        if end < 0:
            end = dim_size + end

        # End must cover the entire dimension
        # Common patterns: end == dim_size, or end is a large value like INT_MAX
        if end < dim_size:
            return False

    return True
