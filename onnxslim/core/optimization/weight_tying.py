import logging
import os

logger = logging.getLogger("onnxslim")
from collections import defaultdict

import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs

THRESHOLD = int(os.getenv("ONNXSLIM_THRESHOLD")) if os.getenv("ONNXSLIM_THRESHOLD") else 1000


def tie_weights(graph):
    """Tie weights in a computational graph to reduce the number of parameters."""
    tensor_map = graph.tensors()
    constant_tensors = [tensor for tensor in tensor_map.values() if isinstance(tensor, gs.Constant)]

    sub_graphs = graph.subgraphs(recursive=True)
    sub_graphs_constant_tensors = [
        [tensor for name, tensor in sub_graph.tensors().items() if isinstance(tensor, gs.Constant)]
        for sub_graph in sub_graphs
    ]

    constant_tensors.extend([tensor for tensors in sub_graphs_constant_tensors for tensor in tensors])

    constant_by_shape = defaultdict(list)

    for constant_tensor in constant_tensors:
        shape = tuple(constant_tensor.shape)
        if np.prod(shape) < THRESHOLD:
            constant_by_shape[shape].append(constant_tensor)

    for nodes in constant_by_shape.values():
        find_and_remove_replaceable_constants(nodes)


def find_and_remove_replaceable_constants(constant_tensors):
    def replace_constant_references(existing_constant, to_be_removed_constant):
        users = list(to_be_removed_constant.outputs)

        for user in users:
            for idx, inp in enumerate(user.inputs):
                if (inp == to_be_removed_constant) and (inp.name == to_be_removed_constant.name):
                    user.inputs.pop(idx)
                    user.inputs.insert(idx, existing_constant)

    if len(constant_tensors) > 1:
        keep_constants = [True] * len(constant_tensors)
        for i, constant_tensor in enumerate(constant_tensors):
            if keep_constants[i]:
                for j in range(i + 1, len(constant_tensors)):
                    if keep_constants[j]:
                        if constant_tensor == constant_tensors[j]:
                            keep_constants[j] = False
                            replace_constant_references(constant_tensor, constant_tensors[j])
                            logger.debug(
                                f"Constant {constant_tensors[j].name} can be replaced by {constant_tensor.name}"
                            )
