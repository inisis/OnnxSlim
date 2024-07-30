from collections import Counter
from typing import List, Union

import onnx

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import get_node_feeds
from onnxslim.core.pattern.registry import get_fusion_patterns
from onnxslim.third_party.onnx_graphsurgeon.ir.graph import Graph
from onnxslim.utils import logger

from .dead_node_elimination import dead_node_elimination
from .subexpression_elimination import subexpression_elimination


def optimize_model(model: Union[onnx.ModelProto, gs.Graph], skip_fusion_patterns: str = None) -> onnx.ModelProto:
    """Optimize and transform the given ONNX model using various fusion patterns and graph rewriting techniques."""
    graph = model if isinstance(model, gs.Graph) else gs.import_onnx(model)
    fusion_patterns = get_fusion_patterns(skip_fusion_patterns)
    fusion_pairs = find_matches(graph, fusion_patterns)
    for match in fusion_pairs.values():
        graph.replace_custom_layer(**match)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    dead_node_elimination(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    subexpression_elimination(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    model = gs.export_onnx(graph)

    return model


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
    """Replace a custom layer in the computational graph with specified parameters and domain."""
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
    match_map = {}
    counter = Counter()
    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for layer_type, pattern_matcher in fusion_patterns.items():
                match = pattern_matcher.match(node)
                if match:
                    match_case = pattern_matcher.rewrite(opset=graph.opset)
                    logger.debug(f"matched pattern {layer_type}")
                    for _, match in match_case.items():
                        if "op" not in match:
                            match.update({"op": layer_type})
                        if "name" not in match:
                            match.update({"name": f"{layer_type.lower()}_{counter[layer_type]}"})
                        counter.update([layer_type])
                    match_map.update(match_case)

    return match_map


def tie_weights(graph, threshold=1 * 1024 * 1024):
    """Tie weights in a computational graph to reduce the number of parameters."""

    tensor_map = graph.tensors()
    constant_tensors = [tensor for tensor in tensor_map.values() if isinstance(tensor, gs.Constant)]

    sub_graphs = graph.subgraphs(recursive=True)
    sub_graphs_constant_tensors = [
        [tensor for name, tensor in sub_graph.tensors().items() if isinstance(tensor, gs.Constant)]
        for sub_graph in sub_graphs
    ]

    constant_tensors.extend([tensor for tensors in sub_graphs_constant_tensors for tensor in tensors])

    def replace_constant_references(existing_constant, to_be_removed_constant):
        users = to_be_removed_constant.outputs
        for user in users:
            for inp in user.inputs:
                if inp in to_be_removed_constant.outputs:
                    index = user.inputs.index(inp)
                    user.inputs.pop(index)
                    user.inputs.insert(index, existing_constant)

        to_be_removed_constant.inputs.clear()
        to_be_removed_constant.outputs.clear()

    filtered_constant_tensors = [tensor for tensor in constant_tensors if tensor.values.size > threshold]

    if len(filtered_constant_tensors) > 1:
        keep_constants = [True] * len(filtered_constant_tensors)
        for i, constant_tensor in enumerate(filtered_constant_tensors):
            if keep_constants[i]:
                for j in range(i + 1, len(filtered_constant_tensors)):
                    if keep_constants[j]:
                        if constant_tensor == filtered_constant_tensors[j]:
                            keep_constants[j] = False
                            replace_constant_references(constant_tensor, filtered_constant_tensors[j])
                            logger.debug(
                                f"Constant {filtered_constant_tensors[j].name} can be replaced by {constant_tensor.name}"
                            )


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
