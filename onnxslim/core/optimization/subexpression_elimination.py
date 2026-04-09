import logging

from onnxslim.third_party.onnx_graphsurgeon.ir.tensor import Variable

logger = logging.getLogger("onnxslim")


def has_graph_output(node, graph_output_ids=None):
    """Return whether any output tensor of the node is a graph output."""
    if graph_output_ids is None:
        return any(output.is_output for output in node.outputs)

    return any(id(output) in graph_output_ids for output in node.outputs)


def find_and_remove_replaceable_nodes(nodes, graph_output_ids=None):
    """Find and remove duplicate or replaceable nodes in a given list of computational graph nodes."""

    def get_node_key(node):
        input_names = []
        for input_node in node.inputs:
            if isinstance(input_node, Variable):
                input_names.append(input_node.name)
        return "_".join(input_names) if input_names else None

    node_dict = {}
    for node in nodes:
        key = get_node_key(node)
        if key:
            if key in node_dict:
                node_dict[key].append(node)
            else:
                node_dict[key] = [node]

    for key, bucketed_nodes in node_dict.items():
        bucketed_nodes = sorted(
            bucketed_nodes,
            key=lambda node: has_graph_output(node, graph_output_ids),
            reverse=True,
        )
        if len(bucketed_nodes) > 1:
            keep_nodes = [True] * len(bucketed_nodes)
            for i, node in enumerate(bucketed_nodes):
                if keep_nodes[i]:
                    for j in range(i + 1, len(bucketed_nodes)):
                        other_node = bucketed_nodes[j]
                        if (
                            keep_nodes[j]
                            and not (
                                has_graph_output(node, graph_output_ids)
                                and has_graph_output(other_node, graph_output_ids)
                            )
                            and can_be_replaced(node, other_node)
                        ):
                            keep_nodes[j] = False
                            existing_node = node
                            to_be_removed_node = other_node
                            to_be_removed_node.replace_all_uses_with(existing_node)
                            logger.debug(
                                f"Node {to_be_removed_node.name} Op {to_be_removed_node.op} can be replaced by {existing_node.name}"
                            )


def sequences_equal(seq1, seq2):
    """Check if two sequences are equal by comparing their lengths and elements."""
    length_match = len(seq1) == len(seq2)
    if not length_match:
        return False

    return all(elem1 == elem2 for elem1, elem2 in zip(seq1, seq2))


def can_be_replaced(node, other_node):
    """Check if two nodes can be replaced based on their operations, attributes, and inputs."""
    attrs_match = node.op == other_node.op and node.attrs == other_node.attrs
    node_input = [input for input in node.inputs if not input.is_empty()]
    other_node_input = [input for input in other_node.inputs if not input.is_empty()]
    inputs_match = sequences_equal(node_input, other_node_input)

    return attrs_match and inputs_match


def subexpression_elimination(graph):
    """Perform subexpression elimination on a computational graph to optimize node operations."""
    nodes_by_op = {}
    graph_output_ids = {id(output) for output in graph.outputs}

    for subgraph in graph.subgraphs():
        subexpression_elimination(subgraph)

    for node in graph.nodes:
        op = node.op
        if op not in nodes_by_op:
            nodes_by_op[op] = []
        nodes_by_op[op].append(node)

    for nodes in nodes_by_op.values():
        find_and_remove_replaceable_nodes(nodes, graph_output_ids)
