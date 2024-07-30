from onnxslim.core.pattern import get_node_users
from onnxslim.third_party.onnx_graphsurgeon.ir.tensor import Variable


def delete_node(node, input_var_idx=0, output_var_idx=0):
    """Delete a node from the computation graph while re-linking its input and output to maintain graph integrity."""
    input_variable = node.inputs[input_var_idx]
    node_variable = node.outputs[output_var_idx]
    next_nodes = get_node_users(node)
    if next_nodes:
        for next_node in next_nodes:
            if isinstance(next_node, Variable) and next_node.is_output:
                continue
            index = next_node.inputs.index(node_variable)
            next_node.inputs.pop(index)
            next_node.inputs.insert(index, input_variable)
    else:
        input_node = node.i()
        input_node.outputs.remove(node.inputs[input_var_idx])
        input_node.outputs.append(node.outputs[output_var_idx])
        node.outputs.clear()
