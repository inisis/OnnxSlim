import numpy as np
import onnx

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.optimization.subexpression_elimination import subexpression_elimination


def test_duplicate_node_with_graph_output_keeps_output_node():
    """Duplicate nodes should keep the graph-output producer to avoid dangling consumers."""
    input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 1, 1, 1))
    input2 = gs.Variable(name="input2", dtype=np.float32, shape=(1, 1, 1, 1))

    reshape1_shape = gs.Constant(name="reshape1_shape", values=np.array([1, 1], dtype=np.int64))
    reshape2_shape = gs.Constant(name="reshape2_shape", values=np.array([1, 1], dtype=np.int64))

    reshape1_out = gs.Variable(name="reshape1_out", dtype=np.float32, shape=(1, 1))
    reshape2_out = gs.Variable(name="reshape2_out", dtype=np.float32, shape=(1, 1))
    concat_internal_out = gs.Variable(name="concat_internal_out", dtype=np.float32, shape=(1, 2))
    concat_graph_out = gs.Variable(name="concat_graph_out", dtype=np.float32, shape=(1, 2))
    sigmoid_out = gs.Variable(name="sigmoid_out", dtype=np.float32, shape=(1, 2))

    reshape1 = gs.Node(op="Reshape", name="reshape1", inputs=[input1, reshape1_shape], outputs=[reshape1_out])
    reshape2 = gs.Node(op="Reshape", name="reshape2", inputs=[input2, reshape2_shape], outputs=[reshape2_out])
    concat_internal = gs.Node(
        op="Concat",
        name="concat_internal",
        inputs=[reshape1_out, reshape2_out],
        outputs=[concat_internal_out],
        attrs={"axis": 1},
    )
    concat_graph = gs.Node(
        op="Concat",
        name="concat_graph",
        inputs=[reshape1_out, reshape2_out],
        outputs=[concat_graph_out],
        attrs={"axis": 1},
    )
    sigmoid = gs.Node(op="Sigmoid", name="sigmoid", inputs=[concat_internal_out], outputs=[sigmoid_out])

    graph = gs.Graph(
        nodes=[reshape1, reshape2, concat_internal, concat_graph, sigmoid],
        inputs=[input1, input2],
        outputs=[concat_graph_out, sigmoid_out],
    )

    subexpression_elimination(graph)
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.checker.check_model(model)

    concat_nodes = [node for node in model.graph.node if node.op_type == "Concat"]
    sigmoid_nodes = [node for node in model.graph.node if node.op_type == "Sigmoid"]

    assert len(concat_nodes) == 1
    assert len(sigmoid_nodes) == 1
    assert sigmoid_nodes[0].input[0] == model.graph.output[0].name
