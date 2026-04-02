import os

import numpy as np
import onnx
import torch
import torch.nn as nn

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.optimization.subexpression_elimination import (
    can_be_replaced,
    sequences_equal,
    subexpression_elimination,
)
from onnxslim.core.optimization.weight_tying import tie_weights


class TestSubexpressionElimination:
    def test_duplicate_operations_elimination(self, request):
        """Test that duplicate operations are properly eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20, bias=False)
                self.linear2 = nn.Linear(10, 20, bias=False)
                # Both linear layers will have the same weights
                self.linear2.weight = nn.Parameter(self.linear1.weight.detach().clone())

            def forward(self, x):
                # Apply the same operation twice on the same input
                y1 = self.linear1(x)
                y2 = self.linear2(x)
                return y1 + y2

        input_tensor = torch.randn(5, 10)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename, opset_version=14, dynamo=False)

        # Import graph and apply subexpression_elimination
        graph = gs.import_onnx(onnx.load(filename))
        graph.fold_constants().cleanup().toposort()
        tie_weights(graph)
        graph.cleanup().toposort()
        initial_node_count = len(graph.nodes)
        subexpression_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # One of the duplicate operations should be eliminated
        assert final_node_count < initial_node_count

    def test_sequences_equal_function(self):
        """Test the sequences_equal function."""
        # Test with equal sequences
        seq1 = [1, 2, 3]
        seq2 = [1, 2, 3]
        assert sequences_equal(seq1, seq2)

        # Test with different length sequences
        seq3 = [1, 2, 3, 4]
        assert not sequences_equal(seq1, seq3)

        # Test with same length but different elements
        seq4 = [1, 2, 4]
        assert not sequences_equal(seq1, seq4)

        # Test with empty sequences
        assert sequences_equal([], [])

    def test_can_be_replaced_function(self):
        """Test the can_be_replaced function with mock nodes."""
        # Create mock nodes with the same attributes and inputs
        node1 = gs.Node(op="Add", name="add1")
        node1.attrs = {"axis": 1}

        node2 = gs.Node(op="Add", name="add2")
        node2.attrs = {"axis": 1}

        # Create mock inputs
        input1 = gs.Variable(name="input1")
        input2 = gs.Variable(name="input2")

        # Set same inputs for both nodes
        node1.inputs = [input1, input2]
        node2.inputs = [input1, input2]

        # Nodes should be replaceable
        assert can_be_replaced(node1, node2)

        # Test with different operations
        node3 = gs.Node(op="Mul", name="mul1")
        node3.attrs = {"axis": 1}
        node3.inputs = [input1, input2]
        assert not can_be_replaced(node1, node3)

        # Test with different attributes
        node4 = gs.Node(op="Add", name="add3")
        node4.attrs = {"axis": 2}  # Different axis
        node4.inputs = [input1, input2]
        assert not can_be_replaced(node1, node4)

        # Test with different inputs
        input3 = gs.Variable(name="input3")
        node5 = gs.Node(op="Add", name="add4")
        node5.attrs = {"axis": 1}
        node5.inputs = [input1, input3]  # Different second input
        assert not can_be_replaced(node1, node5)

    def test_duplicate_node_with_graph_output_keeps_output_node(self):
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
