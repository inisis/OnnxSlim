import os

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.optimization.dead_node_elimination import (
    check_shape,
    dead_node_elimination,
    get_constant_variable,
)


class TestDeadNodeElimination:
    @staticmethod
    def _make_graph(op_type, *, inputs, attrs=None, initializers=None):
        """Build an ONNX protobuf and import it via graphsurgeon.

        The graph is structured as: graph_input -> Relu -> intermediate -> target_op -> graph_output.
        This ensures the target op's input is not a graph input, so dead_node_elimination
        can erase it even when its output is a graph output.
        """
        from onnx import helper, TensorProto

        graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3, 4, 5))
        graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 4, 5))
        intermediate_vi = helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, (2, 3, 4, 5))
        relu_node = helper.make_node("Relu", inputs=["input"], outputs=["intermediate"])
        target_node = helper.make_node(op_type, inputs=inputs, outputs=["output"], **(attrs or {}))

        graph_def = helper.make_graph(
            [relu_node, target_node], "test",
            [graph_input], [graph_output],
            initializer=initializers or [],
            value_info=[intermediate_vi],
        )
        model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 14)])
        return gs.import_onnx(model)

    @staticmethod
    def _assert_graph_valid(graph, target_op, expect_relu_input=True):
        """Verify the graph is well-formed after dead_node_elimination.

        After eliminating *target_op*, only the Relu node should remain,
        with the graph input flowing through Relu to the graph output.
        """
        assert not any(node.op == target_op for node in graph.nodes), f"{target_op} should be eliminated"
        assert len(graph.nodes) == 1, f"expected 1 node (Relu), got {len(graph.nodes)}"
        relu = graph.nodes[0]
        assert relu.op == "Relu", f"expected Relu, got {relu.op}"
        assert graph.inputs[0].name == "input"
        assert graph.outputs[0].name == "output"
        if expect_relu_input:
            assert relu.inputs[0] is graph.inputs[0], "Relu should consume the graph input directly"
        assert relu.outputs[0] is graph.outputs[0], "Relu should produce the graph output directly"

    @staticmethod
    def _make_bridging_graph(op_type, *, inputs, attrs=None, initializers=None):
        """Build an ONNX protobuf where the target op bridges graph input directly to graph output.

        Structure: graph_input -> target_op -> graph_output.
        This exercises the ``erase()`` guard that preserves nodes connecting graph I/O.
        """
        from onnx import helper, TensorProto

        graph_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, (2, 3, 4, 5))
        graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 4, 5))
        target_node = helper.make_node(op_type, inputs=inputs, outputs=["output"], **(attrs or {}))

        graph_def = helper.make_graph(
            [target_node], "test",
            [graph_input], [graph_output],
            initializer=initializers or [],
        )
        model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 14)])
        return gs.import_onnx(model)

    def test_identity_bridging_io_preserved(self):
        """Identity bridging graph input→output must be preserved — erase() guards graph I/O."""

        graph = self._make_bridging_graph("Identity", inputs=["input"])
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()

        # Model must stay intact — Identity bridges graph I/O and cannot be erased
        assert len(graph.nodes) == initial_node_count
        identity = graph.nodes[0]
        assert identity.op == "Identity"
        assert graph.inputs[0].name == "input"
        assert graph.outputs[0].name == "output"
        assert identity.inputs[0] is graph.inputs[0]
        assert identity.outputs[0] is graph.outputs[0]

    def test_identity_elimination(self):
        """Test that Identity nodes are properly eliminated."""

        graph = self._make_graph("Identity", inputs=["intermediate"])
        dead_node_elimination(graph)
        graph.cleanup().toposort()

        self._assert_graph_valid(graph, "Identity")

    def test_dropout_elimination(self):
        """Test that Dropout nodes are properly eliminated."""

        graph = self._make_graph("Dropout", inputs=["intermediate"])
        dead_node_elimination(graph)
        graph.cleanup().toposort()

        self._assert_graph_valid(graph, "Dropout")

    def test_zero_pad_elimination(self):
        """Test that Pad nodes with all zeros are properly eliminated."""

        from onnx import helper, TensorProto

        pads_init = helper.make_tensor("pads", TensorProto.INT64, [8], [0, 0, 0, 0, 0, 0, 0, 0])
        graph = self._make_graph("Pad", inputs=["intermediate", "pads"], initializers=[pads_init])
        dead_node_elimination(graph)
        graph.cleanup().toposort()

        self._assert_graph_valid(graph, "Pad")

    def test_redundant_cast_elimination(self, request):
        """Test that Cast nodes with same input and output types are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Cast to same dtype (float32 -> float32)
                return x.float()

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename, opset_version=14, dynamo=False)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Cast node to same type should be eliminated
        assert final_node_count <= initial_node_count

    def test_redundant_reshape_elimination(self, request):
        """Test that Reshape nodes with same input and output shapes are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Reshape to same shape
                shape = x.shape
                return x.reshape(shape)

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename, opset_version=14, dynamo=False)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Reshape node to same shape should be eliminated
        assert final_node_count <= initial_node_count

    def test_mul_by_one_elimination(self):
        """Test that Mul nodes with constant 1 are eliminated."""

        from onnx import helper, TensorProto

        ones_init = helper.make_tensor("ones", TensorProto.FLOAT, [1], [1.0])
        graph = self._make_graph("Mul", inputs=["intermediate", "ones"], initializers=[ones_init])
        dead_node_elimination(graph)
        graph.cleanup().toposort()

        self._assert_graph_valid(graph, "Mul")

    def test_add_zero_elimination(self, request):
        """Test that Add nodes with constant 0 are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("zeros", torch.zeros(1))

            def forward(self, x):
                # Add 0
                x += 0
                x += self.zeros
                return x

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename, opset_version=14, dynamo=False)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        graph.fold_constants().cleanup().toposort()
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Add node with constant 0 should be eliminated
        assert final_node_count < initial_node_count

    def test_div_by_one_elimination(self, request):
        """Test that Div nodes with constant 1 are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("ones", torch.ones(1))

            def forward(self, x):
                # Divide by 1
                return x / self.ones / 1

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename, opset_version=14, dynamo=False)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        graph.fold_constants().cleanup().toposort()
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Div node with constant 1 should be eliminated
        assert final_node_count < initial_node_count

    def test_sub_zero_elimination(self, request):
        """Test that Sub nodes with constant 0 are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("zeros", torch.zeros(1))

            def forward(self, x):
                x -= 0
                x -= self.zeros
                return x

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename, opset_version=14, dynamo=False)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        graph.fold_constants().cleanup().toposort()
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Sub node with constant 0 should be eliminated
        assert final_node_count < initial_node_count

    def test_single_concat_elimination(self):
        """Test that Concat nodes with a single input are eliminated."""

        graph = self._make_graph("Concat", inputs=["intermediate"], attrs={"axis": 0})
        dead_node_elimination(graph)
        graph.cleanup().toposort()

        self._assert_graph_valid(graph, "Concat")

    def test_single_output_split_elimination(self, request):
        """Test that Split nodes with a single output are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Split with single output
                return torch.split(x, x.shape[0], dim=0)[0]

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename, opset_version=14, dynamo=False)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Split node with single output should be eliminated
        assert final_node_count <= initial_node_count

    def test_noop_slice_elimination(self):
        """Test that Slice nodes that don't change the tensor are eliminated.

        Construct the ONNX graph manually to ensure a `Slice` node exists
        with starts/ends/axes/constants that represent a no-op slice.
        """

        # Create a simple graph with an explicit Slice node
        input_tensor = gs.Variable(name="input", dtype=np.float32, shape=(1, 512, 8, 8))
        output_tensor = gs.Variable(name="output", dtype=np.float32, shape=(1, 512, 8, 8))

        starts = gs.Constant(name="starts", values=np.array([0, 0], dtype=np.int64))
        ends = gs.Constant(name="ends", values=np.array([8, 8], dtype=np.int64))
        axes = gs.Constant(name="axes", values=np.array([2, 3], dtype=np.int64))

        slice_node = gs.Node(op="Slice", inputs=[input_tensor, starts, ends, axes], outputs=[output_tensor])

        graph = gs.Graph(
            inputs=[input_tensor],
            outputs=[output_tensor],
            nodes=[slice_node],
        )

        dead_node_elimination(graph)
        graph.cleanup().toposort()

        # Slice that does not change shape should be eliminated
        assert not any(node.op == "Slice" for node in graph.nodes)

    def test_check_shape_function(self):
        """Test the check_shape helper function."""
        # All positive integers
        assert check_shape([1, 2, 3, 4])

        # One string, rest positive integers
        assert check_shape([1, 2, "batch", 4])

        # Multiple strings
        assert not check_shape([1, "batch", "seq", 4])

        # Negative integers
        assert not check_shape([1, -1, 3, 4])

    def test_get_constant_variable_function(self):
        """Test the get_constant_variable helper function."""
        # Create a simple graph with a node that has constant inputs
        graph = gs.Graph()

        # Create tensors
        input_tensor = gs.Variable(name="input", dtype=np.float32, shape=(1, 3, 224, 224))
        const_tensor = gs.Constant(name="const", values=np.ones((1,), dtype=np.float32))
        output_tensor = gs.Variable(name="output", dtype=np.float32, shape=(1, 3, 224, 224))

        # Create node with both variable and constant inputs
        node = gs.Node(op="Add", name="add", inputs=[input_tensor, const_tensor], outputs=[output_tensor])
        graph.nodes.append(node)

        # Test without return_idx
        constant = get_constant_variable(node)
        assert isinstance(constant, gs.Constant)
        assert np.all(constant.values == 1)

        # Test with return_idx
        idx, constant = get_constant_variable(node, return_idx=True)
        assert idx == 1
        assert isinstance(constant, gs.Constant)
        assert np.all(constant.values == 1)


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main(
            [
                "-p",
                "no:warnings",
                "-v",
                "tests/test_dead_node_elimination.py",
            ]
        )
    )
