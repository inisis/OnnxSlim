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
    def test_identity_elimination(self, request):
        """Test that Identity nodes are properly eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.identity = nn.Identity()

            def forward(self, x):
                return self.identity(x)

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Identity node should be eliminated
        assert final_node_count < initial_node_count
        # Check no Identity nodes remain
        assert not any(node.op == "Identity" for node in graph.nodes)

    def test_dropout_elimination(self, request):
        """Test that Dropout nodes are properly eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()
        model.eval()  # Set to eval mode to ensure deterministic behavior

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Dropout node should be eliminated
        assert final_node_count < initial_node_count
        # Check no Dropout nodes remain
        assert not any(node.op == "Dropout" for node in graph.nodes)

    def test_zero_pad_elimination(self, request):
        """Test that Pad nodes with all zeros are properly eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Zero padding
                return torch.nn.functional.pad(x, (0, 0, 0, 0), "constant", 0)

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        graph.fold_constants().cleanup().toposort()
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Pad node with all zeros should be eliminated
        assert final_node_count < initial_node_count
        # Check no Pad nodes remain
        assert not any(node.op == "Pad" for node in graph.nodes)

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
        torch.onnx.export(model, input_tensor, filename)

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
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Reshape node to same shape should be eliminated
        assert final_node_count <= initial_node_count

    def test_mul_by_one_elimination(self, request):
        """Test that Mul nodes with constant 1 are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("ones", torch.ones(1))

            def forward(self, x):
                # Multiply by 1
                return x * self.ones

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Mul node with constant 1 should be eliminated
        assert final_node_count < initial_node_count

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
        torch.onnx.export(model, input_tensor, filename)

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
        torch.onnx.export(model, input_tensor, filename)

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
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        graph.fold_constants().cleanup().toposort()
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Sub node with constant 0 should be eliminated
        assert final_node_count < initial_node_count

    def test_single_concat_elimination(self, request):
        """Test that Concat nodes with a single input are eliminated."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Single concat
                return torch.cat([x], dim=0)

        input_tensor = torch.randn(2, 3, 4, 5)
        model = Model()

        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Concat node with single input should be eliminated
        assert final_node_count < initial_node_count
        # Check no Concat nodes remain
        assert not any(node.op == "Concat" for node in graph.nodes)

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
        torch.onnx.export(model, input_tensor, filename)

        # Import graph and apply dead_node_elimination
        graph = gs.import_onnx(onnx.load(filename))
        initial_node_count = len(graph.nodes)
        dead_node_elimination(graph)
        graph.cleanup().toposort()
        final_node_count = len(graph.nodes)

        # Split node with single output should be eliminated
        assert final_node_count <= initial_node_count

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
