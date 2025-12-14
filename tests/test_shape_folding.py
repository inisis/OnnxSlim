import numpy as np
import onnx
import pytest

import onnxslim.third_party.onnx_graphsurgeon as gs


class TestShapeFolding:
    """Test cases for fold_shape, fold_shape_gather, and fold_shape_slice functions
    with Shape node start/end attributes (opset 15+).
    """

    def test_fold_shape_without_start_end(self):
        """Test fold_shape with a basic Shape node (no start/end attributes)."""
        # Create input variable with known shape
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node output
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(4,))

        # Create Shape node
        shape_node = gs.Node(op="Shape", inputs=[input_var], outputs=[shape_output])

        # Create a dummy output using the shape
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=(4,))
        identity_node = gs.Node(op="Identity", inputs=[shape_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(nodes=[shape_node, identity_node], inputs=[input_var], outputs=[final_output])

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # The shape should be folded to a constant [2, 3, 4, 5]
        # Check that Shape node is removed after folding
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        assert len(shape_nodes) == 0, "Shape node should be folded"

    def test_fold_shape_with_start_end(self):
        """Test fold_shape with Shape node having start and end attributes."""
        # Create input variable with known shape
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node output (should output shape[1:3] = [3, 4])
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(2,))

        # Create Shape node with start=1, end=3
        shape_node = gs.Node(
            op="Shape",
            inputs=[input_var],
            outputs=[shape_output],
            attrs={"start": 1, "end": 3},
        )

        # Create a dummy output using the shape
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=(2,))
        identity_node = gs.Node(op="Identity", inputs=[shape_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(nodes=[shape_node, identity_node], inputs=[input_var], outputs=[final_output])

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # The shape should be folded to a constant [3, 4]
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        assert len(shape_nodes) == 0, "Shape node should be folded"

        # Verify the constant value
        for node in graph.nodes:
            for inp in node.inputs:
                if isinstance(inp, gs.Constant) and inp.name == "shape_output":
                    np.testing.assert_array_equal(inp.values, np.array([3, 4], dtype=np.int64))

    def test_fold_shape_with_negative_start(self):
        """Test fold_shape with Shape node having negative start index."""
        # Create input variable with known shape
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node output (start=-2 means last 2 dims = [4, 5])
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(2,))

        # Create Shape node with start=-2 (equivalent to start=2 for 4D tensor)
        shape_node = gs.Node(
            op="Shape",
            inputs=[input_var],
            outputs=[shape_output],
            attrs={"start": -2},
        )

        # Create a dummy output using the shape
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=(2,))
        identity_node = gs.Node(op="Identity", inputs=[shape_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(nodes=[shape_node, identity_node], inputs=[input_var], outputs=[final_output])

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # The shape should be folded to a constant [4, 5]
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        assert len(shape_nodes) == 0, "Shape node should be folded"

    def test_fold_shape_gather_without_start_end(self):
        """Test fold_shape_gather with basic Shape + Gather pattern."""
        # Create input variable with known shape
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node output
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(4,))
        shape_node = gs.Node(op="Shape", inputs=[input_var], outputs=[shape_output])

        # Create Gather to get dimension at index 2 (should be 4)
        indices = gs.Constant(name="indices", values=np.array(2, dtype=np.int64))
        gather_output = gs.Variable(name="gather_output", dtype=np.int64, shape=())
        gather_node = gs.Node(
            op="Gather",
            inputs=[shape_output, indices],
            outputs=[gather_output],
            attrs={"axis": 0},
        )

        # Create a dummy output
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=())
        identity_node = gs.Node(op="Identity", inputs=[gather_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(
            nodes=[shape_node, gather_node, identity_node],
            inputs=[input_var],
            outputs=[final_output],
        )

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # Both Shape and Gather should be folded
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        gather_nodes = [n for n in graph.nodes if n.op == "Gather"]
        assert len(shape_nodes) == 0, "Shape node should be folded"
        assert len(gather_nodes) == 0, "Gather node should be folded"

    def test_fold_shape_gather_with_start_end(self):
        """Test fold_shape_gather with Shape node having start/end + Gather."""
        # Create input variable with known shape (2, 3, 4, 5)
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node with start=1, end=3, output = [3, 4]
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(2,))
        shape_node = gs.Node(
            op="Shape",
            inputs=[input_var],
            outputs=[shape_output],
            attrs={"start": 1, "end": 3},
        )

        # Create Gather to get dimension at index 1 of the sliced shape (should be 4)
        indices = gs.Constant(name="indices", values=np.array(1, dtype=np.int64))
        gather_output = gs.Variable(name="gather_output", dtype=np.int64, shape=())
        gather_node = gs.Node(
            op="Gather",
            inputs=[shape_output, indices],
            outputs=[gather_output],
            attrs={"axis": 0},
        )

        # Create a dummy output
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=())
        identity_node = gs.Node(op="Identity", inputs=[gather_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(
            nodes=[shape_node, gather_node, identity_node],
            inputs=[input_var],
            outputs=[final_output],
        )

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # Both Shape and Gather should be folded
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        gather_nodes = [n for n in graph.nodes if n.op == "Gather"]
        assert len(shape_nodes) == 0, "Shape node should be folded"
        assert len(gather_nodes) == 0, "Gather node should be folded"

    def test_fold_shape_gather_with_negative_index(self):
        """Test fold_shape_gather with negative gather index."""
        # Create input variable with known shape
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node with start=1 (output = [3, 4, 5])
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(3,))
        shape_node = gs.Node(
            op="Shape",
            inputs=[input_var],
            outputs=[shape_output],
            attrs={"start": 1},
        )

        # Create Gather with index -1 (last element, should be 5)
        indices = gs.Constant(name="indices", values=np.array(-1, dtype=np.int64))
        gather_output = gs.Variable(name="gather_output", dtype=np.int64, shape=())
        gather_node = gs.Node(
            op="Gather",
            inputs=[shape_output, indices],
            outputs=[gather_output],
            attrs={"axis": 0},
        )

        # Create a dummy output
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=())
        identity_node = gs.Node(op="Identity", inputs=[gather_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(
            nodes=[shape_node, gather_node, identity_node],
            inputs=[input_var],
            outputs=[final_output],
        )

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # Both Shape and Gather should be folded
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        gather_nodes = [n for n in graph.nodes if n.op == "Gather"]
        assert len(shape_nodes) == 0, "Shape node should be folded"
        assert len(gather_nodes) == 0, "Gather node should be folded"

    def test_fold_shape_slice_without_start_end(self):
        """Test fold_shape_slice with basic Shape + Slice pattern."""
        # Create input variable with known shape
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node output
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(4,))
        shape_node = gs.Node(op="Shape", inputs=[input_var], outputs=[shape_output])

        # Create Slice to get shape[1:3] = [3, 4]
        starts = gs.Constant(name="starts", values=np.array([1], dtype=np.int64))
        ends = gs.Constant(name="ends", values=np.array([3], dtype=np.int64))
        axes = gs.Constant(name="axes", values=np.array([0], dtype=np.int64))
        slice_output = gs.Variable(name="slice_output", dtype=np.int64, shape=(2,))
        slice_node = gs.Node(
            op="Slice",
            inputs=[shape_output, starts, ends, axes],
            outputs=[slice_output],
        )

        # Create a dummy output
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=(2,))
        identity_node = gs.Node(op="Identity", inputs=[slice_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(
            nodes=[shape_node, slice_node, identity_node],
            inputs=[input_var],
            outputs=[final_output],
        )

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # Both Shape and Slice should be folded
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        slice_nodes = [n for n in graph.nodes if n.op == "Slice"]
        assert len(shape_nodes) == 0, "Shape node should be folded"
        assert len(slice_nodes) == 0, "Slice node should be folded"

    def test_fold_shape_slice_with_start_end(self):
        """Test fold_shape_slice with Shape node having start/end + Slice."""
        # Create input variable with known shape (2, 3, 4, 5)
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5))

        # Create Shape node with start=1 (output = [3, 4, 5])
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(3,))
        shape_node = gs.Node(
            op="Shape",
            inputs=[input_var],
            outputs=[shape_output],
            attrs={"start": 1},
        )

        # Create Slice to get shape_output[0:2] = [3, 4]
        starts = gs.Constant(name="starts", values=np.array([0], dtype=np.int64))
        ends = gs.Constant(name="ends", values=np.array([2], dtype=np.int64))
        axes = gs.Constant(name="axes", values=np.array([0], dtype=np.int64))
        slice_output = gs.Variable(name="slice_output", dtype=np.int64, shape=(2,))
        slice_node = gs.Node(
            op="Slice",
            inputs=[shape_output, starts, ends, axes],
            outputs=[slice_output],
        )

        # Create a dummy output
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=(2,))
        identity_node = gs.Node(op="Identity", inputs=[slice_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(
            nodes=[shape_node, slice_node, identity_node],
            inputs=[input_var],
            outputs=[final_output],
        )

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # Both Shape and Slice should be folded
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        slice_nodes = [n for n in graph.nodes if n.op == "Slice"]
        assert len(shape_nodes) == 0, "Shape node should be folded"
        assert len(slice_nodes) == 0, "Slice node should be folded"

    def test_fold_shape_slice_with_step(self):
        """Test fold_shape_slice with Slice having step parameter."""
        # Create input variable with known shape
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(2, 3, 4, 5, 6))

        # Create Shape node (output = [2, 3, 4, 5, 6])
        shape_output = gs.Variable(name="shape_output", dtype=np.int64, shape=(5,))
        shape_node = gs.Node(op="Shape", inputs=[input_var], outputs=[shape_output])

        # Create Slice to get shape[0:5:2] = [2, 4, 6]
        starts = gs.Constant(name="starts", values=np.array([0], dtype=np.int64))
        ends = gs.Constant(name="ends", values=np.array([5], dtype=np.int64))
        axes = gs.Constant(name="axes", values=np.array([0], dtype=np.int64))
        steps = gs.Constant(name="steps", values=np.array([2], dtype=np.int64))
        slice_output = gs.Variable(name="slice_output", dtype=np.int64, shape=(3,))
        slice_node = gs.Node(
            op="Slice",
            inputs=[shape_output, starts, ends, axes, steps],
            outputs=[slice_output],
        )

        # Create a dummy output
        final_output = gs.Variable(name="final_output", dtype=np.int64, shape=(3,))
        identity_node = gs.Node(op="Identity", inputs=[slice_output], outputs=[final_output])

        # Build graph
        graph = gs.Graph(
            nodes=[shape_node, slice_node, identity_node],
            inputs=[input_var],
            outputs=[final_output],
        )

        # Run fold_constants
        graph.fold_constants()
        graph.cleanup().toposort()

        # Both Shape and Slice should be folded
        shape_nodes = [n for n in graph.nodes if n.op == "Shape"]
        slice_nodes = [n for n in graph.nodes if n.op == "Slice"]
        assert len(shape_nodes) == 0, "Shape node should be folded"
        assert len(slice_nodes) == 0, "Slice node should be folded"


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main(
            [
                "-p",
                "no:warnings",
                "-v",
                "tests/test_shape_folding.py",
            ]
        )
    )
