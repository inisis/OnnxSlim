import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto
from utils import run_onnx

import onnxslim
from onnxslim.core.pattern.elimination.concat import ConcatPatternMatcher
from onnxslim.core.pattern.elimination.reshape import ReshapePatternMatcher
from onnxslim.core.pattern.elimination.reshape_as import ReshapeAsPatternMatcher
from onnxslim.core.pattern.elimination.slice import SlicePatternMatcher
from onnxslim.core.pattern.elimination.unsqueeze import UnsqueezePatternMatcher


class TestEliminationPatterns(unittest.TestCase):
    def test_concat_pattern(self):
        # Create a model with two sequential concat operations
        # Input -> Concat1 -> Concat2 -> Output
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        node1 = helper.make_node("Concat", ["input"], ["intermediate"], axis=1)

        node2 = helper.make_node("Concat", ["intermediate"], ["output"], axis=1)

        graph = helper.make_graph([node1, node2], "concat-test", [input_tensor], [output_tensor])

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11

        # Test the pattern matcher directly
        matcher = ConcatPatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Test with onnxslim optimization
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"input": input_data})
            # Optimize the model
            optimized_model = onnxslim.slim(model, skip_optimizations=["dead_node_elimination"])
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})

            # Check that the outputs are the same
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], rtol=1e-5)

            # Check that the concat nodes were eliminated or simplified
            optimized_graph = optimized_model.graph
            # The pattern should eliminate at least one of the concat nodes
            self.assertLess(len(optimized_graph.node), 2)

        os.unlink(f.name)

    def test_reshape_pattern(self):
        # Create a model with a reshape pattern that can be eliminated
        # Input -> Reshape -> Reshape -> Output
        # where the second Reshape reverses the first
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        # Create shape tensors
        shape1 = numpy_helper.from_array(np.array([1, 3 * 224, 224], dtype=np.int64), name="shape1")
        shape2 = numpy_helper.from_array(np.array([1, 12, 112, 112], dtype=np.int64), name="shape2")

        node1 = helper.make_node(
            "Reshape",
            ["input", "shape1"],
            ["intermediate"],
        )

        node2 = helper.make_node(
            "Reshape",
            ["intermediate", "shape2"],
            ["output"],
        )

        graph = helper.make_graph(
            [node1, node2], "reshape-test", [input_tensor], [output_tensor], initializer=[shape1, shape2]
        )

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11

        # Test the pattern matcher directly
        matcher = ReshapePatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Test with onnxslim optimization
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"input": input_data})
            # Optimize the model
            optimized_model = onnxslim.slim(model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})

            # Check that the outputs are the same
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], rtol=1e-5)

        os.unlink(f.name)

    def test_slice_pattern(self):
        # Create a model with a slice pattern that can be eliminated
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10, 10])

        # Create constant tensors for Slice parameters
        starts = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="starts")
        ends = numpy_helper.from_array(np.array([10, 10], dtype=np.int64), name="ends")
        axes = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="axes")

        node = helper.make_node(
            "Slice",
            ["input", "starts", "ends", "axes"],
            ["output"],
        )

        graph = helper.make_graph(
            [node], "slice-test", [input_tensor], [output_tensor], initializer=[starts, ends, axes]
        )

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11

        # Test the pattern matcher directly
        matcher = SlicePatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Test with onnxslim optimization
        input_data = np.random.randn(10, 10).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"input": input_data})

            # Optimize the model
            optimized_model = onnxslim.slim(model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})

            # Check that the outputs are the same
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], rtol=1e-5)

        os.unlink(f.name)

    def test_unsqueeze_pattern(self):
        # Create a model with an unsqueeze pattern that can be eliminated
        # Input -> Unsqueeze -> Squeeze -> Output
        # where Squeeze reverses Unsqueeze
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4, 5])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4, 5])

        # Create axes tensors
        axes1 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes1")
        axes2 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes2")

        node1 = helper.make_node(
            "Unsqueeze",
            ["input", "axes1"],
            ["intermediate"],
        )

        node2 = helper.make_node(
            "Squeeze",
            ["intermediate", "axes2"],
            ["output"],
        )

        graph = helper.make_graph(
            [node1, node2], "unsqueeze-test", [input_tensor], [output_tensor], initializer=[axes1, axes2]
        )

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 14

        # Test the pattern matcher directly
        matcher = UnsqueezePatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Test with onnxslim optimization
        input_data = np.random.randn(3, 4, 5).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"input": input_data})

            # Optimize the model
            optimized_model = onnxslim.slim(model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})

            # Check that the outputs are the same
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], rtol=1e-5)

        os.unlink(f.name)

    def test_reshape_as_pattern(self):
        # Test the ReshapeAs pattern matcher
        matcher = ReshapeAsPatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))


if __name__ == "__main__":
    unittest.main()
