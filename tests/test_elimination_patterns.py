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

        model.ir_version = 7
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

    def test_concat_pattern_preserves_intermediate_graph_output(self):
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
        intermediate_tensor = helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, [1, 3, 4, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])

        node1 = helper.make_node("Concat", ["input"], ["intermediate"], axis=1)
        node2 = helper.make_node("Concat", ["intermediate"], ["output"], axis=1)
        graph = helper.make_graph(
            [node1, node2],
            "concat-intermediate-output-test",
            [input_tensor],
            [intermediate_tensor, output_tensor],
        )
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11
        model.ir_version = 7

        optimized_model = onnxslim.slim(model, skip_optimizations=["dead_node_elimination"])

        onnx.checker.check_model(optimized_model)
        output_names = {output.name for output in optimized_model.graph.output}
        produced_names = {output for node in optimized_model.graph.node for output in node.output}
        self.assertEqual(output_names, {"intermediate", "output"})
        self.assertTrue(output_names.issubset(produced_names))

    def test_reshape_pattern_preserves_intermediate_graph_output(self):
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
        intermediate_tensor = helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, [3, 2])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [6])
        shape1 = numpy_helper.from_array(np.array([3, 2], dtype=np.int64), name="shape1")
        shape2 = numpy_helper.from_array(np.array([6], dtype=np.int64), name="shape2")

        node1 = helper.make_node("Reshape", ["input", "shape1"], ["intermediate"])
        node2 = helper.make_node("Reshape", ["intermediate", "shape2"], ["output"])
        graph = helper.make_graph(
            [node1, node2],
            "reshape-intermediate-output-test",
            [input_tensor],
            [intermediate_tensor, output_tensor],
            initializer=[shape1, shape2],
        )
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11
        model.ir_version = 7

        optimized_model = onnxslim.slim(model, skip_optimizations=["dead_node_elimination"])

        onnx.checker.check_model(optimized_model)
        output_names = {output.name for output in optimized_model.graph.output}
        produced_names = {output for node in optimized_model.graph.node for output in node.output}
        self.assertEqual(output_names, {"intermediate", "output"})
        self.assertTrue(output_names.issubset(produced_names))

    def test_reshape_pattern(self):
        # Create a model with a reshape pattern that can be eliminated
        # Input -> Reshape -> Reshape -> Output
        # Two consecutive reshapes are merged into a single reshape
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 12, 112, 112])

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

        model.ir_version = 7
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
        ends = numpy_helper.from_array(np.array([9, 9], dtype=np.int64), name="ends")
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

        model.ir_version = 7
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

        model.ir_version = 7
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

    def test_consecutive_unsqueeze_opset11(self):
        # Consecutive Unsqueeze elimination at opset 11: axes carried as attributes.
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4, 1, 1])

        node1 = helper.make_node("Unsqueeze", ["input"], ["intermediate"], axes=[-1])
        node2 = helper.make_node("Unsqueeze", ["intermediate"], ["output"], axes=[-1])

        graph = helper.make_graph(
            [node1, node2], "consecutive-unsqueeze-opset11", [input_tensor], [output_tensor]
        )
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11
        model.ir_version = 7

        input_data = np.random.randn(3, 4).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"input": input_data})

            optimized_model = onnxslim.slim(model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})

            np.testing.assert_allclose(original_output["output"], optimized_output["output"], rtol=1e-5)
            unsqueeze_count = sum(1 for n in optimized_model.graph.node if n.op_type == "Unsqueeze")
            self.assertEqual(unsqueeze_count, 1)
        os.unlink(f.name)

    def test_consecutive_unsqueeze_opset13(self):
        # Consecutive Unsqueeze elimination at opset 13: axes carried as Constant inputs.
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4, 1, 1])
        axes1 = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes1")
        axes2 = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes2")

        node1 = helper.make_node("Unsqueeze", ["input", "axes1"], ["intermediate"])
        node2 = helper.make_node("Unsqueeze", ["intermediate", "axes2"], ["output"])

        graph = helper.make_graph(
            [node1, node2],
            "consecutive-unsqueeze-opset13",
            [input_tensor],
            [output_tensor],
            initializer=[axes1, axes2],
        )
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7

        input_data = np.random.randn(3, 4).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"input": input_data})

            optimized_model = onnxslim.slim(model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})

            np.testing.assert_allclose(original_output["output"], optimized_output["output"], rtol=1e-5)
            unsqueeze_count = sum(1 for n in optimized_model.graph.node if n.op_type == "Unsqueeze")
            self.assertEqual(unsqueeze_count, 1)
        os.unlink(f.name)


class TestTransposeViewTransposePattern(unittest.TestCase):
    """Transpose -> ViewOp -> Transpose: the ViewOp is pulled before the first Transpose so the two
    Transposes end up adjacent and are merged by EliminationTranspose on a later iteration.
    """

    def _check(self, model, input_shape, expect_transpose_count=None):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"a": input_data})

            optimized_model = onnxslim.slim(model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"a": input_data})

            out_name = list(original_output.keys())[0]
            np.testing.assert_allclose(original_output[out_name], optimized_output[out_name], rtol=1e-5, atol=1e-6)
            if expect_transpose_count is not None:
                transpose_count = sum(1 for n in optimized_model.graph.node if n.op_type == "Transpose")
                self.assertEqual(transpose_count, expect_transpose_count)
        os.unlink(f.name)
        return optimized_model

    def test_reshape_split_across_transpose(self):
        # Only Reshapes that *split* every post-transpose axis into its own contiguous run of output
        # axes are supported (matches onnxruntime's HandleReshapeSplit) - output rank must exceed the
        # post-transpose rank.
        # a:[1,6,4,5] -T0(perm=0,3,1,2)-> b:[1,5,6,4] -Reshape[1,5,2,3,4] (splits the "6" into 2*3)->
        # c:[1,5,2,3,4] -T1(perm=0,1,2,4,3)-> d:[1,5,2,4,3]
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 6, 4, 5])
        d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [1, 5, 2, 4, 3])
        shape_const = helper.make_tensor("shape", TensorProto.INT64, [5], [1, 5, 2, 3, 4])
        nodes = [
            helper.make_node("Transpose", ["a"], ["b"], perm=[0, 3, 1, 2]),
            helper.make_node("Reshape", ["b", "shape"], ["c"]),
            helper.make_node("Transpose", ["c"], ["d"], perm=[0, 1, 2, 4, 3]),
        ]
        graph = helper.make_graph(nodes, "reshape-split", [a], [d], initializer=[shape_const])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (1, 6, 4, 5), expect_transpose_count=1)

    def test_reshape_equal_rank_across_transpose(self):
        # a:[2,3,1] -T0(perm=1,0,2)-> b:[3,2,1] -Reshape[1,3,2]-> c
        # The Reshape only moves the size-1 axis, so it is equivalent to perm=[2,0,1].
        # It first merges with T0; the following identity Transpose is eliminated on the next iteration.
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 1])
        d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [1, 3, 2])
        shape_const = helper.make_tensor("shape", TensorProto.INT64, [3], [1, 3, 2])
        nodes = [
            helper.make_node("Transpose", ["a"], ["b"], perm=[1, 0, 2]),
            helper.make_node("Reshape", ["b", "shape"], ["c"]),
            helper.make_node("Transpose", ["c"], ["d"], perm=[0, 1, 2]),
        ]
        graph = helper.make_graph(nodes, "reshape-as-transpose", [a], [d], initializer=[shape_const])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        optimized = self._check(model, (2, 3, 1), expect_transpose_count=1)
        self.assertFalse(any(n.op_type == "Reshape" for n in optimized.graph.node))

    def test_unsqueeze_before_transpose(self):
        # a:[1024,8,96] -T0(perm=1,0,2)-> b:[8,1024,96] -Unsqueeze(axes=[1])-> c:[8,1,1024,96] -T1(identity)-> d
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1024, 8, 96])
        d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [8, 1, 1024, 96])
        axes_const = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
        nodes = [
            helper.make_node("Transpose", ["a"], ["b"], perm=[1, 0, 2]),
            helper.make_node("Unsqueeze", ["b", "axes"], ["c"]),
            helper.make_node("Transpose", ["c"], ["d"], perm=[0, 1, 2, 3]),
        ]
        graph = helper.make_graph(nodes, "unsqueeze-before-transpose", [a], [d], initializer=[axes_const])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (1024, 8, 96), expect_transpose_count=1)

    def test_squeeze_before_transpose(self):
        # a:[8,1,1024,96] -T0(perm=1,0,2,3)-> b:[1,8,1024,96] -Squeeze(axes=[0])-> c:[8,1024,96]
        # -T1(perm=1,0,2)-> d:[1024,8,96]
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [8, 1, 1024, 96])
        d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [1024, 8, 96])
        axes_const = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
        nodes = [
            helper.make_node("Transpose", ["a"], ["b"], perm=[1, 0, 2, 3]),
            helper.make_node("Squeeze", ["b", "axes"], ["c"]),
            helper.make_node("Transpose", ["c"], ["d"], perm=[1, 0, 2]),
        ]
        graph = helper.make_graph(nodes, "squeeze-before-transpose", [a], [d], initializer=[axes_const])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (8, 1, 1024, 96), expect_transpose_count=1)

    def test_squeeze_omitted_axes_before_transpose(self):
        # Same shapes as test_squeeze_before_transpose, but `axes` is omitted (opset >= 13 infers every
        # statically-known size-1 axis) - a common real export pattern, not just the explicit-axes form.
        a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [8, 1, 1024, 96])
        d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [1024, 8, 96])
        nodes = [
            helper.make_node("Transpose", ["a"], ["b"], perm=[1, 0, 2, 3]),
            helper.make_node("Squeeze", ["b"], ["c"]),
            helper.make_node("Transpose", ["c"], ["d"], perm=[1, 0, 2]),
        ]
        graph = helper.make_graph(nodes, "squeeze-omitted-axes", [a], [d])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (8, 1, 1024, 96), expect_transpose_count=1)


class TestTransposeAsReshapePattern(unittest.TestCase):
    """A Transpose that only moves size-1 axes around (never reorders any real data) is equivalent to a
    Reshape; rewriting it as one lets it merge with an adjacent Reshape via EliminationReshape.
    """

    def _check(self, model, input_shape, expect_transpose_count, expect_reshape_count=None):
        input_data = np.random.randn(*input_shape).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            original_output = run_onnx(f.name, {"x": input_data})

            optimized_model = onnxslim.slim(model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"x": input_data})

            out_name = list(original_output.keys())[0]
            np.testing.assert_allclose(original_output[out_name], optimized_output[out_name], rtol=1e-5, atol=1e-6)
            transpose_count = sum(1 for n in optimized_model.graph.node if n.op_type == "Transpose")
            self.assertEqual(transpose_count, expect_transpose_count)
            if expect_reshape_count is not None:
                reshape_count = sum(1 for n in optimized_model.graph.node if n.op_type == "Reshape")
                self.assertEqual(reshape_count, expect_reshape_count)
        os.unlink(f.name)

    def test_transpose_as_reshape_merges_with_adjacent_reshape(self):
        # x:[10,768] -> Reshape[10,1,768] -> Transpose(perm=[1,0,2]) -> [1,10,768]
        # perm=[1,0,2] only swaps a size-1 axis with a real one, so it never reorders data: the whole
        # chain collapses into a single Reshape.
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [10, 768])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 10, 768])
        shape_const = helper.make_tensor("shape", TensorProto.INT64, [3], [10, 1, 768])
        nodes = [
            helper.make_node("Reshape", ["x", "shape"], ["mid"]),
            helper.make_node("Transpose", ["mid"], ["out"], perm=[1, 0, 2]),
        ]
        graph = helper.make_graph(nodes, "transpose-as-reshape", [x], [out], initializer=[shape_const])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (10, 768), expect_transpose_count=0, expect_reshape_count=1)

    def test_data_reordering_transpose_is_unchanged(self):
        # An adjacent Reshape makes the pattern eligible, but swapping two non-1 axes still requires
        # a real data reorder and must remain a Transpose.
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 3])
        shape_const = helper.make_tensor("shape", TensorProto.INT64, [3], [1, 3, 4])
        nodes = [
            helper.make_node("Reshape", ["x", "shape"], ["mid"]),
            helper.make_node("Transpose", ["mid"], ["out"], perm=[0, 2, 1]),
        ]
        graph = helper.make_graph(nodes, "real-transpose", [x], [out], initializer=[shape_const])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (3, 4), expect_transpose_count=1, expect_reshape_count=1)

    def test_layout_preserving_transpose_without_reshape_chain_is_unchanged(self):
        # Although this Transpose only moves a size-1 axis, replacing it with a standalone Reshape
        # provides no simplification and may prevent a backend from recognizing the Transpose itself.
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 1])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])
        nodes = [helper.make_node("Transpose", ["x"], ["out"], perm=[1, 0])]
        graph = helper.make_graph(nodes, "standalone-layout-transpose", [x], [out])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (2, 1), expect_transpose_count=1, expect_reshape_count=0)

    def test_transpose_as_reshape_merges_with_following_reshape(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [10, 1, 768])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [10, 768])
        shape_const = helper.make_tensor("shape", TensorProto.INT64, [2], [10, 768])
        nodes = [
            helper.make_node("Transpose", ["x"], ["mid"], perm=[1, 0, 2]),
            helper.make_node("Reshape", ["mid", "shape"], ["out"]),
        ]
        graph = helper.make_graph(nodes, "transpose-before-reshape", [x], [out], initializer=[shape_const])
        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 13
        model.ir_version = 7
        self._check(model, (10, 1, 768), expect_transpose_count=0, expect_reshape_count=1)

if __name__ == "__main__":
    unittest.main()
