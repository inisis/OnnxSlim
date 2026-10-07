import tempfile
import unittest

import numpy as np
import onnx
import sympy
from onnx import TensorProto, helper, numpy_helper
from utils import run_onnx

import onnxslim
import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern.elimination.slice import SlicePatternMatcher
from onnxslim.core.shape_inference.standard_ops.misc.resize import _resize_dim


def make_model(name, nodes, inputs, outputs, initializers, opset=13):
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, shape) for tensor_name, shape in inputs],
        [helper.make_tensor_value_info(tensor_name, dtype, shape) for tensor_name, dtype, shape in outputs],
        [numpy_helper.from_array(value, tensor_name) for tensor_name, value in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 7
    return model


class TestGraphValidityRegressions(unittest.TestCase):
    def assert_slim_preserves(self, model, feeds):
        onnx.checker.check_model(model, full_check=True)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as model_file:
            onnx.save(model, model_file.name)
            expected = run_onnx(model_file.name, feeds)
            optimized = onnxslim.slim(model)
            onnx.checker.check_model(optimized, full_check=True)
            onnx.save(optimized, model_file.name)
            actual = run_onnx(model_file.name, feeds)

        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            np.testing.assert_allclose(expected[name], actual[name], rtol=1e-5, atol=1e-6)
        return optimized

    def test_consecutive_reshape_keeps_zero_copy_semantics(self):
        model = make_model(
            "reshape-zero-copy",
            [
                helper.make_node("Reshape", ["X", "shape1"], ["R"]),
                helper.make_node("Reshape", ["R", "shape2"], ["Y"]),
            ],
            [("X", [2, 3])],
            [("Y", TensorProto.FLOAT, [3, 2])],
            {
                "shape1": np.array([3, 2], dtype=np.int64),
                "shape2": np.array([0, -1], dtype=np.int64),
            },
        )
        self.assert_slim_preserves(model, {"X": np.arange(6, dtype=np.float32).reshape(2, 3)})

    def test_consecutive_slice_normalizes_negative_axes_before_merging(self):
        model = make_model(
            "slice-negative-axis-alias",
            [
                helper.make_node("Slice", ["X", "begin0", "end0", "axis0", "step"], ["S"]),
                helper.make_node("Slice", ["S", "begin1", "end1", "axis1", "step"], ["Y"]),
            ],
            [("X", [4, 4])],
            [("Y", TensorProto.FLOAT, [4, 2])],
            {
                "begin0": np.array([1], dtype=np.int64),
                "end0": np.array([4], dtype=np.int64),
                "axis0": np.array([1], dtype=np.int64),
                "begin1": np.array([1], dtype=np.int64),
                "end1": np.array([3], dtype=np.int64),
                "axis1": np.array([-1], dtype=np.int64),
                "step": np.array([1], dtype=np.int64),
            },
        )
        optimized = self.assert_slim_preserves(model, {"X": np.arange(16, dtype=np.float32).reshape(4, 4)})
        self.assertEqual(sum(node.op_type == "Slice" for node in optimized.graph.node), 2)

    def test_consecutive_slice_with_unknown_rank_rejects_negative_axes(self):
        for first_axis, second_axis in ((-1, 0), (0, -1)):
            input_tensor = gs.Variable("X", dtype=np.float32, shape=None)
            intermediate = gs.Variable("S", dtype=np.float32)
            output = gs.Variable("Y", dtype=np.float32)
            step = gs.Constant("step", np.array([1], dtype=np.int64))
            gs.Node(
                "Slice",
                inputs=[
                    input_tensor,
                    gs.Constant("begin0", np.array([0], dtype=np.int64)),
                    gs.Constant("end0", np.array([2], dtype=np.int64)),
                    gs.Constant("axis0", np.array([first_axis], dtype=np.int64)),
                    step,
                ],
                outputs=[intermediate],
            )
            second = gs.Node(
                "Slice",
                inputs=[
                    intermediate,
                    gs.Constant("begin1", np.array([0], dtype=np.int64)),
                    gs.Constant("end1", np.array([1], dtype=np.int64)),
                    gs.Constant("axis1", np.array([second_axis], dtype=np.int64)),
                    step,
                ],
                outputs=[output],
            )
            matcher = SlicePatternMatcher(1)
            self.assertTrue(matcher.match(second))
            self.assertEqual(matcher.rewrite(), {})

    def test_resize_dim_symbolic_fallbacks(self):
        dim = sympy.Symbol("dim", integer=True, positive=True)
        scale = sympy.Symbol("scale", positive=True)
        self.assertEqual(_resize_dim(2, scale), sympy.floor(2 * scale))
        self.assertEqual(_resize_dim(dim, 0.5), sympy.floor(dim * sympy.Float(np.float32(0.5))))
        self.assertEqual(_resize_dim(dim, scale), sympy.floor(dim * scale))

    def test_slice_shape_inference_handles_empty_and_reverse_ranges(self):
        empty = make_model(
            "empty-slice",
            [helper.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"])],
            [("X", [1, 6, 10, 10])],
            [("Y", TensorProto.FLOAT, [1, 6, 0, 10])],
            {
                "starts": np.array([8], dtype=np.int64),
                "ends": np.array([1], dtype=np.int64),
                "axes": np.array([2], dtype=np.int64),
                "steps": np.array([1], dtype=np.int64),
            },
        )
        self.assert_slim_preserves(empty, {"X": np.ones((1, 6, 10, 10), dtype=np.float32)})

        reverse = make_model(
            "shape-slice-reverse",
            [
                helper.make_node("Shape", ["X"], ["S"], start=1, end=3),
                helper.make_node("Slice", ["S", "starts", "ends", "axes", "steps"], ["Y"]),
            ],
            [("X", [2, 3, 4])],
            [("Y", TensorProto.INT64, [2])],
            {
                "starts": np.array([1], dtype=np.int64),
                "ends": np.array([-3], dtype=np.int64),
                "axes": np.array([0], dtype=np.int64),
                "steps": np.array([-1], dtype=np.int64),
            },
            opset=15,
        )
        self.assert_slim_preserves(reverse, {"X": np.ones((2, 3, 4), dtype=np.float32)})

    def test_resize_scale_shape_uses_floor(self):
        model = make_model(
            "resize-floor",
            [helper.make_node("Resize", ["X", "", "scales"], ["Y"], mode="nearest")],
            [("X", [2, 3])],
            [("Y", TensorProto.FLOAT, [2, 1])],
            {"scales": np.array([1.0, 0.5], dtype=np.float32)},
        )
        self.assert_slim_preserves(model, {"X": np.arange(6, dtype=np.float32).reshape(2, 3)})

    def test_resize_dim_literal_scale_uses_double_precision(self):
        # 27 * 0.5185185f == 14.0f in float32, but onnx's shape inference (double
        # precision) floors to 13 and the onnx checker enforces that value.
        self.assertEqual(_resize_dim(27, np.float32(0.5185185)), 13)

    def test_overlapping_transpose_matches_preserve_shared_output(self):
        model = make_model(
            "overlapping-transposes",
            [
                helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0]),
                helper.make_node("Transpose", ["Y"], ["Z"], perm=[1, 0]),
                helper.make_node("Transpose", ["Y"], ["D"], perm=[1, 0]),
                helper.make_node("Transpose", ["Z"], ["A"], perm=[1, 0]),
            ],
            [("X", [2, 3])],
            [("A", TensorProto.FLOAT, [3, 2]), ("D", TensorProto.FLOAT, [2, 3])],
            {},
        )
        optimized = self.assert_slim_preserves(model, {"X": np.arange(6, dtype=np.float32).reshape(2, 3)})
        produced = {name for node in optimized.graph.node for name in node.output}
        self.assertTrue({"A", "D"}.issubset(produced))


if __name__ == "__main__":
    unittest.main()
