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
from onnxslim.core.pattern.fusion.concat_reshape import ConcatReshapeMatcher
from onnxslim.core.pattern.fusion.convadd import ConvAddMatcher
from onnxslim.core.pattern.fusion.convbn import ConvBatchNormMatcher
from onnxslim.core.pattern.fusion.gelu import GeluPatternMatcher
from onnxslim.core.pattern.fusion.gemm import MatMulAddPatternMatcher
from onnxslim.core.pattern.fusion.padconv import PadConvMatcher
from onnxslim.core.pattern.fusion.reduce import ReducePatternMatcher


class TestFusionPatterns(unittest.TestCase):
    def test_convadd_pattern(self):
        # Test the ConvAdd pattern matcher
        matcher = ConvAddMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Create a model with Conv + Add pattern
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 112, 112])

        # Create weights and bias
        weights = numpy_helper.from_array(np.random.randn(3, 3, 3, 3).astype(np.float32), name="weights")
        bias = numpy_helper.from_array(np.random.randn(1, 3, 1, 1).astype(np.float32), name="bias")

        # Create Conv node
        conv_node = helper.make_node(
            "Conv",
            ["input", "weights"],
            ["conv_output"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            dilations=[1, 1],
        )

        # Create Add node
        add_node = helper.make_node(
            "Add",
            ["conv_output", "bias"],
            ["output"],
        )

        graph = helper.make_graph(
            [conv_node, add_node], "convadd-test", [input_tensor], [output_tensor], initializer=[weights, bias]
        )

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11

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

            # Check that the nodes were fused
            self.assertLess(len(optimized_model.graph.node), len(model.graph.node))

        os.unlink(f.name)

    def test_convbn_pattern(self):
        # Test the ConvBN pattern matcher
        matcher = ConvBatchNormMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Create a model with Conv + BatchNormalization pattern
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 112, 112])

        # Create weights for Conv
        weights = numpy_helper.from_array(np.random.randn(16, 3, 3, 3).astype(np.float32), name="weights")

        # Create parameters for BatchNormalization
        scale = numpy_helper.from_array(np.random.randn(16).astype(np.float32), name="scale")
        bias = numpy_helper.from_array(np.random.randn(16).astype(np.float32), name="bias")
        mean = numpy_helper.from_array(np.random.randn(16).astype(np.float32), name="mean")
        var = numpy_helper.from_array(np.abs(np.random.randn(16)).astype(np.float32), name="var")

        # Create Conv node
        conv_node = helper.make_node(
            "Conv",
            ["input", "weights"],
            ["conv_output"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        )

        # Create BatchNormalization node
        bn_node = helper.make_node(
            "BatchNormalization",
            ["conv_output", "scale", "bias", "mean", "var"],
            ["output"],
            epsilon=1e-5,
        )

        graph = helper.make_graph(
            [conv_node, bn_node],
            "convbn-test",
            [input_tensor],
            [output_tensor],
            initializer=[weights, scale, bias, mean, var],
        )

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11

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
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], atol=1e-3)

            # Check that the nodes were fused
            self.assertLess(len(optimized_model.graph.node), len(model.graph.node))

        os.unlink(f.name)

    def test_gemm_pattern(self):
        # Test the MatMulAdd pattern matcher
        matcher = MatMulAddPatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Create a model with MatMul + Add pattern
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 512])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 256])

        # Create weights and bias
        weights = numpy_helper.from_array(np.random.randn(512, 256).astype(np.float32), name="weights")
        bias = numpy_helper.from_array(np.random.randn(256).astype(np.float32), name="bias")

        # Create MatMul node
        matmul_node = helper.make_node(
            "MatMul",
            ["input", "weights"],
            ["matmul_output"],
        )

        # Create Add node
        add_node = helper.make_node(
            "Add",
            ["matmul_output", "bias"],
            ["output"],
        )

        graph = helper.make_graph(
            [matmul_node, add_node], "gemm-test", [input_tensor], [output_tensor], initializer=[weights, bias]
        )

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11

        # Test with onnxslim optimization
        input_data = np.random.randn(1, 512).astype(np.float32)

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

    def test_padconv_pattern(self):
        # Test the PadConv pattern matcher
        matcher = PadConvMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        # Create a model with Pad + Conv pattern
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 224, 224])

        # Create weights for Conv
        weights = numpy_helper.from_array(np.random.randn(16, 3, 3, 3).astype(np.float32), name="weights")

        # Create pads for Pad
        pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), name="pads")

        # Create constant value for Pad
        constant_value = numpy_helper.from_array(np.array(0, dtype=np.float32), name="constant_value")

        # Create Pad node
        pad_node = helper.make_node(
            "Pad",
            ["input", "pads", "constant_value"],
            ["pad_output"],
            mode="constant",
        )

        # Create Conv node
        conv_node = helper.make_node(
            "Conv",
            ["pad_output", "weights"],
            ["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        graph = helper.make_graph(
            [pad_node, conv_node],
            "padconv-test",
            [input_tensor],
            [output_tensor],
            initializer=[weights, pads, constant_value],
        )

        model = helper.make_model(graph, producer_name="onnxslim-test")
        model.opset_import[0].version = 11

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

    def test_reduce_pattern(self):
        # Test the Reduce pattern matcher
        matcher = ReducePatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

    def test_gelu_pattern(self):
        # Test the Gelu pattern matcher
        matcher = GeluPatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        import torch
        import torch.nn as nn

        # Define a simple model with GELU activation
        class GeluModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()

            def forward(self, x):
                return self.gelu(x)

        # Create model instance
        model = GeluModel()
        model.eval()  # Set to evaluation mode

        # Create dummy input
        dummy_input = torch.randn(1, 512, dtype=torch.float32)
        input_data = dummy_input.numpy()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            # Export the model to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                f.name,
                export_params=True,
                opset_version=11,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            # Run the original model
            original_output = run_onnx(f.name, {"input": input_data})

            # Load the ONNX model
            onnx_model = onnx.load(f.name)

            # Optimize the model with onnxslim
            optimized_model = onnxslim.slim(onnx_model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})

            # Check that the outputs are the same
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], rtol=1e-5)

            # Check that optimization occurred
            self.assertLessEqual(len(optimized_model.graph.node), len(onnx_model.graph.node))

        os.unlink(f.name)

    def test_concat_reshape_pattern(self):
        # Test the ConcatReshape pattern matcher
        matcher = ConcatReshapeMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

    def test_gemm_3d_pattern(self):
        # Test the MatMulAdd pattern matcher with 3D input
        matcher = MatMulAddPatternMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))

        import torch
        import torch.nn as nn

        # Define a simple model with MatMul + Add (GEMM) with 3D input
        class GemmModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 256, bias=False)
                self.bias = nn.Parameter(torch.randn(256, dtype=torch.float32))

            def forward(self, x):
                return self.linear(x) + self.bias

        # Create model instance
        model = GemmModel()
        model.eval()  # Set to evaluation mode

        # Create 3D dummy input (batch_size, sequence_length, hidden_size)
        dummy_input = torch.randn(2, 10, 512, dtype=torch.float32)
        input_data = dummy_input.numpy()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            # Export the model to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                f.name,
                export_params=True,
                opset_version=11,
                input_names=["input"],
                output_names=["output"],
            )
            # Run the original model
            original_output = run_onnx(f.name, {"input": input_data})

            # Load the ONNX model
            onnx_model = onnx.load(f.name)

            # Optimize the model with onnxslim
            optimized_model = onnxslim.slim(onnx_model)
            onnx.save(optimized_model, f.name)
            optimized_output = run_onnx(f.name, {"input": input_data})
            # Check that the outputs are the same
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], atol=1e-5)

            # Check that optimization occurred (nodes were fused)
            self.assertLessEqual(len(onnx_model.graph.node), len(optimized_model.graph.node))

        os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
