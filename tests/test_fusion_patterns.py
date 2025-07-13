import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto

import onnxslim
from onnxslim.core.pattern.fusion.convadd import ConvAddMatcher
from onnxslim.core.pattern.fusion.convbn import ConvBatchNormMatcher
from onnxslim.core.pattern.fusion.gemm import MatMulAddPatternMatcher
from onnxslim.core.pattern.fusion.gelu import GeluPatternMatcher
from onnxslim.core.pattern.fusion.padconv import PadConvMatcher
from onnxslim.core.pattern.fusion.reduce import ReducePatternMatcher
from onnxslim.core.pattern.fusion.concat_reshape import ConcatReshapeMatcher
from utils import create_model, run_onnx


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
        weights = numpy_helper.from_array(
            np.random.randn(3, 3, 3, 3).astype(np.float32), 
            name="weights"
        )
        bias = numpy_helper.from_array(
            np.random.randn(1, 3, 1 ,1).astype(np.float32), 
            name="bias"
        )
        
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
            [conv_node, add_node],
            "convadd-test",
            [input_tensor],
            [output_tensor],
            initializer=[weights, bias]
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
        weights = numpy_helper.from_array(
            np.random.randn(16, 3, 3, 3).astype(np.float32), 
            name="weights"
        )
        
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
            initializer=[weights, scale, bias, mean, var]
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
            np.testing.assert_allclose(original_output["output"], optimized_output["output"], atol=1e-5)
            
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
        weights = numpy_helper.from_array(
            np.random.randn(512, 256).astype(np.float32), 
            name="weights"
        )
        bias = numpy_helper.from_array(
            np.random.randn(256).astype(np.float32), 
            name="bias"
        )
        
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
            [matmul_node, add_node],
            "gemm-test",
            [input_tensor],
            [output_tensor],
            initializer=[weights, bias]
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
        weights = numpy_helper.from_array(
            np.random.randn(16, 3, 3, 3).astype(np.float32), 
            name="weights"
        )
        
        # Create pads for Pad
        pads = numpy_helper.from_array(
            np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), 
            name="pads"
        )
        
        # Create constant value for Pad
        constant_value = numpy_helper.from_array(
            np.array(0, dtype=np.float32), 
            name="constant_value"
        )
        
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
            initializer=[weights, pads, constant_value]
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
    
    def test_concat_reshape_pattern(self):
        # Test the ConcatReshape pattern matcher
        matcher = ConcatReshapeMatcher(1)
        self.assertTrue(hasattr(matcher, "match"))
        self.assertTrue(hasattr(matcher, "rewrite"))


if __name__ == "__main__":
    unittest.main() 