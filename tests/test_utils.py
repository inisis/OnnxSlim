import os
import tempfile
import unittest

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from onnxslim.utils import (
    ModelInfo,
    TensorInfo,
    calculate_tensor_size,
    check_onnx,
    check_point,
    check_result,
    dump_model_info_to_disk,
    format_bytes,
    format_model_info,
    gen_onnxruntime_input_data,
    get_ir_version,
    get_max_tensor,
    get_opset,
    init_logging,
    is_onnxruntime_available,
    model_save_as_external_data,
    onnx_dtype_to_numpy,
    onnxruntime_inference,
    save,
    summarize_model,
)


def create_simple_model():
    """Create a simple ONNX model with two Add operations (second uses a constant)."""
    # Define inputs/outputs
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 224, 224])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3, 224, 224])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # First Add: input1 + input2 -> add_out
    add1 = helper.make_node(
        "Add",
        ["input1", "input2"],
        ["add_out"],
        name="add_node1",
    )

    const_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Make tensor with raw data
    const_tensor = helper.make_tensor(
        name="const_tensor",
        data_type=TensorProto.FLOAT,
        dims=const_tensor.shape,
        vals=const_tensor.tobytes(),  # raw bytes
        raw=True,
    )

    # Second Add: add_out + const_tensor -> output
    add2 = helper.make_node(
        "Add",
        ["add_out", "const_tensor"],
        ["output"],
        name="add_node2",
    )

    # Build graph
    graph_def = helper.make_graph(
        [add1, add2],
        "test-model",
        [input1, input2],
        [output],
        initializer=[const_tensor],
    )

    # Build model
    model_def = helper.make_model(graph_def, producer_name="onnxslim-test")
    model_def.opset_import[0].version = 13
    model_def.ir_version = 10

    return model_def


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.model = create_simple_model()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_model.onnx")
        onnx.save(self.model, self.model_path)

    def tearDown(self):
        """Clean up test environment."""
        # self.temp_dir.cleanup()

    def test_init_logging(self):
        """Test init_logging function."""
        logger = init_logging(verbose=True)
        self.assertEqual(logger.level, 10)  # DEBUG level

        logger = init_logging(verbose=False)
        self.assertEqual(logger.level, 40)  # ERROR level

    def test_format_bytes(self):
        """Test format_bytes function."""
        # Test single size
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024), "1.00 GB")

        # Test tuple of sizes
        self.assertEqual(format_bytes((1024, 2048)), "1.00 KB (2.00 KB)")

    def test_onnx_dtype_to_numpy(self):
        """Test onnx_dtype_to_numpy function."""
        self.assertEqual(onnx_dtype_to_numpy(TensorProto.FLOAT), np.float32)
        self.assertEqual(onnx_dtype_to_numpy(TensorProto.INT64), np.int64)

    def test_gen_onnxruntime_input_data(self):
        """Test gen_onnxruntime_input_data function."""
        input_data = gen_onnxruntime_input_data(self.model)

        # Check if input data is generated for all inputs
        self.assertIn("input1", input_data)
        self.assertIn("input2", input_data)

        # Check shapes
        self.assertEqual(input_data["input1"].shape, (1, 3, 224, 224))
        self.assertEqual(input_data["input2"].shape, (1, 3, 224, 224))

        # Check dtypes
        self.assertEqual(input_data["input1"].dtype, np.float32)
        self.assertEqual(input_data["input2"].dtype, np.float32)

    @pytest.mark.skipif(not is_onnxruntime_available(), reason="ONNX Runtime not available")
    def test_onnxruntime_inference(self):
        """Test onnxruntime_inference function."""
        input_data = gen_onnxruntime_input_data(self.model)
        output, model = onnxruntime_inference(self.model, input_data)

        # Check if output contains the expected key
        self.assertIn("output", output)

        # Check if output has the expected shape
        self.assertEqual(output["output"].shape, (1, 3, 224, 224))

        # Check if model is returned correctly
        self.assertEqual(model.graph.name, "test-model")

    def test_get_opset(self):
        """Test get_opset function."""
        opset = get_opset(self.model)
        self.assertEqual(opset, 13)

    def test_get_ir_version(self):
        """Test get_ir_version function."""
        ir_version = get_ir_version(self.model)
        self.assertEqual(ir_version, self.model.ir_version)

    def test_tensor_info(self):
        """Test TensorInfo class."""
        input_tensor = self.model.graph.input[0]
        tensor_info = TensorInfo(input_tensor)

        self.assertEqual(tensor_info.dtype, np.float32)
        self.assertEqual(tensor_info.shape, (1, 3, 224, 224))
        self.assertEqual(tensor_info.name, "input1")

    def test_model_info(self):
        """Test ModelInfo class."""
        model_info = ModelInfo(self.model, "test_model")

        # Check basic attributes
        self.assertEqual(model_info.tag, "test_model")
        self.assertEqual(model_info.op_set, "13")

        # Check op_type_counts
        self.assertEqual(model_info.op_type_counts["Add"], 2)

        # Check input_info
        self.assertEqual(len(model_info.input_info), 2)
        self.assertEqual(model_info.input_info[0].name, "input1")
        self.assertEqual(model_info.input_info[1].name, "input2")

        # Check output_info
        self.assertEqual(len(model_info.output_info), 1)
        self.assertEqual(model_info.output_info[0].name, "output")

        # Check input_maps and output_maps
        self.assertIn("input1", model_info.input_maps)
        self.assertIn("input2", model_info.input_maps)
        self.assertIn("output", model_info.output_maps)

    def test_summarize_model(self):
        """Test summarize_model function."""
        model_info = summarize_model(self.model, "test_model")
        # Check if model_info is returned correctly
        self.assertEqual(model_info.tag, "test_model")
        self.assertEqual(model_info.op_set, "13")
        self.assertEqual(model_info.op_type_counts["Add"], 2)

    def test_dump_model_info_to_disk(self):
        import onnxslim

        model = onnxslim.slim(self.model)
        model_info = summarize_model(model, "test_model")
        dump_model_info_to_disk(model_info)

    def test_save(self):
        """Test save function."""
        output_path = os.path.join(self.temp_dir.name, "saved_model.onnx")
        save(self.model, output_path, model_check=True)

        # Check if file exists
        self.assertTrue(os.path.exists(output_path))

        # Load the saved model and check if it's valid
        saved_model = onnx.load(output_path)
        self.assertEqual(saved_model.graph.name, "test-model")
        self.assertEqual(len(saved_model.graph.node), 2)
        self.assertEqual(saved_model.graph.node[0].op_type, "Add")

    def test_save_as_external_data(self):
        """Test save with external data."""
        output_path = os.path.join(self.temp_dir.name, "external_data_model.onnx")
        save(self.model, output_path, save_as_external_data=True)

        # Check if file exists
        self.assertTrue(os.path.exists(output_path))

        # Check if external data file exists
        external_data_file = f"{os.path.basename(output_path)}.data"
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, external_data_file)))

    def test_model_save_as_external_data(self):
        """Test model_save_as_external_data function."""
        output_path = os.path.join(self.temp_dir.name, "external_data_model2.onnx")
        model_save_as_external_data(self.model, output_path)

        # Check if file exists
        self.assertTrue(os.path.exists(output_path))

        # Check if external data file exists
        external_data_file = f"{os.path.basename(output_path)}.data"
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, external_data_file)))

    def test_check_point(self):
        """Test check_point function."""
        graph = check_point(self.model)

        # Check if graph is returned correctly
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(graph.nodes[0].op, "Add")

    def test_calculate_tensor_size(self):
        """Test calculate_tensor_size function."""
        # Create a tensor with known size
        tensor = helper.make_tensor(
            name="test_tensor",
            data_type=TensorProto.FLOAT,
            dims=[2, 3, 4],
            vals=np.zeros(24, dtype=np.float32),
        )

        # Calculate size
        size = calculate_tensor_size(tensor)

        # Expected size: 2*3*4*4 bytes (float32 is 4 bytes)
        self.assertEqual(size, 96)

    def test_format_model_info(self):
        """Test format_model_info function."""
        model_info = ModelInfo(self.model, "test_model")
        formatted_info = format_model_info([model_info])

        # Check if formatted_info is a list
        self.assertIsInstance(formatted_info, list)
        print(formatted_info)
        # Check if it contains expected sections
        self.assertIn("Model Name", formatted_info[0])
        self.assertIn("Model Info", formatted_info[2])
        self.assertIn("Model Size", formatted_info[-1])

    @pytest.mark.skipif(not is_onnxruntime_available(), reason="ONNX Runtime not available")
    def test_check_result(self):
        """Test check_result function."""
        # Create identical outputs
        output1 = {"output": np.array([1.0, 2.0, 3.0])}
        output2 = {"output": np.array([1.0, 2.0, 3.0])}

        # Check result
        result = check_result(output1, output2)
        self.assertTrue(result)

        # Create different outputs
        output3 = {"output": np.array([1.0, 2.0, 3.0])}
        output4 = {"output": np.array([1.1, 2.2, 3.3])}

        # Check result
        result = check_result(output3, output4)
        self.assertFalse(result)

        # Create outputs with different keys
        output5 = {"output1": np.array([1.0, 2.0, 3.0])}
        output6 = {"output2": np.array([1.0, 2.0, 3.0])}

        # Check result
        result = check_result(output5, output6)
        self.assertFalse(result)

    @pytest.mark.skipif(not is_onnxruntime_available(), reason="ONNX Runtime not available")
    def test_check_onnx(self):
        """Test check_onnx function."""
        input_data, output, model = check_onnx(self.model)

        # Check if input_data contains expected keys
        self.assertIn("input1", input_data)
        self.assertIn("input2", input_data)

        # Check if output contains expected key
        self.assertIn("output", output)

        # Check if model is returned correctly
        self.assertEqual(model.graph.name, "test-model")

    def test_is_onnxruntime_available(self):
        """Test is_onnxruntime_available function."""
        # This is just a simple test to check if the function runs without errors
        result = is_onnxruntime_available()
        self.assertIsInstance(result, bool)

    def test_get_max_tensor(self):
        get_max_tensor(self.model)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_utils.py"])
