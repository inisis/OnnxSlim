import os
import tempfile
from unittest.mock import patch

import numpy as np
import onnx
import pytest

from onnxslim.cli._main import main, slim
from onnxslim.utils import is_onnxruntime_available

# Skip tests if onnxruntime is not available
pytestmark = pytest.mark.skipif(not is_onnxruntime_available(), reason="ONNXRuntime not available")


# Use a simple model for testing
def create_test_model():
    # Create a simple model with Add op
    input1 = onnx.helper.make_tensor_value_info("input:1", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    input2 = onnx.helper.make_tensor_value_info("input:2", onnx.TensorProto.FLOAT, [1, 3, 224, 224])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 224, 224])

    add_node = onnx.helper.make_node(
        "Add",
        ["input:1", "input:2"],
        ["output"],
    )

    graph = onnx.helper.make_graph(
        [add_node],
        "test-model",
        [input1, input2],
        [output],
    )

    model = onnx.helper.make_model(graph, producer_name="onnxslim-test")
    model.opset_import[0].version = 13

    return model


class TestCliMain:
    def test_slim_basic(self):
        """Test basic functionality of slim function."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Test slim function
            slim(input_path, output_path)

            # Verify the output model exists
            assert os.path.exists(output_path)

            # Load the output model and verify
            output_model = onnx.load(output_path)
            assert len(output_model.graph.node) > 0
            assert output_model.graph.node[0].op_type == "Add"

    def test_slim_with_input_shapes(self):
        """Test slim function with input shape modification."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Test slim function with input shape modification
            slim(input_path, output_path, input_shapes=["input:1:1,3,112,112", "input:2:1,3,112,112"])

            # Load the output model and verify
            output_model = onnx.load(output_path)
            assert output_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value == 112
            assert output_model.graph.input[1].type.tensor_type.shape.dim[2].dim_value == 112

    def test_slim_with_dtype_conversion(self):
        """Test slim function with data type conversion."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Test slim function with dtype conversion to fp16
            slim(input_path, output_path, dtype="fp16")

            # Load the output model and verify
            output_model = onnx.load(output_path)
            assert output_model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16

    def test_slim_with_model_check(self):
        """Test slim function with model check."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Create test input data
            input1_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
            input2_data = np.random.random((1, 3, 224, 224)).astype(np.float32)

            input1_path = os.path.join(tempdir, "input1.npy")
            input2_path = os.path.join(tempdir, "input2.npy")

            np.save(input1_path, input1_data)
            np.save(input2_path, input2_data)

            # Test slim function with model check
            slim(
                input_path,
                output_path,
                model_check=True,
                model_check_inputs=[f"input:1:{input1_path}", f"input:2:{input2_path}"],
            )

            # Verify the output model exists
            assert os.path.exists(output_path)

    def test_slim_with_no_shape_infer(self):
        """Test slim function with no shape inference."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Test slim function with no shape inference
            slim(input_path, output_path, no_shape_infer=True)

            # Verify the output model exists
            assert os.path.exists(output_path)

    def test_slim_return_model(self):
        """Test slim function returning model without saving."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Test slim function returning model
            output_model = slim(input_path)

            # Verify the returned model
            assert isinstance(output_model, onnx.ModelProto)
            assert len(output_model.graph.node) > 0
            assert output_model.graph.node[0].op_type == "Add"


class TestCliMainEntryPoint:
    def test_main_basic(self):
        """Test basic functionality of main function."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Mock sys.argv for main function
            with patch("sys.argv", ["onnxslim", input_path, output_path]):
                # Test main function
                exit_code = main()

                # Verify exit code
                assert exit_code == 0

                # Verify the output model exists
                assert os.path.exists(output_path)

    def test_main_with_input_shapes(self):
        """Test main function with input shape modification."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Mock sys.argv for main function with input shapes
            with patch(
                "sys.argv",
                ["onnxslim", input_path, output_path, "--input-shapes", "input:1:1,3,112,112", "input:2:1,3,112,112"],
            ):
                # Test main function
                exit_code = main()

                # Verify exit code
                assert exit_code == 0

                # Verify the output model exists
                assert os.path.exists(output_path)

                # Load the output model and verify
                output_model = onnx.load(output_path)
                assert output_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value == 112

    def test_main_with_dtype_conversion(self):
        """Test main function with data type conversion."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Mock sys.argv for main function with dtype conversion
            with patch("sys.argv", ["onnxslim", input_path, output_path, "--dtype", "fp16"]):
                # Test main function
                exit_code = main()

                # Verify exit code
                assert exit_code == 0

                # Verify the output model exists
                assert os.path.exists(output_path)

                # Load the output model and verify
                output_model = onnx.load(output_path)
                assert output_model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16

    def test_main_with_no_shape_infer(self):
        """Test main function with no shape inference."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Mock sys.argv for main function with no shape inference
            with patch("sys.argv", ["onnxslim", input_path, output_path, "--no-shape-infer"]):
                # Test main function
                exit_code = main()

                # Verify exit code
                assert exit_code == 0

                # Verify the output model exists
                assert os.path.exists(output_path)

    def test_main_with_inspect(self):
        """Test main function with inspect option."""
        model = create_test_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")

            # Save the model
            onnx.save(model, input_path)

            # Mock sys.argv for main function with inspect
            with patch("sys.argv", ["onnxslim", input_path, "--inspect"]):
                # Test main function
                exit_code = main()

                # Verify exit code
                assert exit_code == 0
