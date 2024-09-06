import os
import subprocess
import tempfile

import pytest

from onnxslim import slim
from onnxslim.utils import print_model_info_as_table, summarize_model

MODELZOO_PATH = "/data/modelzoo"
FILENAME = f"{MODELZOO_PATH}/resnet18/resnet18.onnx"


class TestFunctional:
    """Tests the functionality of the 'slim' function for optimizing ONNX models using temporary directories."""

    def test_basic(self, request):
        """Test the basic functionality of the slim function."""
        with tempfile.TemporaryDirectory() as tempdir:
            summary = summarize_model(slim(FILENAME))
            print_model_info_as_table(request.node.name, summary)
            output_name = os.path.join(tempdir, "resnet18.onnx")
            slim(FILENAME, output_name)
            slim(FILENAME, output_name, model_check=True)

            command = f"onnxslim {FILENAME} {output_name}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stderr.strip()
            # Assert the expected return code
            print(output)
            assert result.returncode == 0


class TestFeature:
    """Tests ONNX model modifications like input shape, precision conversion, and input/output adjustments."""

    def test_input_shape_modification(self, request):
        """Test the modification of input shapes."""
        summary = summarize_model(slim(FILENAME, input_shapes=["input:1,3,224,224"]))
        print_model_info_as_table(request.node.name, summary)
        assert summary["op_input_info"]["input"][1] == (1, 3, 224, 224)

        with tempfile.TemporaryDirectory() as tempdir:
            output_name = os.path.join(tempdir, "resnet18.onnx")
            slim(FILENAME, output_name, input_shapes=["input:1,3,224,224"])
            summary = summarize_model(output_name)
            print_model_info_as_table(request.node.name, summary)
            assert summary["op_input_info"]["input"][1] == (1, 3, 224, 224)

    def test_fp162fp32_conversion(self, request):
        """Test the conversion of an ONNX model from FP16 to FP32 precision."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tempdir:
            output_name = os.path.join(tempdir, "resnet18.onnx")
            slim(FILENAME, output_name, input_shapes=["input:1,3,224,224"], dtype="fp16")
            summary = summarize_model(output_name)
            print_model_info_as_table(request.node.name, summary)
            assert summary["op_input_info"]["input"][0] == np.float16
            assert summary["op_input_info"]["input"][1] == (1, 3, 224, 224)

            slim(output_name, output_name, dtype="fp32")
            summary = summarize_model(output_name)
            print_model_info_as_table(request.node.name, summary)
            assert summary["op_input_info"]["input"][0] == np.float32
            assert summary["op_input_info"]["input"][1] == (1, 3, 224, 224)

    def test_output_modification(self, request):
        """Tests output modification."""
        summary = summarize_model(slim(FILENAME, outputs=["/Flatten_output_0"]))
        print_model_info_as_table(request.node.name, summary)
        assert "/Flatten_output_0" in summary["op_output_info"]

        with tempfile.TemporaryDirectory() as tempdir:
            output_name = os.path.join(tempdir, "resnet18.onnx")
            slim(FILENAME, output_name, outputs=["/Flatten_output_0"])
            summary = summarize_model(output_name)
            print_model_info_as_table(request.node.name, summary)
            assert "/Flatten_output_0" in summary["op_output_info"]

    def test_input_modification(self, request):
        """Tests input modification."""
        summary = summarize_model(
            slim(FILENAME, inputs=["/maxpool/MaxPool_output_0", "/layer1/layer1.0/relu/Relu_output_0"])
        )
        print_model_info_as_table(request.node.name, summary)
        assert "/maxpool/MaxPool_output_0" in summary["op_input_info"]
        assert "/layer1/layer1.0/relu/Relu_output_0" in summary["op_input_info"]

        with tempfile.TemporaryDirectory() as tempdir:
            output_name = os.path.join(tempdir, "resnet18.onnx")
            slim(FILENAME, output_name, inputs=["/maxpool/MaxPool_output_0", "/layer1/layer1.0/relu/Relu_output_0"])
            summary = summarize_model(output_name)
            print_model_info_as_table(request.node.name, summary)
            assert "/maxpool/MaxPool_output_0" in summary["op_input_info"]
            assert "/layer1/layer1.0/relu/Relu_output_0" in summary["op_input_info"]


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "tests/test_onnxslim.py",
        ]
    )
