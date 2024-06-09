import subprocess

import pytest
from utils import download_onnx_from_url


@pytest.mark.parametrize(
    "name",
    (
        "swin_tiny",
        "glm_block_0",
        "mobilenet_v2",
        "resnet18",
        "tf_efficientnetv2_s",
        "UNetModel-fp16",
        "dinov2",
        "unet",
    ),
)
class TestOnnxModel:
    def test_onnx_model(self, request, name):
        """Test downloading an ONNX model by its name and running 'onnxslim' command to slim the model."""
        filename = download_onnx_from_url(f"http://120.224.26.32:15030/aifarm/onnx/{name}.onnx")
        command = f"onnxslim {filename} {name}_slim.onnx"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stderr.strip()
        # Assert the expected return code
        print(output)
        assert result.returncode == 0

    def test_onnxslim_python_api(self, request, name):
        """Tests the ONNX model slimming Python API using the 'onnxslim' command for a given model name."""
        import onnx

        from onnxslim import slim

        filename = download_onnx_from_url(f"http://120.224.26.32:15030/aifarm/onnx/{name}.onnx")
        model_slim = slim(filename)
        onnx.save(model_slim, f"{name}_slim.onnx")


class TestFeat:
    def test_input_shape_modification(self, request):
        """Test the modification of input shapes for a UNet model and assert the expected return code."""
        filename = download_onnx_from_url("http://120.224.26.32:15030/aifarm/onnx/UNetModel-fp16.onnx")
        command = f"onnxslim {filename} UNetModel-fp16_slim.onnx --input_shapes cc:1,1,768"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stderr.strip()
        # Assert the expected return code
        print(output)
        assert result.returncode == 0

    def test_fp162fp32_conversion(self, request):
        """Test the conversion of an ONNX model from FP16 to FP32 precision."""
        filename = download_onnx_from_url("http://120.224.26.32:15030/aifarm/onnx/UNetModel-fp16.onnx")
        command = f"onnxslim {filename} UNetModel-fp16_slim.onnx --input_shapes cc:1,1,768 --dtype fp32"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stderr.strip()
        # Assert the expected return code
        print(output)
        assert result.returncode == 0

    def test_output_modification(self, request):
        """Tests output modification of an ONNX model by running a slimming command and checking for successful
        execution.
        """
        filename = download_onnx_from_url("http://120.224.26.32:15030/aifarm/onnx/yolov5m.onnx")
        command = f"onnxslim {filename} yolov5m_slim.onnx --outputs 591 739 443"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stderr.strip()
        # Assert the expected return code
        print(output)
        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "tests/test_onnxslim.py",
        ]
    )
