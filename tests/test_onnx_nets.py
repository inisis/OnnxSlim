import os
import shutil
import subprocess
import warnings

import pytest
import timm
import torch
import torchvision.models as models

FUSE = True
PRETRAINED = False

os.makedirs("tmp", exist_ok=True)


class TestTorchVisionClass:
    @pytest.mark.parametrize(
        "model",
        (
            models.resnet18,
            models.alexnet,
            models.squeezenet1_0,
            models.googlenet,
        ),
    )
    def test_torchvision(self, request, model, shape=(1, 3, 224, 224)):
        """Test various TorchVision models with random input tensors of a specified shape."""
        model = model(pretrained=PRETRAINED)
        x = torch.rand(shape)
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        slim_filename = f"{directory}/{request.node.name}_slim.onnx"

        torch.onnx.export(model, x, filename)

        command = f"onnxslim {filename} {slim_filename}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stderr.strip()
        # Assert the expected return code
        print(output)
        assert result.returncode == 0

        shutil.rmtree(directory, ignore_errors=True)


class TestTimmClass:
    @pytest.fixture(params=timm.list_models())
    def model_name(self, request):
        """Yields names of models available in TIMM (https://github.com/rwightman/pytorch-image-models) for pytest fixture parameterization."""
        yield request.param

    def test_timm(self, request, model_name):
        """Tests a TIMM model's forward pass with a random input tensor of the appropriate size."""
        model = timm.create_model(model_name, pretrained=PRETRAINED)
        input_size = model.default_cfg.get("input_size")
        x = torch.randn((1,) + input_size)
        directory = f"tmp/{request.node.name}"
        try:
            os.makedirs(directory, exist_ok=True)

            filename = f"{directory}/{request.node.name}.onnx"
            slim_filename = f"{directory}/{request.node.name}_slim.onnx"
            torch.onnx.export(model, x, filename)
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return
        if not os.path.exists(filename):
            return

        command = f"onnxslim {filename} {slim_filename}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stderr.strip()
        # Assert the expected return code
        print(output)
        assert result.returncode == 0

        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-n", "10", "-v", "tests/test_onnx_nets.py"])
