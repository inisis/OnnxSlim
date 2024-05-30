import os
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
        model = model(pretrained=PRETRAINED)
        x = torch.rand(shape)
        os.makedirs("tmp/" + request.node.name, exist_ok=True)

        filename = f"tmp/{request.node.name}/{request.node.name}.onnx"
        slim_filename = f"tmp/{request.node.name}/{request.node.name}_slim.onnx"

        torch.onnx.export(model, x, filename)

        command = f"onnxslim {filename} {slim_filename}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stderr.strip()
        # Assert the expected return code
        print(output)
        assert result.returncode == 0

        os.remove(filename)


class TestTimmClass:
    @pytest.fixture(params=timm.list_models())
    def model_name(self, request):
        yield request.param

    def test_timm(self, request, model_name):
        model = timm.create_model(model_name, pretrained=PRETRAINED)
        input_size = model.default_cfg.get("input_size")
        x = torch.randn((1,) + input_size)

        try:
            os.makedirs("tmp/" + request.node.name, exist_ok=True)

            filename = f"tmp/{request.node.name}/{request.node.name}.onnx"
            slim_filename = f"tmp/{request.node.name}/{request.node.name}_slim.onnx"
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

        os.remove(filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-n", "10", "-v", "tests/test_onnx_nets.py"])
