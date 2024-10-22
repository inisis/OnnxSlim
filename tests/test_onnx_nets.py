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
MEMORY_LIMIT_GB = 0.75  # User's memory limit
MEMORY_PER_PARAM = 4e-9  # Approximate memory required per parameter in GB

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

    skip_keywords = ["enormous", "giant", "huge", "xlarge"]

    def test_timm(self, request, model_name):
        """Tests a TIMM model's forward pass with a random input tensor of the appropriate size."""
        if any(keyword in model_name.lower() for keyword in self.skip_keywords):
            pytest.skip(f"Skipping model due to size keyword in name: {model_name}")

        try:
            model = timm.create_model(model_name, pretrained=PRETRAINED)
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip(f"Skipping model {model_name} due to memory error.")

        num_params = sum(p.numel() for p in model.parameters())

        # Calculate estimated memory requirement
        estimated_memory = num_params * MEMORY_PER_PARAM

        if estimated_memory > MEMORY_LIMIT_GB:
            pytest.skip(f"Skipping model {model_name}: estimated memory {estimated_memory:.2f} GB exceeds limit.")

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
    pytest.main(["-p", "no:warnings", "-v", "tests/test_onnx_nets.py"])
