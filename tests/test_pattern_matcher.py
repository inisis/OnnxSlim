import os

import pytest
import torch
import torch.nn as nn

from onnxslim import slim
from onnxslim.utils import print_model_info_as_table, summarize_model


class TestPatternMatcher:
    def test_gelu(self, request):
        """Test the GELU activation function in a neural network model using an instance of nn.Module."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu0 = nn.ReLU()
                self.gelu = nn.GELU()
                self.relu1 = nn.ReLU()

            def forward(self, x):
                """Performs a forward pass through the model applying ReLU, GELU, and ReLU activations sequentially to
                the input tensor x.
                """
                x = self.relu0(x)
                x = self.gelu(x)
                x = self.relu1(x)
                return x

        input = torch.randn(2)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)

    def test_pad_conv(self, request):
        """Test padding followed by 2D convolution within a neural network module."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pad_0 = nn.ConstantPad2d(3, 0)
                self.conv_0 = nn.Conv2d(1, 1, 3)

                self.pad_1 = nn.ConstantPad2d(3, 2)
                self.conv_1 = nn.Conv2d(1, 1, 3, bias=False)

            def forward(self, x):
                """Applies padding and convolutional layers to the input tensor x."""
                x0 = self.pad_0(x)
                x0 = self.conv_0(x0)

                x1 = self.pad_1(x)
                x1 = self.conv_1(x1)

                return x0 + x1

        input = torch.randn(1, 1, 24, 24)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)

        assert summary.op_type_counts["Conv"] == 2
        assert summary.op_type_counts["Add"] == 1

    def test_conv_bn(self, request):
        """Test the convolutional layer followed by batch normalization export and re-import via ONNX."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)
                self.bn = nn.BatchNorm2d(1)

            def forward(self, x):
                """Perform convolution followed by batch normalization on input tensor x."""
                x = self.conv(x)
                x = self.bn(x)
                return x

        input = torch.randn(1, 1, 24, 24)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename, do_constant_folding=False)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Conv"] == 1

    def test_consecutive_slice(self, request):
        """Tests consecutive slicing operations on a model by exporting it to ONNX format and then slimming the ONNX
        file.
        """

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)
                self.bn = nn.BatchNorm2d(1)

            def forward(self, x):
                """Performs slicing operation on the input tensor x by returning the section x[1:2, :2]."""
                return x[1:2, :2]

        input = torch.randn(3, 4)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Slice"] == 1

    def test_consecutive_reshape(self, request):
        """Test the functionality of consecutive reshape operations in a model and export it to ONNX format."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                """Reshape tensor sequentially to (2, 6) and then to (12, 1)."""
                return x.view(2, 6).view(12, 1)

        input = torch.randn(3, 4)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Reshape"] == 1

    def test_matmul_add(self, request):
        """Tests matrix multiplication followed by an addition operation within a neural network model."""

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.data = torch.randn(4, 3)

            def forward(self, x):
                """Performs matrix multiplication of input 'x' with pre-defined data, adds 1, and returns the result."""
                x = torch.matmul(x, self.data)
                x += 1
                return x

        input = torch.randn(3, 4)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Gemm"] == 1

    def test_reduce(self, request):
        """Tests model reduction by exporting a PyTorch model to ONNX format, slimming it, and saving to a specified
        directory.
        """

        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                """Performs a reduction summing over the last dimension of the input tensor and then unsqueezes the
                tensor along the same dimension.
                """
                x = torch.sum(x, dim=[-1], keepdim=False)
                x = x.unsqueeze(-1)
                return x

        input = torch.randn(3, 4)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename, opset_version=11)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["ReduceSum"] == 1

    @pytest.mark.parametrize(
        "opset",
        (
            11,
            13,
        ),
    )
    def test_consecutive_unsqueeze(self, request, opset):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x.unsqueeze(-1)
                x = x.unsqueeze(-1)
                x = x.unsqueeze(1)
                x = x.unsqueeze(0)
                return x

        input = torch.randn(3, 4)
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, filename, opset_version=opset)

        summary = summarize_model(slim(filename, model_check=True), request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Unsqueeze"] == 1


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-sv",
            "tests/test_pattern_matcher.py",
        ]
    )
