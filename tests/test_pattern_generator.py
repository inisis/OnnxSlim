import os

import onnx
import pytest
import torch
import torch.nn as nn

from onnxslim import register_fusion_pattern, slim
from onnxslim.core.pattern import Pattern, PatternGenerator, PatternMatcher
from onnxslim.utils import summarize_model

MODELZOO_PATH = "/data/modelzoo"


class TestPatternGenerator:
    def test_gelu(self, request):
        """Test the GELU activation function within the PatternModel class."""

        class PatternModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()

            def forward(self, x):
                """Applies the GELU activation function to the input tensor."""
                x = self.gelu(x)
                return x

        class Model(nn.Module):
            def __init__(self):
                """Initializes the Model class with ReLU and PatternModel components."""
                super().__init__()
                self.relu0 = nn.ReLU()
                self.pattern = PatternModel()
                self.relu1 = nn.ReLU()

            def forward(self, x):
                """Applies the ReLU activation function, the PatternModel, and another ReLU activation sequentially to
                the input tensor.
                """
                x = self.relu0(x)
                x = self.pattern(x)
                x = self.relu1(x)
                return x

        input = torch.randn(2)
        p = PatternModel()
        m = Model()
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)

        pattern_filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(p, input, pattern_filename, opset_version=17, dynamo=False)

        model_filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, model_filename, opset_version=17, dynamo=False)

        model = onnx.load(pattern_filename)
        pgen = PatternGenerator(model)
        template = pgen.generate()
        Pattern(template)

        class GeluMatcher(PatternMatcher):
            def __init__(self, pattern, priority):
                """Initialize a GeluMatcher with a given pattern and priority."""
                super().__init__(pattern, priority)

            @property
            def name(self):
                """Return the name of the matcher as 'FusionGelu'."""
                return "FusionGeluTest"

            def rewrite(self, opset=11):
                """Raise an exception indicating a pattern match in GeluMatcher."""
                raise Exception("Pattern Matched")

        from onnxslim.core.pattern.fusion import GeluPatternMatcher

        register_fusion_pattern(GeluPatternMatcher(1))
        slim(model_filename, f"{directory}/{request.node.name}_slim.onnx")
        summary = summarize_model(f"{directory}/{request.node.name}_slim.onnx")
        assert summary.op_type_counts["Gelu"] == 1

        # assert str(excinfo.value) == "Pattern Matched"

    def test_quick_gelu(self, request):
        """Test the Quick GELU activation function within the PatternModel class."""

        class SiluPatternMatcher(PatternMatcher):
            def __init__(self, priority):
                r"""
                Initializes a `SiluPatternMatcher` to identify and fuse Silu patterns in a computational graph.

                input
                /     \
                |    Sigmoid
                \     /
                Mul.
                """
                pattern = Pattern(
                    """
                    input    input      0 2 mul_0 sigmoid_0
                    Sigmoid  sigmoid_0  1 1 input mul_0
                    Mul      mul_0      2 1 input sigmoid_0 output
                    output output 1 0 mul_0
                    """
                )
                super().__init__(pattern, priority)

            @property
            def name(self):
                """Returns the name of the fusion pattern, 'FusionSilu'."""
                return "FusionSilu"

            def rewrite(self, opset=11):
                """Rewrite the computation graph pattern to fuse Silu operations."""
                input_variable = self.sigmoid_0.inputs[0]
                mul_node = self.mul_0
                sigmoid_node = self.sigmoid_0

                input_variable.outputs.remove(mul_node)
                input_variable.outputs.remove(sigmoid_node)

                output_variable = self.mul_0.outputs[0]
                output_variable.inputs.clear()

                return {
                    self.mul_0.name: {
                        "op": "Silu",
                        "inputs": [input_variable],
                        "outputs": [output_variable],
                        "domain": None,
                    }
                }

        register_fusion_pattern(SiluPatternMatcher(1))
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        directory = f"tmp/{request.node.name}"
        os.makedirs(directory, exist_ok=True)
        slim(filename, f"{directory}/{request.node.name}_slim.onnx")
        summary = summarize_model(f"{directory}/{request.node.name}_slim.onnx")
        assert summary.op_type_counts["Silu"] == 0


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main(
            [
                "-p",
                "no:warnings",
                "-sv",
                "tests/test_pattern_generator.py",
            ]
        )
    )
