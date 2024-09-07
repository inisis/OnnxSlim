import os

import onnx
import pytest
import torch
import torch.nn as nn

from onnxslim import register_fusion_pattern, slim
from onnxslim.core.pattern import Pattern, PatternGenerator, PatternMatcher


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
        torch.onnx.export(p, input, pattern_filename)

        model_filename = f"{directory}/{request.node.name}.onnx"
        torch.onnx.export(m, input, model_filename)

        model = onnx.load(pattern_filename)
        pgen = PatternGenerator(model)
        template = pgen.generate()
        pattern = Pattern(template)

        class GeluMatcher(PatternMatcher):
            def __init__(self, pattern, priority):
                """Initialize a GeluMatcher with a given pattern and priority."""
                super().__init__(pattern, priority)

            @property
            def name(self):
                """Return the name of the matcher as 'FusionGelu'."""
                return "FusionGelu"

            def rewrite(self, opset=11):
                """Raise an exception indicating a pattern match in GeluMatcher."""
                raise Exception("Pattern Matched")

        register_fusion_pattern(GeluMatcher(pattern, 1))
        with pytest.raises(Exception) as excinfo:
            slim(model_filename, f"{directory}/{request.node.name}_slim.onnx")

        assert str(excinfo.value) == "Pattern Matched"


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-sv",
            "tests/test_pattern_generator.py",
        ]
    )
