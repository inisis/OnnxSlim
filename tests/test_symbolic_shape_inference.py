import os
import tempfile

import onnx
import pytest
import torch
import torch.nn as nn

from onnxslim.third_party.symbolic_shape_infer import SymbolicShapeInference
from onnxslim.utils import print_model_info_as_table, summarize_model


class TestSymbolicShapeInference:
    def test_einsum(self, request):
        class Model(nn.Module):
            def __init__(self, equation):
                super().__init__()
                self.equation = equation

            def forward(self, *args):
                out = torch.einsum(self.equation, *args)
                return out

        def transform(shapes, equation, dynamic_axes=None, verbose=True):
            inputs = [torch.randn(shape) for shape in shapes]
            m = Model(equation)
            with tempfile.TemporaryDirectory() as tempdir:
                filename = os.path.join(tempdir, "model.onnx")
                input_names = [f"input_{i}" for i in range(len(inputs))]
                output_names = ["output"]
                torch.onnx.export(
                    m,
                    tuple(inputs),
                    filename,
                    dynamic_axes=dynamic_axes,
                    input_names=input_names,
                    output_names=output_names,
                )
                model = onnx.load(filename)
                for output in model.graph.output:
                    if output.type.HasField("tensor_type"):
                        output.type.tensor_type.shape.Clear()

                model = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
                summary = summarize_model(model)
                if verbose:
                    print_model_info_as_table(summary)
                args = tuple(inputs)
                output = m(*args)
                return summary, output

        summary, _ = transform((36, 64), "..., f -> ... f", dynamic_axes={"input_0": {0: "N"}})
        assert summary.output_info[0].shape[0] == "N"

        summary, output = transform(((3, 2, 5), (3, 5, 4)), "bij,bjk->bik")
        assert summary.output_info[0].shape == output.shape

        summary, output = transform(((2, 4, 2, 5), (2, 5, 4)), "bchw,bkc->bkhw")
        assert summary.output_info[0].shape == output.shape

        summary, output = transform(((4, 4),), "ii")
        assert summary.output_info[0].shape == output.shape

        summary, output = transform(((4, 4),), "ii->i")
        assert summary.output_info[0].shape == output.shape

        summary, output = transform(((2, 3, 4, 5),), "...ij->...ji")
        assert summary.output_info[0].shape == output.shape

        summary, output = transform(((2, 5), (3, 5, 4), (2, 4)), "bn,anm,bm->ba")
        assert summary.output_info[0].shape == output.shape


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main(
            [
                "-p",
                "no:warnings",
                "-sv",
                "tests/test_symbolic_shape_inference.py",
            ]
        )
    )
