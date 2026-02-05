import os
import tempfile

import numpy as np
import onnxruntime as ort
import pytest

import onnx
from onnxslim import slim
from onnxslim.utils import print_model_info_as_table, summarize_model

MODELZOO_PATH = "/data/modelzoo/amd"

def get_input_shapes(graph):
    input_shapes = {}
    for input in graph.input:
        input_shapes[input.name] = [
            s.dim_value for s in input.type.tensor_type.shape.dim
        ]
    return input_shapes


def load_dummy_data(input_shape, data_size=5):
    """Generate dummy input dicts."""
    for _ in range(data_size):
        data = {}
        for name, shape in input_shape.items():
            data[name] = np.random.rand(*shape).astype(np.float32)
        yield data


def run_slim_and_compare(original_path: str, slim_path: str, data_size=5):
    """
    Compare ORT inference results between original and slim models.
    Slimming must already be done outside this function.
    """
    # ==== Prepare input shape ====
    model = onnx.load(original_path)
    input_shape = get_input_shapes(model.graph)

    # fix dynamic dims → 1
    for name, shape in input_shape.items():
        input_shape[name] = [1 if s < 1 else s for s in shape]

    data_gen = load_dummy_data(input_shape, data_size=data_size)

    # ==== ORT sessions ====
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    opts.log_severity_level = 3
    EP = ["CPUExecutionProvider"]

    sess_orig = ort.InferenceSession(original_path, sess_options=opts, providers=EP)
    sess_slim = ort.InferenceSession(slim_path, sess_options=opts, providers=EP)

    # ==== Compare outputs ====
    for inp in data_gen:
        out1 = sess_orig.run([], inp)
        out2 = sess_slim.run([], inp)

        for a, b in zip(out1, out2):
            if not np.array_equal(a, b):
                return False
    return True

class TestModelZoo:
    def test_EliminationSlice(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim_path = os.path.join(tempdir, f"{name}_slim.onnx")
            slim(filename, slim_path)
            ok = run_slim_and_compare(filename, slim_path)
            assert ok, f"onnxslim output mismatch for model {name}!"

    def test_sub_model(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim_path = os.path.join(tempdir, f"{name}_slim.onnx")
            slim(filename, slim_path)

            # Check that the slimmed model has exactly 2 Cast nodes
            model_info = summarize_model(slim_path)
            cast_count = model_info.op_type_counts.get("Cast", 0)
            assert cast_count == 2, f"Expected 2 Cast nodes, but found {cast_count}"


if __name__ == "__main__":
    import sys

    sys.exit(
        pytest.main(
            [
                "-p",
                "no:warnings",
                "-sv",
                "tests/test_amd.py",
            ]
        )
    )
