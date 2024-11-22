import os
import tempfile

import numpy as np
import onnxruntime as ort
import pytest

from onnxslim import slim
from onnxslim.utils import print_model_info_as_table, summarize_model

MODELZOO_PATH = "/data/modelzoo"


class TestModelZoo:
    def test_silero_vad(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            batch_size = 2
            input = np.zeros((batch_size, 256), dtype=np.float32)
            sr = np.array(16000)
            state = np.zeros((2, batch_size, 128), dtype=np.float32)

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            ort_sess.run(None, {"input": input, "sr": sr, "state": state})

    def test_decoder_with_past_model(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"))
            batch_size = 2
            input_ids = np.ones((batch_size, 256), dtype=np.int64)
            encoder_hidden_states = np.zeros((batch_size, 128, 16), dtype=np.float32)

            ort_sess = ort.InferenceSession(os.path.join(tempdir, f"{name}_slim.onnx"))
            ort_sess.run(None, {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states})

    def test_tiny_en_decoder(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"

        with tempfile.TemporaryDirectory() as tempdir:
            slim(filename, os.path.join(tempdir, f"{name}_slim.onnx"), model_check=True)

    def test_transformer_encoder(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        summary = summarize_model(slim(filename), tag=request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Mul"] == 57
        assert summary.op_type_counts["Div"] == 53

    def test_uiex(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        summary = summarize_model(slim(filename), tag=request.node.name)
        print_model_info_as_table(summary)
        assert summary.op_type_counts["Range"] == 0
        assert summary.op_type_counts["Floor"] == 0
        assert summary.op_type_counts["Concat"] == 54


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-sv",
            "tests/test_modelzoo.py",
        ]
    )
