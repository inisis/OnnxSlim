import os
import subprocess
import tempfile

import numpy as np
import onnxruntime as ort
import pytest

ort.set_default_logger_severity(3)

from onnxslim.utils import print_model_info_as_table, summarize_model

MODELZOO_PATH = "/data/modelzoo"


def bench_main(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result


def bench_onnxslim(input, output):
    command = f"onnxslim {input} {output}"
    result = bench_main(command)
    return result


def bench_onnxsim(input, output):
    command = f"onnxsim {input} {output}"
    result = bench_main(command)
    return result


def bench_polygraphy(input, output):
    command = f"polygraphy surgeon sanitize --fold-constants {input} -o {output}"
    result = bench_main(command)
    return result

def bench_onnxruntime(input, output):
    try:
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = output
        session = rt.InferenceSession(input, sess_options)
        return True

    except Exception as e:
        print(e)
        return None


def bench_onnxruntime(input, output):
    try:
        import onnxruntime as rt

        sess_options = rt.SessionOptions()
        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = output
        rt.InferenceSession(input, sess_options)
        return True

    except Exception as e:
        print(e)
        return None


def bench_onnxruntime(input, output):
    try:
        import onnxruntime as rt

        sess_options = rt.SessionOptions()
        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # To enable model serialization after graph optimization set this
        sess_options.optimized_model_filepath = output
        rt.InferenceSession(input, sess_options)
        return True

    except Exception as e:
        print(e)
        return None


class TestModelZoo:
    def transform_and_check(self, name, filename, transformation_func, suffix, check_func):
        with tempfile.TemporaryDirectory() as tempdir:
            output_file = os.path.join(tempdir, f"{name}_{suffix}.onnx")
            result = transformation_func(filename, output_file)
            if result is None:
                return None
            if result or result.returncode == 0:
                if check_func:
                    try:
                        check_func(output_file)
                    except:
                        return None
                return summarize_model(output_file, suffix)
        return None

    def run_model_test(self, name, filename, check_func=None):
        summary_list = [summarize_model(filename)]
        summary_list.append(self.transform_and_check(name, filename, bench_onnxslim, "onnxslim", check_func))
        summary_list.append(self.transform_and_check(name, filename, bench_onnxsim, "onnxsim", check_func))
        summary_list.append(self.transform_and_check(name, filename, bench_polygraphy, "polygraphy", check_func))
        summary_list.append(self.transform_and_check(name, filename, bench_onnxruntime, "onnxruntime", check_func))

        summary_list = [summary for summary in summary_list if summary is not None]

        print()
        print_model_info_as_table(name, summary_list)

    def test_silero_vad(self, request):
        def check_model_inference(model_path):
            batch_size = 2
            input_data = np.zeros((batch_size, 256), dtype=np.float32)
            sr = np.array(16000)
            state = np.zeros((2, batch_size, 128), dtype=np.float32)

            ort_sess = ort.InferenceSession(model_path)
            outputs = ort_sess.run(None, {"input": input_data, "sr": sr, "state": state})
            assert outputs is not None, "Inference failed on transformed model."

        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        self.run_model_test(name, filename, check_model_inference)

    def test_decoder_with_past_model(self, request):
        def check_model_inference(model_path):
            batch_size = 2
            input_ids = np.ones((batch_size, 256), dtype=np.int64)
            encoder_hidden_states = np.zeros((batch_size, 128, 16), dtype=np.float32)

            ort_sess = ort.InferenceSession(model_path)
            ort_sess.run(None, {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states})

        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        self.run_model_test(name, filename, check_model_inference)

    def test_tiny_en_decoder(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        self.run_model_test(name, filename)

    def test_transformer_encoder(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        self.run_model_test(name, filename)

    def test_uiex(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        self.run_model_test(name, filename)

    def test_en_number_mobile_v2_0_rec_infer(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        self.run_model_test(name, filename)

    def test_paddleocr(self, request):
        name = request.node.originalname[len("test_") :]
        filename = f"{MODELZOO_PATH}/{name}/{name}.onnx"
        self.run_model_test(name, filename)


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-sv",
            "tests/test_benchmark.py",
        ]
    )
