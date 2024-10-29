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


class TestModelZoo:
    def transform_and_check(self, name, filename, transformation_func, suffix, check_func):
        with tempfile.TemporaryDirectory() as tempdir:
            output_file = os.path.join(tempdir, f"{name}_{suffix}.onnx")
            result = transformation_func(filename, output_file)
            if result.returncode == 0:
                if check_func:
                    try:
                        check_func(output_file)
                    except:
                        return None
                return summarize_model(output_file, suffix)
        return None

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

        summary_list = [summarize_model(filename)]
        summary_list.append(self.transform_and_check(name, filename, bench_onnxslim, "onnxslim", check_model_inference))
        summary_list.append(self.transform_and_check(name, filename, bench_onnxsim, "onnxsim", check_model_inference))
        summary_list.append(
            self.transform_and_check(name, filename, bench_polygraphy, "polygraphy", check_model_inference)
        )

        summary_list = [summary for summary in summary_list if summary is not None]

        print()
        print_model_info_as_table(request.node.name, summary_list)


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-sv",
            "tests/test_benchmark.py",
        ]
    )
