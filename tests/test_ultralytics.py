import os
import subprocess
import tempfile
from itertools import product

import numpy as np
import onnxruntime as ort
import pytest

from utils import run_onnx

# Import ultralytics configurations
try:
    from ultralytics import YOLO
    from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    TASKS = frozenset()
    TASK2MODEL = {}
    TASK2DATA = {}


def get_model_input_shape(model_path):
    """Get input shape from ONNX model."""
    session = ort.InferenceSession(model_path)
    input_info = session.get_inputs()[0]
    shape = input_info.shape
    # Replace dynamic dimensions with sensible defaults for YOLO models
    resolved_shape = []
    for i, s in enumerate(shape):
        if isinstance(s, int):
            resolved_shape.append(s)
        elif i == 0:  # batch dimension
            resolved_shape.append(1)
        elif i == 1:  # channels
            resolved_shape.append(3)
        else:  # height/width
            resolved_shape.append(32)
    return input_info.name, resolved_shape


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics package not installed")
class TestUltralyticsONNX:
    """Test OnnxSlim with Ultralytics YOLO ONNX exports."""

    @pytest.mark.parametrize("task", TASKS)
    def test_export_onnx_slim(self, task):
        """Test onnxslim optimization on ONNX exports for all task types."""
        model = YOLO(TASK2MODEL[task])
        onnx_path = model.export(format="onnx", imgsz=32, simplify=True)

        with tempfile.TemporaryDirectory() as tempdir:
            slim_filename = os.path.join(tempdir, f"{task}_slim.onnx")

            command = f"onnxslim {onnx_path} {slim_filename}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stderr.strip())
            assert result.returncode == 0
            assert os.path.exists(slim_filename)

        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    @pytest.mark.parametrize(
        "task, dynamic, batch",
        [
            (task, dynamic, batch)
            for task, dynamic, batch in product(
                TASKS, [True, False], [1, 2]
            )
        ],
    )
    def test_export_onnx_matrix(self, task, dynamic, batch):
        """Test onnxslim on ONNX exports with various configurations."""
        model = YOLO(TASK2MODEL[task])
        onnx_path = model.export(
            format="onnx",
            imgsz=32,
            dynamic=dynamic,
            batch=batch,
            simplify=True,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            slim_filename = os.path.join(tempdir, f"{task}_slim.onnx")

            command = f"onnxslim {onnx_path} {slim_filename}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stderr.strip())
            assert result.returncode == 0
            assert os.path.exists(slim_filename)

        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    @pytest.mark.parametrize("task", TASKS)
    def test_output_consistency(self, task):
        """Test that onnxslim optimization preserves model output consistency."""
        model = YOLO(TASK2MODEL[task])
        onnx_path = model.export(format="onnx", imgsz=32, simplify=True)

        with tempfile.TemporaryDirectory() as tempdir:
            slim_filename = os.path.join(tempdir, f"{task}_slim.onnx")

            command = f"onnxslim {onnx_path} {slim_filename}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            assert result.returncode == 0

            # Get input shape from model
            input_name, input_shape = get_model_input_shape(onnx_path)
            input_data = {input_name: np.random.randn(*input_shape).astype(np.float32)}

            original_output = run_onnx(onnx_path, input_data)
            slim_output = run_onnx(slim_filename, input_data)

            for key in original_output:
                np.testing.assert_allclose(
                    original_output[key],
                    slim_output[key],
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Output mismatch for {key} in task {task}",
                )

        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    @pytest.mark.parametrize("task", TASKS)
    def test_dynamic_export_consistency(self, task):
        """Test output consistency with dynamic ONNX exports."""
        model = YOLO(TASK2MODEL[task])
        onnx_path = model.export(format="onnx", imgsz=32, dynamic=True, simplify=True)

        with tempfile.TemporaryDirectory() as tempdir:
            slim_filename = os.path.join(tempdir, f"{task}_slim.onnx")

            command = f"onnxslim {onnx_path} {slim_filename}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            assert result.returncode == 0

            # Get input shape from model
            input_name, input_shape = get_model_input_shape(onnx_path)
            input_data = {input_name: np.random.randn(*input_shape).astype(np.float32)}

            original_output = run_onnx(onnx_path, input_data)
            slim_output = run_onnx(slim_filename, input_data)

            for key in original_output:
                np.testing.assert_allclose(
                    original_output[key],
                    slim_output[key],
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Output mismatch for {key} in task {task}",
                )

        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-p", "no:warnings", "-v", "tests/test_ultralytics.py"]))
