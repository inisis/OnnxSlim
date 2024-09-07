from itertools import product

import pytest
from ultralytics import YOLO
from ultralytics.cfg import TASK2MODEL, TASKS
from ultralytics.utils import ASSETS

SOURCE = ASSETS / "bus.jpg"


class TestYolo:
    @pytest.mark.parametrize(
        "task, dynamic, int8, half, batch", product(TASKS, [True, False], [False], [False], [1, 2])
    )
    def test_yolov8_export_onnx_matrix(self, request, task, dynamic, int8, half, batch):
        """Tests YOLOv8 ONNX export functionality with various parameter configurations for different tasks."""
        file = YOLO(TASK2MODEL[task]).export(
            format="onnx",
            imgsz=32,
            dynamic=dynamic,
            int8=int8,
            half=half,
            batch=batch,
            simplify=True,
        )
        YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference

    @pytest.mark.parametrize(
        "task, dynamic, int8, half, batch",
        product(["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"], [True, False], [False], [False], [1, 2]),
    )
    def test_yolov10_export_onnx_matrix(self, request, task, dynamic, int8, half, batch):
        """Exports YOLOv10 models to ONNX and tests inference with varying configurations."""
        file = YOLO(task).export(
            format="onnx",
            imgsz=32,
            dynamic=dynamic,
            int8=int8,
            half=half,
            batch=batch,
            simplify=True,
        )
        YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference

    @pytest.mark.parametrize(
        "task, dynamic, int8, half, batch",
        product(["yolov10n"], [False], [False], [False], [1, 2]),
    )
    def test_yolov10_export_onnx_matrix(self, request, task, dynamic, int8, half, batch):
        """Exports YOLOv10 models to TFLite and tests inference with varying configurations."""
        file = YOLO(task).export(
            format="tflite",
            imgsz=32,
            dynamic=dynamic,
            int8=int8,
            half=half,
            batch=batch,
            simplify=True,
        )
        YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference


if __name__ == "__main__":
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-sv",
            "tests/test_yolo.py",
        ]
    )
