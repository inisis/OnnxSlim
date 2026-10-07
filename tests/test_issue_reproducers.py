"""Regression checks for the public graph-optimizer-issue-reproducers cases.

The fixture contains all 20 OnnxSlim cases and their one-condition controls from
https://github.com/Yuhx141/graph-optimizer-issue-reproducers/tree/main/onnxslim.
Fixture revision: 42efcdefddc847e12ebd3c0e441ae71b6da39912.
Unlike reproduce.py (which asserts that a bug occurs), these tests require a
valid optimized graph and exact output preservation. Models use readable ONNX
text; integer-looking float attributes are spelled with a decimal point for
compatibility with older ONNX parsers.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

import onnxslim

CASES = json.loads((Path(__file__).parent / "data" / "onnxslim_issue_reproducers.json").read_text())


def check_model(model):
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)


def run_model(model, feeds):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {item.name: feeds[item.name] for item in session.get_inputs()})
    return {item.name: value for item, value in zip(session.get_outputs(), outputs)}


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("disable_pass", [False, True], ids=["optimized", "pass-disabled"])
def test_issue_reproducer(case_name, disable_pass):
    case = CASES[case_name]
    model = onnx.parser.parse_model(case["model"])
    feeds = {
        name: np.asarray(item["values"], dtype=item["dtype"]).reshape(item["shape"])
        for name, item in case["inputs"].items()
    }
    check_model(model)
    expected = run_model(model, feeds)
    optimized = onnxslim.slim(model, **(case["disable"] if disable_pass else {}))
    check_model(optimized)
    actual = run_model(optimized, feeds)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
