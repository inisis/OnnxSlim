import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnxslim.core import shape_infer


def value_info_shape(value_info):
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def output_shape(model):
    inferred = shape_infer(model)
    onnx.checker.check_model(inferred)
    return value_info_shape(inferred.graph.output[0])


def onnx_output_shape(model):
    return value_info_shape(
        onnx.shape_inference.infer_shapes(copy.deepcopy(model)).graph.output[0]
    )


def unary_model(op_type, input_shape, **attrs):
    graph = helper.make_graph(
        [helper.make_node(op_type, ["input"], ["output"], **attrs)],
        op_type,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.UNDEFINED, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_flatten_preserves_symbolic_products():
    model = unary_model("Flatten", ["N", 3, "H", 5], axis=2)

    assert output_shape(model) == ["3*N", "5*H"]


def test_space_depth_preserves_symbolic_relations():
    space_to_depth = unary_model("SpaceToDepth", ["N", "C", "H", "W"], blocksize=2)
    depth_to_space = unary_model("DepthToSpace", ["N", "C", "H", "W"], blocksize=2)

    assert output_shape(space_to_depth) == ["N", "4*C", "floor(H/2)", "floor(W/2)"]
    assert output_shape(depth_to_space) == ["N", "floor(C/4)", "2*H", "2*W"]


def conv_transpose_model(weight_shape=(3, 4, 3, 3), input_shape=None, **attrs):
    weights = numpy_helper.from_array(
        np.zeros(weight_shape, dtype=np.float32), name="weights"
    )
    input_shape = input_shape or ["N", weight_shape[0], "H", "W"]
    graph = helper.make_graph(
        [helper.make_node("ConvTranspose", ["input", "weights"], ["output"], **attrs)],
        "ConvTranspose",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.UNDEFINED, None)],
        [weights],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_conv_transpose_preserves_symbolic_spatial_formula():
    explicit_pads = conv_transpose_model(strides=[2, 2], pads=[1, 1, 1, 1])
    same_padding = conv_transpose_model(
        weight_shape=(4, 3, 3, 3), group=2, strides=[2, 2], auto_pad="SAME_UPPER"
    )
    explicit_shape = conv_transpose_model(output_shape=[11, 13])

    assert output_shape(explicit_pads) == ["N", 4, "2*H - 1", "2*W - 1"]
    assert output_shape(same_padding) == ["N", 6, "2*H", "2*W"]
    assert output_shape(explicit_shape) == ["N", 4, 11, 13]


def test_handlers_match_onnx_for_concrete_shapes():
    models = [
        unary_model("Flatten", [2, 3, 4, 5], axis=0),
        unary_model("Flatten", [2, 3, 4, 5], axis=-1),
        unary_model("SpaceToDepth", [2, 3, 8, 10], blocksize=2),
        unary_model("DepthToSpace", [2, 12, 4, 5], blocksize=2),
        conv_transpose_model(
            weight_shape=(2, 3, 3),
            input_shape=[1, 2, 5],
            strides=[2],
            pads=[1, 1],
        ),
        conv_transpose_model(
            weight_shape=(2, 3, 3, 3, 3),
            input_shape=[1, 2, 4, 5, 6],
            strides=[2, 1, 2],
            pads=[1, 1, 1, 1, 1, 1],
        ),
    ]

    for model in models:
        assert output_shape(model) == onnx_output_shape(model)
