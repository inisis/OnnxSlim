import logging

import os
import sys
import tempfile
from typing import Dict, List

import numpy as np
import onnx
from onnx import checker

import onnxslim.onnx_graphsurgeon as gs
from onnxslim.onnx_graphsurgeon.ir.tensor import Constant
from onnxslim.onnx_graphsurgeon.logger.logger import G_LOGGER

from ..utils.utils import (
    dump_model_info_to_disk,
    gen_onnxruntime_input_data,
    onnxruntime_inference,
    print_model_info_as_table,
    logger
)

from .optimizer import delete_node, optimize_model
from .symbolic_shape_infer import SymbolicShapeInference

DEBUG = bool(os.getenv("ONNXSLIM_DEBUG"))
AUTO_MERGE = (
    True
    if os.getenv("ONNXSLIM_AUTO_MERGE") is None
    else bool(int(os.getenv("ONNXSLIM_AUTO_MERGE")))
)


def init_logging(verbose=False):
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if verbose:  # DEBUG
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stderr)])
        G_LOGGER.severity = logging.DEBUG
    else:  # ERROR
        logging.basicConfig(level=logging.ERROR,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stderr)])
        G_LOGGER.severity = logging.ERROR

    G_LOGGER.colors = False

    import onnxruntime as ort

    ort.set_default_logger_severity(3)


def get_opset(model: onnx.ModelProto) -> int:
    try:
        for importer in model.opset_import:
            if importer.domain == "" or importer.domain == "ai.onnx":
                return importer.version

        return None
    except:
        return None


def summarize_model(model: onnx.ModelProto) -> Dict:
    logger.debug("Start summarizing model.")
    model_info = {}

    model_size = model.ByteSize()
    model_info["model_size"] = model_size

    op_info = {}
    op_type_counts = {}

    def get_tensor_dtype_shape(tensor):
        type_str = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(
            tensor.type.tensor_type.elem_type, "Unknown"
        )
        shape = None
        if tensor.type.tensor_type.HasField("shape"):
            shape = []
            for dim in tensor.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                elif dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)

        return (type_str, shape)

    def get_shape(inputs: onnx.ModelProto) -> Dict[str, List[int]]:
        op_shape_info = {}
        for input in inputs:
            type_str, shape = get_tensor_dtype_shape(input)
            if shape:
                op_shape_info[input.name] = str(type_str) + ": " + str(tuple(shape))
            else:
                op_shape_info[input.name] = str(type_str) + ": None"

        return op_shape_info

    value_info_dict = {
        value_info.name: value_info for value_info in model.graph.value_info
    }

    for node in model.graph.node:
        op_type = node.op_type
        if op_type in op_type_counts:
            op_type_counts[op_type] += 1
        else:
            op_type_counts[op_type] = 1

        for output in node.output:
            shapes = []
            if output in value_info_dict:
                tensor = value_info_dict[output]
                type_str, shape = get_tensor_dtype_shape(tensor)
                shapes.append([type_str, shape])

        op_info[node.name] = [node.op_type, shapes]

    model_info["op_set"] = str(get_opset(model))
    model_info["op_info"] = op_info
    model_info["op_type_counts"] = op_type_counts

    model_info["op_input_info"] = get_shape(model.graph.input)
    model_info["op_output_info"] = get_shape(model.graph.output)

    logger.debug("Finish summarizing model.")
    return model_info


def model_save_as_external_data(model: onnx.ModelProto, model_path: str):
    location = os.path.basename(model_path) + ".data"
    if os.path.exists(location):
        os.remove(location)
    onnx.save(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
    )


def input_shape_modification(
    model: onnx.ModelProto, input_shapes: str
) -> onnx.ModelProto:
    if not input_shapes:
        return

    graph = gs.import_onnx(model)
    input_names = [input.name for input in graph.inputs]
    tensors = graph.tensors()

    for input_shape in input_shapes:
        key, values = input_shape.rsplit(":", 1)
        values_list = [int(value) for value in values.split(",")]
        if key not in input_names:
            raise Exception(
                f"Input name {key} not found in model, available keys: {' '.join(input_names)}"
            )
        tensors[key].shape = values_list

    for _, tensor in tensors.items():
        if tensor.name not in input_names:
            if isinstance(tensor, Constant):
                continue
            tensor.shape = None

    model = gs.export_onnx(graph)

    return model


def output_modification(model: onnx.ModelProto, outputs: str) -> onnx.ModelProto:
    graph = gs.import_onnx(model)
    graph.outputs.clear()
    tensors = graph.tensors()
    for output in outputs:
        values = output.rsplit(":", 1)
        if len(values) == 1:
            key = values[0]
            if key not in tensors.keys():
                raise Exception(
                    f"Output name {key} not found in model, available keys: {' '.join(tensors.keys())}"
                )
            dtype = tensors[key].dtype
            if dtype == None:
                dtype = np.float32
                logger.warning(
                    f"Output layer {key} has no dtype, set to default {dtype}"
                )
        else:
            key, dtype = values
            if dtype == "fp16":
                dtype = np.float16
            elif dtype == "fp32":
                dtype = np.float32
            elif dtype == "int32":
                dtype = np.int32
            elif dtype == "bool":
                dtype = bool
            else:
                raise Exception(
                    f"Output layer {key} assigned unsupported dtype {dtype}"
                )

        graph.outputs.append(
            tensors[key].to_variable(dtype=dtype, shape=tensors[key].shape)
        )

    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    model = gs.export_onnx(graph)

    return model


def check_onnx(model: onnx.ModelProto, model_check_inputs=None):
    input_data_dict = gen_onnxruntime_input_data(model, model_check_inputs)
    raw_onnx_output = onnxruntime_inference(model, input_data_dict)

    return input_data_dict, raw_onnx_output


def shape_infer(model: onnx.ModelProto):
    logger.debug("Start shape inference.")
    try:
        logger.debug("try onnxruntime shape infer.")
        model = SymbolicShapeInference.infer_shapes(model, auto_merge=AUTO_MERGE)
    except:
        logger.debug("onnxruntime shape infer failed, try onnx shape infer.")
        if model.ByteSize() >= checker.MAXIMUM_PROTOBUF:
            tmp_dir = tempfile.TemporaryDirectory()
            tmp_path = os.path.join(tmp_dir.name, "tmp.onnx")
            tmp_infer_path = os.path.join(tmp_dir.name, "tmp_infer.onnx")
            save(model, tmp_path)
            onnx.shape_inference.infer_shapes_path(tmp_path, tmp_infer_path)
            model = onnx.load(tmp_infer_path)
        else:
            model = onnx.shape_inference.infer_shapes(model)
    if DEBUG:
        onnx.save(model, "debug_shape_infer.onnx")
    logger.debug("Finish shape inference.")
    return model


def optimize(model: onnx.ModelProto, skip_fusion_patterns: str = None):
    logger.debug("Start converting model to gs.")
    graph = gs.import_onnx(model).toposort()
    logger.debug("Finish converting model to gs.")
    logger.debug("Start constant folding.")
    graph.fold_constants().cleanup().toposort()
    logger.debug("Finish constant folding.")
    logger.debug("Start optimize model.")
    model = optimize_model(graph, skip_fusion_patterns)
    logger.debug("Finish optimize model.")
    if DEBUG:
        onnx.save(model, "debug_slim.onnx")

    return model


def check_point(model: onnx.ModelProto):
    graph_check_point = gs.import_onnx(model)

    return graph_check_point


def is_converged(model: onnx.ModelProto, graph_ckpt, iter: int) -> bool:
    logger.debug(f"optimization iter: {iter}")
    graph = gs.import_onnx(model)
    if graph == graph_ckpt:
        print(f"converged at iter: {iter}")
        return None
    else:
        graph_ckpt = graph
        return False


def convert_data_format(model: onnx.ModelProto, dtype: str) -> onnx.ModelProto:
    if dtype == "fp16":
        from onnxconverter_common import float16

        model = float16.convert_float_to_float16(model)
    elif dtype == "fp32":
        graph = gs.import_onnx(model).toposort()

        for node in graph.nodes:
            if node.op == "Cast":
                inp_dtype = [input.dtype for input in node.inputs][0]
                if inp_dtype == np.float16 or inp_dtype == np.float32:
                    delete_node(node)

        for tensor in graph.tensors().values():
            if isinstance(tensor, gs.Variable) and tensor.dtype == np.float16:
                tensor.dtype = np.float32
            elif isinstance(tensor, gs.Constant) and tensor.dtype == np.float16:
                tensor.values = tensor.values.astype(np.float32)

        graph.cleanup(remove_unused_graph_inputs=True).toposort()
        model = gs.export_onnx(graph)

    return model


def save(model: onnx.ModelProto, model_path: str, model_check: bool = False):
    if model_check:
        try:
            checker.check_model(model)
        except ValueError:
            logger.warning("Model too large and cannot be checked.")

    if model_path:
        if (
            model.ByteSize() <= checker.MAXIMUM_PROTOBUF
        ):  # model larger than 2GB can be saved, but compiler like trtexec won't parse it
            onnx.save(model, model_path)
        else:
            import os

            location = os.path.basename(model_path) + ".data"
            if os.path.exists(location):
                os.remove(location)
            onnx.save(
                model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=location,
            )
            logger.debug("Model too large and saved as external data automatically.")


def check_result(raw_onnx_output, slimmed_onnx_output):
    if set(raw_onnx_output.keys()) != set(slimmed_onnx_output.keys()):
        logger.warning("Model output mismatch after slimming.")
        logger.warning("Raw model output keys: {}".format(raw_onnx_output.keys()))
        logger.warning(
            "Slimmed model output keys: {}".format(slimmed_onnx_output.keys())
        )
        logger.warning("Please check the model carefully.")
        return
    else:
        for key in raw_onnx_output.keys():
            if not np.allclose(
                raw_onnx_output[key],
                slimmed_onnx_output[key],
                rtol=1e-03,
                atol=1e-04,
                equal_nan=True,
            ):
                logger.warning("Model output mismatch after slimming.")
                logger.warning("Please check the model carefully.")
                return


def freeze(model: onnx.ModelProto):
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        if input.name in name_to_input.keys():
            logger.warning(f"Duplicate input name: {input.name}")
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
            name_to_input.pop(initializer.name)
