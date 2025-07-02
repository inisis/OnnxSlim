import importlib.util
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import checker, helper

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.misc.font import GREEN, WHITE
from onnxslim.misc.tabulate import SEPARATING_LINE, tabulate
from onnxslim.third_party.onnx_graphsurgeon.logger.logger import G_LOGGER

logger = logging.getLogger("onnxslim")


def init_logging(verbose=False):
    """Configure the logging settings for the application based on the verbosity level."""
    logger = logging.getLogger("onnxslim")

    if verbose:  # DEBUG
        logger.setLevel(logging.DEBUG)
        G_LOGGER.severity = logging.DEBUG
    else:  # ERROR
        logger.setLevel(logging.ERROR)
        G_LOGGER.severity = logging.ERROR

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    G_LOGGER.colors = False

    if is_onnxruntime_available():
        import onnxruntime as ort

        ort.set_default_logger_severity(3)

    return logger


def format_bytes(size: Union[int, Tuple[int, ...]]) -> str:
    """Convert byte sizes into human-readable format with appropriate units (B, KB, MB, GB)."""
    if isinstance(size, int):
        size = (size,)

    units = ["B", "KB", "MB", "GB"]
    formatted_sizes = []

    for size_in_bytes in size:
        unit_index = 0
        while size_in_bytes >= 1024 and unit_index < len(units) - 1:
            size_in_bytes /= 1024
            unit_index += 1

        formatted_size = f"{size_in_bytes:.2f} {units[unit_index]}"
        formatted_sizes.append(formatted_size)

    if len(formatted_sizes) == 1:
        return formatted_sizes[0]
    else:
        return f"{formatted_sizes[0]} ({formatted_sizes[1]})"


def onnx_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    """Maps an ONNX dtype to its corresponding NumPy dtype."""
    return np.dtype(helper.tensor_dtype_to_np_dtype(onnx_dtype))


def gen_onnxruntime_input_data(
    model: onnx.ModelProto, model_check_inputs: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Generate random input data for an ONNX model considering potential specific input shapes and types."""
    input_info = {}
    for input_tensor in model.graph.input:
        name = input_tensor.name
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            elif dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(None)
        dtype = onnx_dtype_to_numpy(input_tensor.type.tensor_type.elem_type)

        input_info[name] = {"shape": shape, "dtype": dtype}

    if model_check_inputs:
        for model_check_input in model_check_inputs:
            key, value = model_check_input.rsplit(":", 1)
            if value.endswith(".npy"):
                if key not in input_info:
                    raise Exception(
                        f"model_check_input name:{key} not found in model, available keys: {' '.join(input_info.keys())}"
                    )
                data = np.load(value)
                input_info[key] = {"data": data}
            else:
                values_list = [int(val) for val in value.split(",")]
                if key in input_info:
                    input_info[key]["shape"] = values_list
                else:
                    raise Exception(
                        f"model_check_input name:{key} not found in model, available keys: {' '.join(input_info.keys())}"
                    )

    input_data_dict = {}
    for name, info in input_info.items():
        if "data" in info:
            input_data_dict[name] = info["data"]
        else:
            shapes = [shape if (shape != -1 and not isinstance(shape, str)) else 1 for shape in info["shape"]]
            shapes = shapes or [1]
            dtype = info["dtype"]

            if dtype in {np.int32, np.int64}:
                random_data = np.random.randint(10, size=shapes).astype(dtype)
            else:
                random_data = np.random.rand(*shapes).astype(dtype)
            input_data_dict[name] = random_data

    return input_data_dict


def onnxruntime_inference(model: onnx.ModelProto, input_data: dict) -> Dict[str, np.array]:
    """Perform inference using ONNX Runtime on the given model and input data."""
    import os
    import tempfile

    import onnx
    import onnxruntime as rt

    if model.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = os.path.join(tmp_dir.name, "tmp.onnx")
        location = f"{os.path.basename(tmp_path)}.data"
        if os.path.exists(location):
            os.remove(location)
        onnx.save(
            model,
            tmp_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=location,
        )
        onnx_model = tmp_path
    else:
        onnx_model = model.SerializeToString()

    sess = rt.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    onnx_output = sess.run(None, input_data)

    output_names = [output.name for output in sess.get_outputs()]
    onnx_output = dict(zip(output_names, onnx_output))

    if isinstance(onnx_model, str):
        model = onnx.load(onnx_model)

    return onnx_output, model


def format_model_info(model_info_list: Union[Dict, List[Dict]], elapsed_time: float = None):
    assert model_info_list, "model_info_list must contain more than one model info"
    if not isinstance(model_info_list, (list, tuple)):
        model_info_list = [model_info_list]

    final_op_info = []
    final_op_info.extend(
        (
            ["Model Name"] + [item.tag for item in model_info_list],
            [SEPARATING_LINE] * (len(model_info_list) + 1),
            ["Model Info"]
            + ["Op Set: " + item.op_set + " / IR Version: " + item.ir_version for item in model_info_list],
            [SEPARATING_LINE] * (len(model_info_list) + 1),
        )
    )

    def get_io_info(model_info_list, tag=None):
        if tag == "OUT":
            ios = [op_type for model_info in model_info_list for op_type in model_info.output_info]
        else:
            ios = [op_type for model_info in model_info_list for op_type in model_info.input_info]
        ios = list(dict.fromkeys([io.name for io in ios]))
        io_info = []
        for io in ios:
            input_info_list = [f"{tag}: {io}"]
            for model_info in model_info_list:
                if tag == "OUT":
                    io_tensor = model_info.output_maps.get(io, None)
                else:
                    io_tensor = model_info.input_maps.get(io, None)
                inputs_shape = (io_tensor.dtype, io_tensor.shape) if io_tensor else ""
                if isinstance(inputs_shape, (list, tuple)):
                    inputs_shape = ": ".join([str(i) for i in inputs_shape])
                input_info_list.append(inputs_shape)
            io_info.append(input_info_list)

        return io_info

    final_op_info.extend(get_io_info(model_info_list, "IN"))
    final_op_info.extend(get_io_info(model_info_list, "OUT"))

    final_op_info.append([SEPARATING_LINE] * (len(model_info_list) + 1))

    all_ops = {op_type for model_info in model_info_list for op_type in model_info.op_type_counts}
    sorted_ops = sorted(all_ops)
    for op in sorted_ops:
        op_info_list = [op]
        float_number = model_info_list[0].op_type_counts.get(op, 0)
        op_info_list.append(float_number)
        for model_info in model_info_list[1:]:
            slimmed_number = model_info.op_type_counts.get(op, 0)
            if float_number > slimmed_number:
                slimmed_number = GREEN + str(slimmed_number) + WHITE
            op_info_list.append(slimmed_number)

        final_op_info.append(op_info_list)
    final_op_info.extend(
        (
            [SEPARATING_LINE] * (len(model_info_list) + 1),
            ["Model Size"] + [format_bytes(model_info.model_size) for model_info in model_info_list],
        )
    )
    if elapsed_time:
        final_op_info.extend(
            (
                [SEPARATING_LINE] * (len(model_info_list) + 1),
                ["Elapsed Time"] + [f"{elapsed_time:.2f} s"],
            )
        )

    return final_op_info


def print_model_info_as_table(model_info_list: Union[Dict, List[Dict]], elapsed_time: float = None):
    """Prints the model information as a formatted table for the given model name and list of model details."""
    if not isinstance(model_info_list, (list, tuple)):
        model_info_list = [model_info_list]

    final_op_info = format_model_info(model_info_list, elapsed_time)
    lines = tabulate(
        final_op_info,
        headers=[],
        tablefmt="pretty",
        maxcolwidths=[None] + [40] * len(model_info_list),
    ).split("\n")
    if elapsed_time:
        time_row = lines[-2].split("|")
        time_row[-3] = (
            time_row[-2][: len(time_row[-2]) // 2 + 1] + time_row[-3] + time_row[-2][len(time_row[-2]) // 2 :]
        )
        time_row.pop(-2)
        lines[-2] = "|".join(time_row)
    output = "\n".join([line if line != "| \x01 |" else lines[0] for line in lines])

    print(output)


def dump_model_info_to_disk(model_info: Dict):
    """Writes model information to a CSV file for a given model name and dictionary of model info."""
    import csv

    csv_file_path = f"{model_info.tag}_model_info.csv"
    with open(csv_file_path, "a", newline="") as csvfile:  # Use 'a' for append mode
        fieldnames = ["NodeName", "OpType", "OutputDtype", "OutputShape"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is empty, write the header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write the data
        for node_name, info in model_info.op_info.items():
            op_type, output_info_list = info.op, info.outputs
            # Write the first row with actual NodeName and OpType
            row_data_first = {
                "NodeName": node_name,
                "OpType": op_type,
                "OutputDtype": output_info_list[0].dtype,  # First entry in the list
                "OutputShape": output_info_list[0].shape,  # First entry in the list
            }
            writer.writerow(row_data_first)

            # Write subsequent rows with empty strings for NodeName and OpType
            for output_dtype, output_shape in output_info_list[1:]:
                row_data_empty = {
                    "NodeName": "",
                    "OpType": "",
                    "OutputDtype": output_dtype,
                    "OutputShape": output_shape,
                }
                writer.writerow(row_data_empty)
    print(f"Model info written to {csv_file_path}")


def get_opset(model: onnx.ModelProto) -> int:
    """Returns the ONNX opset version for a given model."""
    try:
        for importer in model.opset_import:
            if importer.domain in {"", "ai.onnx"}:
                return importer.version

        return None
    except Exception:
        return None


def get_ir_version(model: onnx.ModelProto) -> int:
    """Returns the ONNX ir version for a given model."""
    try:
        return model.ir_version
    except Exception:
        return None


class TensorInfo:
    def __init__(self, tensor):
        self.dtype: np.dtype = np.float32
        self.shape: Tuple[Union[str, int]] = None

        self._extract_info(tensor)

    def _extract_info(self, tensor):
        """Extract the data type and shape of an ONNX tensor."""
        self.dtype = (
            onnx.helper.tensor_dtype_to_np_dtype(tensor.type.tensor_type.elem_type)
            if tensor.type.tensor_type.elem_type in onnx.helper.get_all_tensor_dtypes()
            else "Unknown"
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
                    shape.append("?")

        self.shape = tuple(shape) if shape is not None else None
        self.name = tensor.name


class OperatorInfo:
    def __init__(self, operator, outputs=None):
        self.name: str = None
        self.op: str = None

        self._extract_info(operator)
        self.outputs = outputs

    def _extract_info(self, operator):
        self.name: str = operator.name
        self.op: str = operator.op_type


class ModelInfo:
    def __init__(self, model: Union[str, onnx.ModelProto], tag: str = "OnnxSlim"):
        if isinstance(model, str):
            tag = Path(model).name
            model = onnx.load(model)

        self.tag: str = tag
        self.model_size: int = -1
        self.op_set: str = None
        self.ir_version: str = None
        self.op_type_counts: Dict[str, int] = defaultdict(int)
        self.op_info: Dict[str, Dict] = {}
        self.input_info: List[str, Tuple[str, Tuple]] = []
        self.output_info: List[str, Tuple[str, Tuple]] = []

        self._summarize_model(model)

    def _summarize_model(self, model):
        self.op_set = str(get_opset(model))
        self.ir_version = str(get_ir_version(model))
        self.model_size = model.ByteSize()

        for input in model.graph.input:
            self.input_info.append(TensorInfo(input))

        for output in model.graph.output:
            self.output_info.append(TensorInfo(output))

        value_info_dict = {value_info.name: value_info for value_info in model.graph.value_info}

        def get_graph_node_info(graph: onnx.GraphProto) -> Dict[str, List[str]]:
            for node in graph.node:
                op_type = node.op_type
                self.op_type_counts[op_type] += 1
                output_tensor_info = []
                for output in node.output:
                    if output in value_info_dict:
                        tensor = value_info_dict[output]
                        tensor_info = TensorInfo(tensor)
                        output_tensor_info.append(tensor_info)

                self.op_info[node.name] = OperatorInfo(node, output_tensor_info)

                for attr in node.attribute:
                    ATTR_TYPE_MAPPING = {v: k for k, v in onnx.AttributeProto.AttributeType.items()}
                    if attr.type in ATTR_TYPE_MAPPING:
                        attr_str = ATTR_TYPE_MAPPING[attr.type]
                        if attr_str == "GRAPH":
                            get_graph_node_info(attr.g)

        get_graph_node_info(model.graph)

    @property
    def input_maps(self):
        self.input_dict = {input_info.name: input_info for input_info in self.input_info}

        return self.input_dict

    @property
    def output_maps(self):
        self.output_dict = {output_info.name: output_info for output_info in self.output_info}

        return self.output_dict


def summarize_model(model: Union[str, onnx.ModelProto], tag="OnnxModel") -> Dict:
    """Generates a summary of the ONNX model, including model size, operations, and tensor shapes."""
    logger.debug("Start summarizing model.")
    model_info = ModelInfo(model, tag)
    logger.debug("Finish summarizing model.")
    return model_info


def model_save_as_external_data(model: onnx.ModelProto, model_path: str):
    """Save an ONNX model with tensor data as an external file."""
    location = f"{os.path.basename(model_path)}.data"
    if os.path.exists(location):
        os.remove(location)
    onnx.save(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
    )


def check_onnx(model: onnx.ModelProto, model_check_inputs=None):
    """Validates an ONNX model by generating input data and performing inference to check outputs."""
    input_data_dict = gen_onnxruntime_input_data(model, model_check_inputs)
    raw_onnx_output, model = onnxruntime_inference(model, input_data_dict)

    return input_data_dict, raw_onnx_output, model


def check_point(model: onnx.ModelProto):
    """Imports an ONNX model checkpoint into a Graphsurgeon graph representation."""
    return gs.import_onnx(model)


def is_converged(model: onnx.ModelProto, graph_ckpt, iter: int) -> bool:
    """Checks if the model optimization has converged by comparing the current graph to the checkpoint."""
    logger.debug(f"optimization iter: {iter}")
    graph = gs.import_onnx(model)
    if graph == graph_ckpt:
        print(f"converged at iter: {iter}")
        return None
    else:
        graph_ckpt = graph
        return False


def save(
    model: onnx.ModelProto,
    model_path: str,
    model_check: bool = False,
    save_as_external_data: bool = False,
    model_info: Dict = None,
):
    """Save an ONNX model to a specified path, with optional model checking for validity."""
    if model_check:
        try:
            checker.check_model(model)
        except ValueError:
            logger.warning("Model too large and cannot be checked.")

    if model_path:  # model larger than 2GB can be saved, but compiler like trtexec won't parse it
        if model.ByteSize() <= checker.MAXIMUM_PROTOBUF and not save_as_external_data:
            onnx.save(model, model_path)
        else:
            import os

            location = f"{os.path.basename(model_path)}.data"
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

            if model_info:
                model_size = model.ByteSize()
                model_info.model_size = [model_size, model_info.model_size]


def check_result(raw_onnx_output, slimmed_onnx_output):
    """Verify the consistency of outputs between the raw and slimmed ONNX models, logging warnings if discrepancies are
    detected.
    """
    if set(raw_onnx_output.keys()) != set(slimmed_onnx_output.keys()):
        print("Model output mismatch after slimming.")
        print(f"Raw model output keys: {raw_onnx_output.keys()}")
        print(f"Slimmed model output keys: {slimmed_onnx_output.keys()}")
        print("Please check the model carefully.")
        return False
    else:
        for key in raw_onnx_output.keys():
            if not np.allclose(
                raw_onnx_output[key],
                slimmed_onnx_output[key],
                rtol=1e-03,
                atol=1e-04,
                equal_nan=True,
            ):
                print(f"\033[31mModel output {key} mismatch after slimming.")
                print("\033[31mPlease check the model carefully.")
                return False

    return True


data_type_sizes = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT16: 2,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.BOOL: 1,
}


def calculate_tensor_size(tensor):
    """Calculates the size of an ONNX tensor in bytes based on its shape and data type size."""
    shape = tensor.dims
    num_elements = np.prod(shape) if shape else 0
    element_size = data_type_sizes.get(tensor.data_type, 0)
    return num_elements * element_size


def get_model_size_and_initializer_size(model):
    """Calculates and prints the model size and initializer size for an ONNX model in bytes."""
    initializer_size = 0
    for tensor in model.graph.initializer:
        tensor_size = calculate_tensor_size(tensor)
        initializer_size += tensor_size

    print("model size", model.ByteSize())
    print("initializer size", initializer_size)


def get_model_subgraph_size(model):
    """Calculate and print the size of subgraphs in an ONNX model in bytes."""
    graph = model.graph
    for node in graph.node:
        for attr in node.attribute:
            ATTR_TYPE_MAPPING = {v: k for k, v in onnx.AttributeProto.AttributeType.items()}
            if attr.type in ATTR_TYPE_MAPPING:
                attr_str = ATTR_TYPE_MAPPING[attr.type]
                if attr_str == "GRAPH":
                    print("subgraph", attr.g.ByteSize())


def is_onnxruntime_available():
    if importlib.util.find_spec("onnxruntime") is None:
        logger = logging.getLogger("onnxslim")
        logger.debug("onnxruntime is not available, please install it first for better optimization")
        return False
    else:
        try:
            # in case of onnxruntime import error
            import onnxruntime as ort

            if hasattr(ort, "__version__"):
                return True
            else:
                return False
        except:
            logger = logging.getLogger("onnxslim")
            logger.debug("onnxruntime is not available, please install it first for better optimization")
            return False


def check_onnx_compatibility():
    """Ensure ONNX Runtime and ONNX versions are compatible for model inference."""
    compatibility_dict = {
        "1.20": "1.16",
        "1.19": "1.16",
        "1.18": "1.16",
        "1.17": "1.15",
        "1.16": "1.14.1",
        "1.15": "1.14",
        "1.14": "1.13",
        "1.13": "1.12",
        "1.12": "1.12",
        "1.11": "1.11",
        "1.10": "1.10",
        "1.9": "1.10",
        "1.8": "1.9",
        "1.7": "1.8",
        "1.6": "1.8",
        "1.5": "1.7",
        "1.4": "1.7",
        "1.3": "1.7",
        "1.2": "1.6",
        "1.1": "1.6",
        "1.0": "1.6",
        "0.5": "1.5",
        "0.4": "1.5",
        "0.3": "1.4",
        "0.2": "1.3",
        "0.1": "1.3",
    }
    import onnx
    import onnxruntime

    onnx_version = onnx.__version__
    # ort_version = onnxruntime.__version__
    ort_version = ".".join(onnxruntime.__version__.split("+")[0].split(".")[:2])
    # Check compatibility
    expected_onnx_version = compatibility_dict.get(ort_version)
    if expected_onnx_version is None:
        print(
            f"Warning: Onnx Runtime version {ort_version} has no specified compatible ONNX version. Compatibility issues may occur."
        )
    elif expected_onnx_version == ".".join(onnx_version.split("+")[0].split(".")[:2]):
        logger.info(
            f"Installed Onnx Runtime version {ort_version} is compatible with installed ONNX version {onnx_version}."
        )
    else:
        print(
            f"Warning: Installed Onnx Runtime version {ort_version} is not compatible with installed ONNX version {onnx_version}. Expected ONNX version: {expected_onnx_version}."
        )


def get_max_tensor(model, topk=5):
    graph = gs.import_onnx(model)

    tensor_map = graph.tensors()
    constant_tensors = [tensor for tensor in tensor_map.values() if isinstance(tensor, gs.Constant)]

    sub_graphs = graph.subgraphs(recursive=True)
    sub_graphs_constant_tensors = [
        [tensor for name, tensor in sub_graph.tensors().items() if isinstance(tensor, gs.Constant)]
        for sub_graph in sub_graphs
    ]

    constant_tensors.extend([tensor for tensors in sub_graphs_constant_tensors for tensor in tensors])

    sizes = [tensor.values.size for tensor in constant_tensors]
    sorted_indices = np.argsort(sizes)[::-1][:topk]

    for i in sorted_indices:
        tensor = constant_tensors[i]
        print(
            f"Tensor name: {tensor.name}, shape: {tensor.values.shape}, dtype: {tensor.values.dtype} size: {tensor.values.size}"
        )


# copied from https://onnx.ai/onnx/api/tools.html
def update_outputs_dims(
    model,
    output_dims,
):
    dim_param_set: set[str] = set()

    def init_dim_param_set(dim_param_set, value_infos):
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField("dim_param"):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore

    def update_dim(tensor, dim, j, name) -> None:
        dim_proto = tensor.type.tensor_type.shape.dim[j]

        # if it's int in model, it won't be replaced by original symbol
        if dim_proto.HasField("dim_value"):
            return

        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
                    raise ValueError(
                        f"Unable to set dimension value to {dim} for axis {j} of {name}. Contradicts existing dimension value {dim_proto.dim_value}."
                    )
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + "_" + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError(
                        f"Unable to generate unique dim_param for axis {j} of {name}. Please manually provide a dim_param value."
                    )
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, str):
            dim_proto.dim_param = dim
        else:
            raise ValueError(  # noqa: TRY004
                f"Only int or str is accepted as dimension value, incorrect type: {type(dim)}"
            )

    for output in model.graph.output:
        output_name = output.name
        output_dim_arr = output_dims[output_name]
        if output_dim_arr is None:
            continue
        for j, dim in enumerate(output_dim_arr):
            update_dim(output, dim, j, output_name)

    return model
