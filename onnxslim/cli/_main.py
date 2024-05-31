from typing import Union

import onnx

from onnxslim.utils.utils import logger


def slim(
    model: Union[str, onnx.ModelProto],
    output_model: str = None,
    model_check: bool = False,
    input_shapes: str = None,
    outputs: str = None,
    no_shape_infer: bool = False,
    no_constant_folding: bool = False,
    dtype: str = None,
    skip_fusion_patterns: str = None,
    inspect: bool = False,
    dump_to_disk: bool = False,
    save_as_external_data: bool = False,
    model_check_inputs: str = None,
    verbose: bool = False,
):
    """
    Slims down or optimizes an ONNX model.

    Args:
        model (Union[str, onnx.ModelProto]): The ONNX model to be slimmed. It can be either a file path or an `onnx.ModelProto` object.

        output_model (str, optional): File path to save the slimmed model. If None, the model will not be saved.

        model_check (bool, optional): Flag indicating whether to perform model checking. Default is False.

        input_shapes (str, optional): String representing the input shapes. Default is None.

        outputs (str, optional): String representing the outputs. Default is None.

        no_shape_infer (bool, optional): Flag indicating whether to perform shape inference. Default is False.

        no_constant_folding (bool, optional): Flag indicating whether to perform constant folding. Default is False.

        dtype (str, optional): Data type. Default is None.

        skip_fusion_patterns (str, optional): String representing fusion patterns to skip. Default is None.

        inspect (bool, optional): Flag indicating whether to inspect the model. Default is False.

        dump_to_disk (bool, optional): Flag indicating whether to dump the model detail to disk. Default is False.

        save_as_external_data (bool, optional): Flag indicating whether to split onnx as model and weight. Default is False.

        model_check_inputs (str, optional): The shape or tensor used for model check. Default is None.

        verbose (bool, optional): Flag indicating whether to print verbose logs. Default is False.

    Returns:
        onnx.ModelProto/None: If `output_model` is None, return slimmed model else return None.
    """
    import os
    import time
    from pathlib import Path

    from onnxslim.core.slim import (
        check_onnx,
        check_point,
        check_result,
        convert_data_format,
        freeze,
        init_logging,
        input_shape_modification,
        model_save_as_external_data,
        optimize,
        output_modification,
        save,
        shape_infer,
        summarize_model,
    )

    from ..utils.utils import (
        dump_model_info_to_disk,
        onnxruntime_inference,
        print_model_info_as_table,
    )

    init_logging(verbose)

    MAX_ITER = 10 if not os.getenv("ONNXSLIM_MAX_ITER") else int(os.getenv("ONNXSLIM_MAX_ITER"))

    if isinstance(model, str):
        model_name = Path(model).name
        model = onnx.load(model)
    else:
        model = model
        model_name = "ONNX_Model"

    freeze(model)

    start_time = time.time()

    if save_as_external_data:
        model_save_as_external_data(model, output_model)
        return None

    if output_model or inspect:
        float_info = summarize_model(model)

    if inspect:
        print_model_info_as_table(model_name, [float_info])
        if dump_to_disk:
            dump_model_info_to_disk(model_name, float_info)
        return None

    if input_shapes:
        model = input_shape_modification(model, input_shapes)

    if outputs:
        model = output_modification(model, outputs)

    if model_check:
        input_data_dict, raw_onnx_output = check_onnx(model, model_check_inputs)

    if not no_shape_infer:
        model = shape_infer(model)

    if not no_constant_folding:
        graph_check_point = check_point(model)
        while MAX_ITER > 0:
            logger.debug(f"iter: {MAX_ITER}")
            model = optimize(model, skip_fusion_patterns)
            model = shape_infer(model)
            graph = check_point(model)
            if graph == graph_check_point:
                logger.debug(f"converged at iter: {MAX_ITER}")
                break
            else:
                graph_check_point = graph

            MAX_ITER -= 1

    if dtype:
        model = convert_data_format(model, dtype)

    if model_check:
        slimmed_onnx_output = onnxruntime_inference(model, input_data_dict)
        check_result(raw_onnx_output, slimmed_onnx_output)

    if not output_model:
        return model
    else:
        slimmed_info = summarize_model(model)
        save(model, output_model, model_check)
        if slimmed_info["model_size"] >= onnx.checker.MAXIMUM_PROTOBUF:
            model_size = model.ByteSize()
            slimmed_info["model_size"] = [model_size, slimmed_info["model_size"]]
        end_time = time.time()
        elapsed_time = end_time - start_time

        print_model_info_as_table(
            model_name,
            [float_info, slimmed_info],
            elapsed_time,
        )


def main():
    """Entry point for the OnnxSlim toolkit, processes command-line arguments and passes them to the slim function."""
    import argparse

    import onnxslim

    parser = argparse.ArgumentParser(
        description="OnnxSlim: A Toolkit to Help Optimizer Onnx Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_model", help="input onnx model")
    parser.add_argument("output_model", nargs="?", default=None, help="output onnx model")

    parser.add_argument("--model_check", action="store_true", help="enable model check")
    parser.add_argument("-v", "--version", action="version", version=onnxslim.__version__)

    # Input Shape Modification
    parser.add_argument(
        "--input_shapes",
        nargs="+",
        type=str,
        help="input shape of the model, INPUT_NAME:SHAPE, e.g. x:1,3,224,224 or x1:1,3,224,224 x2:1,3,224,224",
    )
    # Output Modification
    parser.add_argument(
        "--outputs",
        nargs="+",
        type=str,
        help="output of the model, OUTPUT_NAME:DTYPE, e.g. y:fp32 or y1:fp32 y2:fp32. \
                                                                             If dtype is not specified, the dtype of the output will be the same as the original model \
                                                                             if it has dtype, otherwise it will be fp32, available dtype: fp16, fp32, int32",
    )
    # Shape Inference
    parser.add_argument(
        "--no_shape_infer",
        action="store_true",
        help="whether to disable shape_infer, default false.",
    )

    # Constant Folding
    parser.add_argument(
        "--no_constant_folding",
        action="store_true",
        help="whether to disable constant_folding, default false.",
    )

    # Data Format Conversion
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        help="convert data format to fp16 or fp32.",
    )

    parser.add_argument(
        "--skip_fusion_patterns",
        nargs="+",
        choices=list(onnxslim.DEFAULT_FUSION_PATTERNS.keys()),
        help="whether to skip the fusion of some patterns",
    )

    # Inspect Model
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="inspect model, default False.",
    )

    # Dump Model Info to Disk
    parser.add_argument(
        "--dump_to_disk",
        action="store_true",
        help="dump model info to disk, default False.",
    )

    # Dump Model Info to Disk
    parser.add_argument(
        "--save_as_external_data",
        action="store_true",
        help="split onnx as model and weight, default False.",
    )

    # Model Check Inputs
    parser.add_argument(
        "--model_check_inputs",
        nargs="+",
        type=str,
        help="Works only when model_check is enabled, Input shape of the model or numpy data path, INPUT_NAME:SHAPE or INPUT_NAME:DATAPATH, "
        "e.g. x:1,3,224,224 or x1:1,3,224,224 x2:data.npy. Useful when input shapes are dynamic.",
    )

    # Verbose
    parser.add_argument("--verbose", action="store_true", help="verbose mode, default False.")

    args, unknown = parser.parse_known_args()

    if unknown:
        logger.error(f"unrecognized options: {unknown}")
        return 1

    if args.inspect and args.output_model:
        parser.error("--inspect and output_model are mutually exclusive")

    if not args.inspect and args.dump_to_disk:
        parser.error("dump_to_disk can only be used with --inspect")

    if args.save_as_external_data and args.output_model:
        parser.error("--save_as_external_data can only be used for single model")

    slim(
        args.input_model,
        args.output_model,
        args.model_check,
        args.input_shapes,
        args.outputs,
        args.no_shape_infer,
        args.no_constant_folding,
        args.dtype,
        args.skip_fusion_patterns,
        args.inspect,
        args.dump_to_disk,
        args.save_as_external_data,
        args.model_check_inputs,
        args.verbose,
    )

    return 0
