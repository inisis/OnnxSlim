# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Backward-compatible wrapper for symbolic shape inference.

This module provides backward compatibility for code that imports from
`onnxslim.third_party.symbolic_shape_infer`. The actual implementation
has been moved to `onnxslim.core.shape_inference`.

Usage:
    from onnxslim.third_party.symbolic_shape_infer import SymbolicShapeInference

    model = onnx.load("model.onnx")
    model_with_shapes = SymbolicShapeInference.infer_shapes(model)
"""

import argparse
import logging

import onnx
from packaging import version

from onnxslim.core.shape_inference import ShapeInferencer

assert version.parse(onnx.__version__) >= version.parse("1.8.0")

logger = logging.getLogger(__name__)


from onnxslim.core.shape_inference.utils import (
    as_list,
    as_scalar,
    get_attribute,
    get_dim_from_proto,
    get_elem_type_from_type_proto,
    get_opset,
    get_shape_from_sympy_shape,
    get_shape_from_type_proto,
    get_shape_from_value_info,
    handle_negative_axis,
    is_literal,
    is_sequence,
    make_named_value_info,
    sympy_reduce_product,
)

from onnxslim.core.shape_inference.context import PythonPrinter, pexpr


class SymbolicShapeInference:
    """Backward-compatible wrapper for SymbolicShapeInference.

    This class wraps the new ShapeInferencer class to provide backward
    compatibility for existing code that uses SymbolicShapeInference directly.
    """

    def __init__(self, int_max, auto_merge, guess_output_rank, verbose, prefix=""):
        """Initializes the SymbolicShapeInference class with configuration parameters."""
        self._inferencer = ShapeInferencer(
            int_max=int_max,
            auto_merge=auto_merge,
            guess_output_rank=guess_output_rank,
            verbose=verbose,
            prefix=prefix,
        )
        self.int_max_ = int_max
        self.auto_merge_ = auto_merge
        self.guess_output_rank_ = guess_output_rank
        self.verbose_ = verbose
        self.prefix_ = prefix

        # These will be set during preprocessing
        self.out_mp_ = None
        self.run_ = True
        self.suggested_merge_ = {}
        self.symbolic_dims_ = {}
        self.input_symbols_ = {}
        self.subgraph_id_ = 0
        self.known_vi_ = {}
        self.sympy_data_ = {}
        self.initializers_ = {}
        self.graph_inputs_ = {}

    @staticmethod
    def infer_shapes(in_mp, int_max=2**31 - 1, auto_merge=False, guess_output_rank=False, verbose=0):
        """Perform symbolic shape inference on an ONNX model.

        Args:
            in_mp: The input ONNX ModelProto.
            int_max: Maximum value for unbounded integers.
            auto_merge: Whether to automatically merge conflicting dimensions.
            guess_output_rank: Whether to guess output rank from input.
            verbose: Logging verbosity level.

        Returns:
            The model with inferred shapes.
        """
        return ShapeInferencer.infer_shapes(
            in_mp,
            int_max=int_max,
            auto_merge=auto_merge,
            guess_output_rank=guess_output_rank,
            verbose=verbose,
        )


def parse_arguments():
    """Parses command-line arguments for ONNX model transformation options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="The input model file")
    parser.add_argument("--output", help="The output model file")
    parser.add_argument(
        "--auto_merge",
        help="Automatically merge symbolic dims when confliction happens",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--int_max",
        help="maximum value for integer to be treated as boundless for ops like slice",
        type=int,
        default=2**31 - 1,
    )
    parser.add_argument(
        "--guess_output_rank",
        help="guess output rank to be the same as input 0 for unknown ops",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_as_external_data",
        help="Saving an ONNX model to external data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--all_tensors_to_one_file",
        help="Saving all the external data to one file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--external_data_location",
        help="The file location to save the external file",
        default="./",
    )
    parser.add_argument(
        "--external_data_size_threshold",
        help="The size threshold for external data",
        type=int,
        default=1024,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"input model: {args.input}")
    if args.output:
        logger.info(f"output model {args.output}")
    logger.info("Doing symbolic shape inference...")
    out_mp = SymbolicShapeInference.infer_shapes(
        onnx.load(args.input),
        args.int_max,
        args.auto_merge,
        args.guess_output_rank,
        args.verbose,
    )
    if args.output and out_mp:
        if args.save_as_external_data:
            onnx.save_model(
                out_mp,
                args.output,
                save_as_external_data=True,
                all_tensors_to_one_file=args.all_tensors_to_one_file,
                location=args.external_data_location,
                size_threshold=args.external_data_size_threshold,
                convert_attribute=False,
            )
        else:
            onnx.save(out_mp, args.output)
        logger.info("Done!")
