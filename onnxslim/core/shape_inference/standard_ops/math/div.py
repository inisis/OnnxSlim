# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Div operator."""

from ...base import MultiOpHandler
from ...registry import register_shape_handler
from ._symbolic_compute import infer_symbolic_compute_ops

register_shape_handler(MultiOpHandler("Div", infer_symbolic_compute_ops))
