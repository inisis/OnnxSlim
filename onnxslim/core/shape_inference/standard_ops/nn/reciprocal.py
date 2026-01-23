# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for Reciprocal operator."""

from ...base import PassthroughHandler
from ...registry import register_shape_handler

register_shape_handler(PassthroughHandler("Reciprocal"))
