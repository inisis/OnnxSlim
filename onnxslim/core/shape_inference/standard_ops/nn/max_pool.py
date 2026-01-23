# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Shape handler for MaxPool operator."""

from ...registry import register_shape_handler
from .average_pool import PoolHandler

register_shape_handler(PoolHandler("MaxPool"))
