# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Registry for shape inference handlers."""

from collections import OrderedDict

# Global registries for shape handlers
SHAPE_HANDLERS = OrderedDict()
ATEN_SHAPE_HANDLERS = OrderedDict()


def register_shape_handler(handler):
    """Register a shape handler for a specific ONNX operator type.

    Args:
        handler: A ShapeHandler instance to register.

    Returns:
        The registered handler.

    Raises:
        ValueError: If a handler for the same op_type is already registered.
    """
    op_type = handler.op_type
    if op_type in SHAPE_HANDLERS:
        raise ValueError(f"Handler for op_type '{op_type}' is already registered")
    SHAPE_HANDLERS[op_type] = handler
    return handler


def register_aten_handler(handler):
    """Register a shape handler for a PyTorch ATen operator.

    Args:
        handler: A ShapeHandler instance to register.

    Returns:
        The registered handler.

    Raises:
        ValueError: If a handler for the same op_name is already registered.
    """
    op_name = handler.op_type
    if op_name in ATEN_SHAPE_HANDLERS:
        raise ValueError(f"Handler for ATen op '{op_name}' is already registered")
    ATEN_SHAPE_HANDLERS[op_name] = handler
    return handler


def get_shape_handler(op_type):
    """Get the shape handler for a given ONNX operator type.

    Args:
        op_type: The ONNX operator type string.

    Returns:
        The registered ShapeHandler or None if not found.
    """
    return SHAPE_HANDLERS.get(op_type)


def get_aten_handler(op_name):
    """Get the shape handler for a given ATen operator name.

    Args:
        op_name: The ATen operator name string.

    Returns:
        The registered ShapeHandler or None if not found.
    """
    return ATEN_SHAPE_HANDLERS.get(op_name)


def get_all_shape_handlers():
    """Get all registered shape handlers.

    Returns:
        OrderedDict of all registered shape handlers.
    """
    return SHAPE_HANDLERS.copy()


def get_all_aten_handlers():
    """Get all registered ATen shape handlers.

    Returns:
        OrderedDict of all registered ATen shape handlers.
    """
    return ATEN_SHAPE_HANDLERS.copy()
