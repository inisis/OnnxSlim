# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utility functions for symbolic shape inference."""

import numpy as np
import onnx
import sympy
from onnx import helper, numpy_helper


def get_attribute(node, attr_name, default_value=None):
    """Retrieve the value of an attribute from an ONNX node.

    Args:
        node: The ONNX node.
        attr_name: The name of the attribute to retrieve.
        default_value: The default value if the attribute is not found.

    Returns:
        The attribute value or the default value.
    """
    found = [attr for attr in node.attribute if attr.name == attr_name]
    return helper.get_attribute_value(found[0]) if found else default_value


def get_dim_from_proto(dim):
    """Retrieve the dimension value from the ONNX protobuf object.

    Args:
        dim: The ONNX TensorShapeProto.Dimension.

    Returns:
        The dimension value (int or str) or None.
    """
    return getattr(dim, dim.WhichOneof("value")) if type(dim.WhichOneof("value")) is str else None


def is_sequence(type_proto):
    """Check if the given ONNX proto type is a sequence.

    Args:
        type_proto: The ONNX TypeProto.

    Returns:
        True if the type is a sequence type.
    """
    cls_type = type_proto.WhichOneof("value")
    assert cls_type in {"tensor_type", "sequence_type"}
    return cls_type == "sequence_type"


def get_shape_from_type_proto(type_proto):
    """Extract the shape of a tensor from an ONNX type proto.

    Args:
        type_proto: The ONNX TypeProto.

    Returns:
        A list of dimension values or None if no shape is available.
    """
    assert not is_sequence(type_proto)
    if type_proto.tensor_type.HasField("shape"):
        return [get_dim_from_proto(d) for d in type_proto.tensor_type.shape.dim]
    else:
        return None


def get_elem_type_from_type_proto(type_proto):
    """Return the element type from a given TypeProto object.

    Args:
        type_proto: The ONNX TypeProto.

    Returns:
        The element type (e.g., TensorProto.FLOAT).
    """
    if is_sequence(type_proto):
        return type_proto.sequence_type.elem_type.tensor_type.elem_type
    else:
        return type_proto.tensor_type.elem_type


def get_shape_from_value_info(vi):
    """Return the shape from the given ValueInfoProto object.

    Args:
        vi: The ONNX ValueInfoProto.

    Returns:
        A list of dimension values or None.
    """
    cls_type = vi.type.WhichOneof("value")
    if cls_type is None:
        return None
    if not is_sequence(vi.type):
        return get_shape_from_type_proto(vi.type)
    if vi.type.sequence_type.elem_type.WhichOneof("value") == "tensor_type":
        return get_shape_from_type_proto(vi.type.sequence_type.elem_type)
    else:
        return None


def make_named_value_info(name):
    """Create and return an ONNX ValueInfoProto object with the specified name.

    Args:
        name: The name for the ValueInfoProto.

    Returns:
        A new ValueInfoProto with the given name.
    """
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def get_shape_from_sympy_shape(sympy_shape):
    """Convert a sympy shape to a list with int, str, or None elements.

    Args:
        sympy_shape: A list of sympy expressions.

    Returns:
        A list of int, str, or None values.
    """
    return [None if i is None else (int(i) if is_literal(i) else str(i)) for i in sympy_shape]


def is_literal(dim):
    """Check if a dimension is a literal number.

    Args:
        dim: The dimension value to check.

    Returns:
        True if the dimension is a literal number.
    """
    return type(dim) in {int, np.int64, np.int32, sympy.Integer} or (hasattr(dim, "is_number") and dim.is_number)


def handle_negative_axis(axis, rank):
    """Convert a potentially negative axis to a positive axis.

    Args:
        axis: The axis value (can be negative).
        rank: The total rank of the tensor.

    Returns:
        A non-negative axis value.
    """
    assert axis < rank and axis >= -rank
    return axis if axis >= 0 else rank + axis


def get_opset(mp, domain=None):
    """Retrieve the opset version for a given model namespace.

    Args:
        mp: The ONNX ModelProto.
        domain: The domain(s) to check. Defaults to common ONNX domains.

    Returns:
        The opset version or None if not found.
    """
    domain = domain or ["", "onnx", "ai.onnx"]
    if type(domain) != list:
        domain = [domain]
    for opset in mp.opset_import:
        if opset.domain in domain:
            return opset.version
    return None


def as_scalar(x):
    """Convert input to scalar if input is a list with a single item or a NumPy ndarray.

    Args:
        x: The input value.

    Returns:
        A scalar value.
    """
    if type(x) == list:
        assert len(x) == 1
        return x[0]
    elif type(x) == np.ndarray:
        return x.item()
    else:
        return x


def as_list(x, keep_none):
    """Convert input to list, optionally preserving None values.

    Args:
        x: The input value.
        keep_none: If True, return None as-is instead of wrapping in list.

    Returns:
        A list or None.
    """
    if type(x) == list:
        return x
    elif type(x) == np.ndarray:
        return list(x)
    elif keep_none and x is None:
        return None
    else:
        return [x]


def sympy_reduce_product(x):
    """Reduce a list or element to a product using Sympy's Integer.

    Args:
        x: A list or single value.

    Returns:
        The product as a sympy expression.
    """
    if type(x) == list:
        value = sympy.Integer(1)
        for v in x:
            value = value * v
    else:
        value = x
    return value


def numpy_to_sympy(array):
    """Convert a numpy array to a list of sympy values.

    Args:
        array: A numpy array.

    Returns:
        The converted list or value.
    """
    if isinstance(array, np.ndarray):
        if array.ndim == 0:
            return int(array.item())
        return [int(x) for x in array.flatten()]
    return array
