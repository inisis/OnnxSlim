# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for shape inference handlers."""

from abc import ABC, abstractmethod


class ShapeHandler(ABC):
    """Abstract base class for shape inference handlers.

    Each handler is responsible for inferring the output shapes for a specific
    ONNX operator type.
    """

    def __init__(self, min_opset=1, max_opset=999):
        """Initialize the shape handler.

        Args:
            min_opset: Minimum ONNX opset version this handler supports.
            max_opset: Maximum ONNX opset version this handler supports.
        """
        self.min_opset = min_opset
        self.max_opset = max_opset

    @property
    @abstractmethod
    def op_type(self) -> str:
        """Return the ONNX operator type this handler supports.

        Returns:
            The operator type string (e.g., "Reshape", "MatMul").
        """
        raise NotImplementedError

    def supports_opset(self, opset: int) -> bool:
        """Check if this handler supports the given opset version.

        Args:
            opset: The ONNX opset version to check.

        Returns:
            True if the handler supports this opset version.
        """
        return self.min_opset <= opset <= self.max_opset

    @abstractmethod
    def infer_shape(self, node, ctx) -> None:
        """Infer the output shapes for the given node.

        Args:
            node: The ONNX node to infer shapes for.
            ctx: The InferenceContext providing access to shape information.
        """
        raise NotImplementedError


class PassthroughHandler(ShapeHandler):
    """Handler for operators that pass through input shape to output unchanged.

    This is used for operators like Identity, Reciprocal, Round, etc.
    """

    def __init__(self, op_type_name, min_opset=1, max_opset=999):
        """Initialize the passthrough handler.

        Args:
            op_type_name: The operator type name.
            min_opset: Minimum ONNX opset version this handler supports.
            max_opset: Maximum ONNX opset version this handler supports.
        """
        super().__init__(min_opset, max_opset)
        self._op_type = op_type_name

    @property
    def op_type(self) -> str:
        """Return the ONNX operator type this handler supports."""
        return self._op_type

    def infer_shape(self, node, ctx) -> None:
        """Pass through shape and type from input to output."""
        ctx.pass_on_shape_and_type(node)


class MultiOpHandler(ShapeHandler):
    """Handler that supports multiple operator types with the same logic.

    This is useful when multiple operators share the same shape inference logic.
    """

    def __init__(self, op_type_name, handler_func, min_opset=1, max_opset=999):
        """Initialize the multi-op handler.

        Args:
            op_type_name: The operator type name.
            handler_func: The function to call for shape inference.
            min_opset: Minimum ONNX opset version this handler supports.
            max_opset: Maximum ONNX opset version this handler supports.
        """
        super().__init__(min_opset, max_opset)
        self._op_type = op_type_name
        self._handler_func = handler_func

    @property
    def op_type(self) -> str:
        """Return the ONNX operator type this handler supports."""
        return self._op_type

    def infer_shape(self, node, ctx) -> None:
        """Call the handler function for shape inference."""
        self._handler_func(node, ctx)
