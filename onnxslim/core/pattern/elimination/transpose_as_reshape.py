import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


def _is_static(dim) -> bool:
    return isinstance(dim, int) and dim > 0


@register_fusion_pattern(priority=1)
class TransposeAsReshapeMatcher(PatternMatcher):
    """Replaces a Transpose with an equivalent Reshape when its perm only reorders size-1 axes.
    Real(non-1) axes must keep their relative order, or it's a genuine data reorder and not safe to rewrite.
    A dim with unknown static value (e.g. a dynamic batch axis) is conservatively treated as non-1; at
    most one such dim can still be expressed via ONNX Reshape's single `-1`.
    """

    def __init__(self, priority):
        pattern = Pattern(
            """
            input      input        0  1  transpose_0
            Transpose  transpose_0  1  1  input output
            output     output       1  0  transpose_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "EliminationTransposeAsReshape"

    def parameter_check(self):
        # EliminationTranspose may have already consumed this node earlier in the same pass.
        return bool(self.transpose_0.inputs) and bool(self.transpose_0.outputs)

    def rewrite(self, opset=11):
        node = self.transpose_0
        data = node.inputs[0]
        shape = data.shape
        if not shape:
            return {}
        if any(isinstance(d, int) and d == 0 for d in shape):
            # ONNX Reshape's shape encoding can't safely express a real empty axis here: a literal `0`
            # means "copy from the input at this same position" (not "set to zero"), and combining it with
            # `-1` for some other axis is undefined (division by zero) - so a 0-sized tensor is out of scope.
            return {}

        perm = list(node.attrs["perm"])
        real_axes = [p for p in perm if not (_is_static(shape[p]) and shape[p] == 1)]
        if real_axes != sorted(real_axes):
            return {}

        dynamic_count = sum(1 for p in perm if not _is_static(shape[p]))
        if dynamic_count > 1:
            return {}  # ONNX Reshape allows at most one inferred (-1) dimension

        new_shape = [shape[p] if _is_static(shape[p]) else -1 for p in perm]
        shape_const = gs.Constant(
            name=f"{node.outputs[0].name}_shape",
            values=np.array(new_shape, dtype=np.int64),
        )

        inputs = [data, shape_const]
        outputs = list(node.outputs)
        name = node.outputs[0].name
        node.inputs.clear()
        node.outputs.clear()

        return {
            name: {
                "op": "Reshape",
                "inputs": inputs,
                "outputs": outputs,
                "name": name,
                "attrs": {},
                "domain": None,
            }
        }
