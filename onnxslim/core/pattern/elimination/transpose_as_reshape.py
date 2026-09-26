import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


def _is_static(dim) -> bool:
    return isinstance(dim, int) and dim > 0


@register_fusion_pattern(priority=1, min_opset=5)
class TransposeAsReshapeMatcher(PatternMatcher):
    """Replaces a Transpose with an equivalent Reshape when its perm only reorders size-1 axes.
    Real(non-1) axes must keep their relative order, or it's a genuine data reorder and not safe to rewrite.
    A dim with unknown static value (e.g. a dynamic batch axis) is conservatively treated as non-1; at
    most one such dim can still be expressed via ONNX Reshape's single `-1`. The rewrite is only applied
    when the Transpose is adjacent to a Reshape; isolated Transposes are left to backends.
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
        node = self.transpose_0
        if not node.inputs or not node.outputs:
            return False

        has_previous_reshape = any(producer.op == "Reshape" for producer in node.inputs[0].inputs)
        has_next_reshape = any(user.op == "Reshape" for user in node.users)
        return has_previous_reshape or has_next_reshape

    def rewrite(self, opset=11):
        node = self.transpose_0
        data = node.inputs[0]
        shape = data.shape
        if shape is None:
            return {}
        if any(isinstance(dim, int) and dim == 0 for dim in shape):
            return {}

        if "perm" in node.attrs:
            perm = list(node.attrs["perm"])
        else:
            perm = list(reversed(range(len(shape))))
        axes = [(p, shape[p], _is_static(shape[p])) for p in perm]
        real_axes = [p for p, dim, is_static in axes if not (is_static and dim == 1)]
        if real_axes != sorted(real_axes):
            return {}

        dynamic_count = sum(1 for _, _, is_static in axes if not is_static)
        if dynamic_count > 1:
            return {}  # ONNX Reshape allows at most one inferred (-1) dimension

        new_shape = [dim if is_static else -1 for _, dim, is_static in axes]
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
