import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern

_UNSQUEEZE_PATTERN = """
    input      input       0  1 unsqueeze_0
    Unsqueeze  unsqueeze_0 1+ 1 input unsqueeze_1
    Unsqueeze  unsqueeze_1 1+ 1 unsqueeze_0 output
    output     output      1  0 unsqueeze_1
    """


class _UnsqueezePatternMatcherBase(PatternMatcher):
    def __init__(self, priority):
        super().__init__(Pattern(_UNSQUEEZE_PATTERN), priority)

    @property
    def name(self):
        return "EliminationUnsqueeze"

    def _get_axes(self, unsqueeze_node):
        """Return list[int] axes (normalized to non-negative) for this opset, or None if unavailable."""
        raise NotImplementedError

    def _build_axes_payload(self, merged_axes, source_name):
        """Return (attrs, extra_inputs) describing how to attach the merged axes for this opset."""
        raise NotImplementedError

    def rewrite(self, opset=11):
        match_case = {}
        n0 = self.unsqueeze_0
        n1 = self.unsqueeze_1
        users_n0 = n0.users
        if not (len(users_n0) == 1 and n0.inputs[0].shape and n1.inputs[0].shape):
            return match_case

        axes_n0 = self._get_axes(n0)
        axes_n1 = self._get_axes(n1)
        if axes_n0 is None or axes_n1 is None:
            return match_case

        axes_n0 = [a + sum(1 for b in axes_n1 if b <= a) for a in axes_n0]
        merged_axes = axes_n0 + axes_n1

        index = n1.inputs.index(n0.outputs[0])
        n1.inputs.pop(index)
        for i, item in enumerate(n0.inputs):
            n1.inputs.insert(index + i, item)
        inputs = [next(iter(n1.inputs))]
        outputs = list(n1.outputs)
        n1.inputs.clear()
        n1.outputs.clear()
        if len(users_n0) == 0:
            n0.inputs.clear()
            n0.outputs.clear()

        attrs, extra_inputs = self._build_axes_payload(merged_axes, n0.name)
        inputs.extend(extra_inputs)

        match_case[n0.name] = {
            "op": "Unsqueeze",
            "inputs": inputs,
            "outputs": outputs,
            "name": n0.name,
            "attrs": attrs,
            "domain": None,
        }
        return match_case


@register_fusion_pattern(priority=1, max_opset=12)
class UnsqueezePatternMatcher(_UnsqueezePatternMatcherBase):
    """Consecutive-Unsqueeze elimination for opsets <= 12 (axes as attribute)."""

    def _get_axes(self, unsqueeze_node):
        dim = len(unsqueeze_node.inputs[0].shape)
        axes = unsqueeze_node.attrs["axes"]
        return [a + dim + len(axes) if a < 0 else a for a in axes]

    def _build_axes_payload(self, merged_axes, source_name):
        return {"axes": merged_axes}, []


@register_fusion_pattern(priority=1, min_opset=13)
class UnsqueezePatternMatcherV13(_UnsqueezePatternMatcherBase):
    """Consecutive-Unsqueeze elimination for opsets >= 13 (axes as optional Constant input)."""

    def _get_axes(self, unsqueeze_node):
        if len(unsqueeze_node.inputs) < 2:
            return None
        axes_input = unsqueeze_node.inputs[1]
        if not isinstance(axes_input, gs.Constant):
            return None
        dim = len(unsqueeze_node.inputs[0].shape)
        axes = axes_input.values
        return [a + dim + len(axes) if a < 0 else a for a in axes]

    def _build_axes_payload(self, merged_axes, source_name):
        extra = [
            gs.Constant(
                name=f"{source_name}_axes",
                values=np.array(merged_axes, dtype=np.int64),
            )
        ]
        return None, extra
