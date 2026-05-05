import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern

_REDUCE_PATTERN = """
    input     input       0   1 reduce_0
    ReduceSum reduce_0    1+  1 input unsqueeze_0
    Unsqueeze unsqueeze_0 1+  1 reduce_0 output
    output    output      1   0 unsqueeze_0
    """


def _build_match_case(reduce_node, unsqueeze_node):
    inputs = list(reduce_node.inputs)
    outputs = list(unsqueeze_node.outputs)
    attrs = reduce_node.attrs
    reduce_node.outputs.clear()
    unsqueeze_node.inputs.clear()
    unsqueeze_node.outputs.clear()
    attrs["keepdims"] = 1
    return {
        reduce_node.name: {
            "op": reduce_node.op,
            "inputs": inputs,
            "outputs": outputs,
            "name": reduce_node.name,
            "attrs": attrs,
            "domain": None,
        }
    }


@register_fusion_pattern(priority=1, max_opset=12)
class ReducePatternMatcher(PatternMatcher):
    """Reduce/Unsqueeze fusion for opsets <= 12, where ReduceSum/Unsqueeze axes are node attributes."""

    def __init__(self, priority):
        super().__init__(Pattern(_REDUCE_PATTERN), priority)

    @property
    def name(self):
        return "FusionReduce"

    def rewrite(self, opset=11):
        match_case = {}
        reduce_node = self.reduce_0
        unsqueeze_node = self.unsqueeze_0
        if len(reduce_node.users) != 1:
            return match_case

        reduce_node_axes = reduce_node.attrs.get("axes", None)
        reduce_node_keepdims = reduce_node.attrs.get("keepdims", 1)
        unsqueeze_node_axes = unsqueeze_node.attrs.get("axes", None)

        if reduce_node_axes == unsqueeze_node_axes and reduce_node_keepdims == 0:
            return _build_match_case(reduce_node, unsqueeze_node)
        return match_case


@register_fusion_pattern(priority=1, min_opset=13)
class ReducePatternMatcherV13(PatternMatcher):
    """Reduce/Unsqueeze fusion for opsets >= 13, where axes is an optional input.

    In opset 13+ a ReduceSum without an axes input is valid (semantics controlled by
    `noop_with_empty_axes`); the fusion only applies when both the ReduceSum and the
    following Unsqueeze carry an explicit Constant axes input.
    """

    def __init__(self, priority):
        super().__init__(Pattern(_REDUCE_PATTERN), priority)

    @property
    def name(self):
        return "FusionReduce"

    def rewrite(self, opset=13):
        match_case = {}
        reduce_node = self.reduce_0
        unsqueeze_node = self.unsqueeze_0
        if len(reduce_node.users) != 1:
            return match_case

        if len(reduce_node.inputs) < 2 or len(unsqueeze_node.inputs) < 2:
            return match_case
        reduce_axes_input = reduce_node.inputs[1]
        unsqueeze_axes_input = unsqueeze_node.inputs[1]
        if not (isinstance(reduce_axes_input, gs.Constant) and isinstance(unsqueeze_axes_input, gs.Constant)):
            return match_case

        reduce_node_axes = reduce_axes_input.values
        unsqueeze_node_axes = unsqueeze_axes_input.values
        reduce_node_keepdims = reduce_node.attrs.get("keepdims", 1)

        if reduce_node_axes == unsqueeze_node_axes and reduce_node_keepdims == 0:
            return _build_match_case(reduce_node, unsqueeze_node)
        return match_case
