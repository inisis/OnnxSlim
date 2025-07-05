import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class ConcatReshapeMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input    input  0   1 concat_0
            Concat   concat_0   1+ 1 input reshape_0
            Reshape  reshape_0  2  1 ? concat_0 output
            output   output     1  0 reshape_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "FusionConcatReshape"

    def parameter_check(self):
        concat_node = self.concat_0

        def check_inputs(inputs):
            vars = [i for i in inputs if isinstance(i, gs.Variable)]
            consts = [i for i in inputs if isinstance(i, gs.Constant)]
            return (
                len(vars) == 1 and all(c.values.size == 1 and c.values != -1 for c in consts) and vars[0].shape == [1]
            )

        return check_inputs(concat_node.inputs)

    def rewrite(self, opset=11):
        match_case = {}
        concat_node = self.concat_0
        index = [idx for idx, i in enumerate(concat_node.inputs) if isinstance(i, gs.Variable)][0]
        constant = gs.Constant(
            concat_node.inputs[index].name + "_fixed",
            values=np.array([-1], dtype=np.int64),
        )
        concat_node.inputs.pop(index)
        concat_node.inputs.insert(index, constant)

        return match_case


register_fusion_pattern(ConcatReshapeMatcher(1))
