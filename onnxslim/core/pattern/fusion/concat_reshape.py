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
        inputs = concat_node.inputs
        variables = [i for i in inputs if isinstance(i, gs.Variable)]
        consts = [i for i in inputs if isinstance(i, gs.Constant)]

        if len(variables) != 1:
            return False

        if not all(c.values.size == 1 and int(c.values.flatten()[0]) != -1 for c in consts):
            return False

        var = variables[0]
        if var.shape is None or len(var.shape) != 1 or var.shape[0] != 1:
            return False

        return True

    def rewrite(self, opset=11):
        match_case = {}
        concat_node = self.concat_0
        reshape_node = self.reshape_0

        # Build the fused constant shape: collect values from concat inputs,
        # replacing the single dynamic (Variable) input with -1.
        shape_values = []
        for inp in concat_node.inputs:
            if isinstance(inp, gs.Constant):
                shape_values.append(int(inp.values.flatten()[0]))
            else:
                shape_values.append(-1)

        shape_constant = gs.Constant(
            reshape_node.name + "_shape",
            values=np.array(shape_values, dtype=np.int64),
        )

        # Rewire: replace Reshape's shape input (from Concat output) with the
        # new constant, effectively eliminating the Concat node.
        data_input = reshape_node.inputs[0]
        outputs = list(reshape_node.outputs)

        reshape_node.inputs.clear()
        reshape_node.outputs.clear()
        concat_node.inputs.clear()
        concat_node.outputs.clear()

        match_case[reshape_node.name] = {
            "op": "Reshape",
            "inputs": [data_input, shape_constant],
            "outputs": outputs,
            "name": reshape_node.name,
            "attrs": reshape_node.attrs,
            "domain": None,
        }

        return match_case


register_fusion_pattern(ConcatReshapeMatcher(1))
