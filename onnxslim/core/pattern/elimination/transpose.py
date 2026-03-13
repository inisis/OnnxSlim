import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class TransposePatternMatcher(PatternMatcher):
    def __init__(self, priority):
        pattern = Pattern(
            """
            input      input        0  1 transpose_0
            Transpose  transpose_0  1  1 input transpose_1
            Transpose  transpose_1  1  1 transpose_0 output
            output     output       1  0 transpose_1
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        return "EliminationTranspose"

    def rewrite(self, opset=11):
        match_case = {}
        node_0 = self.transpose_0
        node_1 = self.transpose_1

        if len(node_0.users) != 1:
            return match_case

        perm_0 = list(node_0.attrs["perm"])
        perm_1 = list(node_1.attrs["perm"])

        # Compose: combined[i] = perm_0[perm_1[i]]
        combined = [perm_0[p] for p in perm_1]

        inputs = [node_0.inputs[0]]
        outputs = list(node_1.outputs)

        node_0.inputs.clear()
        node_0.outputs.clear()
        node_1.inputs.clear()
        node_1.outputs.clear()

        # If the combined perm is identity, just wire input to output directly
        if combined == list(range(len(combined))):
            match_case[node_0.name] = {
                "op": "Identity",
                "inputs": inputs,
                "outputs": outputs,
                "name": node_0.name,
                "attrs": {},
                "domain": None,
            }
        else:
            match_case[node_0.name] = {
                "op": "Transpose",
                "inputs": inputs,
                "outputs": outputs,
                "name": node_0.name,
                "attrs": {"perm": combined},
                "domain": None,
            }

        return match_case


register_fusion_pattern(TransposePatternMatcher(1))
