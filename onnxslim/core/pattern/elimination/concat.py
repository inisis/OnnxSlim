from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class ConcatPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes the ConcatPatternMatcher with a specified priority using a predefined graph pattern."""
        pattern = Pattern(
            """
            input      input       0  1 concat_0
            Concat     concat_0    1+ 1 input concat_1
            Concat     concat_1    1* 1 concat_0 output
            output     output      1  0 concat_1
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the elimination pattern, 'EliminationConcat'."""
        return "EliminationConcat"

    def rewrite(self, opset=11):
        """Rewrites an elimination pattern for concat nodes by optimizing nested slice operations."""
        match_case = {}

        node_concat_0 = self.concat_0
        users_node_concat_0 = node_concat_0.users
        node_concat_1 = self.concat_1
        node_concat_0_axis = node_concat_0.attrs.get("axis", 0)
        node_concat_1.attrs.get("axis", 0)

        if all(user.op == "Concat" and user.attrs.get("axis", 0) == node_concat_0_axis for user in users_node_concat_0):
            index = node_concat_1.inputs.index(node_concat_0.outputs[0])
            node_concat_1.inputs.pop(index)
            for i, item in enumerate(node_concat_0.inputs):
                node_concat_1.inputs.insert(index + i, item)
            inputs = list(node_concat_1.inputs)
            outputs = list(node_concat_1.outputs)
            node_concat_1.inputs.clear()
            node_concat_1.outputs.clear()

            if len(users_node_concat_0) == 0:
                node_concat_0.inputs.clear()
                node_concat_0.outputs.clear()

            attrs = {"axis": node_concat_0_axis}

            match_case[node_concat_1.name] = {
                "op": "Concat",
                "inputs": inputs,
                "outputs": outputs,
                "name": node_concat_1.name,
                "attrs": attrs,
                "domain": None,
            }

        return match_case


register_fusion_pattern(ConcatPatternMatcher(1))
