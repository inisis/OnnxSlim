import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


@register_fusion_pattern(priority=1)
class ReshapeAsPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes the ReshapeAsPatternMatcher with a priority and a specific pattern for reshape as operations."""
        pattern = Pattern(
            """
            input     input     0  1 shape
            Shape     shape     1+ 1 input gather
            Gather    gather    1+ 1 shape unsqueeze
            Unsqueeze unsqueeze 1+ 1 gather output
            Concat    concat    1+ 1 unsqueeze output
            output    output    1  0 concat
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name 'EliminationReshapeAs'."""
        return "EliminationReshapeAs"

    def parameter_check(self) -> bool:
        shape_node = self.shape
        if shape_node.outputs[0].shape is None:
            return False

        if len(shape_node.users) != shape_node.outputs[0].shape[0]:
            return False

        if not all([user.op == "Gather" for user in shape_node.users]):
            return False

        gather_nodes = []
        for idx, user in enumerate(shape_node.users):
            if not isinstance(user.inputs[1], gs.Constant):
                return False

            if user.inputs[1].values.shape != ():
                return False

            if user.inputs[1].values != idx:
                return False

            gather_nodes.append(user)

        concat_node = self.concat
        if len(concat_node.inputs) != len(shape_node.users):
            return False

        # Verify that each Concat input traces back to the corresponding Gather via Unsqueeze
        for idx, concat_input in enumerate(concat_node.inputs):
            if not concat_input.inputs:
                return False
            unsqueeze_node = concat_input.inputs[0]
            if unsqueeze_node.op != "Unsqueeze":
                return False
            if unsqueeze_node.inputs[0] not in gather_nodes[idx].outputs:
                return False

        return True

    def rewrite(self, opset=11):
        """Rewrites the pattern by replacing the Concat node with the Shape node."""
        match_case = {}
        shape_node = self.shape
        concat_node = self.concat

        concat_node.replace_all_uses_with(shape_node)

        return match_case
