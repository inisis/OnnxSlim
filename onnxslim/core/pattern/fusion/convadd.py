import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class ConvAddMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes the ConvAddMatcher for fusing Conv and Add layers in an ONNX graph."""
        pattern = Pattern(
            """
            input    input  0  1 conv_0
            Conv     conv_0 1+ 1 input bn_0
            Add      add_0  2  1 conv_0 ? output
            output   output 1  0 add_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the FusionConvAdd pattern."""
        return "FusionConvAdd"

    def rewrite(self, opset=11):
        match_case = {}
        conv_node = self.conv_0
        conv_weight = list(conv_node.inputs)[1]
        conv_node_users = conv_node.users
        node = self.add_0
        if (
            len(conv_node_users) == 1
            and isinstance(node.inputs[1], gs.Constant)
            and isinstance(conv_weight, gs.Constant)
            and node.inputs[1].values.squeeze().ndim == 1
            and node.inputs[1].values.squeeze().shape[0] == conv_weight.shape[0]
        ):
            add_node = node
            if len(conv_node.inputs) == 2:
                conv_bias = node.inputs[1].values.squeeze()
            else:
                conv_bias = conv_node.inputs[2].values + node.inputs[1].values.squeeze()

            inputs = []
            inputs.append(list(conv_node.inputs)[0])
            inputs.append(conv_weight)
            weight_name = list(conv_node.inputs)[1].name
            if weight_name.endswith("weight"):
                bias_name = f"{weight_name[:-6]}bias"
            else:
                bias_name = f"{weight_name}_bias"
            inputs.append(gs.Constant(bias_name, values=conv_bias))
            outputs = list(add_node.outputs)

            conv_node.outputs.clear()
            add_node.inputs.clear()
            add_node.outputs.clear()

            match_case[conv_node.name] = {
                "op": conv_node.op,
                "inputs": inputs,
                "outputs": outputs,
                "name": conv_node.name,
                "attrs": conv_node.attrs,
                "domain": None,
            }

        return match_case


register_fusion_pattern(ConvAddMatcher(1))
