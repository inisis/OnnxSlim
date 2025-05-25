import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class PadConvMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes the PadConvMatcher with a specified priority and defines its matching pattern."""
        pattern = Pattern(
            """
            input  input  0  1 pad_0
            Pad    pad_0  1+ 1 input conv_0
            Conv   conv_0 1+ 1 pad_0 output
            output output 1  0 conv_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern used."""
        return "FusionPadConv"

    def parameter_check(self) -> bool:
        """Validates if the padding parameter for a convolutional node is a constant."""
        pad_node = self.pad_0
        return isinstance(pad_node.inputs[1], gs.Constant)

    def rewrite(self, opset=11):
        """Rewrites the padding parameter for a convolutional node to use a constant if the current parameter is not a
        constant.
        """
        match_case = {}
        conv_node = self.conv_0
        pad_node = self.pad_0
        pad_node_users = pad_node.users

        pad_inputs = len(pad_node.inputs)
        if pad_inputs < 3 or (
            pad_inputs >= 3 and (isinstance(pad_node.inputs[2], gs.Constant) and pad_node.inputs[2].values == 0)
        ):
            if (
                isinstance(pad_node.inputs[1], gs.Constant)
                and pad_node.attrs["mode"] == "constant"
                and conv_node.inputs[1].shape
            ):
                conv_weight_dim = len(conv_node.inputs[1].shape)
                pad_value = pad_node.inputs[1].values.tolist()

                if all(pad == 0 for pad in (pad_value[:2] + pad_value[conv_weight_dim : conv_weight_dim + 2])):
                    conv_weight_dim - 2
                    input_variable = self.pad_0.inputs[0]
                    pad_variable = pad_node.outputs[0]  # pad output variable
                    index = conv_node.inputs.index(pad_variable)
                    conv_node.inputs.pop(index)
                    conv_node.inputs.insert(index, input_variable)

                    inputs = list(conv_node.inputs)
                    outputs = list(conv_node.outputs)
                    attrs = conv_node.attrs

                    conv_node.inputs.clear()
                    conv_node.outputs.clear()
                    # remove pad node if it has only one user
                    if len(pad_node_users) == 0:
                        input_variable.outputs.remove(pad_node)
                        pad_node.inputs.clear()
                        pad_node.outputs.clear()

                    conv_pads = attrs["pads"]
                    pads = pad_value[2:conv_weight_dim] + pad_value[conv_weight_dim + 2 :]
                    pads = [pad + conv_pad for pad, conv_pad in zip(pads, conv_pads)]

                    attrs["pads"] = pads
                    match_case[conv_node.name] = {
                        "op": "Conv",
                        "inputs": inputs,
                        "outputs": outputs,
                        "name": conv_node.name,
                        "attrs": conv_node.attrs,
                        "domain": None,
                    }

        return match_case


register_fusion_pattern(PadConvMatcher(1))
