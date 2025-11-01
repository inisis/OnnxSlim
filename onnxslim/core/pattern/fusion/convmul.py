import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class ConvMulMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes the ConvMulMatcher for fusing Conv and Mul layers in an ONNX graph."""
        pattern = Pattern(
            """
            input    input  0  1 conv_0
            Conv     conv_0 1+ 1 input mul_0
            Mul      mul_0  2  1 conv_0 ? output
            output   output 1  0 mul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the FusionConvMul pattern."""
        return "FusionConvMul"

    def rewrite(self, opset=11):
        match_case = {}
        conv_node = self.conv_0
        mul_node = self.mul_0
        conv_weight = list(conv_node.inputs)[1]
        if len(conv_node.users) == 1 and conv_node.users[0] == mul_node and isinstance(mul_node.inputs[1], gs.Constant):
            mul_constant = mul_node.inputs[1].values

            if mul_constant.squeeze().ndim == 1 and mul_constant.squeeze().shape[0] == conv_weight.shape[0]:
                weight_shape = conv_weight.values.shape
                reshape_shape = [-1] + [1] * (len(weight_shape) - 1)

                mul_scale_reshaped = mul_constant.squeeze().reshape(reshape_shape)
                new_weight = conv_weight.values * mul_scale_reshaped

                inputs = []
                inputs.append(next(iter(conv_node.inputs)))

                weight_name = list(conv_node.inputs)[1].name
                inputs.append(gs.Constant(weight_name, values=new_weight))

                if len(conv_node.inputs) == 3:
                    conv_bias = conv_node.inputs[2].values
                    new_bias = conv_bias * mul_constant.squeeze()
                    bias_name = list(conv_node.inputs)[2].name
                    inputs.append(gs.Constant(bias_name, values=new_bias))

                outputs = list(mul_node.outputs)

                conv_node.outputs.clear()
                mul_node.inputs.clear()
                mul_node.outputs.clear()

                match_case[conv_node.name] = {
                    "op": conv_node.op,
                    "inputs": inputs,
                    "outputs": outputs,
                    "name": conv_node.name,
                    "attrs": conv_node.attrs,
                    "domain": None,
                }

        return match_case


register_fusion_pattern(ConvMulMatcher(1))
