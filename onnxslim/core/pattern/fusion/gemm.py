import numpy as np

import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim.core.optimization.dead_node_elimination import get_constant_variable
from onnxslim.core.pattern import Pattern, PatternMatcher
from onnxslim.core.pattern.registry import register_fusion_pattern


class MatMulAddPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes a matcher for fusing MatMul and Add operations in ONNX graph optimization."""
        pattern = Pattern(
            """
            input    input    0 1 matmul_0
            MatMul   matmul_0 2 1 input ? add_0
            Add      add_0    1* 1 matmul_0 output
            output   output   1 0 add_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern as a string 'FusionGemm'."""
        return "FusionGemm"

    def rewrite(self, opset=11):
        """Rewrites the graph for the fusion pattern 'FusionGemm' based on matching criteria and constant variables in
        matmul nodes.
        """
        match_case = {}
        node = self.add_0
        matmul_node = self.matmul_0
        matmul_bias_variable = get_constant_variable(matmul_node)
        add_bias_variable = get_constant_variable(node)
        input_variable = (
            matmul_node.inputs[0] if isinstance(matmul_node.inputs[1], gs.Constant) else matmul_node.inputs[1]
        )
        users = matmul_node.users
        if len(users) == 1 and matmul_bias_variable and add_bias_variable and len(matmul_bias_variable.shape) == 2:
            if (
                input_variable.shape
                and len(input_variable.shape) > 2
                and all([isinstance(value, int) for value in input_variable.shape])
            ):
                pre_reshape_const = gs.Constant(
                    f"{matmul_node.name}_pre_reshape_in",
                    values=np.array([-1, matmul_bias_variable.values.shape[0]], dtype=np.int64),
                )
                inputs = []
                inputs.append(input_variable)
                inputs.append(pre_reshape_const)

                reshape_out_variable = gs.Variable(
                    f"{matmul_node.name}_pre_reshape_out",
                    dtype=input_variable.dtype,
                )
                outputs = [reshape_out_variable]

                match_case.update(
                    {
                        f"{matmul_node.name}_pre_reshape": {
                            "op": "Reshape",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": f"{matmul_node.name}_pre_reshape",
                            "domain": None,
                        }
                    }
                )

                add_node = node
                add_bias_variable = get_constant_variable(add_node)

                output_variable = add_node.inputs[0]
                output_variable.outputs.remove(add_node)

                matmul_bias_transpose_constant = gs.Constant(
                    matmul_bias_variable.name, values=matmul_bias_variable.values.T
                )

                inputs = []
                inputs.append(reshape_out_variable)
                inputs.append(matmul_bias_transpose_constant)
                inputs.append(add_bias_variable)

                gemm_out_variable = gs.Variable(f"{matmul_node.name}_gemm_out", dtype=output_variable.dtype)
                outputs = [gemm_out_variable]

                match_case.update(
                    {
                        matmul_node.name: {
                            "op": "Gemm",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": matmul_node.name,
                            "attrs": {
                                "alpha": 1.0,
                                "beta": 1.0,
                                "transA": 0,
                                "transB": 1,
                            },
                            "domain": None,
                        }
                    }
                )

                values = list(input_variable.shape[:-1]) + [matmul_bias_variable.values.shape[-1]]
                post_reshape_const = gs.Constant(
                    f"{matmul_node.name}_post_reshape_in",
                    values=np.array(values, dtype=np.int64),
                )

                inputs = []
                inputs.append(gemm_out_variable)
                inputs.append(post_reshape_const)
                outputs = list(add_node.outputs)

                matmul_node.outputs.clear()
                add_node.inputs.clear()
                add_node.outputs.clear()

                match_case.update(
                    {
                        f"{matmul_node.name}_post_reshape": {
                            "op": "Reshape",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": f"{matmul_node.name}_post_reshape",
                            "domain": None,
                        }
                    }
                )
            elif (
                input_variable.shape
                and len(input_variable.shape) == 2
                and all([isinstance(value, int) for value in input_variable.shape])
            ):
                add_node = node
                add_bias_variable = get_constant_variable(add_node)

                output_variable = add_node.inputs[0]
                output_variable.outputs.remove(add_node)

                matmul_bias_transpose_constant = gs.Constant(
                    matmul_bias_variable.name, values=matmul_bias_variable.values.T
                )

                inputs = []
                inputs.append(input_variable)
                inputs.append(matmul_bias_transpose_constant)
                inputs.append(add_bias_variable)

                outputs = list(add_node.outputs)
                add_node.inputs.clear()
                add_node.outputs.clear()
                match_case.update(
                    {
                        matmul_node.name: {
                            "op": "Gemm",
                            "inputs": inputs,
                            "outputs": outputs,
                            "name": matmul_node.name,
                            "attrs": {
                                "alpha": 1.0,
                                "beta": 1.0,
                                "transA": 0,
                                "transB": 1,
                            },
                            "domain": None,
                        }
                    }
                )
        return match_case


register_fusion_pattern(MatMulAddPatternMatcher(1))


class GemmMulPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes a matcher for fusing MatMul and Add operations in ONNX graph optimization."""
        pattern = Pattern(
            """
            input    input     0  1 gemm_0
            Gemm     gemm_0    1+ 1 input reshape_0
            Reshape  reshape_0 2  1 gemm_0 ? mul_0
            Mul      mul_0     1* 1 reshape_0 output
            output   output    1  0 mul_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern as a string 'FusionGemmMul'."""
        return "FusionGemmMul"

    def rewrite(self, opset=11):
        """Rewrites the graph for the fusion pattern 'FusionGemmMul' based on matching criteria and constant variables in
        gemm nodes.
        """
        match_case = {}
        gemm_node = self.gemm_0
        reshape_node = self.reshape_0
        mul_node = self.mul_0
        mul_bias_variable = get_constant_variable(mul_node)

        if (
            (
                (len(gemm_node.inputs) == 2 and isinstance(gemm_node.inputs[1], gs.Constant))
                or (
                    len(gemm_node.inputs) == 3
                    and isinstance(gemm_node.inputs[1], gs.Constant)
                    and isinstance(gemm_node.inputs[2], gs.Constant)
                )
            )
            and mul_bias_variable
            and len(reshape_node.users) == 1
        ):
            gemm_attr = gemm_node.attrs
            gemm_weight_constant = gemm_node.inputs[1]
            gemm_bias_constant = gemm_node.inputs[2] if len(gemm_node.inputs) == 3 else None
            if (
                gemm_attr["transA"] == 0
                and gemm_attr["transB"] == 1
                and (
                    (mul_bias_variable.values.ndim == 1 and gemm_weight_constant.shape[0] == mul_bias_variable.shape[0])
                    or mul_bias_variable.values.ndim == 0
                )
            ):
                gemm_weight = gemm_weight_constant.values
                mul_weight = mul_bias_variable.values
                if mul_bias_variable.values.ndim == 1:
                    gemm_weight_fused = gemm_weight * mul_weight[:, None]
                else:
                    gemm_weight_fused = gemm_weight * mul_weight
                gemm_weight_fused_constant = gs.Constant(gemm_weight_constant.name + "_fused", values=gemm_weight_fused)
                gemm_node.inputs[1] = gemm_weight_fused_constant

                if gemm_bias_constant:
                    gemm_bias = gemm_bias_constant.values
                    mul_bias = mul_bias_variable.values
                    gemm_bias_fused = gemm_bias * mul_bias
                    gemm_bias_fused_constant = gs.Constant(gemm_bias_constant.name + "_fused", values=gemm_bias_fused)
                    gemm_node.inputs[2] = gemm_bias_fused_constant

                mul_node.replace_all_uses_with(reshape_node)

        return match_case


register_fusion_pattern(GemmMulPatternMatcher(1))


class GemmAddPatternMatcher(PatternMatcher):
    def __init__(self, priority):
        """Initializes a matcher for fusing MatMul and Add operations in ONNX graph optimization."""
        pattern = Pattern(
            """
            input    input     0  1 gemm_0
            Gemm     gemm_0    1+ 1 input reshape_0
            Reshape  reshape_0 2  1 gemm_0 ? add_0
            Add      add_0     1* 1 reshape_0 output
            output   output    1  0 add_0
            """
        )
        super().__init__(pattern, priority)

    @property
    def name(self):
        """Returns the name of the fusion pattern as a string 'FusionGemmAdd'."""
        return "FusionGemmAdd"

    def rewrite(self, opset=11):
        """Rewrites the graph for the fusion pattern 'FusionGemmAdd' based on matching criteria and constant variables in
        gemm nodes.
        """
        match_case = {}
        gemm_node = self.gemm_0
        reshape_node = self.reshape_0
        add_node = self.add_0
        add_bias_variable = get_constant_variable(add_node)

        if (
            (
                (len(gemm_node.inputs) == 2)
                or (len(gemm_node.inputs) == 3 and isinstance(gemm_node.inputs[2], gs.Constant))
            )
            and add_bias_variable
            and len(reshape_node.users) == 1
        ):
            gemm_bias_constant = gemm_node.inputs[2] if len(gemm_node.inputs) == 3 else None
            if gemm_bias_constant:
                gemm_bias = gemm_bias_constant.values
                add_bias = add_bias_variable.values
                gemm_bias_fused = gemm_bias + add_bias
                gemm_bias_fused_constant = gs.Constant(gemm_bias_constant.name + "_fused", values=gemm_bias_fused)
                gemm_node.inputs[2] = gemm_bias_fused_constant
            else:
                gemm_node.inputs[2] = add_bias_variable

            add_node.replace_all_uses_with(reshape_node)

        return match_case


register_fusion_pattern(GemmAddPatternMatcher(1))
