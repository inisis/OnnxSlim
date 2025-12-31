import os
import tempfile

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto

import onnxslim
import onnxslim.third_party.onnx_graphsurgeon as gs
from onnxslim import slim
from onnxslim.core import (
    convert_data_format,
    freeze,
    input_modification,
    input_shape_modification,
    optimize,
    output_modification,
    shape_infer,
)
from onnxslim.core.optimization import OptimizationSettings
from onnxslim.utils import (
    TensorInfo,
    ModelInfo,
    OperatorInfo,
    check_onnx,
    check_point,
    check_result,
    format_bytes,
    format_model_info,
    get_initializer_size,
    get_ir_version,
    get_itemsize,
    get_numpy_type,
    get_opset,
    init_logging,
    is_onnxruntime_available,
    onnx_dtype_to_numpy,
    onnxruntime_inference,
    print_model_info_as_table,
    save,
    summarize_model,
    gen_onnxruntime_input_data,
    calculate_tensor_size,
    get_graph_initializer_size,
    dump_model_info_to_disk,
    check_onnx_compatibility,
    update_outputs_dims,
)


class TestUtilFunctions:
    """Tests for utility functions in utils.py"""

    def test_format_bytes_int(self):
        """Test format_bytes with integer input."""
        assert "0.00 B" == format_bytes(0)
        assert "100.00 B" == format_bytes(100)
        assert "1.00 KB" == format_bytes(1024)
        assert "1.00 MB" == format_bytes(1024 * 1024)
        assert "1.00 GB" == format_bytes(1024 * 1024 * 1024)

    def test_format_bytes_tuple(self):
        """Test format_bytes with tuple input."""
        result = format_bytes((1024, 2048))
        assert "1.00 KB" in result
        assert "2.00 KB" in result

    def test_format_bytes_numpy_int(self):
        """Test format_bytes with numpy integer."""
        result = format_bytes(np.int64(1024))
        assert "1.00 KB" == result

    def test_onnx_dtype_to_numpy(self):
        """Test onnx_dtype_to_numpy conversion."""
        assert onnx_dtype_to_numpy(TensorProto.FLOAT) == np.float32
        assert onnx_dtype_to_numpy(TensorProto.INT32) == np.int32
        assert onnx_dtype_to_numpy(TensorProto.INT64) == np.int64
        assert onnx_dtype_to_numpy(TensorProto.DOUBLE) == np.float64

    def test_onnx_dtype_to_numpy_undefined(self):
        """Test onnx_dtype_to_numpy with undefined type."""
        result = onnx_dtype_to_numpy(9999)  # Invalid dtype
        assert result == "UNDEFINED"

    def test_get_numpy_type(self):
        """Test get_numpy_type function."""
        # Already numpy type
        assert get_numpy_type(np.float32) == np.float32

        # ONNX type
        result = get_numpy_type(TensorProto.FLOAT)
        assert result == np.float32

    def test_get_itemsize(self):
        """Test get_itemsize function."""
        assert get_itemsize(TensorProto.FLOAT) == 4
        assert get_itemsize(TensorProto.DOUBLE) == 8
        assert get_itemsize(TensorProto.INT32) == 4
        assert get_itemsize(TensorProto.INT64) == 8
        assert get_itemsize(TensorProto.INT16) == 2
        assert get_itemsize(TensorProto.BFLOAT16) == 2

    def test_get_itemsize_float8(self):
        """Test get_itemsize for float8 types."""
        if hasattr(TensorProto, 'FLOAT8E4M3FN'):
            assert get_itemsize(TensorProto.FLOAT8E4M3FN) == 1

    def test_init_logging(self):
        """Test init_logging function."""
        logger = init_logging(verbose=False)
        assert logger is not None

        logger_verbose = init_logging(verbose=True)
        assert logger_verbose is not None

    def test_is_onnxruntime_available(self):
        """Test is_onnxruntime_available function."""
        result = is_onnxruntime_available()
        assert isinstance(result, bool)
        assert result is True  # Should be available in test environment

    def test_check_result_matching(self):
        """Test check_result with matching outputs."""
        raw_output = {"output": np.array([1.0, 2.0, 3.0])}
        slimmed_output = {"output": np.array([1.0, 2.0, 3.0])}
        assert check_result(raw_output, slimmed_output) is True

    def test_check_result_mismatched_keys(self):
        """Test check_result with mismatched keys."""
        raw_output = {"output1": np.array([1.0, 2.0, 3.0])}
        slimmed_output = {"output2": np.array([1.0, 2.0, 3.0])}
        assert check_result(raw_output, slimmed_output) is False

    def test_check_result_mismatched_values(self):
        """Test check_result with mismatched values."""
        raw_output = {"output": np.array([1.0, 2.0, 3.0])}
        slimmed_output = {"output": np.array([4.0, 5.0, 6.0])}
        assert check_result(raw_output, slimmed_output) is False


class TestModelCreation:
    """Helper class to create test models."""

    @staticmethod
    def create_simple_conv_model():
        """Create a simple Conv model for testing."""
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 222, 222])

        weights = numpy_helper.from_array(
            np.random.randn(16, 3, 3, 3).astype(np.float32), name="weights"
        )

        conv_node = helper.make_node(
            "Conv",
            ["input", "weights"],
            ["output"],
            kernel_shape=[3, 3],
        )

        graph = helper.make_graph(
            [conv_node],
            "test-conv",
            [input_tensor],
            [output_tensor],
            initializer=[weights],
        )

        model = helper.make_model(graph, producer_name="test")
        model.opset_import[0].version = 11
        return model

    @staticmethod
    def create_linear_model():
        """Create a simple linear model for testing."""
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        return LinearModel()


class TestModelInfo:
    """Tests for ModelInfo class."""

    def test_model_info_from_model(self):
        """Test ModelInfo creation from model."""
        model = TestModelCreation.create_simple_conv_model()
        info = ModelInfo(model, "test_model")

        assert info.tag == "test_model"
        assert info.op_set is not None
        assert info.ir_version is not None
        assert "Conv" in info.op_type_counts
        assert len(info.input_info) > 0
        assert len(info.output_info) > 0

    def test_model_info_from_file(self):
        """Test ModelInfo creation from file path."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            info = ModelInfo(f.name)

            assert info.tag is not None
            assert info.model_size >= 0

            os.unlink(f.name)

    def test_model_info_input_maps(self):
        """Test ModelInfo input_maps property."""
        model = TestModelCreation.create_simple_conv_model()
        info = ModelInfo(model)

        input_maps = info.input_maps
        assert "input" in input_maps

    def test_model_info_output_maps(self):
        """Test ModelInfo output_maps property."""
        model = TestModelCreation.create_simple_conv_model()
        info = ModelInfo(model)

        output_maps = info.output_maps
        assert "output" in output_maps


class TestTensorInfo:
    """Tests for TensorInfo class."""

    def test_tensor_info_extraction(self):
        """Test TensorInfo extraction from tensor."""
        model = TestModelCreation.create_simple_conv_model()
        tensor = model.graph.input[0]

        info = TensorInfo(tensor)

        assert info.name == "input"
        assert info.dtype == np.float32
        assert info.shape == (1, 3, 224, 224)


class TestOperatorInfo:
    """Tests for OperatorInfo class."""

    def test_operator_info_extraction(self):
        """Test OperatorInfo extraction from node."""
        model = TestModelCreation.create_simple_conv_model()
        node = model.graph.node[0]

        info = OperatorInfo(node)

        assert info.op == "Conv"


class TestCoreFunctions:
    """Tests for core functions."""

    def test_get_opset(self):
        """Test get_opset function."""
        model = TestModelCreation.create_simple_conv_model()
        opset = get_opset(model)
        assert opset == 11

    def test_get_ir_version(self):
        """Test get_ir_version function."""
        model = TestModelCreation.create_simple_conv_model()
        ir_version = get_ir_version(model)
        assert ir_version is not None

    def test_freeze(self):
        """Test freeze function."""
        model = TestModelCreation.create_simple_conv_model()
        freeze(model)
        # Should not raise any errors

    def test_shape_infer(self):
        """Test shape_infer function."""
        model = TestModelCreation.create_simple_conv_model()
        inferred_model = shape_infer(model)
        assert inferred_model is not None

    def test_optimize(self):
        """Test optimize function."""
        model = TestModelCreation.create_simple_conv_model()
        OptimizationSettings.reset(None)
        optimized_model = optimize(model)
        assert optimized_model is not None

    def test_check_point(self):
        """Test check_point function."""
        model = TestModelCreation.create_simple_conv_model()
        graph = check_point(model)
        assert graph is not None

    def test_input_shape_modification(self):
        """Test input_shape_modification function."""
        model = TestModelCreation.create_simple_conv_model()
        modified = input_shape_modification(model, ["input:1,3,128,128"])

        assert modified is not None
        # Check input shape was modified
        graph = gs.import_onnx(modified)
        # Shape could be list or tuple
        assert list(graph.inputs[0].shape) == [1, 3, 128, 128]

    def test_input_shape_modification_invalid_key(self):
        """Test input_shape_modification with invalid key."""
        model = TestModelCreation.create_simple_conv_model()

        with pytest.raises(Exception, match="not found in model"):
            input_shape_modification(model, ["invalid_input:1,3,128,128"])

    def test_output_modification(self):
        """Test output_modification function."""
        model = TestModelCreation.create_simple_conv_model()
        modified = output_modification(model, ["output"])

        assert modified is not None

    def test_output_modification_with_dtype(self):
        """Test output_modification with dtype specification."""
        model = TestModelCreation.create_simple_conv_model()
        modified = output_modification(model, ["output:fp32"])

        assert modified is not None

    def test_output_modification_invalid_key(self):
        """Test output_modification with invalid key."""
        model = TestModelCreation.create_simple_conv_model()

        with pytest.raises(Exception, match="not found in model"):
            output_modification(model, ["invalid_output"])

    def test_input_modification(self):
        """Test input_modification function."""
        model = TestModelCreation.create_simple_conv_model()
        modified = input_modification(model, ["input"])

        assert modified is not None

    def test_input_modification_with_dtype(self):
        """Test input_modification with dtype specification."""
        model = TestModelCreation.create_simple_conv_model()
        modified = input_modification(model, ["input:fp32"])

        assert modified is not None

    def test_convert_data_format_fp16(self):
        """Test convert_data_format to fp16."""
        model = TestModelCreation.create_simple_conv_model()
        converted = convert_data_format(model, "fp16")

        assert converted is not None

    def test_convert_data_format_fp32(self):
        """Test convert_data_format to fp32."""
        model = TestModelCreation.create_simple_conv_model()
        # First convert to fp16
        fp16_model = convert_data_format(model, "fp16")
        # Then convert back to fp32
        fp32_model = convert_data_format(fp16_model, "fp32")

        assert fp32_model is not None


class TestSaveFunction:
    """Tests for save function."""

    def test_save_model(self):
        """Test save function."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            save(model, f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_save_model_with_check(self):
        """Test save function with model check."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            save(model, f.name, model_check=True)
            assert os.path.exists(f.name)
            os.unlink(f.name)


class TestOnnxRuntimeInference:
    """Tests for onnxruntime inference functions."""

    def test_gen_onnxruntime_input_data(self):
        """Test gen_onnxruntime_input_data function."""
        model = TestModelCreation.create_simple_conv_model()
        input_data = gen_onnxruntime_input_data(model)

        assert "input" in input_data
        assert input_data["input"].shape == (1, 3, 224, 224)

    def test_onnxruntime_inference(self):
        """Test onnxruntime_inference function."""
        model = TestModelCreation.create_simple_conv_model()
        input_data = gen_onnxruntime_input_data(model)

        output, model = onnxruntime_inference(model, input_data)

        assert "output" in output

    def test_check_onnx(self):
        """Test check_onnx function."""
        model = TestModelCreation.create_simple_conv_model()

        input_data, output, model = check_onnx(model)

        assert input_data is not None
        assert output is not None


class TestSummarizeModel:
    """Tests for summarize_model function."""

    def test_summarize_model(self):
        """Test summarize_model function."""
        model = TestModelCreation.create_simple_conv_model()
        summary = summarize_model(model)

        assert summary is not None
        assert "Conv" in summary.op_type_counts

    def test_summarize_model_from_file(self):
        """Test summarize_model from file."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            summary = summarize_model(f.name)

            assert summary is not None
            os.unlink(f.name)


class TestFormatModelInfo:
    """Tests for format_model_info function."""

    def test_format_model_info(self):
        """Test format_model_info function."""
        model = TestModelCreation.create_simple_conv_model()
        info = summarize_model(model, "test")

        formatted = format_model_info(info)
        assert formatted is not None
        assert len(formatted) > 0

    def test_format_model_info_list(self):
        """Test format_model_info with list input."""
        model = TestModelCreation.create_simple_conv_model()
        info1 = summarize_model(model, "test1")
        info2 = summarize_model(model, "test2")

        formatted = format_model_info([info1, info2])
        assert formatted is not None

    def test_format_model_info_with_time(self):
        """Test format_model_info with elapsed time."""
        model = TestModelCreation.create_simple_conv_model()
        info = summarize_model(model, "test")

        formatted = format_model_info(info, elapsed_time=1.5)
        assert formatted is not None


class TestPrintModelInfo:
    """Tests for print_model_info_as_table function."""

    def test_print_model_info_as_table(self, capsys):
        """Test print_model_info_as_table function."""
        model = TestModelCreation.create_simple_conv_model()
        info = summarize_model(model, "test")

        print_model_info_as_table(info)

        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_print_model_info_as_table_with_time(self, capsys):
        """Test print_model_info_as_table with elapsed time."""
        model = TestModelCreation.create_simple_conv_model()
        info = summarize_model(model, "test")

        print_model_info_as_table(info, elapsed_time=1.5)

        captured = capsys.readouterr()
        assert "1.5" in captured.out or "1.50" in captured.out


class TestInitializerSize:
    """Tests for initializer size functions."""

    def test_get_initializer_size(self):
        """Test get_initializer_size function."""
        model = TestModelCreation.create_simple_conv_model()
        size = get_initializer_size(model)

        assert size > 0

    def test_calculate_tensor_size(self):
        """Test calculate_tensor_size function."""
        tensor = numpy_helper.from_array(
            np.random.randn(16, 3, 3, 3).astype(np.float32), name="test"
        )

        size = calculate_tensor_size(tensor)
        expected_size = 16 * 3 * 3 * 3 * 4  # float32 = 4 bytes
        assert size == expected_size


class TestSlimFunction:
    """Tests for the main slim function."""

    def test_slim_model_object(self):
        """Test slim with model object."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model)

        assert slimmed is not None

    def test_slim_model_file(self):
        """Test slim with model file."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            slimmed = slim(f.name)

            assert slimmed is not None
            os.unlink(f.name)

    def test_slim_with_output(self):
        """Test slim with output file."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)
            slim(input_path, output_path)

            assert os.path.exists(output_path)

    def test_slim_with_model_check(self):
        """Test slim with model check enabled."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model, model_check=True)

        assert slimmed is not None

    def test_slim_with_dtype(self):
        """Test slim with dtype conversion."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model, dtype="fp16")

        assert slimmed is not None

    def test_slim_with_no_shape_infer(self):
        """Test slim with shape inference disabled."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model, no_shape_infer=True)

        assert slimmed is not None

    def test_slim_inspect_mode(self):
        """Test slim in inspect mode."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            # Inspect mode with list triggers special handling
            result = slim([f.name], inspect=True)
            assert result is None
            os.unlink(f.name)

    def test_slim_list_of_models(self):
        """Test slim with list of models (inspect mode)."""
        model1 = TestModelCreation.create_simple_conv_model()
        model2 = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "model1.onnx")
            path2 = os.path.join(tempdir, "model2.onnx")

            onnx.save(model1, path1)
            onnx.save(model2, path2)

            # List of models triggers inspect mode
            result = slim([path1, path2])
            assert result is None


class TestOptimizationSettings:
    """Tests for OptimizationSettings."""

    def test_reset_optimization_settings(self):
        """Test reset of optimization settings."""
        OptimizationSettings.reset(None)
        assert OptimizationSettings.enabled() is True

    def test_skip_optimizations(self):
        """Test skipping optimizations."""
        OptimizationSettings.reset(["constant_folding"])
        assert OptimizationSettings.constant_folding is False

    def test_all_optimizations_disabled(self):
        """Test all optimizations disabled."""
        OptimizationSettings.reset(["constant_folding", "dead_node_elimination",
                                    "graph_fusion", "subexpression_elimination",
                                    "weight_tying"])
        assert OptimizationSettings.enabled() is False


class TestDumpModelInfo:
    """Tests for dump_model_info_to_disk function."""

    def test_dump_model_info(self):
        """Test dump_model_info_to_disk function."""
        model = TestModelCreation.create_simple_conv_model()
        info = summarize_model(model, "test_dump")

        with tempfile.TemporaryDirectory() as tempdir:
            old_cwd = os.getcwd()
            os.chdir(tempdir)
            try:
                dump_model_info_to_disk(info)
                csv_path = "test_dump_model_info.csv"
                assert os.path.exists(csv_path)
            finally:
                os.chdir(old_cwd)


class TestUpdateOutputsDims:
    """Tests for update_outputs_dims function."""

    def test_update_outputs_dims(self):
        """Test update_outputs_dims function."""
        model = TestModelCreation.create_simple_conv_model()
        output_dims = {"output": (1, 16, 222, 222)}

        updated = update_outputs_dims(model, output_dims)
        assert updated is not None

    def test_update_outputs_dims_with_string(self):
        """Test update_outputs_dims with string dimension."""
        model = TestModelCreation.create_simple_conv_model()
        output_dims = {"output": ("batch", 16, 222, 222)}

        updated = update_outputs_dims(model, output_dims)
        assert updated is not None


class TestCheckOnnxCompatibility:
    """Tests for check_onnx_compatibility function."""

    def test_check_onnx_compatibility(self, capsys):
        """Test check_onnx_compatibility function."""
        # This should not raise any errors
        check_onnx_compatibility()
        # Just verify it runs without error


class TestPytorchModels:
    """Tests using PyTorch models for higher coverage."""

    def test_relu_model(self, request):
        """Test ReLU model."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        model = Model()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_bn_model(self, request):
        """Test BatchNorm model."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                return self.bn(x)

        model = Model()
        model.eval()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_sequential_model(self, request):
        """Test Sequential model with multiple layers."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                )

            def forward(self, x):
                return self.seq(x)

        model = Model()
        model.eval()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_concat_model(self, request):
        """Test model with concatenation."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
                self.conv2 = nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x):
                y1 = self.conv1(x)
                y2 = self.conv2(x)
                return torch.cat([y1, y2], dim=1)

        model = Model()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_reshape_model(self, request):
        """Test model with reshape operations."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                batch_size = x.shape[0]
                return x.view(batch_size, -1)

        model = Model()
        input_tensor = torch.randn(2, 3, 4, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_transpose_model(self, request):
        """Test model with transpose operations."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        model = Model()
        input_tensor = torch.randn(2, 3, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_softmax_model(self, request):
        """Test model with softmax."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, x):
                return self.softmax(x)

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_sigmoid_model(self, request):
        """Test model with sigmoid."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(x)

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_maxpool_model(self, request):
        """Test model with MaxPool."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x):
                return self.pool(x)

        model = Model()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_avgpool_model(self, request):
        """Test model with AvgPool."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AvgPool2d(2)

            def forward(self, x):
                return self.pool(x)

        model = Model()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_add_model(self, request):
        """Test model with add operations."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("bias", torch.ones(1, 3, 1, 1))

            def forward(self, x):
                return x + self.bias

        model = Model()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_mul_model(self, request):
        """Test model with mul operations."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(1, 3, 1, 1) * 2)

            def forward(self, x):
                return x * self.scale

        model = Model()
        input_tensor = torch.randn(1, 3, 32, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)


class TestGenOnnxRuntimeInputData:
    """Tests for gen_onnxruntime_input_data with various scenarios."""

    def test_gen_input_with_int_dtype(self):
        """Test generating input data with integer dtype."""
        input_tensor = helper.make_tensor_value_info("input", TensorProto.INT64, [1, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.INT64, [1, 10])

        identity_node = helper.make_node("Identity", ["input"], ["output"])

        graph = helper.make_graph(
            [identity_node],
            "test-int",
            [input_tensor],
            [output_tensor],
        )

        model = helper.make_model(graph)
        input_data = gen_onnxruntime_input_data(model)

        assert input_data["input"].dtype == np.int64

    def test_gen_input_with_dynamic_shape(self):
        """Test generating input data with dynamic shape."""
        # Create input with dim_param for dynamic shape
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 10])

        identity_node = helper.make_node("Identity", ["input"], ["output"])

        graph = helper.make_graph(
            [identity_node],
            "test-dynamic",
            [input_tensor],
            [output_tensor],
        )

        model = helper.make_model(graph)
        input_data = gen_onnxruntime_input_data(model)

        assert "input" in input_data

    def test_gen_input_with_model_check_inputs_shape(self):
        """Test generating input data with model_check_inputs specifying shape."""
        model = TestModelCreation.create_simple_conv_model()
        input_data = gen_onnxruntime_input_data(model, ["input:1,3,112,112"])

        assert input_data["input"].shape == (1, 3, 112, 112)


class TestArgparser:
    """Tests for argument parser."""

    def test_argument_parser_creation(self):
        """Test OnnxSlimArgumentParser creation."""
        from onnxslim.argparser import (
            OnnxSlimArgumentParser,
            ModelArguments,
            OptimizationArguments,
            ModificationArguments,
            CheckerArguments,
        )

        parser = OnnxSlimArgumentParser(
            ModelArguments, OptimizationArguments, ModificationArguments, CheckerArguments
        )
        assert parser is not None

    def test_cli_with_dtype(self):
        """Test CLI with dtype conversion."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, output_path, "--dtype", "fp16"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    def test_cli_with_input_shapes(self):
        """Test CLI with input shapes."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, output_path, "--input-shapes", "input:1,3,224,224"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    def test_cli_with_skip_optimizations(self):
        """Test CLI with skip optimizations."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, output_path, "--skip-optimizations", "constant_folding"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    def test_cli_multiple_models_inspect(self):
        """Test CLI with multiple models in inspect mode."""
        import subprocess

        model1 = TestModelCreation.create_simple_conv_model()
        model2 = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            path1 = os.path.join(tempdir, "model1.onnx")
            path2 = os.path.join(tempdir, "model2.onnx")

            onnx.save(model1, path1)
            onnx.save(model2, path2)

            result = subprocess.run(
                ["uv", "run", "onnxslim", path1, path2, "--inspect"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0


class TestAdditionalPatterns:
    """Additional tests for pattern matchers."""

    def test_concat_reshape_pattern(self, request):
        """Test concat-reshape pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                y = self.conv(x)
                # Concat then reshape
                z = torch.cat([y, y], dim=1)
                return z.reshape(z.shape[0], -1)

        model = Model()
        input_tensor = torch.randn(1, 3, 8, 8)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_reduce_sum_pattern(self, request):
        """Test reduce sum pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.sum(dim=-1, keepdim=True)

        model = Model()
        input_tensor = torch.randn(2, 3, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_reduce_mean_pattern(self, request):
        """Test reduce mean pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.mean(dim=-1, keepdim=True)

        model = Model()
        input_tensor = torch.randn(2, 3, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_slice_pattern(self, request):
        """Test slice pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Multiple slice operations
                y = x[:, :2]
                z = y[:, 1:]
                return z

        model = Model()
        input_tensor = torch.randn(4, 8)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_gemm_without_bias(self, request):
        """Test GEMM without bias."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_gemm_with_bias(self, request):
        """Test GEMM with bias."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5, bias=True)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_layernorm_model(self, request):
        """Test model with LayerNorm."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(32)

            def forward(self, x):
                return self.ln(x)

        model = Model()
        input_tensor = torch.randn(2, 8, 32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_attention_like_model(self, request):
        """Test attention-like model with matmul and softmax."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v):
                # Simple attention-like computation
                scores = torch.matmul(q, k.transpose(-2, -1))
                weights = torch.softmax(scores, dim=-1)
                return torch.matmul(weights, v)

        model = Model()
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, (q, k, v), f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_residual_block(self, request):
        """Test residual block pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x):
                residual = x
                x = self.relu(self.conv1(x))
                x = self.conv2(x)
                return x + residual

        model = Model()
        input_tensor = torch.randn(1, 16, 8, 8)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_squeeze_unsqueeze(self, request):
        """Test squeeze and unsqueeze operations."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x.unsqueeze(0)
                x = x.squeeze(0)
                return x

        model = Model()
        input_tensor = torch.randn(3, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_split_concat(self, request):
        """Test split and concat operations."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                parts = torch.split(x, 2, dim=1)
                return torch.cat(parts, dim=1)

        model = Model()
        input_tensor = torch.randn(2, 6, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_expand_model(self, request):
        """Test expand operation."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.expand(2, 3, 4)

        model = Model()
        input_tensor = torch.randn(1, 3, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_gather_model(self, request):
        """Test gather operation."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("indices", torch.tensor([0, 2]))

            def forward(self, x):
                return torch.index_select(x, 1, self.indices)

        model = Model()
        input_tensor = torch.randn(2, 4, 3)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)


class TestMoreCoreFeatures:
    """More tests for core features."""

    def test_output_modification_fp16(self):
        """Test output modification with fp16."""
        model = TestModelCreation.create_simple_conv_model()
        modified = output_modification(model, ["output:fp16"])
        assert modified is not None

    def test_output_modification_int32(self):
        """Test output modification with int32."""
        model = TestModelCreation.create_simple_conv_model()
        modified = output_modification(model, ["output:int32"])
        assert modified is not None

    def test_input_modification_fp16(self):
        """Test input modification with fp16."""
        model = TestModelCreation.create_simple_conv_model()
        modified = input_modification(model, ["input:fp16"])
        assert modified is not None

    def test_input_modification_int32(self):
        """Test input modification with int32."""
        model = TestModelCreation.create_simple_conv_model()
        modified = input_modification(model, ["input:int32"])
        assert modified is not None

    def test_output_modification_bool(self):
        """Test output modification with bool dtype."""
        # Create a model with bool output
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10])

        greater_node = helper.make_node(
            "Greater",
            ["input", "threshold"],
            ["output"]
        )

        threshold = numpy_helper.from_array(np.zeros((1, 10), dtype=np.float32), name="threshold")

        graph = helper.make_graph(
            [greater_node],
            "test-bool",
            [input_tensor],
            [output_tensor],
            initializer=[threshold]
        )

        model = helper.make_model(graph)
        modified = output_modification(model, ["output:bool"])
        assert modified is not None

    def test_input_modification_bool(self):
        """Test input modification with bool dtype."""
        input_tensor = helper.make_tensor_value_info("input", TensorProto.BOOL, [1, 10])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10])

        not_node = helper.make_node("Not", ["input"], ["output"])

        graph = helper.make_graph(
            [not_node],
            "test-bool-input",
            [input_tensor],
            [output_tensor],
        )

        model = helper.make_model(graph)
        modified = input_modification(model, ["input:bool"])
        assert modified is not None

    def test_slim_with_skip_optimizations(self):
        """Test slim with skip_optimizations."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model, skip_optimizations=["constant_folding"])
        assert slimmed is not None

    def test_slim_with_skip_fusion_patterns(self):
        """Test slim with skip_fusion_patterns."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model, skip_fusion_patterns=["FusionConvBN"])
        assert slimmed is not None

    def test_slim_verbose(self):
        """Test slim with verbose mode."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model, verbose=True)
        assert slimmed is not None

    def test_slim_size_threshold(self):
        """Test slim with size_threshold."""
        model = TestModelCreation.create_simple_conv_model()
        slimmed = slim(model, size_threshold=1000)
        assert slimmed is not None


class TestDeadNodeEliminationDirect:
    """Direct tests for dead_node_elimination."""

    def test_identity_with_multiple_outputs(self):
        """Test identity elimination with node having multiple outputs."""
        # Create a graph with identity node that has multiple consumers
        input_var = gs.Variable(name="input", dtype=np.float32, shape=(1, 3, 4))
        output_var1 = gs.Variable(name="output1", dtype=np.float32, shape=(1, 3, 4))
        output_var2 = gs.Variable(name="output2", dtype=np.float32, shape=(1, 3, 4))

        identity_out = gs.Variable(name="id_out", dtype=np.float32, shape=(1, 3, 4))
        id_node = gs.Node(op="Identity", inputs=[input_var], outputs=[identity_out])

        relu_node1 = gs.Node(op="Relu", inputs=[identity_out], outputs=[output_var1])
        relu_node2 = gs.Node(op="Relu", inputs=[identity_out], outputs=[output_var2])

        graph = gs.Graph(
            inputs=[input_var],
            outputs=[output_var1, output_var2],
            nodes=[id_node, relu_node1, relu_node2],
        )

        from onnxslim.core.optimization.dead_node_elimination import dead_node_elimination
        dead_node_elimination(graph)
        graph.cleanup().toposort()

        # Identity should still be eliminated
        assert not any(node.op == "Identity" for node in graph.nodes)


class TestPatternMatchersDirect:
    """Direct tests for pattern matchers with specific ONNX patterns."""

    def test_concat_elimination_pattern(self):
        """Test concat elimination pattern - nested concats."""
        # Create graph with nested concat pattern
        input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 3, 4))
        input2 = gs.Variable(name="input2", dtype=np.float32, shape=(1, 3, 4))
        input3 = gs.Variable(name="input3", dtype=np.float32, shape=(1, 3, 4))

        concat1_out = gs.Variable(name="concat1_out", dtype=np.float32, shape=(1, 6, 4))
        concat2_out = gs.Variable(name="concat2_out", dtype=np.float32, shape=(1, 9, 4))

        concat1 = gs.Node(op="Concat", inputs=[input1, input2], outputs=[concat1_out], attrs={"axis": 1})
        concat2 = gs.Node(op="Concat", inputs=[concat1_out, input3], outputs=[concat2_out], attrs={"axis": 1})

        graph = gs.Graph(
            inputs=[input1, input2, input3],
            outputs=[concat2_out],
            nodes=[concat1, concat2],
        )

        model = gs.export_onnx(graph)
        slimmed = slim(model, model_check=True)
        assert slimmed is not None

    def test_reduce_sum_elimination(self):
        """Test ReduceSum + Unsqueeze pattern (opset < 13)."""
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 1, 4])

        reduce_node = helper.make_node(
            "ReduceSum",
            ["input"],
            ["reduce_out"],
            axes=[1],
            keepdims=0
        )

        unsqueeze_node = helper.make_node(
            "Unsqueeze",
            ["reduce_out"],
            ["output"],
            axes=[1]
        )

        graph = helper.make_graph(
            [reduce_node, unsqueeze_node],
            "test-reduce",
            [input_tensor],
            [output_tensor],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        slimmed = slim(model)
        assert slimmed is not None


class TestUtilsAdditional:
    """Additional utility tests."""

    def test_get_opset_none(self):
        """Test get_opset with model without opset."""
        # Create minimal model without proper opset
        graph = helper.make_graph([], "empty", [], [])
        model = helper.make_model(graph)
        model.opset_import.pop()  # Remove opset

        opset = get_opset(model)
        assert opset is None

    def test_format_bytes_large(self):
        """Test format_bytes with very large values."""
        # More than 1 GB
        result = format_bytes(2 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_get_itemsize_unsupported(self):
        """Test get_itemsize with unsupported dtype."""
        with pytest.raises(ValueError):
            get_itemsize(9999)  # Invalid dtype

    def test_summarize_model_with_subgraph(self):
        """Test summarize_model with model containing subgraph (If node)."""
        # Create a model with If node
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
        cond_tensor = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

        # Then branch
        then_out = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [1])
        then_node = helper.make_node("Identity", ["input"], ["then_out"])
        then_graph = helper.make_graph([then_node], "then_graph", [], [then_out])

        # Else branch
        else_out = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [1])
        else_node = helper.make_node("Neg", ["input"], ["else_out"])
        else_graph = helper.make_graph([else_node], "else_graph", [], [else_out])

        if_node = helper.make_node(
            "If",
            ["cond"],
            ["output"],
            then_branch=then_graph,
            else_branch=else_graph
        )

        graph = helper.make_graph(
            [if_node],
            "if_test",
            [cond_tensor, input_tensor],
            [output_tensor]
        )

        model = helper.make_model(graph)
        summary = summarize_model(model)

        assert "If" in summary.op_type_counts

    def test_check_result_with_nan(self):
        """Test check_result with NaN values (should use equal_nan)."""
        raw_output = {"output": np.array([1.0, np.nan, 3.0])}
        slimmed_output = {"output": np.array([1.0, np.nan, 3.0])}
        assert check_result(raw_output, slimmed_output) is True

    def test_gen_input_data_with_npy_file(self):
        """Test generating input data with .npy file."""
        model = TestModelCreation.create_simple_conv_model()

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            np.save(f.name, test_data)

            input_data = gen_onnxruntime_input_data(model, [f"input:{f.name}"])

            assert "input" in input_data
            np.testing.assert_array_equal(input_data["input"], test_data)
            os.unlink(f.name)

    def test_gen_input_data_invalid_key(self):
        """Test generating input data with invalid key raises error."""
        model = TestModelCreation.create_simple_conv_model()

        with pytest.raises(Exception, match="not found in model"):
            gen_onnxruntime_input_data(model, ["invalid_input:1,3,224,224"])


class TestCLIMain:
    """Tests for CLI main function."""

    def test_cli_basic(self):
        """Test basic CLI invocation."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, output_path],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert os.path.exists(output_path)

    def test_cli_inspect(self):
        """Test CLI inspect mode."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, "--inspect"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    def test_cli_model_check(self):
        """Test CLI with model check."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, output_path, "--model-check"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    def test_cli_no_shape_infer(self):
        """Test CLI with no shape inference."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, output_path, "--no-shape-infer"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

    def test_cli_verbose(self):
        """Test CLI with verbose mode."""
        import subprocess

        model = TestModelCreation.create_simple_conv_model()

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = os.path.join(tempdir, "input.onnx")
            output_path = os.path.join(tempdir, "output.onnx")

            onnx.save(model, input_path)

            result = subprocess.run(
                ["uv", "run", "onnxslim", input_path, output_path, "--verbose"],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0


class TestEdgeCases:
    """Edge case tests for higher coverage."""

    def test_output_modification_unsupported_dtype(self):
        """Test output modification with unsupported dtype raises error."""
        model = TestModelCreation.create_simple_conv_model()
        with pytest.raises(Exception, match="unsupported dtype"):
            output_modification(model, ["output:invalid_type"])

    def test_input_modification_unsupported_dtype(self):
        """Test input modification with unsupported dtype raises error."""
        model = TestModelCreation.create_simple_conv_model()
        with pytest.raises(Exception, match="unsupported dtype"):
            input_modification(model, ["input:invalid_type"])

    def test_output_modification_no_dtype(self):
        """Test output modification when tensor has no dtype."""
        # Create model where intermediate tensor has no dtype
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

        # Create intermediate without explicit dtype in value_info
        relu_node = helper.make_node("Relu", ["input"], ["intermediate"])
        identity_node = helper.make_node("Identity", ["intermediate"], ["output"])

        graph = helper.make_graph(
            [relu_node, identity_node],
            "test",
            [input_tensor],
            [output_tensor]
        )

        model = helper.make_model(graph)
        # Modify to use intermediate as output
        modified = output_modification(model, ["intermediate"])
        assert modified is not None

    def test_input_modification_no_dtype(self):
        """Test input modification when tensor has no dtype."""
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

        relu_node = helper.make_node("Relu", ["input"], ["intermediate"])
        identity_node = helper.make_node("Identity", ["intermediate"], ["output"])

        graph = helper.make_graph(
            [relu_node, identity_node],
            "test",
            [input_tensor],
            [output_tensor]
        )

        model = helper.make_model(graph)
        # Modify to use intermediate as input (this will trigger the no dtype path)
        modified = input_modification(model, ["intermediate"])
        assert modified is not None

    def test_slim_with_duplicate_input_names(self):
        """Test freeze with duplicate input names."""
        # Create model with duplicate input names in initializer
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

        weight = numpy_helper.from_array(np.ones((1, 3), dtype=np.float32), name="weight")

        add_node = helper.make_node("Add", ["input", "weight"], ["output"])

        graph = helper.make_graph(
            [add_node],
            "test",
            [input_tensor, helper.make_tensor_value_info("weight", TensorProto.FLOAT, [1, 3])],
            [output_tensor],
            initializer=[weight]
        )

        model = helper.make_model(graph)
        slimmed = slim(model)
        assert slimmed is not None

    def test_optimization_settings_stats(self):
        """Test OptimizationSettings stats method."""
        from onnxslim.core.optimization import OptimizationSettings

        OptimizationSettings.reset(None)
        stats = OptimizationSettings.stats()

        assert "constant_folding" in stats
        assert stats["constant_folding"] is True


class TestMorePatterns:
    """More pattern tests."""

    def test_gelu_pattern(self, request):
        """Test GELU activation pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()

            def forward(self, x):
                return self.gelu(x)

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name)
            assert slimmed is not None
            os.unlink(f.name)

    def test_conv_add_pattern(self, request):
        """Test Conv + Add pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
                self.register_buffer("bias", torch.randn(1, 16, 1, 1))

            def forward(self, x):
                return self.conv(x) + self.bias

        model = Model()
        input_tensor = torch.randn(1, 3, 8, 8)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_conv_mul_pattern(self, request):
        """Test Conv + Mul pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.register_buffer("scale", torch.ones(1, 16, 1, 1) * 2)

            def forward(self, x):
                return self.conv(x) * self.scale

        model = Model()
        input_tensor = torch.randn(1, 3, 8, 8)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_pad_conv_pattern(self, request):
        """Test Pad + Conv pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.pad = nn.ZeroPad2d(1)
                self.conv = nn.Conv2d(3, 16, 3, padding=0)

            def forward(self, x):
                return self.conv(self.pad(x))

        model = Model()
        input_tensor = torch.randn(1, 3, 8, 8)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_two_matmul_add_pattern(self, request):
        """Test two consecutive linear layers (MatMul + Add)."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_multiple_unsqueeze(self, request):
        """Test multiple unsqueeze operations."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Multiple unsqueeze
                x = x.unsqueeze(0)
                x = x.unsqueeze(0)
                return x

        model = Model()
        input_tensor = torch.randn(3, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_matmul_add_2d(self, request):
        """Test MatMul + Add pattern with 2D input."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 5))
                self.bias = nn.Parameter(torch.randn(5))

            def forward(self, x):
                return torch.matmul(x, self.weight) + self.bias

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_matmul_add_3d(self, request):
        """Test MatMul + Add pattern with 3D input."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 5))
                self.bias = nn.Parameter(torch.randn(5))

            def forward(self, x):
                return torch.matmul(x, self.weight) + self.bias

        model = Model()
        input_tensor = torch.randn(2, 4, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_flatten_linear(self, request):
        """Test Flatten + Linear pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(48, 10)

            def forward(self, x):
                return self.linear(self.flatten(x))

        model = Model()
        input_tensor = torch.randn(2, 3, 4, 4)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_leaky_relu(self, request):
        """Test LeakyReLU pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lrelu = nn.LeakyReLU(0.1)

            def forward(self, x):
                return self.lrelu(x)

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)

    def test_elu(self, request):
        """Test ELU pattern."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.elu = nn.ELU()

            def forward(self, x):
                return self.elu(x)

        model = Model()
        input_tensor = torch.randn(2, 10)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(model, input_tensor, f.name, opset_version=14, dynamo=False)
            slimmed = slim(f.name, model_check=True)
            assert slimmed is not None
            os.unlink(f.name)



if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-p", "no:warnings", "-v", __file__]))
