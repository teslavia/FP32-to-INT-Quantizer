import pytest
import numpy as np
import tempfile
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_test_data
from fp32_to_int_quantizer.visualization.metrics import evaluate_precision


class TestVisualization:
    def test_evaluate_precision_basic(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)

        metrics = quantizer.evaluate_precision(data, fp32_recovered, sig_analysis=False)

        assert "MSE" in metrics
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "PSNR" in metrics
        assert "SNR" in metrics
        assert metrics["quant_bit"] == 8

    def test_evaluate_precision_with_statistics(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)

        metrics = quantizer.evaluate_precision(data, fp32_recovered, sig_analysis=True)

        assert "Error_Std" in metrics
        assert "Error_Var" in metrics
        assert "Error_Skewness" in metrics
        assert "Error_Kurtosis" in metrics
        assert "Error_Entropy" in metrics

    def test_evaluate_precision_int4(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=4)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)

        metrics = quantizer.evaluate_precision(data, fp32_recovered, sig_analysis=True)
        assert metrics["quant_bit"] == 4

    def test_visualize_analysis_save_png(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            quantizer.visualize_analysis(
                fp32_original=data,
                fp32_recovered=fp32_recovered,
                save_path=temp_path,
                sig_analysis=False,
            )
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_visualize_analysis_with_reference(self):
        quantizer_int8 = FP32ToLowBitQuantizer(quant_bit=8)
        quantizer_int4 = FP32ToLowBitQuantizer(quant_bit=4)

        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))

        _, fp32_recovered_int8 = quantizer_int8.quantize(data)
        _, fp32_recovered_int4 = quantizer_int4.quantize(data)

        int8_metrics = quantizer_int8.evaluate_precision(data, fp32_recovered_int8, sig_analysis=False)
        int8_metrics["scales"] = quantizer_int8.scales
        int8_metrics["zero_points"] = quantizer_int8.zero_points

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            quantizer_int4.visualize_analysis(
                fp32_original=data,
                fp32_recovered=fp32_recovered_int4,
                save_path=temp_path,
                ref_metrics=int8_metrics,
                sig_analysis=False,
            )
            assert os.path.exists(temp_path)
        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
