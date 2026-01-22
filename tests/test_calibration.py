import pytest
import numpy as np
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_test_data


class TestCalibration:
    def test_extremes_calibration_per_tensor(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, calib_mode=None)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_extremes_calibration_per_channel(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, quant_level="per_channel", calib_mode=None)
        data = generate_test_data(shape=(16, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_percentile_calibration_per_tensor(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, calib_mode="percentile", calib_percentile=99.9)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_percentile_calibration_per_channel(self):
        quantizer = FP32ToLowBitQuantizer(
            quant_bit=8, quant_level="per_channel", calib_mode="percentile", calib_percentile=99.9
        )
        data = generate_test_data(shape=(16, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_entropy_calibration_per_tensor(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, calib_mode="entropy", entropy_bins=2048)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_entropy_calibration_per_channel(self):
        quantizer = FP32ToLowBitQuantizer(
            quant_bit=8, quant_level="per_channel", calib_mode="entropy", entropy_bins=2048
        )
        data = generate_test_data(shape=(16, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_entropy_calibration_int4(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=4, calib_mode="entropy", entropy_bins=2048)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_invalid_calibration_mode(self):
        with pytest.raises(ValueError):
            quantizer = FP32ToLowBitQuantizer(quant_bit=8, calib_mode="invalid_mode")
            data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
            quantizer.quantize(data)
