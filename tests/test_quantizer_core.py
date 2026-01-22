import pytest
import numpy as np
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_test_data


class TestQuantizerCore:
    def test_int8_initialization(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        assert quantizer.quant_bit == 8
        assert quantizer.int_min == -128
        assert quantizer.int_max == 127

    def test_int4_initialization(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=4)
        assert quantizer.quant_bit == 4
        assert quantizer.int_min == -8
        assert quantizer.int_max == 7

    def test_invalid_quant_bit(self):
        with pytest.raises(ValueError):
            FP32ToLowBitQuantizer(quant_bit=16)

    def test_symmetric_quantization(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, quant_mode="symmetric")
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.dtype == np.int8
        assert fp32_recovered.shape == data.shape

    def test_asymmetric_quantization(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, quant_mode="asymmetric")
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.dtype == np.int8
        assert fp32_recovered.shape == data.shape

    def test_per_tensor_quantization(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, quant_level="per_tensor")
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_per_channel_quantization(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, quant_level="per_channel")
        data = generate_test_data(shape=(16, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)
        assert quant_data.shape == data.shape

    def test_batch_quantization(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        batch_data = [
            generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0)) for _ in range(4)
        ]
        results = quantizer.quantize(batch_data)
        assert len(results) == 4
        for quant_data, fp32_recovered in results:
            assert quant_data.shape == (3, 32, 32)

    def test_custom_quant_range(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, int_min=-64, int_max=63)
        assert quantizer.int_min == -64
        assert quantizer.int_max == 63

    def test_input_validation_dtype(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        data = np.random.rand(3, 32, 32).astype(np.float64)
        with pytest.raises(TypeError):
            quantizer.quantize(data)

    def test_input_validation_nan(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        data[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            quantizer.quantize(data)

    def test_input_validation_inf(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        data[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            quantizer.quantize(data)
