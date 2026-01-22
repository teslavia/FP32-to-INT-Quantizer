import pytest
import numpy as np
import os
import tempfile
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_test_data


class TestSerialization:
    def test_save_load_int8_quant_data(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name

        try:
            quantizer.save_quant_data(quant_data, temp_path)
            loaded_data = quantizer.load_quant_data(temp_path, data.shape)
            assert loaded_data.shape == data.shape
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_load_int4_quant_data(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=4)
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quant_data, fp32_recovered = quantizer.quantize(data)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name

        try:
            quantizer.save_quant_data(quant_data, temp_path)
            loaded_data = quantizer.load_quant_data(temp_path, data.shape)
            assert loaded_data.shape == data.shape
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_load_quant_params_int8(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8, quant_mode="symmetric", quant_level="per_tensor")
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quantizer.quantize(data)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            temp_path = f.name

        try:
            quantizer.save_quant_params(temp_path)
            new_quantizer = FP32ToLowBitQuantizer(quant_bit=8)
            new_quantizer.load_quant_params(temp_path)
            assert new_quantizer.quant_bit == quantizer.quant_bit
            assert new_quantizer.quant_mode == quantizer.quant_mode
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_load_quant_params_int4(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=4, quant_mode="symmetric", quant_level="per_tensor")
        data = generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0))
        quantizer.quantize(data)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            temp_path = f.name

        try:
            quantizer.save_quant_params(temp_path)
            new_quantizer = FP32ToLowBitQuantizer(quant_bit=4)
            new_quantizer.load_quant_params(temp_path)
            assert new_quantizer.quant_bit == quantizer.quant_bit
            assert new_quantizer.quant_mode == quantizer.quant_mode
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_batch_save_int8(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        batch_data = [generate_test_data(shape=(3, 32, 32), data_range=(0.0, 1.0)) for _ in range(4)]
        results = quantizer.quantize(batch_data)
        quant_list = [res[0] for res in results]

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name

        try:
            quantizer.save_quant_data(quant_list, temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
