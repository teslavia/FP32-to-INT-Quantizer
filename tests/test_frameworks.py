import pytest
import torch
import numpy as np
from fp32_to_int_quantizer import FP32ToLowBitQuantizer


class TestPyTorch:
    def test_quantize_torch_tensor(self):
        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        torch_tensor = torch.randn(3, 32, 32, dtype=torch.float32)

        quant_data, fp32_recovered = quantizer.quantize_torch_tensor(torch_tensor)

        assert isinstance(quant_data, torch.Tensor)
        assert isinstance(fp32_recovered, torch.Tensor)
        assert quant_data.shape == torch_tensor.shape

    def test_quantize_torch_model(self):
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.fc1 = torch.nn.Linear(16 * 32 * 32, 10)

            def forward(self, x):
                return self.fc1(self.conv1(x).flatten(1))

        model = SimpleCNN()
        dummy_input = torch.randn(1, 3, 32, 32).float()

        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        quantized_model, layer_params = quantizer.quantize_torch_model(model, dummy_input)

        assert isinstance(quantized_model, torch.nn.Module)
        assert isinstance(layer_params, dict)
        assert len(layer_params) > 0

    def test_quantize_torch_model_int4(self):
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.fc1 = torch.nn.Linear(16 * 32 * 32, 10)

            def forward(self, x):
                return self.fc1(self.conv1(x).flatten(1))

        model = SimpleCNN()
        dummy_input = torch.randn(1, 3, 32, 32).float()

        quantizer = FP32ToLowBitQuantizer(quant_bit=4)
        quantized_model, layer_params = quantizer.quantize_torch_model(model, dummy_input)

        assert isinstance(quantized_model, torch.nn.Module)
        assert len(layer_params) > 0

    def test_quantize_torch_model_empty(self):
        class EmptyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)

        model = EmptyModel()
        dummy_input = torch.randn(1, 10).float()

        quantizer = FP32ToLowBitQuantizer(quant_bit=8)
        quantized_model, layer_params = quantizer.quantize_torch_model(model, dummy_input)

        assert isinstance(quantized_model, torch.nn.Module)
        assert len(layer_params) > 0
