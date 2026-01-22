"""
PyTorch model quantization example.
PyTorch模型量化示例
"""
import torch
from fp32_to_int_quantizer import FP32ToLowBitQuantizer


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.fc1 = torch.nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        return self.fc1(self.conv1(x).flatten(1))


def main():
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 32, 32).float()

    quantizer = FP32ToLowBitQuantizer(
        quant_bit=4,
        quant_mode="symmetric",
        quant_level="per_channel",
        calib_mode="entropy",
    )

    quantized_model, layer_quant_params = quantizer.quantize_torch_model(model, dummy_input)
    print("Quantized layers and parameters:", list(layer_quant_params.keys()))


if __name__ == "__main__":
    main()
