import torch
import numpy as np
from typing import Tuple, Optional, Dict
from fp32_to_int_quantizer import FP32ToLowBitQuantizer


def quantize_torch_tensor(
    torch_tensor: torch.Tensor,
    quantizer: FP32ToLowBitQuantizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch_tensor.device
    fp32_data = torch_tensor.cpu().detach().numpy().astype(np.float32)

    quant_data, fp32_recovered = quantizer.quantize(fp32_data)

    return torch.from_numpy(quant_data).to(device), torch.from_numpy(fp32_recovered).to(device)


def quantize_torch_model(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    quantizer: FP32ToLowBitQuantizer,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, dict]]:
    if device is None:
        device = dummy_input.device
    model = model.to(device).eval()

    layer_quant_params = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = module.weight.data
            print(f"Quantizing layer: {name} (Shape: {weight.shape}), Precision: INT{quantizer.quant_bit}")

            layer_quantizer = FP32ToLowBitQuantizer(
                quant_bit=quantizer.quant_bit,
                quant_mode=quantizer.quant_mode,
                quant_level=quantizer.quant_level,
                calib_mode=quantizer.calib_mode,
                calib_percentile=quantizer.calib_percentile,
                entropy_bins=quantizer.entropy_bins,
                int_min=quantizer.int_min,
                int_max=quantizer.int_max,
            )

            _, recovered_weight = quantize_torch_tensor(weight, layer_quantizer)
            module.weight.data = recovered_weight

            layer_quant_params[name] = {
                "scale": layer_quantizer.scales,
                "zero_point": layer_quantizer.zero_points,
                "quant_bit": quantizer.quant_bit,
            }

    print(f"PyTorch model weight quantization completed (Precision: INT{quantizer.quant_bit})")
    return model, layer_quant_params
