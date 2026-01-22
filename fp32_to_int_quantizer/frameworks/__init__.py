from fp32_to_int_quantizer.frameworks.torch import quantize_torch_tensor, quantize_torch_model
from fp32_to_int_quantizer.frameworks.tensorflow import quantize_tf_tensor, quantize_tf_model

__all__ = [
    "quantize_torch_tensor",
    "quantize_torch_model",
    "quantize_tf_tensor",
    "quantize_tf_model",
]
