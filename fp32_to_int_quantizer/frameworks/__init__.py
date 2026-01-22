__all__ = []

try:
    from fp32_to_int_quantizer.frameworks.torch import quantize_torch_tensor, quantize_torch_model
    __all__.extend(["quantize_torch_tensor", "quantize_torch_model"])
except ImportError:
    pass

try:
    from fp32_to_int_quantizer.frameworks.tensorflow import quantize_tf_tensor, quantize_tf_model
    __all__.extend(["quantize_tf_tensor", "quantize_tf_model"])
except ImportError:
    pass

