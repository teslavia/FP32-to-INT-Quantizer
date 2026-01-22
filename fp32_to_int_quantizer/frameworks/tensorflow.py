import tensorflow as tf
import numpy as np
from typing import Tuple, Dict
from fp32_to_int_quantizer.core.quantizer import FP32ToLowBitQuantizer


def quantize_tf_tensor(
    tf_tensor: tf.Tensor,
    quantizer: FP32ToLowBitQuantizer,
) -> Tuple[tf.Tensor, tf.Tensor]:
    fp32_data = tf_tensor.numpy().astype(np.float32)
    quant_data, fp32_recovered = quantizer.quantize(fp32_data)
    return tf.convert_to_tensor(quant_data), tf.convert_to_tensor(fp32_recovered)


def quantize_tf_model(
    model: tf.keras.Model,
    dummy_input: tf.Tensor,
    quantizer: FP32ToLowBitQuantizer,
) -> Tuple[tf.keras.Model, Dict[str, dict]]:
    model = model.copy()
    layer_quant_params = {}

    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            if layer.weights:
                weight = layer.weights[0].numpy().astype(np.float32)
                print(f"Quantizing layer: {layer.name} (Shape: {weight.shape}), Precision: INT{quantizer.quant_bit}")

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

                _, recovered_weight = layer_quantizer.quantize(weight)
                layer.set_weights([recovered_weight] + layer.weights[1:])

                layer_quant_params[layer.name] = {
                    "scale": layer_quantizer.scales,
                    "zero_point": layer_quantizer.zero_points,
                    "quant_bit": quantizer.quant_bit,
                }

    print(f"TensorFlow model weight quantization completed (Precision: INT{quantizer.quant_bit})")
    return model, layer_quant_params
