"""
Basic quantization example for INT4/INT8 comparison.
INT4/INT4量化对比基础示例
"""
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_batch_test_data


def main():
    config = {
        "quant_mode": "symmetric",
        "quant_level": "per_channel",
        "calib_mode": "entropy",
        "entropy_bins": 2048,
        "seed": 42,
    }

    batch_data = generate_batch_test_data(
        batch_size=4,
        data_shape=(3, 32, 32),
        data_range=(0.0, 1.0),
        is_image_like=True,
    )
    print(f"Generated batch test data: {len(batch_data)} samples, shape: {batch_data[0].shape}")

    int8_quantizer = FP32ToLowBitQuantizer(quant_bit=8, **config)
    int8_results = int8_quantizer.quantize(batch_data)
    int8_metrics = int8_quantizer.evaluate_precision(batch_data[0], int8_results[0][1])
    int8_metrics["scales"] = int8_quantizer.scales
    int8_metrics["zero_points"] = int8_quantizer.zero_points

    int4_quantizer = FP32ToLowBitQuantizer(quant_bit=4, **config)
    int4_results = int4_quantizer.quantize(batch_data)
    int4_metrics = int4_quantizer.evaluate_precision(batch_data[0], int4_results[0][1])
    int4_metrics["scales"] = int4_quantizer.scales
    int4_metrics["zero_points"] = int4_quantizer.zero_points

    int4_quantizer.visualize_analysis(
        fp32_original=batch_data[0],
        fp32_recovered=int4_results[0][1],
        save_path="int4_int8_comparison.png",
        ref_metrics=int8_metrics,
        sig_analysis=True,
    )

    int4_quantizer.save_quant_data([res[0] for res in int4_results], "int4_quantized.bin")
    int4_quantizer.save_quant_params("int4_quant_params.npy")
    int8_quantizer.save_quant_data([res[0] for res in int8_results], "int8_quantized.bin")
    int8_quantizer.save_quant_params("int8_quant_params.npy")

    print("\n" + "=" * 80)
    print("INT4/INT8 Quantization Comparison Completed!")
    print(f"INT4 Quantized Data Size:  bytes")
    print(f"INT8 Quantized Data Size:  bytes")
    print("Generated Files:")
    print("- int4_int8_comparison.png (Multi-Precision Visualization Report)")
    print("- int4_quantized.bin / int8_quantized.bin (Quantized Binary Data)")
    print("- int4_quant_params.npy / int8_quant_params.npy (Quantization Parameters)")
    print("=" * 80)


if __name__ == "__main__":
    main()
