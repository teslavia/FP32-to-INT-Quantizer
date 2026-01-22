"""
INT4/INT8 precision comparison report example.
INT4/INT8精度对比报告示例
"""
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_batch_test_data
import os


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

    # Save quantized data for storage comparison
    int8_quantizer.save_quant_data([res[0] for res in int8_results], "int8_quantized.bin")
    int4_quantizer.save_quant_data([res[0] for res in int4_results], "int4_quantized.bin")

    print("\n" + "=" * 80)
    print("INT4/INT8 Quantization Precision Comparison Report")
    print("=" * 80)

    print("\n--- INT8 Metrics ---")
    for key, value in int8_metrics.items():
        if key not in ["scales", "zero_points"]:
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n--- INT4 Metrics ---")
    for key, value in int4_metrics.items():
        if key not in ["scales", "zero_points"]:
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n--- Storage Comparison ---")
    int4_size = os.path.getsize("int4_quantized.bin") if os.path.exists("int4_quantized.bin") else 0
    int8_size = os.path.getsize("int8_quantized.bin") if os.path.exists("int8_quantized.bin") else 0
    print(f"  INT4 Size: {int4_size} bytes")
    print(f"  INT8 Size: {int8_size} bytes")
    if int8_size > 0:
        print(f"  Compression Ratio: {int8_size / int4_size:.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
