import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from scipy import stats

plt.rcParams.update(plt.rcParamsDefault)


def visualize_analysis(
    fp32_original: np.ndarray,
    fp32_recovered: np.ndarray,
    save_path: Optional[str] = None,
    ref_metrics: Optional[Dict] = None,
    sig_analysis: bool = True,
    quant_bit: int = 8,
    scales=None,
    zero_points=None,
) -> None:
    from fp32_to_int_quantizer.visualization.metrics import evaluate_precision

    original_flat = fp32_original.flatten()
    recovered_flat = fp32_recovered.flatten()
    error_flat = original_flat - recovered_flat

    n_cols = 3 if ref_metrics is None else 4
    fig, axes = plt.subplots(2, n_cols, figsize=(15 if ref_metrics is None else 20, 10))
    fig.suptitle(
        f"FP32→INT{quant_bit} Quantization Analysis Report" + (f" (vs INT{ref_metrics['quant_bit']})" if ref_metrics else ""),
        fontsize=16,
    )

    axes[0, 0].hist(original_flat, bins=50, alpha=0.7, label="Original FP32", color="blue")
    axes[0, 0].hist(recovered_flat, bins=50, alpha=0.7, label=f"INT{quant_bit} Dequantized", color="orange")
    axes[0, 0].set_title("Data Distribution Comparison")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(error_flat, bins=50, color="red", alpha=0.7)
    axes[0, 1].set_title(f"INT{quant_bit} Quantization Error Distribution")
    axes[0, 1].set_xlabel("Error (Original - Dequantized)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    if len(fp32_original.shape) == 4:
        error_2d = error_flat.reshape(fp32_original.shape)[0, 0, :, :]
    else:
        error_2d = error_flat.reshape(fp32_original.shape)
    im = axes[0, 2].imshow(error_2d, cmap="RdYlBu_r", aspect="auto")
    axes[0, 2].set_title(f"INT{quant_bit} Error Heatmap")
    axes[0, 2].set_xlabel("Dimension 2")
    axes[0, 2].set_ylabel("Dimension 1")
    plt.colorbar(im, ax=axes[0, 2], label="Error Value")

    if ref_metrics is not None:
        metrics = evaluate_precision(fp32_original, fp32_recovered, quant_bit=quant_bit, sig_analysis=False)
        compare_metrics = ["MSE", "MAE", "RMSE", "PSNR", "SNR"]
        x = np.arange(len(compare_metrics))
        width = 0.35

        int4_vals = [metrics[m] if metrics[m] != np.inf else 100 for m in compare_metrics]
        int8_vals = [ref_metrics[m] if ref_metrics[m] != np.inf else 100 for m in compare_metrics]

        axes[0, 3].bar(x - width / 2, int4_vals, width, label=f"INT4", alpha=0.7)
        axes[0, 3].bar(x + width / 2, int8_vals, width, label=f"INT8", alpha=0.7)
        axes[0, 3].set_title("Multi-Precision Metric Comparison")
        axes[0, 3].set_xlabel("Metric Type")
        axes[0, 3].set_ylabel("Value")
        axes[0, 3].set_xticks(x)
        axes[0, 3].set_xticklabels(compare_metrics)
        axes[0, 3].legend()
        axes[0, 3].grid(True, alpha=0.3)

    metrics = evaluate_precision(fp32_original, fp32_recovered, quant_bit=quant_bit, sig_analysis=False)
    basic_text = f"INT{quant_bit} Basic Metrics\n" + "\n".join([f"{k}: {v:.6f}" for k, v in metrics.items() if k != "quant_bit"])
    axes[1, 0].text(
        0.05,
        0.95,
        basic_text,
        transform=axes[1, 0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        fontsize=9,
    )
    axes[1, 0].set_title("Basic Precision Metrics")
    axes[1, 0].axis("off")

    if sig_analysis:
        sig_metrics = evaluate_precision(fp32_original, fp32_recovered, quant_bit=quant_bit, sig_analysis=True)
        sig_text = f"INT{quant_bit} Statistical Metrics\n" + "\n".join([f"{k}: {v:.6f}" for k, v in sig_metrics.items() if k.startswith("Error_")])
        axes[1, 1].text(
            0.05,
            0.95,
            sig_text,
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            fontsize=9,
        )
        axes[1, 1].set_title("Statistical Significance Metrics")
        axes[1, 1].axis("off")

    stats.probplot(error_flat, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title(f"INT{quant_bit} Error Q-Q Plot (Normality Test)")
    axes[1, 2].grid(True, alpha=0.3)

    if ref_metrics is not None:
        scale_str = np.round(scales[:3], 6) if isinstance(scales, np.ndarray) else np.round(scales, 6)
        zp_str = zero_points[:3] if isinstance(zero_points, np.ndarray) else zero_points
        ref_scale_str = np.round(ref_metrics.get("scales", 0)[:3], 6) if isinstance(ref_metrics.get("scales", 0), np.ndarray) else np.round(ref_metrics.get("scales", 0), 6)
        ref_zp_str = ref_metrics.get("zero_points", 0)[:3] if isinstance(ref_metrics.get("zero_points", 0), np.ndarray) else ref_metrics.get("zero_points", 0)

        param_text = f"Quantization Params Comparison\n"
        param_text += f"INT{quant_bit}: scale={scale_str}, zero_point={zp_str}\n"
        param_text += f"INT{ref_metrics['quant_bit']}: scale={ref_scale_str}, zero_point={ref_zp_str}"
        axes[1, 3].text(
            0.05,
            0.95,
            param_text,
            transform=axes[1, 3].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            fontsize=9,
        )
        axes[1, 3].set_title("Quantization Parameters Comparison")
        axes[1, 3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization report saved to: {save_path}")
    else:
        plt.show()
    plt.close()
