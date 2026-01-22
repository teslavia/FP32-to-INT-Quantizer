import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from typing import Tuple, Optional, Union, List, Dict, Iterable
import warnings
from scipy.stats import entropy as scipy_entropy
from scipy import stats

from fp32_to_int_quantizer.core.calibration import CalibrationStrategy

plt.rcParams.update(plt.rcParamsDefault)


class FP32ToLowBitQuantizer:
    def __init__(
        self,
        quant_bit: int = 8,
        quant_mode: str = "symmetric",
        quant_level: str = "per_tensor",
        calib_mode: Optional[str] = None,
        calib_percentile: float = 99.9,
        entropy_bins: int = 2048,
        int_min: Optional[int] = None,
        int_max: Optional[int] = None,
        seed: int = 42,
    ):
        if quant_bit not in [4, 8]:
            raise ValueError(f"Only support 4/8-bit quantization, current input: {quant_bit}")

        self.quant_bit = quant_bit

        if int_min is not None and int_max is not None:
            self.int_min = int_min
            self.int_max = int_max
        else:
            if self.quant_bit == 8:
                self.int_min = -128
                self.int_max = 127
            else:
                self.int_min = -8
                self.int_max = 7

        self.quant_mode = quant_mode
        self.quant_level = quant_level
        self.calib_mode = calib_mode
        self.calib_percentile = calib_percentile
        self.entropy_bins = entropy_bins
        self.seed = seed
        np.random.seed(seed)

        self.scales: Union[float, np.ndarray] = 1.0
        self.zero_points: Union[int, np.ndarray] = 0
        self._params_computed = False

        self.calibrator = CalibrationStrategy(
            quant_bit=quant_bit,
            quant_level=quant_level,
            calib_mode=calib_mode,
            calib_percentile=calib_percentile,
            entropy_bins=entropy_bins,
            int_min=int_min,
            int_max=int_max,
        )

    def _validate_input(self, fp32_data: np.ndarray) -> None:
        if fp32_data.dtype != np.float32:
            raise TypeError(f"Input data must be np.float32, current: {fp32_data.dtype}")
        if np.any(np.isnan(fp32_data)) or np.any(np.isinf(fp32_data)):
            raise ValueError("Input data contains NaN/Inf, cannot quantize")
        if self.quant_level == "per_channel" and len(fp32_data.shape) < 2:
            raise ValueError("Per-channel quantization only supports 2D+ tensors")

    def _compute_quant_params(self, fp32_data: np.ndarray) -> None:
        calib_min, calib_max = self.calibrator.calibrate(fp32_data)

        if np.all(calib_min == calib_max):
            self.scales = 1.0 if isinstance(calib_min, float) else np.ones_like(calib_min, dtype=np.float32)
            self.zero_points = self.int_min if isinstance(calib_min, float) else np.full_like(calib_min, self.int_min, dtype=np.int32)
            self._params_computed = True
            return

        if self.quant_mode == "symmetric":
            if isinstance(calib_min, float):
                max_abs = max(abs(calib_min), abs(calib_max))
                self.scales = max_abs / self.int_max if max_abs != 0 else 1.0
                self.zero_points = 0
            else:
                max_abs = np.maximum(np.abs(calib_min), np.abs(calib_max))
                self.scales = max_abs / self.int_max
                self.scales[max_abs == 0] = 1.0
                self.zero_points = np.zeros_like(calib_min, dtype=np.int32)
        else:
            if isinstance(calib_min, float):
                self.scales = (calib_max - calib_min) / (self.int_max - self.int_min)
                self.zero_points = round(-calib_min / self.scales)
                self.zero_points = np.clip(self.zero_points, self.int_min, self.int_max)
            else:
                self.scales = (calib_max - calib_min) / (self.int_max - self.int_min)
                self.zero_points = np.round(-calib_min / self.scales).astype(np.int32)
                self.zero_points = np.clip(self.zero_points, self.int_min, self.int_max)

        self._params_computed = True

    def _quantize_single(self, fp32_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_input(fp32_data)
        original_shape = fp32_data.shape
        num_dim = len(original_shape)

        if num_dim == 4:
            n, c, h, w = original_shape
            fp32_data_2d = fp32_data.transpose(0, 2, 3, 1).reshape(-1, c)
        else:
            fp32_data_2d = fp32_data

        if not self._params_computed:
            self._compute_quant_params(fp32_data_2d)

        if self.quant_mode == "symmetric":
            quantized = np.round(fp32_data_2d / self.scales)
        else:
            quantized = np.round((fp32_data_2d / self.scales) + self.zero_points)

        quantized = np.clip(quantized, self.int_min, self.int_max).astype(np.int32)

        if self.quant_mode == "symmetric":
            fp32_recovered_2d = quantized.astype(np.float32) * self.scales
        else:
            fp32_recovered_2d = (quantized.astype(np.float32) - self.zero_points) * self.scales

        if num_dim == 4:
            quantized = quantized.reshape(n, h, w, c).transpose(0, 3, 1, 2)
            fp32_recovered = fp32_recovered_2d.reshape(n, h, w, c).transpose(0, 3, 1, 2)
        else:
            quantized = quantized.reshape(original_shape)
            fp32_recovered = fp32_recovered_2d.reshape(original_shape)

        if self.quant_bit == 4:
            quantized = quantized.astype(np.int8)

        return quantized, fp32_recovered

    def quantize(
        self, fp32_data: Union[np.ndarray, Iterable[np.ndarray]]
    ) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        self._params_computed = False

        if isinstance(fp32_data, np.ndarray):
            return self._quantize_single(fp32_data)
        elif isinstance(fp32_data, (list, tuple, Iterable)):
            for idx, data in enumerate(fp32_data):
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"The {idx+1}-th element in batch data is not np.ndarray")
            return [self._quantize_single(data) for data in fp32_data]
        else:
            raise TypeError("Input data must be np.ndarray or iterable containing np.ndarray")

    def evaluate_precision(
        self,
        fp32_original: np.ndarray,
        fp32_recovered: np.ndarray,
        sig_analysis: bool = True,
    ) -> dict:
        error = fp32_original - fp32_recovered
        squared_error = error ** 2
        absolute_error = np.abs(error)

        metrics = {
            "quant_bit": self.quant_bit,
            "MSE": np.mean(squared_error),
            "MAE": np.mean(absolute_error),
            "RMSE": np.sqrt(np.mean(squared_error)),
            "PSNR": 10 * np.log10((np.max(fp32_original) ** 2) / np.mean(squared_error)) if np.mean(squared_error) != 0 else np.inf,
            "SNR": 10 * np.log10(np.var(fp32_original) / np.var(error)) if np.var(error) != 0 else np.inf,
            "Max_Abs_Error": np.max(absolute_error),
            "Min_Abs_Error": np.min(absolute_error),
            "95th_Percentile_Error": np.percentile(absolute_error, 95),
        }

        if sig_analysis:
            metrics.update({
                "Error_Std": np.std(error),
                "Error_Var": np.var(error),
                "Error_Skewness": np.mean(((error - np.mean(error)) / np.std(error)) ** 3) if np.std(error) != 0 else 0.0,
                "Error_Kurtosis": np.mean(((error - np.mean(error)) / np.std(error)) ** 4) - 3 if np.std(error) != 0 else 0.0,
                "Error_Entropy": scipy_entropy(np.histogram(error, bins=100, density=True)[0] + 1e-10),
            })

        return metrics

    def visualize_analysis(
        self,
        fp32_original: np.ndarray,
        fp32_recovered: np.ndarray,
        save_path: Optional[str] = None,
        ref_metrics: Optional[dict] = None,
        sig_analysis: bool = True,
    ) -> None:
        original_flat = fp32_original.flatten()
        recovered_flat = fp32_recovered.flatten()
        error_flat = original_flat - recovered_flat

        n_cols = 3 if ref_metrics is None else 4
        fig, axes = plt.subplots(2, n_cols, figsize=(15 if ref_metrics is None else 20, 10))
        fig.suptitle(
            f"FP32→INT{self.quant_bit} Quantization Analysis Report" + (f" (vs INT{ref_metrics['quant_bit']})" if ref_metrics else ""),
            fontsize=16,
        )

        axes[0, 0].hist(original_flat, bins=50, alpha=0.7, label="Original FP32", color="blue")
        axes[0, 0].hist(recovered_flat, bins=50, alpha=0.7, label=f"INT{self.quant_bit} Dequantized", color="orange")
        axes[0, 0].set_title("Data Distribution Comparison")
        axes[0, 0].set_xlabel("Value")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(error_flat, bins=50, color="red", alpha=0.7)
        axes[0, 1].set_title(f"INT{self.quant_bit} Quantization Error Distribution")
        axes[0, 1].set_xlabel("Error (Original - Dequantized)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        if len(fp32_original.shape) == 4:
            error_2d = error_flat.reshape(fp32_original.shape)[0, 0, :, :]
        else:
            error_2d = error_flat.reshape(fp32_original.shape)
        im = axes[0, 2].imshow(error_2d, cmap="RdYlBu_r", aspect="auto")
        axes[0, 2].set_title(f"INT{self.quant_bit} Error Heatmap")
        axes[0, 2].set_xlabel("Dimension 2")
        axes[0, 2].set_ylabel("Dimension 1")
        plt.colorbar(im, ax=axes[0, 2], label="Error Value")

        if ref_metrics is not None:
            metrics = self.evaluate_precision(fp32_original, fp32_recovered, sig_analysis=False)
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

        metrics = self.evaluate_precision(fp32_original, fp32_recovered, sig_analysis=False)
        basic_text = f"INT{self.quant_bit} Basic Metrics\n" + "\n".join([f"{k}: {v:.6f}" for k, v in metrics.items() if k != "quant_bit"])
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
            sig_metrics = self.evaluate_precision(fp32_original, fp32_recovered, sig_analysis=True)
            sig_text = f"INT{self.quant_bit} Statistical Metrics\n" + "\n".join([f"{k}: {v:.6f}" for k, v in sig_metrics.items() if k.startswith("Error_")])
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
        axes[1, 2].set_title(f"INT{self.quant_bit} Error Q-Q Plot (Normality Test)")
        axes[1, 2].grid(True, alpha=0.3)

        if ref_metrics is not None:
            scale_str = np.round(self.scales[:3], 6) if isinstance(self.scales, np.ndarray) else np.round(self.scales, 6)
            zp_str = self.zero_points[:3] if isinstance(self.zero_points, np.ndarray) else self.zero_points
            ref_scale_str = np.round(ref_metrics.get("scales", 0)[:3], 6) if isinstance(ref_metrics.get("scales", 0), np.ndarray) else np.round(ref_metrics.get("scales", 0), 6)
            ref_zp_str = ref_metrics.get("zero_points", 0)[:3] if isinstance(ref_metrics.get("zero_points", 0), np.ndarray) else ref_metrics.get("zero_points", 0)

            param_text = f"Quantization Params Comparison\n"
            param_text += f"INT{self.quant_bit}: scale={scale_str}, zero_point={zp_str}\n"
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

    def save_quant_data(self, quant_data: Union[np.ndarray, List[np.ndarray]], save_path: str) -> None:
        if isinstance(quant_data, (list, tuple)):
            quant_data = np.concatenate([data.flatten() for data in quant_data])
        else:
            quant_data = quant_data.flatten()

        if self.quant_bit == 4:
            quant_data = np.clip(quant_data, self.int_min, self.int_max)
            quant_data_uint4 = quant_data + 8
            if len(quant_data_uint4) % 2 != 0:
                quant_data_uint4 = np.append(quant_data_uint4, 0)
            packed_data = (quant_data_uint4[::2] << 4) | quant_data_uint4[1::2]
            packed_data = packed_data.astype(np.uint8)
        else:
            packed_data = quant_data.astype(np.int8)

        with open(save_path, "wb") as f:
            f.write(packed_data.tobytes())
        print(f"Quantized data saved to: {save_path} (Size: {os.path.getsize(save_path)} bytes)")

    def load_quant_data(self, load_path: str, target_shape: Tuple[int, ...]) -> np.ndarray:
        with open(load_path, "rb") as f:
            data_bytes = f.read()

        if self.quant_bit == 4:
            packed_data = np.frombuffer(data_bytes, dtype=np.uint8)
            quant_data_uint4 = np.empty(len(packed_data) * 2, dtype=np.uint8)
            quant_data_uint4[::2] = (packed_data >> 4) & 0x0F
            quant_data_uint4[1::2] = packed_data & 0x0F
            quant_data = quant_data_uint4 - 8
            quant_data = quant_data[: np.prod(target_shape)]
        else:
            quant_data = np.frombuffer(data_bytes, dtype=np.int8)
            if len(quant_data) != np.prod(target_shape):
                raise ValueError(f"Data length {len(quant_data)} does not match target shape {target_shape}")

        quant_data = quant_data.reshape(target_shape)
        print(f"Quantized data loaded, shape: {quant_data.shape}")
        return quant_data

    def save_quant_params(self, save_path: str) -> None:
        quant_params = {
            "quant_bit": self.quant_bit,
            "quant_mode": self.quant_mode,
            "quant_level": self.quant_level,
            "int_min": self.int_min,
            "int_max": self.int_max,
            "calib_mode": self.calib_mode,
            "calib_percentile": self.calib_percentile,
            "entropy_bins": self.entropy_bins,
            "scales": self.scales,
            "zero_points": self.zero_points,
        }
        np.save(save_path, quant_params)
        print(f"Quantization parameters saved to: {save_path}")

    def load_quant_params(self, load_path: str) -> None:
        quant_params = np.load(load_path, allow_pickle=True).item()
        self.quant_bit = quant_params["quant_bit"]
        self.quant_mode = quant_params["quant_mode"]
        self.quant_level = quant_params["quant_level"]
        self.int_min = quant_params["int_min"]
        self.int_max = quant_params["int_max"]
        self.calib_mode = quant_params["calib_mode"]
        self.calib_percentile = quant_params["calib_percentile"]
        self.entropy_bins = quant_params["entropy_bins"]
        self.scales = quant_params["scales"]
        self.zero_points = quant_params["zero_points"]
        self._params_computed = True
        print(f"Quantization parameters loaded (Precision: INT{self.quant_bit})")

    # -------------------------- PyTorch Interface --------------------------
    def quantize_torch_tensor(self, torch_tensor):
        """Quantize PyTorch Tensor"""
        from fp32_to_int_quantizer.frameworks.torch import quantize_torch_tensor

        return quantize_torch_tensor(torch_tensor, self)

    def quantize_torch_model(self, model, dummy_input, device=None):
        """Quantize PyTorch Model Weights (INT4/INT8)"""
        from fp32_to_int_quantizer.frameworks.torch import quantize_torch_model

        return quantize_torch_model(model, dummy_input, self, device)

    # -------------------------- TensorFlow Interface --------------------------
    def quantize_tf_tensor(self, tf_tensor):
        """Quantize TensorFlow Tensor"""
        from fp32_to_int_quantizer.frameworks.tensorflow import quantize_tf_tensor

        return quantize_tf_tensor(tf_tensor, self)

    def quantize_tf_model(self, model, dummy_input):
        """Quantize TensorFlow Model Weights (INT4/INT8)"""
        from fp32_to_int_quantizer.frameworks.tensorflow import quantize_tf_model

        return quantize_tf_model(model, dummy_input, self)

